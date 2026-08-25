"""
FastAPI application serving REST search endpoints and the web frontend.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from semantic_code_intel.api.schemas import (
    ChunkResponse,
    IndexRequest,
    SearchApiResponse,
    SearchRequest,
    SynthesizeApiResponse,
    SynthesizeRequest,
)
from semantic_code_intel.config import CodeIntelConfig
from semantic_code_intel.generation.synthesizer import CodeSynthesizer
from semantic_code_intel.indexing.engine import HybridIndexer
from semantic_code_intel.retrieval.pipeline import HybridRetrievalPipeline

app = FastAPI(
    title="Semantic Code Intelligence API",
    description="Sub-second Hybrid Code Search & RAG with FAISS, BM25, and Cross-Encoder Reranking",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config = CodeIntelConfig()
pipeline: Optional[HybridRetrievalPipeline] = None
synthesizer = CodeSynthesizer()


def get_pipeline() -> HybridRetrievalPipeline:
    global pipeline
    if pipeline is None:
        pipeline = HybridRetrievalPipeline(config)
    return pipeline


@app.get("/api/health")
def health_check():
    p = get_pipeline()
    indexed = p.is_indexed()
    return {"status": "ok", "indexed": indexed, "index_dir": str(config.get_index_dir())}


@app.get("/api/stats")
def get_stats():
    p = get_pipeline()
    stats = p.metadata_store.get_stats()
    manifest = p.metadata_store.get_manifest_val("index_manifest", {})
    return {**stats, "manifest": manifest}


@app.post("/api/search", response_model=SearchApiResponse)
def search_code(req: SearchRequest):
    p = get_pipeline()
    if not p.is_indexed():
        raise HTTPException(status_code=400, detail="Repository is not indexed yet. Run index first.")

    res = p.query(
        query_text=req.query,
        top_k=req.top_k,
        use_reranker=req.use_reranker,
        mode=req.mode
    )

    chunk_responses: list[ChunkResponse] = []
    for r in res.results:
        c = r.chunk
        chunk_responses.append(
            ChunkResponse(
                chunk_id=c.chunk_id,
                file_path=c.file_path,
                absolute_path=c.absolute_path,
                language=c.language,
                symbol_name=c.symbol_name,
                symbol_type=c.symbol_type.value,
                parent_scope=c.parent_scope,
                start_line=c.start_line,
                end_line=c.end_line,
                content=c.content,
                context_header=c.context_header,
                docstring=c.docstring,
                citation=r.citation,
                markdown_link=r.markdown_link,
                score=r.score,
                dense_score=r.dense_score,
                sparse_score=r.sparse_score,
                rrf_score=r.rrf_score,
                rerank_score=r.rerank_score
            )
        )

    return SearchApiResponse(
        query=req.query,
        results=chunk_responses,
        latency_ms=res.latency.model_dump(),
        total_candidates=res.total_candidates_considered
    )


@app.post("/api/synthesize", response_model=SynthesizeApiResponse)
def synthesize_answer(req: SynthesizeRequest):
    p = get_pipeline()
    t0 = time.perf_counter()
    search_res = p.query(query_text=req.query, top_k=req.top_k, use_reranker=True)
    search_latency = (time.perf_counter() - t0) * 1000.0

    synth_res = synthesizer.synthesize(req.query, search_res.results)
    total_latency = (time.perf_counter() - t0) * 1000.0

    return SynthesizeApiResponse(
        query=req.query,
        answer=synth_res.answer,
        citations=synth_res.citations,
        model_used=synth_res.model_used,
        search_latency_ms=search_latency,
        total_latency_ms=total_latency
    )


@app.post("/api/index")
def index_repo(req: IndexRequest, background_tasks: BackgroundTasks):
    target_path = Path(req.target_dir) if req.target_dir else config.project_root
    
    def run_indexing():
        global pipeline
        indexer = HybridIndexer(config)
        indexer.index_codebase(target_dir=target_path, force_reindex=req.force_reindex)
        # Reset cached pipeline
        pipeline = None

    background_tasks.add_task(run_indexing)
    return {"message": "Indexing triggered in background", "target_dir": str(target_path)}


# Mount static directory for frontend
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return "<h1>Semantic Code Intelligence API Running</h1>"
