"""
FastAPI REST API and Web Interface Server for Semantic Code Intelligence Platform.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from semantic_code_intel.config import CodeIntelConfig, DEFAULT_CONFIG
from semantic_code_intel.generation.synthesizer import CodeSynthesizer
from semantic_code_intel.indexing.engine import HybridIndexer
from semantic_code_intel.retrieval.pipeline import HybridRetrievalPipeline

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Semantic Code Intelligence API",
    description="Local-first hybrid code search with FAISS, BM25, and Cross-Encoder reranking",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Module level configuration & pipeline references
config: CodeIntelConfig = DEFAULT_CONFIG
pipeline: Optional[HybridRetrievalPipeline] = None


def get_pipeline() -> HybridRetrievalPipeline:
    global pipeline, config
    if pipeline is None or pipeline.config != config:
        pipeline = HybridRetrievalPipeline(config)
    return pipeline


class SearchRequest(BaseModel):
    query: str = Field(..., description="Query string or question")
    top_k: int = Field(default=5, ge=1, le=50)
    mode: str = Field(default="hybrid", description="'hybrid', 'dense', or 'sparse'")
    rerank: bool = Field(default=True)
    use_reranker: Optional[bool] = Field(default=None)
    directory: Optional[str] = Field(default=None)


class SearchResultItem(BaseModel):
    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    symbol_name: Optional[str] = None
    symbol_type: Optional[str] = None
    score: float
    citation: str
    code: str
    content: Optional[str] = None
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rerank_score: Optional[float] = None


class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[SearchResultItem]
    latency_ms: Dict[str, float]
    index_status: str = "ready"


class SynthesizeRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    provider: str = Field(default="extractive")


class SynthesizeResponse(BaseModel):
    query: str
    answer: str
    citations: List[str]
    latency_ms: Dict[str, float]


class IndexRequest(BaseModel):
    target_dir: Optional[str] = Field(default=".")
    force: bool = Field(default=False)


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the single-page web UI dashboard."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>Semantic Code Intelligence Web UI</h1>", status_code=200)
    return FileResponse(str(index_path))


@app.get("/api/health")
async def health_check():
    """Health check endpoint confirming service and index readiness."""
    p = get_pipeline()
    indexed = p.is_indexed()
    return {
        "status": "ok",
        "service": "semantic-code-intelligence",
        "indexed": indexed
    }


@app.get("/api/stats")
async def get_stats():
    """Get statistics about the indexed repository and vector dimensions."""
    p = get_pipeline()
    if not p.is_indexed():
        return {
            "status": "not_indexed",
            "message": "Repository is not indexed yet.",
            "total_chunks": 0,
            "total_files": 0,
            "total_lines": 0
        }

    stats = p.metadata_store.get_stats()
    manifest = p.metadata_store.get_manifest_val("index_manifest", {})
    return {
        "status": "ready",
        **stats,
        "manifest": manifest,
        "index_dir": str(p.config.get_index_dir())
    }


@app.post("/api/index")
async def trigger_index(req: IndexRequest):
    """Index a codebase directory via API."""
    global pipeline, config
    target_dir = Path(req.target_dir or ".").resolve()
    config = CodeIntelConfig(project_root=target_dir)
    indexer = HybridIndexer(config)
    metrics = indexer.index_codebase(target_dir, force_reindex=req.force)
    pipeline = None
    return {
        "status": "indexed",
        "metrics": metrics
    }


@app.post("/api/search", response_model=SearchResponse)
async def search_code(req: SearchRequest):
    """Execute hybrid code search with line citations and sub-second latency breakdown."""
    p = get_pipeline()
    if not p.is_indexed():
        return SearchResponse(
            query=req.query,
            total_results=0,
            results=[],
            latency_ms={"total_ms": 0.0},
            index_status="not_indexed"
        )

    use_rerank = req.use_reranker if req.use_reranker is not None else req.rerank

    res = p.query(
        query_text=req.query,
        top_k=req.top_k,
        use_reranker=use_rerank,
        mode=req.mode
    )

    items = [
        SearchResultItem(
            chunk_id=r.chunk.chunk_id,
            file_path=r.chunk.file_path,
            start_line=r.chunk.start_line,
            end_line=r.chunk.end_line,
            language=r.chunk.language,
            symbol_name=r.chunk.symbol_name,
            symbol_type=r.chunk.symbol_type.value if r.chunk.symbol_type else None,
            score=round(r.score, 4),
            citation=r.chunk.citation,
            code=r.chunk.content,
            content=r.chunk.content,
            dense_score=round(r.dense_score, 4) if r.dense_score is not None else None,
            sparse_score=round(r.sparse_score, 4) if r.sparse_score is not None else None,
            rerank_score=round(r.rerank_score, 4) if r.rerank_score is not None else None,
        )
        for r in res.results
    ]

    latency_dict = res.latency.to_dict()
    latency_dict["total_ms"] = latency_dict.get("total_end_to_end_ms", 0.0)

    return SearchResponse(
        query=req.query,
        total_results=len(items),
        results=items,
        latency_ms=latency_dict,
        index_status="ready"
    )


@app.post("/api/synthesize", response_model=SynthesizeResponse)
@app.post("/api/ask", response_model=SynthesizeResponse)
async def ask_or_synthesize(req: SynthesizeRequest):
    """Synthesize cited code answer using extractive or local Ollama LLM backend."""
    p = get_pipeline()
    if not p.is_indexed():
        raise HTTPException(
            status_code=400,
            detail="Repository is not indexed yet. Please run `code-intel index` first."
        )

    res = p.query(query_text=req.query, top_k=req.top_k, use_reranker=True)
    synthesizer = CodeSynthesizer(provider=req.provider)
    answer = synthesizer.synthesize(req.query, res.results)

    latency_dict = res.latency.to_dict()
    latency_dict["total_ms"] = latency_dict.get("total_end_to_end_ms", 0.0)

    return SynthesizeResponse(
        query=req.query,
        answer=answer.answer,
        citations=answer.citations,
        latency_ms=latency_dict
    )
