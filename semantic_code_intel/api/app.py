"""
FastAPI REST API and Web Interface Server for Semantic Code Intelligence Platform.
Supports dynamic repository selection, indexing, and querying from the UI.
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

# Dynamic pipeline cache by repository/index path
_PIPELINES: Dict[str, HybridRetrievalPipeline] = {}
_ACTIVE_REPO_PATH: Path = Path.cwd()
_ACTIVE_INDEX_PATH: Optional[Path] = None

# Exported for tests
config: CodeIntelConfig = DEFAULT_CONFIG
pipeline: Optional[HybridRetrievalPipeline] = None

# Default known repositories
KNOWN_PRESETS = [
    {
        "name": "Current Platform Codebase (Python)",
        "path": ".",
        "index_dir": ".code_intel_index",
        "description": "Semantic Code Intelligence engine source code"
    },
    {
        "name": "Kubernetes client-go (Go - 406k LOC)",
        "path": "./oss_evaluation/client-go",
        "index_dir": "./oss_evaluation/k8s_index",
        "description": "Kubernetes official Go distributed systems client"
    },
    {
        "name": "FastAPI (Python - 112k LOC)",
        "path": "./oss_evaluation/fastapi",
        "index_dir": "./oss_evaluation/fastapi_index",
        "description": "FastAPI web framework core"
    },
    {
        "name": "Textualize Rich (Python - 67k LOC)",
        "path": "./oss_evaluation/rich",
        "index_dir": "./oss_evaluation/rich_index",
        "description": "Rich terminal UI and formatting engine"
    }
]


def resolve_paths(target_dir: Optional[str] = None, index_dir: Optional[str] = None) -> tuple[Path, Optional[Path]]:
    """Resolve absolute repository and index directory paths."""
    global _ACTIVE_REPO_PATH, _ACTIVE_INDEX_PATH, config
    
    if target_dir:
        t_path = Path(target_dir).expanduser().resolve()
    else:
        t_path = config.project_root.resolve()

    if index_dir:
        i_path = Path(index_dir).expanduser().resolve()
    elif config.index_dir is not None and target_dir is None:
        i_path = config.index_dir.resolve()
    elif target_dir:
        matched_preset = next((p for p in KNOWN_PRESETS if Path(p["path"]).resolve() == t_path), None)
        if matched_preset and matched_preset.get("index_dir"):
            i_path = Path(matched_preset["index_dir"]).resolve()
        else:
            i_path = (t_path / ".code_intel_index").resolve()
    else:
        i_path = (t_path / ".code_intel_index").resolve()

    return t_path, i_path


def get_pipeline(target_dir: Optional[str] = None, index_dir: Optional[str] = None) -> HybridRetrievalPipeline:
    """Retrieve or instantiate a cached HybridRetrievalPipeline for the target repository."""
    global pipeline, config
    if target_dir is None and index_dir is None:
        if config.project_root != Path.cwd() or config.index_dir is not None:
            if pipeline is None or pipeline.config != config:
                pipeline = HybridRetrievalPipeline(config)
            return pipeline

    t_path, i_path = resolve_paths(target_dir, index_dir)
    cache_key = f"{t_path}::{i_path}"
    
    if cache_key not in _PIPELINES:
        cfg = CodeIntelConfig(project_root=t_path, index_dir=i_path)
        _PIPELINES[cache_key] = HybridRetrievalPipeline(cfg)
    
    return _PIPELINES[cache_key]


# Pydantic Request/Response Models
class SearchRequest(BaseModel):
    query: str = Field(..., description="Query string or question")
    top_k: int = Field(default=5, ge=1, le=50)
    mode: str = Field(default="hybrid", description="'hybrid', 'dense', or 'sparse'")
    rerank: bool = Field(default=True)
    use_reranker: Optional[bool] = Field(default=None)
    repo_path: Optional[str] = Field(default=None)
    index_path: Optional[str] = Field(default=None)


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
    repo_path: str
    total_results: int
    results: List[SearchResultItem]
    latency_ms: Dict[str, float]
    index_status: str = "ready"


class SynthesizeRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    provider: str = Field(default="extractive")
    repo_path: Optional[str] = Field(default=None)
    index_path: Optional[str] = Field(default=None)


class SynthesizeResponse(BaseModel):
    query: str
    answer: str
    citations: List[str]
    latency_ms: Dict[str, float]


class IndexRequest(BaseModel):
    target_dir: str = Field(default=".", description="Local filesystem path to codebase")
    index_dir: Optional[str] = Field(default=None, description="Custom index directory")
    force: bool = Field(default=False)


# Routes
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the single-page web UI dashboard."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>Semantic Code Intelligence Web UI</h1>", status_code=200)
    return FileResponse(str(index_path))


@app.get("/api/presets")
async def get_presets():
    """List available repositories and presets."""
    resolved_presets = []
    for p in KNOWN_PRESETS:
        p_path = Path(p["path"]).resolve()
        i_path = Path(p["index_dir"]).resolve() if p.get("index_dir") else (p_path / ".code_intel_index").resolve()
        is_ready = (i_path / "metadata.sqlite3").exists() or (i_path / "vector_index.faiss.npy").exists()
        resolved_presets.append({
            **p,
            "resolved_path": str(p_path),
            "resolved_index": str(i_path),
            "is_indexed": is_ready,
            "exists": p_path.exists()
        })
    return {
        "active_repo": str(_ACTIVE_REPO_PATH.resolve()),
        "presets": resolved_presets
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    p = get_pipeline()
    return {
        "status": "ok",
        "service": "semantic-code-intelligence",
        "indexed": p.is_indexed()
    }


@app.get("/api/stats")
async def get_stats(repo_path: Optional[str] = None, index_path: Optional[str] = None):
    """Get statistics about the selected repository index."""
    t_path, i_path = resolve_paths(repo_path, index_path)
    p = get_pipeline(str(t_path), str(i_path) if i_path else None)
    
    if not p.is_indexed():
        return {
            "status": "not_indexed",
            "repo_path": str(t_path),
            "index_dir": str(p.index_dir),
            "total_chunks": 0,
            "total_files": 0,
            "total_lines": 0
        }

    stats = p.metadata_store.get_stats()
    manifest = p.metadata_store.get_manifest_val("index_manifest", {})
    return {
        "status": "ready",
        "repo_path": str(t_path),
        "index_dir": str(p.index_dir),
        **stats,
        "manifest": manifest
    }


@app.post("/api/index")
async def trigger_index(req: IndexRequest):
    """Index any codebase directory specified by path."""
    global _ACTIVE_REPO_PATH, _ACTIVE_INDEX_PATH
    t_path, i_path = resolve_paths(req.target_dir, req.index_dir)
    
    if not t_path.exists():
        raise HTTPException(status_code=404, detail=f"Target directory not found: {t_path}")

    cfg = CodeIntelConfig(project_root=t_path, index_dir=i_path)
    indexer = HybridIndexer(cfg)
    metrics = indexer.index_codebase(t_path, force_reindex=req.force)
    
    # Invalidate cache for this repo
    cache_key = f"{t_path}::{cfg.get_index_dir()}"
    if cache_key in _PIPELINES:
        del _PIPELINES[cache_key]

    _ACTIVE_REPO_PATH = t_path
    _ACTIVE_INDEX_PATH = cfg.get_index_dir()

    return {
        "status": "indexed",
        "repo_path": str(t_path),
        "index_dir": str(cfg.get_index_dir()),
        "metrics": metrics
    }


@app.post("/api/search", response_model=SearchResponse)
async def search_code(req: SearchRequest):
    """Execute hybrid search on the selected repository."""
    t_path, i_path = resolve_paths(req.repo_path, req.index_path)
    p = get_pipeline(str(t_path), str(i_path) if i_path else None)

    if not p.is_indexed():
        return SearchResponse(
            query=req.query,
            repo_path=str(t_path),
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

    return SearchResponse(
        query=req.query,
        repo_path=str(t_path),
        total_results=len(items),
        results=items,
        latency_ms=latency_dict,
        index_status="ready"
    )


@app.post("/api/synthesize", response_model=SynthesizeResponse)
@app.post("/api/ask", response_model=SynthesizeResponse)
async def ask_or_synthesize(req: SynthesizeRequest):
    """Synthesize cited code answer on the selected repository."""
    t_path, i_path = resolve_paths(req.repo_path, req.index_path)
    p = get_pipeline(str(t_path), str(i_path) if i_path else None)

    if not p.is_indexed():
        raise HTTPException(
            status_code=400,
            detail=f"Repository {t_path} is not indexed yet. Please run indexing first."
        )

    res = p.query(query_text=req.query, top_k=req.top_k, use_reranker=True)
    synthesizer = CodeSynthesizer(provider=req.provider)
    answer = synthesizer.synthesize(req.query, res.results)

    return SynthesizeResponse(
        query=req.query,
        answer=answer.answer,
        citations=answer.citations,
        latency_ms=res.latency.to_dict()
    )
