"""
FastAPI REST API and Web Interface Server for Semantic Code Intelligence Platform.
Supports real-time SSE progress streaming, dynamic repository selection, indexing,
interactive call-graphs, streaming synthesis, multi-file diff patching, and background file watching.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
te from pydantic import BaseModel, Field

from semantic_code_intel.config import CodeIntelConfig, DEFAULT_CONFIG
from semantic_code_intel.generation.patcher import CodePatcher
from semantic_code_intel.generation.synthesizer import CodeSynthesizer
from semantic_code_intel.graph.symbol_graph import SymbolGraphEngine
from semantic_code_intel.indexing.engine import HybridIndexer
from semantic_code_intel.indexing.watcher import CodebaseWatcher
from semantic_code_intel.retrieval.pipeline import HybridRetrievalPipeline

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Semantic Code Intelligence API",
    description="Local-first hybrid code search with FAISS, BM25, Cross-Encoder reranking, AST graphs, and streaming synthesis",
    version="0.2.0"
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

_PIPELINES: Dict[str, HybridRetrievalPipeline] = {}
_ACTIVE_REPO_PATH: Path = Path.cwd()
_ACTIVE_INDEX_PATH: Optional[Path] = None
_WATCHER: Optional[CodebaseWatcher] = None

# Exported for tests
config: CodeIntelConfig = DEFAULT_CONFIG
pipeline: Optional[HybridRetrievalPipeline] = None

KNOWN_PRESETS = [
    {
        "name": "Current Platform Codebase (Python)",
        "path": ".",
        "index_dir": ".code_intel_index",
        "description": "Semantic Code Intelligence engine source code"
    },
    {
        "name": "Kubernetes client-go (Go — 406k LOC)",
        "path": "./oss_evaluation/client-go",
        "index_dir": "./oss_evaluation/k8s_index",
        "description": "Kubernetes official Go distributed systems client"
    },
    {
        "name": "FastAPI (Python — 112k LOC)",
        "path": "./oss_evaluation/fastapi",
        "index_dir": "./oss_evaluation/fastapi_index",
        "description": "FastAPI web framework core"
    },
    {
        "name": "Textualize Rich (Python — 67k LOC)",
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
        i_path = None

    return t_path, i_path


def get_pipeline(repo_path: Optional[str] = None, index_path: Optional[str] = None) -> HybridRetrievalPipeline:
    """Get or initialize a cached HybridRetrievalPipeline instance for a specific repository."""
    global _PIPELINES, _ACTIVE_REPO_PATH, _ACTIVE_INDEX_PATH, config, pipeline

    t_path, i_path = resolve_paths(repo_path, index_path)
    cache_key = f"{t_path}::{i_path}"

    if cache_key not in _PIPELINES:
        cfg = CodeIntelConfig(project_root=t_path, index_dir=i_path)
        _PIPELINES[cache_key] = HybridRetrievalPipeline(cfg)

    _ACTIVE_REPO_PATH = t_path
    _ACTIVE_INDEX_PATH = i_path
    config = _PIPELINES[cache_key].config
    pipeline = _PIPELINES[cache_key]
    return _PIPELINES[cache_key]


# Pydantic Schemas
class SearchRequest(BaseModel):
    query: str
    repo_path: Optional[str] = Field(default=None, description="Repository directory path")
    index_path: Optional[str] = Field(default=None, description="Custom index directory path")
    top_k: int = Field(default=5, ge=1, le=50)
    mode: str = Field(default="hybrid", description="'hybrid', 'dense', or 'sparse'")
    rerank: bool = Field(default=True)
    use_reranker: Optional[bool] = None


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
    content: str
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
    repo_path: Optional[str] = None
    index_path: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)
    provider: str = Field(default="local")


class SynthesizeResponse(BaseModel):
    query: str
    answer: str
    citations: List[str]
    latency_ms: Dict[str, float]


class PatchGenerateRequest(BaseModel):
    instruction: str
    repo_path: Optional[str] = None
    top_k: int = Field(default=3)


class PatchApplyRequest(BaseModel):
    diff: str
    repo_path: Optional[str] = None


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


@app.get("/api/index/stream")
async def stream_indexing(
    target_dir: str = Query(default=".", description="Target repo directory"),
    force: bool = Query(default=True, description="Force complete re-indexing")
):
    """Real-time SSE progress streaming endpoint for indexing animations."""
    t_path, i_path = resolve_paths(target_dir)

    if not t_path.exists():
        raise HTTPException(status_code=404, detail=f"Target path does not exist: {t_path}")

    event_q: queue.Queue = queue.Queue()

    def progress_handler(stage: str, current: int, total: int, message: str = "", percentage: float = 0.0):
        event_q.put({
            "stage": stage,
            "current": current,
            "total": total,
            "percentage": round(percentage, 1),
            "message": message
        })

    def run_indexer():
        try:
            cfg = CodeIntelConfig(project_root=t_path, index_dir=i_path)
            indexer = HybridIndexer(cfg)
            metrics = indexer.index_codebase(
                target_dir=t_path,
                force_reindex=force,
                progress_callback=progress_handler
            )
            cache_key = f"{t_path}::{cfg.get_index_dir()}"
            if cache_key in _PIPELINES:
                del _PIPELINES[cache_key]

            t_files = getattr(metrics, 'total_files', None) if not isinstance(metrics, dict) else metrics.get('total_files', 0)
            t_lines = getattr(metrics, 'total_lines', None) if not isinstance(metrics, dict) else metrics.get('total_lines', 0)
            t_chunks = getattr(metrics, 'total_chunks', None) if not isinstance(metrics, dict) else metrics.get('total_chunks', 0)
            t_time = getattr(metrics, 'indexing_time_seconds', None) if not isinstance(metrics, dict) else metrics.get('indexing_time_seconds', 0.0)

            event_q.put({
                "stage": "done",
                "percentage": 100.0,
                "message": f"Successfully indexed {t_files} files ({t_lines:,} LOC).",
                "metrics": {
                    "total_files": t_files,
                    "total_lines": t_lines,
                    "total_chunks": t_chunks,
                    "indexing_time_seconds": t_time
                }
            })
        except Exception as e:
            logger.exception("Indexing failed in background thread")
            event_q.put({
                "stage": "error",
                "message": f"Indexing error: {str(e)}",
                "percentage": 0.0
            })
        finally:
            event_q.put(None)

    threading.Thread(target=run_indexer, daemon=True).start()

    async def event_generator():
        while True:
            await asyncio.sleep(0.05)
            while not event_q.empty():
                item = event_q.get()
                if item is None:
                    return
                yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/index")
async def trigger_index(req: IndexRequest):
    """Synchronous indexing fallback endpoint."""
    t_path, i_path = resolve_paths(req.target_dir, req.index_dir)

    if not t_path.exists():
        raise HTTPException(status_code=404, detail=f"Target directory not found: {t_path}")

    cfg = CodeIntelConfig(project_root=t_path, index_dir=i_path)
    indexer = HybridIndexer(cfg)
    metrics = indexer.index_codebase(t_path, force_reindex=req.force)

    cache_key = f"{t_path}::{cfg.get_index_dir()}"
    if cache_key in _PIPELINES:
        del _PIPELINES[cache_key]

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

    return SearchResponse(
        query=req.query,
        repo_path=str(t_path),
        total_results=len(items),
        results=items,
        latency_ms=res.latency.to_dict(),
        index_status="ready"
    )


# --- ADVANCED CAPABILITIES ENDPOINTS ---

@app.get("/api/graph")
async def get_symbol_graph(
    repo_path: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: int = 60
):
    """Retrieve the interactive symbol dependency and call graph."""
    t_path, i_path = resolve_paths(repo_path)
    cfg = CodeIntelConfig(project_root=t_path, index_dir=i_path)
    engine = SymbolGraphEngine(cfg)
    graph_data = engine.extract_graph(target_symbol=symbol, limit_nodes=limit)
    return graph_data


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
    synthesizer = CodeSynthesizer(p.config, provider=req.provider)
    answer = synthesizer.synthesize(req.query, res.results)

    return SynthesizeResponse(
        query=req.query,
        answer=answer.answer,
        citations=answer.citations,
        latency_ms=res.latency.to_dict()
    )


@app.post("/api/synthesize/stream")
async def stream_synthesis(req: SynthesizeRequest):
    """Stream AI / Extractive structured explanation for a query."""
    t_path, i_path = resolve_paths(req.repo_path, req.index_path)
    p = get_pipeline(str(t_path), str(i_path) if i_path else None)

    if not p.is_indexed():
        raise HTTPException(status_code=400, detail=f"Repository {t_path} is not indexed.")

    res = p.query(query_text=req.query, top_k=req.top_k, use_reranker=True)
    synthesizer = CodeSynthesizer(p.config)

    async def event_generator():
        async for chunk in synthesizer.stream_synthesis(req.query, res.results):
            yield f"data: {chunk}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/patch/generate")
async def generate_patch(req: PatchGenerateRequest):
    """Generate a multi-file unified diff based on the user's refactoring instruction."""
    t_path, i_path = resolve_paths(req.repo_path)
    p = get_pipeline(str(t_path), str(i_path) if i_path else None)

    if not p.is_indexed():
        raise HTTPException(status_code=400, detail=f"Repository {t_path} is not indexed.")

    res = p.query(query_text=req.instruction, top_k=req.top_k, use_reranker=True)
    patcher = CodePatcher(p.config)
    result = patcher.generate_refactoring_diff(req.instruction, res.results)
    return result


@app.post("/api/patch/apply")
async def apply_patch(req: PatchApplyRequest):
    """Safely apply a generated diff to the repository with rollback protection."""
    t_path, i_path = resolve_paths(req.repo_path)
    cfg = CodeIntelConfig(project_root=t_path, index_dir=i_path)
    patcher = CodePatcher(cfg)
    result = patcher.apply_patch(req.diff)
    return result


@app.get("/api/watcher/status")
async def get_watcher_status():
    """Check background incremental watcher status."""
    global _WATCHER
    return {
        "running": _WATCHER.is_running if _WATCHER else False,
        "watched_root": str(_WATCHER.config.project_root) if _WATCHER and _WATCHER.is_running else None
    }


@app.post("/api/watcher/toggle")
async def toggle_watcher(repo_path: Optional[str] = None):
    """Toggle background file watcher on/off."""
    global _WATCHER
    t_path, _ = resolve_paths(repo_path)

    if _WATCHER and _WATCHER.is_running:
        _WATCHER.stop()
        return {"running": False, "message": "Watcher stopped"}
    else:
        cfg = CodeIntelConfig(project_root=t_path)
        _WATCHER = CodebaseWatcher(config=cfg)
        _WATCHER.start()
        return {"running": True, "watched_root": str(t_path), "message": "Watcher active"}


@app.post("/api/git/commit/generate")
async def generate_git_commit(repo_path: Optional[str] = None, staged_only: bool = False):
    """Generate a conventional commit message based on local git diff."""
    from semantic_code_intel.git_intel.commit_generator import SemanticCommitGenerator
    t_path, _ = resolve_paths(repo_path)
    generator = SemanticCommitGenerator(repo_path=t_path)
    result = generator.generate_commit_message(staged_only=staged_only)
    return result


@app.get("/api/lsp/inspect")
async def lsp_inspect(
    repo_path: Optional[str] = None,
    symbol: Optional[str] = None,
    file_path: Optional[str] = None,
    line: int = 1
):
    """Inspect LSP definitions, references, and hover data for a symbol or location."""
    from semantic_code_intel.lsp.server import CodeIntelLSPServer
    t_path, i_path = resolve_paths(repo_path)
    cfg = CodeIntelConfig(project_root=t_path, index_dir=i_path)
    lsp_server = CodeIntelLSPServer(config=cfg)

    params = {
        "symbol": symbol,
        "textDocument": {"uri": f"file://{t_path}/{file_path}" if file_path else ""},
        "position": {"line": line - 1, "character": 0}
    }

    definitions = lsp_server._handle_definition(params)
    references = lsp_server._handle_references(params)
    hover = lsp_server._handle_hover(params)

    return {
        "symbol": symbol,
        "definitions": definitions,
        "references": references,
        "hover": hover
    }


class OpenFileRequest(BaseModel):
    file_path: str
    repo_path: Optional[str] = None
    line: Optional[int] = 1
    action: Optional[str] = "editor"  # "editor" or "finder"


@app.post("/api/open")
async def open_file(req: OpenFileRequest):
    """Open a file at a specific line in the default editor (Cursor, VS Code) or macOS Finder."""
    import shutil
    import subprocess
    t_path, _ = resolve_paths(req.repo_path)

    # Resolve target file path
    target = (Path(t_path) / req.file_path).resolve()
    if not target.exists():
        # Try absolute path fallback
        target = Path(req.file_path).resolve()

    if not target.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.file_path}")

    line_num = max(1, req.line or 1)

    try:
        if req.action == "finder":
            subprocess.run(["open", "-R", str(target)], check=False)
            return {"success": True, "message": f"Revealed {target.name} in Finder", "path": str(target)}
        else:
            line_spec = f"{target}:{line_num}"
            opened = False

            # 1. Check Cursor CLI
            if shutil.which("cursor"):
                res = subprocess.run(["cursor", "-g", line_spec], capture_output=True, text=True, check=False)
                if res.returncode == 0:
                    opened = True

            # 2. Check VS Code CLI
            if not opened and shutil.which("code"):
                res = subprocess.run(["code", "-g", line_spec], capture_output=True, text=True, check=False)
                if res.returncode == 0:
                    opened = True

            # 3. Fallback: macOS system open
            if not opened:
                subprocess.run(["open", str(target)], check=False)

            return {
                "success": True,
                "message": f"Opened {target.name}:{line_num}",
                "path": str(target),
                "line": line_num
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


