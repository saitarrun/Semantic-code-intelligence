"""Model Context Protocol server for Semantic Code Intelligence."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from semantic_code_intel.config import CodeIntelConfig
from semantic_code_intel.graph.symbol_graph import SymbolGraphEngine
from semantic_code_intel.indexing.engine import HybridIndexer
from semantic_code_intel.retrieval.pipeline import HybridRetrievalPipeline


class CodeIntelMCPServer:
    """MCP JSON-RPC 2.0 server transported over newline-delimited stdio."""

    PROTOCOL_VERSION = "2024-11-05"
    TOOLS = [
        {
            "name": "code_intel_search",
            "description": "Search indexed code with hybrid semantic and lexical retrieval, exact citations, match reasons, and reliability scoring.",
            "inputSchema": {"type": "object", "properties": {
                "query": {"type": "string", "description": "Question, behavior, or symbol to find."},
                "repo_path": {"type": "string", "description": "Repository root; defaults to the server --dir."},
                "index_dir": {"type": "string", "description": "Optional index storage directory."},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
                "rerank": {"type": "boolean", "default": True},
                "mode": {"type": "string", "enum": ["hybrid", "dense", "sparse"], "default": "hybrid"},
            }, "required": ["query"]},
        },
        {
            "name": "code_intel_symbol_graph",
            "description": "Return repository dependency/call-graph data, optionally centered on a symbol.",
            "inputSchema": {"type": "object", "properties": {
                "symbol": {"type": "string", "description": "Optional function, class, or symbol name."},
                "repo_path": {"type": "string", "description": "Repository root; defaults to the server --dir."},
                "index_dir": {"type": "string", "description": "Optional index storage directory."},
            }},
        },
        {
            "name": "code_intel_index",
            "description": "Build or refresh the semantic and lexical indexes for a local repository.",
            "inputSchema": {"type": "object", "properties": {
                "repo_path": {"type": "string", "description": "Repository root; defaults to the server --dir."},
                "index_dir": {"type": "string", "description": "Optional index storage directory."},
                "force": {"type": "boolean", "default": False},
            }},
        },
        {
            "name": "code_intel_read_file",
            "description": "Read a bounded line range from a source file inside the configured repository.",
            "inputSchema": {"type": "object", "properties": {
                "path": {"type": "string", "description": "Repository-relative or absolute file path."},
                "repo_path": {"type": "string", "description": "Repository root; defaults to the server --dir."},
                "start_line": {"type": "integer", "minimum": 1, "default": 1},
                "end_line": {"type": "integer", "minimum": 1, "description": "Inclusive; at most 400 lines are returned."},
            }, "required": ["path"]},
        },
    ]

    def __init__(self, repo_path: Optional[Path] = None, index_dir: Optional[Path] = None):
        self.repo_path = (repo_path or Path.cwd()).expanduser().resolve()
        self.index_dir = index_dir.expanduser().resolve() if index_dir else None

    def _config(self, args: Dict[str, Any]) -> CodeIntelConfig:
        repo = Path(args.get("repo_path") or self.repo_path).expanduser().resolve()
        if not repo.is_dir():
            raise ValueError(f"Repository directory does not exist: {repo}")
        raw_index = args.get("index_dir")
        index = Path(raw_index).expanduser().resolve() if raw_index else self.index_dir
        return CodeIntelConfig(project_root=repo, index_dir=index)

    @staticmethod
    def _result(req_id: Any, payload: Any, *, is_error: bool = False) -> Dict[str, Any]:
        text = payload if isinstance(payload, str) else json.dumps(payload, indent=2)
        result: Dict[str, Any] = {"content": [{"type": "text", "text": text}]}
        if is_error:
            result["isError"] = True
        elif not isinstance(payload, str):
            result["structuredContent"] = payload
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    def handle_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        req_id, method = request.get("id"), request.get("method")
        params = request.get("params") or {}
        if method == "notifications/initialized" or (req_id is None and str(method).startswith("notifications/")):
            return None
        if method == "initialize":
            return {"jsonrpc": "2.0", "id": req_id, "result": {
                "protocolVersion": self.PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "semantic-code-intelligence", "version": "0.1.0"},
            }}
        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": self.TOOLS}}
        if method == "tools/call":
            return self._dispatch_tool(req_id, params.get("name", ""), params.get("arguments") or {})
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method '{method}' not found"}}

    def _dispatch_tool(self, req_id: Any, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            cfg = self._config(args)
            if name == "code_intel_search":
                query = str(args.get("query", "")).strip()
                if not query:
                    raise ValueError("query must not be empty")
                top_k = int(args.get("top_k", 5))
                if not 1 <= top_k <= 50:
                    raise ValueError("top_k must be between 1 and 50")
                response = HybridRetrievalPipeline(cfg).query(
                    query_text=query, top_k=top_k,
                    use_reranker=bool(args.get("rerank", True)),
                    mode=str(args.get("mode", "hybrid")),
                )
                return self._result(req_id, {
                    "query": response.query,
                    "reliability": response.reliability,
                    "reliability_score": response.reliability_score,
                    "reliability_reasons": response.reliability_reasons,
                    "total_candidates_considered": response.total_candidates_considered,
                    "results": [{
                        "citation": result.citation,
                        "file_path": result.chunk.file_path,
                        "absolute_path": result.chunk.absolute_path,
                        "start_line": result.chunk.start_line,
                        "end_line": result.chunk.end_line,
                        "symbol": result.chunk.symbol_name,
                        "symbol_type": result.chunk.symbol_type,
                        "language": result.chunk.language,
                        "score": round(result.score, 6),
                        "match_reasons": result.match_reasons,
                        "code": result.chunk.content,
                    } for result in response.results],
                })
            if name == "code_intel_symbol_graph":
                return self._result(req_id, SymbolGraphEngine(cfg).extract_graph(target_symbol=args.get("symbol")))
            if name == "code_intel_index":
                metrics = HybridIndexer(cfg).index_codebase(
                    target_dir=cfg.project_root, force_reindex=bool(args.get("force", False)))
                return self._result(req_id, {
                    "message": "Repository indexed successfully.", "repo_path": str(cfg.project_root),
                    "index_dir": str(cfg.get_index_dir()), **metrics,
                })
            if name == "code_intel_read_file":
                requested = Path(str(args.get("path", ""))).expanduser()
                target = requested.resolve() if requested.is_absolute() else (cfg.project_root / requested).resolve()
                try:
                    target.relative_to(cfg.project_root)
                except ValueError as exc:
                    raise ValueError("path must remain inside the configured repository") from exc
                if not target.is_file():
                    raise ValueError(f"File does not exist: {target}")
                start, end = int(args.get("start_line", 1)), int(args.get("end_line", int(args.get("start_line", 1)) + 399))
                if start < 1 or end < start:
                    raise ValueError("line range is invalid")
                end = min(end, start + 399)
                lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
                return self._result(req_id, {
                    "path": str(target.relative_to(cfg.project_root)), "start_line": start,
                    "end_line": min(end, len(lines)), "content": "\n".join(lines[start - 1:end]),
                })
            return self._result(req_id, f"Unknown tool: {name}", is_error=True)
        except Exception as exc:
            return self._result(req_id, str(exc), is_error=True)

    def run_stdio(self) -> None:
        """Read JSON-RPC lines from stdin and write responses only to stdout."""
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                response = self.handle_request(json.loads(line))
                if response is not None:
                    sys.stdout.write(json.dumps(response) + "\n")
                    sys.stdout.flush()
            except Exception as exc:
                sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {exc}"}}) + "\n")
                sys.stdout.flush()


def run_mcp_server(repo_path: Optional[Path] = None, index_dir: Optional[Path] = None) -> None:
    CodeIntelMCPServer(repo_path=repo_path, index_dir=index_dir).run_stdio()


if __name__ == "__main__":
    run_mcp_server()
