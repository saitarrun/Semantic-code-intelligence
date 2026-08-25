"""
Model Context Protocol (MCP) Server for Semantic Code Intelligence.
Exposes JSON-RPC 2.0 tool endpoints for Cursor, Claude Desktop, Antigravity, and AI Agents.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from semantic_code_intel.config import CodeIntelConfig
from semantic_code_intel.graph.symbol_graph import SymbolGraphEngine
from semantic_code_intel.indexing.engine import HybridIndexer
from semantic_code_intel.retrieval.pipeline import HybridRetrievalPipeline


class CodeIntelMCPServer:
    """Standard MCP JSON-RPC 2.0 Server over stdio."""

    TOOLS = [
        {
            "name": "code_intel_search",
            "description": "Performs sub-second hybrid code search (dense FAISS + sparse BM25 + Cross-Encoder reranking) to retrieve exact line citations and code implementations.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query or symbol to search for (e.g. 'Where is LeaderElector lease renew loop?')"
                    },
                    "repo_path": {
                        "type": "string",
                        "description": "Path to the repository folder to search. Defaults to current directory.",
                        "default": "."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of relevant code chunks to return.",
                        "default": 5
                    },
                    "rerank": {
                        "type": "boolean",
                        "description": "Whether to apply Cross-Encoder sequence classification reranking.",
                        "default": True
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "code_intel_symbol_graph",
            "description": "Retrieves the symbol dependency and call graph for a function, class, or file.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Symbol name to center the call graph around (optional)."
                    },
                    "repo_path": {
                        "type": "string",
                        "description": "Path to the repository directory.",
                        "default": "."
                    }
                }
            }
        },
        {
            "name": "code_intel_index",
            "description": "Indexes or re-indexes a local repository codebase into FAISS and BM25.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "target_dir": {
                        "type": "string",
                        "description": "Target folder path to index.",
                        "default": "."
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Whether to force a clean re-index from scratch.",
                        "default": False
                    }
                },
                "required": ["target_dir"]
            }
        }
    ]

    def __init__(self):
        self.config = CodeIntelConfig()

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": self.TOOLS}
            }

        elif method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})
            return self._dispatch_tool(req_id, tool_name, tool_args)

        elif method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "semantic-code-intelligence",
                        "version": "0.1.0"
                    }
                }
            }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method '{method}' not found"}
        }

    def _dispatch_tool(self, req_id: Any, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if name == "code_intel_search":
                repo = args.get("repo_path", ".")
                query = args.get("query", "")
                top_k = args.get("top_k", 5)
                rerank = args.get("rerank", True)

                cfg = CodeIntelConfig(project_root=Path(repo).resolve())
                pipeline = HybridRetrievalPipeline(cfg)
                resp = pipeline.search(query=query, top_k=top_k, rerank=rerank)

                formatted = [
                    {
                        "citation": r.citation,
                        "file_path": r.file_path,
                        "lines": f"{r.start_line}-{r.end_line}",
                        "symbol": r.symbol_name,
                        "score": round(r.score, 4),
                        "code": r.code
                    }
                    for r in resp.results
                ]

                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {"type": "text", "text": json.dumps(formatted, indent=2)}
                        ]
                    }
                }

            elif name == "code_intel_symbol_graph":
                repo = args.get("repo_path", ".")
                symbol = args.get("symbol")
                cfg = CodeIntelConfig(project_root=Path(repo).resolve())
                graph_eng = SymbolGraphEngine(cfg)
                graph_data = graph_eng.extract_graph(target_symbol=symbol)

                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {"type": "text", "text": json.dumps(graph_data, indent=2)}
                        ]
                    }
                }

            elif name == "code_intel_index":
                target = args.get("target_dir", ".")
                force = args.get("force", False)
                cfg = CodeIntelConfig(project_root=Path(target).resolve())
                indexer = HybridIndexer(cfg)
                metrics = indexer.index_codebase(force=force)

                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Successfully indexed {metrics.total_files} files ({metrics.total_lines} LOC, {metrics.total_chunks} chunks) in {metrics.indexing_time_seconds:.2f}s."
                            }
                        ]
                    }
                }

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {name}"}
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": str(e)}
            }

    def run_stdio(self) -> None:
        """Reads JSON-RPC lines from standard input and writes responses to standard output."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                res = self.handle_request(req)
                sys.stdout.write(json.dumps(res) + "\n")
                sys.stdout.flush()
            except Exception as e:
                err = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {e}"}
                }
                sys.stdout.write(json.dumps(err) + "\n")
                sys.stdout.flush()


def run_mcp_server():
    server = CodeIntelMCPServer()
    server.run_stdio()


if __name__ == "__main__":
    run_mcp_server()
