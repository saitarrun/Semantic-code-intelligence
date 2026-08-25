"""
Minimalist, High-Performance Language Server Protocol (LSP) Bridge.
Provides semantic jump-to-definition, find-references, hover documentation,
and symbol completion over standard JSON-RPC 2.0 stdio.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse

from semantic_code_intel.config import CodeIntelConfig
from semantic_code_intel.graph.symbol_graph import SymbolGraphEngine
from semantic_code_intel.indexing.metadata_store import MetadataStore

logger = logging.getLogger(__name__)


def uri_to_path(uri: str) -> Path:
    """Converts a file:// URI to a local filesystem Path."""
    parsed = urlparse(uri)
    path_str = unquote(parsed.path)
    return Path(path_str).resolve()


def path_to_uri(path: Path | str) -> str:
    """Converts a local Path to a file:// URI."""
    return Path(path).resolve().as_uri()


class CodeIntelLSPServer:
    """LSP Server handling textDocument/definition, hover, references, and completion."""

    def __init__(self, config: Optional[CodeIntelConfig] = None):
        self.config = config or CodeIntelConfig()
        self.metadata_store = MetadataStore(self.config.get_metadata_db_path())
        self.graph_engine = SymbolGraphEngine(self.config)
        self.is_running = False

    def handle_request(self, req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Dispatches an LSP JSON-RPC request and returns the response."""
        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "capabilities": {
                        "definitionProvider": True,
                        "referencesProvider": True,
                        "hoverProvider": True,
                        "completionProvider": {
                            "resolveProvider": False,
                            "triggerCharacters": [".", "_", ":", "(", ">"]
                        },
                        "textDocumentSync": 1
                    },
                    "serverInfo": {
                        "name": "semantic-code-intel-lsp",
                        "version": "0.2.0"
                    }
                }
            }

        elif method == "textDocument/definition":
            res = self._handle_definition(params)
            return {"jsonrpc": "2.0", "id": req_id, "result": res}

        elif method == "textDocument/references":
            res = self._handle_references(params)
            return {"jsonrpc": "2.0", "id": req_id, "result": res}

        elif method == "textDocument/hover":
            res = self._handle_hover(params)
            return {"jsonrpc": "2.0", "id": req_id, "result": res}

        elif method == "textDocument/completion":
            res = self._handle_completion(params)
            return {"jsonrpc": "2.0", "id": req_id, "result": res}

        elif method == "shutdown":
            return {"jsonrpc": "2.0", "id": req_id, "result": None}

        elif method == "exit":
            self.is_running = False
            return None

        # Return method not found if an ID is present
        if req_id is not None:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method {method} not supported"}
            }
        return None

    def _extract_symbol_at_position(self, params: Dict[str, Any]) -> Tuple[str, str, int]:
        """Extracts symbol query, target file path, and line number from params."""
        text_doc = params.get("textDocument", {})
        uri = text_doc.get("uri", "")
        pos = params.get("position", {})
        line_num = pos.get("line", 0) + 1  # 0-indexed to 1-indexed

        file_path = str(uri_to_path(uri)) if uri else ""
        symbol_name = params.get("symbol", "")
        return symbol_name, file_path, line_num

    def _handle_definition(self, params: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Finds definition location for a symbol or cursor position."""
        sym_name, f_path, line_num = self._extract_symbol_at_position(params)
        chunks = self.metadata_store.get_all_chunks()

        # If symbol name is not explicitly passed, find chunk at current file/line
        target_symbol = sym_name
        if not target_symbol and f_path:
            for ch in chunks:
                if str(Path(ch.file_path).resolve()) == str(Path(f_path).resolve()) and ch.start_line <= line_num <= ch.end_line:
                    target_symbol = ch.symbol_name
                    break

        if not target_symbol:
            return []

        results = []
        for ch in chunks:
            if ch.symbol_name and ch.symbol_name.lower() == target_symbol.lower():
                resolved_file = (self.config.project_root / ch.file_path).resolve()
                results.append({
                    "uri": path_to_uri(resolved_file),
                    "range": {
                        "start": {"line": max(0, ch.start_line - 1), "character": 0},
                        "end": {"line": max(0, ch.end_line - 1), "character": 80}
                    }
                })
        return results

    def _handle_references(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Finds all referencing locations across polyglot files."""
        sym_name, f_path, line_num = self._extract_symbol_at_position(params)
        chunks = self.metadata_store.get_all_chunks()

        target_symbol = sym_name
        if not target_symbol and f_path:
            for ch in chunks:
                if str(Path(ch.file_path).resolve()) == str(Path(f_path).resolve()) and ch.start_line <= line_num <= ch.end_line:
                    target_symbol = ch.symbol_name
                    break

        if not target_symbol:
            return []

        locations = []
        for ch in chunks:
            # Check if target_symbol is referenced in chunk content (excluding its own definition)
            if target_symbol in ch.content and ch.symbol_name != target_symbol:
                resolved_file = (self.config.project_root / ch.file_path).resolve()
                locations.append({
                    "uri": path_to_uri(resolved_file),
                    "range": {
                        "start": {"line": max(0, ch.start_line - 1), "character": 0},
                        "end": {"line": max(0, ch.end_line - 1), "character": 80}
                    }
                })
        return locations

    def _handle_hover(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Returns rich markdown hover information for a symbol."""
        sym_name, f_path, line_num = self._extract_symbol_at_position(params)
        chunks = self.metadata_store.get_all_chunks()

        for ch in chunks:
            match_pos = f_path and str(Path(ch.file_path).resolve()) == str(Path(f_path).resolve()) and ch.start_line <= line_num <= ch.end_line
            match_sym = sym_name and ch.symbol_name and ch.symbol_name.lower() == sym_name.lower()

            if match_pos or match_sym:
                sym_type = ch.symbol_type or "symbol"
                code_snippet = ch.content[:300].strip()
                doc = ch.docstring or "No documentation string provided."

                markdown = (
                    f"```python\n# {sym_type.upper()}: {ch.symbol_name or Path(ch.file_path).name}\n"
                    f"# Location: {ch.file_path}:{ch.start_line}-{ch.end_line}\n"
                    f"{code_snippet}\n```\n\n"
                    f"**Documentation:**\n{doc}\n"
                )
                return {
                    "contents": {
                        "kind": "markdown",
                        "value": markdown
                    }
                }
        return None

    def _handle_completion(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Returns fuzzy autocomplete suggestions from indexed symbols."""
        chunks = self.metadata_store.get_all_chunks()
        items = []
        seen = set()

        for ch in chunks:
            if not ch.symbol_name or ch.symbol_name.startswith("block_L"):
                continue
            if ch.symbol_name in seen:
                continue

            seen.add(ch.symbol_name)
            sym_kind = 3 if ch.symbol_type == "function" else (7 if ch.symbol_type == "class" else 6)
            items.append({
                "label": ch.symbol_name,
                "kind": sym_kind,
                "detail": f"{ch.symbol_type or 'symbol'} in {ch.file_path}:{ch.start_line}",
                "documentation": ch.docstring or f"Defined in {ch.file_path}"
            })
            if len(items) >= 50:
                break

        return items

    def run_stdio_server(self) -> None:
        """Standard LSP stdio server loop reading Content-Length headers."""
        self.is_running = True
        logger.info("Semantic Code Intel LSP server started on stdio.")

        while self.is_running:
            try:
                line = sys.stdin.readline()
                if not line:
                    break

                line = line.strip()
                if not line.startswith("Content-Length:"):
                    continue

                content_length = int(line.split(":")[1].strip())
                # Read empty newline separator
                sys.stdin.readline()

                # Read JSON-RPC body
                body_raw = sys.stdin.read(content_length)
                if not body_raw:
                    break

                req = json.loads(body_raw)
                res = self.handle_request(req)

                if res is not None:
                    res_json = json.dumps(res)
                    res_bytes = res_json.encode("utf-8")
                    header = f"Content-Length: {len(res_bytes)}\r\n\r\n"
                    sys.stdout.write(header + res_json)
                    sys.stdout.flush()

            except Exception as e:
                logger.error(f"LSP error: {e}", exc_info=True)
                break
