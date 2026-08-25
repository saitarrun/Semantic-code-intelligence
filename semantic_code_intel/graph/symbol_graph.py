"""
Clean, High-Precision AST Symbol Call-Graph and Dependency Engine.
Excludes non-code docs and strips all symbols/emojis for a minimalist, professional graph.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from semantic_code_intel.config import CodeIntelConfig
from semantic_code_intel.indexing.metadata_store import MetadataStore

CODE_EXTENSIONS = {
    ".py", ".go", ".rs", ".ts", ".tsx", ".js", ".jsx",
    ".java", ".c", ".cpp", ".h", ".hpp", ".cs", ".swift",
    ".kt", ".scala", ".rb", ".php"
}

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE
)


def sanitize_label(name: str) -> str:
    """Removes emojis, symbols, and formatting noise from identifiers."""
    cleaned = EMOJI_PATTERN.sub("", name)
    cleaned = re.sub(r'[^a-zA-Z0-9_\.\-\(\)]', ' ', cleaned).strip()
    return re.sub(r'\s+', ' ', cleaned)


class SymbolGraphEngine:
    """Builds and queries symbol dependency and call graphs across codebases."""

    def __init__(self, config: Optional[CodeIntelConfig] = None):
        self.config = config or CodeIntelConfig()
        self.metadata_store = MetadataStore(self.config.get_metadata_db_path())

    def extract_graph(
        self,
        target_symbol: Optional[str] = None,
        max_depth: int = 2,
        limit_nodes: int = 40
    ) -> Dict[str, Any]:
        """
        Extracts clean symbol definition and call nodes.
        Filters out documentation/markdown files and anonymous blocks.
        """
        chunks = self.metadata_store.get_all_chunks()
        if not chunks:
            return {"nodes": [], "edges": [], "metrics": {"total_nodes": 0, "total_edges": 0}}

        nodes_dict: Dict[str, Dict[str, Any]] = {}
        edges_list: List[Dict[str, Any]] = []
        edges_set: Set[Tuple[str, str, str]] = set()

        # 1. Collect only valid code chunks from programming files
        valid_chunks = []
        for ch in chunks:
            ext = Path(ch.file_path).suffix.lower()
            if ext not in CODE_EXTENSIONS:
                continue
            if not ch.symbol_name:
                continue
            if ch.symbol_name.startswith("block_L"):
                continue  # Skip anonymous line chunks
            valid_chunks.append(ch)

        # 2. Register defined symbols and file parent nodes
        defined_symbols: Dict[str, Dict[str, Any]] = {}

        for ch in valid_chunks:
            f_path = ch.file_path
            f_node_id = f"file:{f_path}"

            clean_file_name = sanitize_label(Path(f_path).name)
            if f_node_id not in nodes_dict:
                nodes_dict[f_node_id] = {
                    "id": f_node_id,
                    "label": clean_file_name,
                    "full_path": f_path,
                    "type": "file",
                    "group": 1,
                    "lines": ch.end_line
                }

            clean_sym_name = sanitize_label(ch.symbol_name)
            if not clean_sym_name:
                continue

            sym_id = f"sym:{f_path}:{clean_sym_name}"
            sym_type = ch.symbol_type or "function"
            
            defined_symbols[clean_sym_name] = {
                "id": sym_id,
                "name": clean_sym_name,
                "type": sym_type,
                "file_path": f_path,
                "start_line": ch.start_line,
                "end_line": ch.end_line,
                "content": ch.content
            }

            nodes_dict[sym_id] = {
                "id": sym_id,
                "label": f"{clean_sym_name}()" if sym_type == "function" else clean_sym_name,
                "name": clean_sym_name,
                "type": sym_type,
                "file": clean_file_name,
                "full_path": f_path,
                "start_line": ch.start_line,
                "group": 2
            }

            # Edge: File -> defines -> Symbol
            edge_key = (f_node_id, sym_id, "defines")
            if edge_key not in edges_set:
                edges_set.add(edge_key)
                edges_list.append({
                    "source": f_node_id,
                    "target": sym_id,
                    "relation": "defines"
                })

        # 3. Detect symbol-to-symbol call links
        for ch in valid_chunks:
            caller_sym = sanitize_label(ch.symbol_name) if ch.symbol_name else None
            caller_id = f"sym:{ch.file_path}:{caller_sym}" if caller_sym else f"file:{ch.file_path}"

            for target_name, target_info in defined_symbols.items():
                if target_name == caller_sym:
                    continue  # Skip self call

                # Word-boundary regex matching
                pattern = r'\b' + re.escape(target_name) + r'\b'
                if re.search(pattern, ch.content):
                    target_id = target_info["id"]
                    edge_key = (caller_id, target_id, "calls")
                    if edge_key not in edges_set:
                        edges_set.add(edge_key)
                        edges_list.append({
                            "source": caller_id,
                            "target": target_id,
                            "relation": "calls"
                        })

        # 4. Target symbol filtering
        if target_symbol:
            target_symbol = sanitize_label(target_symbol.strip())
            focus_ids: Set[str] = set()

            for n_id, n_data in nodes_dict.items():
                if target_symbol.lower() in n_data.get("label", "").lower() or \
                   target_symbol.lower() in n_data.get("name", "").lower():
                    focus_ids.add(n_id)

            if focus_ids:
                connected_ids = set(focus_ids)
                current_frontier = set(focus_ids)

                for _ in range(max_depth):
                    next_frontier = set()
                    for edge in edges_list:
                        src, tgt = edge["source"], edge["target"]
                        if src in current_frontier and tgt not in connected_ids:
                            connected_ids.add(tgt)
                            next_frontier.add(tgt)
                        elif tgt in current_frontier and src not in connected_ids:
                            connected_ids.add(src)
                            next_frontier.add(src)
                    current_frontier = next_frontier

                filtered_nodes = [n for n_id, n in nodes_dict.items() if n_id in connected_ids]
                filtered_edges = [
                    e for e in edges_list
                    if e["source"] in connected_ids and e["target"] in connected_ids
                ]
                return {
                    "nodes": filtered_nodes[:limit_nodes],
                    "edges": filtered_edges[:limit_nodes * 2],
                    "metrics": {
                        "total_nodes": len(filtered_nodes[:limit_nodes]),
                        "total_edges": len(filtered_edges[:limit_nodes * 2]),
                        "focus_symbol": target_symbol
                    }
                }

        # Global top nodes
        all_nodes = list(nodes_dict.values())[:limit_nodes]
        active_ids = {n["id"] for n in all_nodes}
        all_edges = [
            e for e in edges_list
            if e["source"] in active_ids and e["target"] in active_ids
        ][:limit_nodes * 2]

        return {
            "nodes": all_nodes,
            "edges": all_edges,
            "metrics": {
                "total_nodes": len(all_nodes),
                "total_edges": len(all_edges)
            }
        }
