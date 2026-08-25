"""
Symbol Call-Graph and Code Dependency Analysis Engine.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from semantic_code_intel.config import CodeIntelConfig
from semantic_code_intel.indexing.metadata_store import MetadataStore


class SymbolGraphEngine:
    """Builds and queries symbol dependency and call graphs across codebases."""

    def __init__(self, config: Optional[CodeIntelConfig] = None):
        self.config = config or CodeIntelConfig()
        self.metadata_store = MetadataStore(self.config.get_metadata_db_path())

    def extract_graph(
        self,
        target_symbol: Optional[str] = None,
        max_depth: int = 2,
        limit_nodes: int = 60
    ) -> Dict[str, Any]:
        """
        Extracts symbol definition, call, and import nodes and edges.
        If target_symbol is provided, builds a subgraph focused on that symbol.
        """
        chunks = self.metadata_store.get_all_chunks()
        if not chunks:
            return {"nodes": [], "edges": [], "metrics": {"total_nodes": 0, "total_edges": 0}}

        nodes_dict: Dict[str, Dict[str, Any]] = {}
        edges_list: List[Dict[str, Any]] = []
        edges_set: Set[Tuple[str, str, str]] = set()

        # 1. Map all defined symbols
        defined_symbols: Dict[str, Dict[str, Any]] = {}
        file_symbols: Dict[str, List[str]] = {}

        for ch in chunks:
            f_path = ch.file_path
            f_node_id = f"file:{f_path}"

            if f_node_id not in nodes_dict:
                nodes_dict[f_node_id] = {
                    "id": f_node_id,
                    "label": Path(f_path).name,
                    "full_path": f_path,
                    "type": "file",
                    "group": 1,
                    "lines": ch.end_line
                }

            if ch.symbol_name:
                sym_id = f"sym:{f_path}:{ch.symbol_name}"
                defined_symbols[ch.symbol_name] = {
                    "id": sym_id,
                    "name": ch.symbol_name,
                    "type": ch.symbol_type or "function",
                    "file_path": f_path,
                    "start_line": ch.start_line,
                    "end_line": ch.end_line,
                    "code": ch.content[:200]
                }
                nodes_dict[sym_id] = {
                    "id": sym_id,
                    "label": f"{ch.symbol_name}()",
                    "name": ch.symbol_name,
                    "type": ch.symbol_type or "symbol",
                    "file": f_path,
                    "start_line": ch.start_line,
                    "group": 2
                }

                # Edge: File contains symbol
                edge_key = (f_node_id, sym_id, "defines")
                if edge_key not in edges_set:
                    edges_set.add(edge_key)
                    edges_list.append({
                        "source": f_node_id,
                        "target": sym_id,
                        "relation": "defines"
                    })

                file_symbols.setdefault(f_path, []).append(ch.symbol_name)

        # 2. Extract call and reference links by scanning code text against known symbols
        for ch in chunks:
            caller_sym = ch.symbol_name
            caller_id = f"sym:{ch.file_path}:{caller_sym}" if caller_sym else f"file:{ch.file_path}"

            for target_name, target_info in defined_symbols.items():
                if target_name == caller_sym:
                    continue  # Skip self

                # Fast regex check: word boundary matching target symbol
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

        # 3. Filter if target_symbol is specified
        if target_symbol:
            target_symbol = target_symbol.strip()
            focus_ids: Set[str] = set()

            for n_id, n_data in nodes_dict.items():
                if target_symbol.lower() in n_data.get("label", "").lower() or \
                   target_symbol.lower() in n_data.get("name", "").lower():
                    focus_ids.add(n_id)

            if focus_ids:
                # Traverse neighbors up to max_depth
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

        # Global overview
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
