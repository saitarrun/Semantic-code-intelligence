"""
Tests for Advanced Capabilities:
1. Model Context Protocol (MCP) JSON-RPC 2.0 Server
2. AST Symbol Call-Graph & Dependency Engine
3. Multi-File Unified Diff Patcher
4. Zero-Latency Background Incremental File Watcher
"""

import tempfile
from pathlib import Path
from semantic_code_intel.config import CodeIntelConfig
from semantic_code_intel.generation.patcher import CodePatcher
from semantic_code_intel.graph.symbol_graph import SymbolGraphEngine
from semantic_code_intel.indexing.engine import HybridIndexer
from semantic_code_intel.indexing.watcher import CodebaseWatcher
from semantic_code_intel.mcp.server import CodeIntelMCPServer
from semantic_code_intel.retrieval.pipeline import HybridRetrievalPipeline


def test_mcp_server_protocol():
    """Test standard MCP server tools listing and dispatching."""
    server = CodeIntelMCPServer()
    
    # 1. Initialize
    init_res = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert init_res["result"]["serverInfo"]["name"] == "semantic-code-intelligence"

    # 2. List tools
    tools_res = server.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    tool_names = [t["name"] for t in tools_res["result"]["tools"]]
    assert "code_intel_search" in tool_names
    assert "code_intel_symbol_graph" in tool_names
    assert "code_intel_index" in tool_names


def test_symbol_graph_and_diff_patcher():
    """Test AST symbol call-graph extraction and multi-file unified diff patching."""
    sample_code = """
def calculate_tax(subtotal: float) -> float:
    return subtotal * 0.08

def checkout(order_id: str, items: list) -> float:
    subtotal = sum(items)
    tax = calculate_tax(subtotal)
    return subtotal + tax
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        index_dir = Path(tmpdir) / "index"
        repo_dir.mkdir()
        (repo_dir / "checkout.py").write_text(sample_code, encoding="utf-8")

        cfg = CodeIntelConfig(project_root=repo_dir, index_dir=index_dir)
        indexer = HybridIndexer(cfg)
        indexer.index_codebase(target_dir=repo_dir, force_reindex=True)

        # 1. Test Symbol Graph Engine
        graph_eng = SymbolGraphEngine(cfg)
        graph = graph_eng.extract_graph(target_symbol="calculate_tax")
        assert len(graph["nodes"]) >= 1
        assert any("calculate_tax" in n.get("label", "") for n in graph["nodes"])

        # 2. Test Diff Patcher
        pipeline = HybridRetrievalPipeline(cfg)
        res = pipeline.query("checkout order calculation", top_k=1)
        
        patcher = CodePatcher(cfg)
        patch = patcher.generate_refactoring_diff("Add type hints to checkout", res.results)
        assert patch["success"] is True
        assert "diff --git" in patch["diff"] or "checkout.py" in patch["diff"]

        # 3. Test Patch Application
        apply_res = patcher.apply_patch(patch["diff"])
        assert apply_res["success"] is True


def test_incremental_watcher():
    """Test real-time filesystem watcher lifecycle."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        (repo_dir / "app.py").write_text("def hello(): pass\n", encoding="utf-8")

        cfg = CodeIntelConfig(project_root=repo_dir)
        events = []

        def callback(event_type: str, path: str):
            events.append((event_type, path))

        watcher = CodebaseWatcher(config=cfg, on_change_callback=callback)
        watcher.start()
        assert watcher.is_running is True
        watcher.stop()
        assert watcher.is_running is False
