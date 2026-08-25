"""
Automated tests for Language Server Protocol (LSP) Bridge
and Semantic Git Commit Generator.
"""

import tempfile
from pathlib import Path
from semantic_code_intel.config import CodeIntelConfig
from semantic_code_intel.git_intel.commit_generator import SemanticCommitGenerator
from semantic_code_intel.indexing.engine import HybridIndexer
from semantic_code_intel.lsp.server import CodeIntelLSPServer


def test_lsp_server_capabilities():
    """Test standard LSP initialize, hover, definition, and completion methods."""
    sample_code = '''
class PaymentGateway:
    """Enterprise Stripe and PayPal payment gateway client."""
    def charge_card(self, token: str, amount_cents: int) -> bool:
        """Charge credit card token with specified amount in cents."""
        return amount_cents > 0

def process_checkout(order_id: str, total_cents: int) -> bool:
    gateway = PaymentGateway()
    return gateway.charge_card("tok_visa", total_cents)
'''
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        index_dir = Path(tmpdir) / "index"
        repo_dir.mkdir()
        pay_file = repo_dir / "gateway.py"
        pay_file.write_text(sample_code, encoding="utf-8")

        cfg = CodeIntelConfig(project_root=repo_dir, index_dir=index_dir)
        indexer = HybridIndexer(cfg)
        indexer.index_codebase(target_dir=repo_dir, force_reindex=True)

        server = CodeIntelLSPServer(config=cfg)

        # 1. Initialize
        init_res = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        caps = init_res["result"]["capabilities"]
        assert caps["definitionProvider"] is True
        assert caps["hoverProvider"] is True
        assert caps["referencesProvider"] is True

        # 2. Definition lookup for PaymentGateway
        def_res = server.handle_request({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "textDocument/definition",
            "params": {
                "symbol": "PaymentGateway",
                "textDocument": {"uri": f"file://{pay_file}"},
                "position": {"line": 1, "character": 6}
            }
        })
        assert len(def_res["result"]) >= 1
        assert "gateway.py" in def_res["result"][0]["uri"]

        # 3. Hover lookup
        hover_res = server.handle_request({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "textDocument/hover",
            "params": {
                "symbol": "charge_card",
                "textDocument": {"uri": f"file://{pay_file}"},
                "position": {"line": 3, "character": 8}
            }
        })
        assert hover_res["result"] is not None
        assert "charge_card" in hover_res["result"]["contents"]["value"]

        # 4. Symbol autocompletion
        comp_res = server.handle_request({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "textDocument/completion",
            "params": {"textDocument": {"uri": f"file://{pay_file}"}}
        })
        labels = [item["label"] for item in comp_res["result"]]
        assert "PaymentGateway" in labels or "charge_card" in labels


def test_semantic_commit_generator():
    """Test git diff analysis and conventional commit construction."""
    generator = SemanticCommitGenerator(repo_path=Path("."))
    result = generator.generate_commit_message()
    
    assert "title" in result
    assert "type" in result
    assert "full_message" in result
    # Guarantee no AI attribution lines
    assert "Co-Authored-By" not in result["full_message"]
    assert "Generated with" not in result["full_message"]
