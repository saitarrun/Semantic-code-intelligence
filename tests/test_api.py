"""Integration tests for FastAPI REST API endpoints."""

import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
from semantic_code_intel.api.app import app, config, pipeline
from semantic_code_intel.indexing.engine import HybridIndexer


def test_api_endpoints():
    sample_code = '''
def process_refund(order_id: str, amount_cents: int) -> bool:
    """Refund order transaction through stripe payment processor."""
    return amount_cents > 0
'''
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        index_dir = Path(tmpdir) / "index"
        repo_dir.mkdir()
        (repo_dir / "payments.py").write_text(sample_code, encoding="utf-8")

        # Set API config
        config.project_root = repo_dir
        config.index_dir = index_dir

        # Index codebase
        indexer = HybridIndexer(config)
        indexer.index_codebase(target_dir=repo_dir, force_reindex=True)

        client = TestClient(app)

        # 1. Health check
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["indexed"] is True

        # 2. Stats endpoint
        res_stats = client.get("/api/stats")
        assert res_stats.status_code == 200
        stats_data = res_stats.json()
        assert stats_data["total_files"] >= 1
        assert stats_data["total_chunks"] >= 1

        # 3. Search endpoint
        res_search = client.post(
            "/api/search",
            json={
                "query": "refund order stripe payment",
                "top_k": 3,
                "mode": "hybrid",
                "use_reranker": True
            }
        )
        assert res_search.status_code == 200
        search_data = res_search.json()
        assert len(search_data["results"]) > 0
        assert "payments.py" in search_data["results"][0]["file_path"]
        assert search_data["latency_ms"]["total_ms"] > 0

        # 4. Synthesize endpoint
        res_synth = client.post(
            "/api/synthesize",
            json={
                "query": "how to process refund?",
                "top_k": 2,
                "provider": "extractive"
            }
        )
        assert res_synth.status_code == 200
        synth_data = res_synth.json()
        assert len(synth_data["citations"]) > 0
        assert "payments.py" in synth_data["citations"][0]
        assert "process_refund" in synth_data["answer"]
