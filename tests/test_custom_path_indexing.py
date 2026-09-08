"""Integration and unit tests for custom path indexing, endpoint flexibility, and incremental updates."""

import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
from semantic_code_intel.api.app import app
from semantic_code_intel.config import CodeIntelConfig
from semantic_code_intel.indexing.engine import HybridIndexer
from semantic_code_intel.retrieval.pipeline import HybridRetrievalPipeline


SAMPLE_MODULE_A = """
class OrderService:
    def __init__(self, db_client):
        self.db = db_client

    def create_order(self, customer_id: str, item_ids: list[str]) -> str:
        \"\"\"Create and persist a new customer order transaction.\"\"\"
        order_id = f"ord_{customer_id}_123"
        return order_id
"""

SAMPLE_MODULE_B = """
def calculate_tax(subtotal: float, state_code: str) -> float:
    \"\"\"Compute state sales tax for e-commerce checkout.\"\"\"
    rates = {"CA": 0.0725, "NY": 0.08875, "TX": 0.0625}
    return subtotal * rates.get(state_code, 0.05)
"""


def test_custom_path_indexing_and_search():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "custom_service"
        repo_dir.mkdir(parents=True)
        (repo_dir / "order.py").write_text(SAMPLE_MODULE_A, encoding="utf-8")
        (repo_dir / "tax.py").write_text(SAMPLE_MODULE_B, encoding="utf-8")

        cfg = CodeIntelConfig(project_root=repo_dir)
        indexer = HybridIndexer(cfg)
        metrics = indexer.index_codebase(target_dir=repo_dir, force_reindex=True)

        assert metrics["total_files"] == 2
        assert metrics["total_chunks"] >= 2
        assert indexer.is_indexed() is True

        pipeline = HybridRetrievalPipeline(cfg)
        res = pipeline.query("create new customer order transaction", top_k=2)
        assert len(res.results) > 0
        assert "order.py" in res.results[0].citation
        assert res.results[0].chunk.symbol_name in ("create_order", "OrderService")


def test_incremental_indexing_changes():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "incremental_repo"
        repo_dir.mkdir(parents=True)
        file1 = repo_dir / "order.py"
        file1.write_text(SAMPLE_MODULE_A, encoding="utf-8")

        cfg = CodeIntelConfig(project_root=repo_dir)
        indexer = HybridIndexer(cfg)
        
        # Initial indexing
        m1 = indexer.index_codebase(target_dir=repo_dir, force_reindex=True)
        assert m1["total_files"] == 1

        # Second indexing without changes -> Cache hit
        m2 = indexer.index_codebase(target_dir=repo_dir, force_reindex=False)
        assert m2["total_files"] == 1

        # Add a new file and re-index incrementally
        (repo_dir / "tax.py").write_text(SAMPLE_MODULE_B, encoding="utf-8")
        m3 = indexer.index_codebase(target_dir=repo_dir, force_reindex=False)
        assert m3["total_files"] == 2


def test_api_index_flexible_payloads():
    client = TestClient(app)
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "api_test_repo"
        repo_dir.mkdir(parents=True)
        (repo_dir / "order.py").write_text(SAMPLE_MODULE_A, encoding="utf-8")

        # 1. Test POST /api/index with JSON body using target_dir
        res1 = client.post("/api/index", json={"target_dir": str(repo_dir), "force": True})
        assert res1.status_code == 200
        data1 = res1.json()
        assert data1["status"] == "indexed"
        assert data1["total_chunks"] >= 1

        # 2. Test POST /api/index with JSON body using repo_path
        res2 = client.post("/api/index", json={"repo_path": str(repo_dir), "force": False})
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["status"] == "indexed"

        # 3. Test POST /api/index with query parameter repo_path
        res3 = client.post(f"/api/index?repo_path={repo_dir}&force=true")
        assert res3.status_code == 200
        data3 = res3.json()
        assert data3["status"] == "indexed"

        # 4. Test GET /api/status and GET /api/stats
        res_status = client.get(f"/api/status?repo_path={repo_dir}")
        assert res_status.status_code == 200
        status_data = res_status.json()
        assert status_data["indexed"] is True
        assert status_data["total_files"] >= 1

        res_stats = client.get(f"/api/stats?target_dir={repo_dir}")
        assert res_stats.status_code == 200
        stats_data = res_stats.json()
        assert stats_data["indexed"] is True

        # 5. Test search on this custom indexed repo
        res_search = client.post(
            "/api/search",
            json={
                "query": "OrderService create_order",
                "repo_path": str(repo_dir),
                "top_k": 2
            }
        )
        assert res_search.status_code == 200
        search_results = res_search.json()
        assert search_results["total_results"] > 0
        assert "order.py" in search_results["results"][0]["file_path"]

        # 6. Test GET /api/repos and GET /api/presets
        res_repos = client.get("/api/repos")
        assert res_repos.status_code == 200
        assert "presets" in res_repos.json()


def test_watcher_endpoints():
    client = TestClient(app)
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "watched_repo"
        repo_dir.mkdir(parents=True)

        res_start = client.post(f"/api/watcher/start?repo_path={repo_dir}")
        assert res_start.status_code == 200
        assert res_start.json()["running"] is True

        res_status = client.get("/api/watcher/status")
        assert res_status.status_code == 200
        assert res_status.json()["running"] is True

        res_stop = client.post("/api/watcher/stop")
        assert res_stop.status_code == 200
        assert res_stop.json()["running"] is False


def test_browse_folder_endpoint():
    client = TestClient(app)
    # Testing that /api/browse/folder is a valid endpoint
    # In headless/CI mode where dialog cancels or completes, returns status 200
    res = client.post("/api/browse/folder")
    assert res.status_code == 200
    assert "canceled" in res.json() or "path" in res.json()

