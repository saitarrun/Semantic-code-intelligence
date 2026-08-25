"""Unit tests for Hybrid Retrieval Pipeline, RRF Fusion, and Line Citations."""

import tempfile
from pathlib import Path
from semantic_code_intel.config import CodeIntelConfig
from semantic_code_intel.indexing.engine import HybridIndexer
from semantic_code_intel.retrieval.citation import CitationFormatter, SearchResult
from semantic_code_intel.retrieval.pipeline import HybridRetrievalPipeline
from semantic_code_intel.retrieval.rrf import ReciprocalRankFusion


def test_rrf_fusion():
    rrf = ReciprocalRankFusion(k=60, dense_weight=0.5, sparse_weight=0.5)

    dense_results = [("chunk_a", 0.95), ("chunk_b", 0.85), ("chunk_c", 0.75)]
    sparse_results = [("chunk_b", 12.5), ("chunk_a", 8.2), ("chunk_d", 5.1)]

    fused = rrf.fuse(dense_results, sparse_results, top_k=3)
    assert len(fused) == 3

    # chunk_a and chunk_b appear in both, so they should rank highest
    top_ids = [c.chunk_id for c in fused]
    assert "chunk_a" in top_ids[:2]
    assert "chunk_b" in top_ids[:2]


def test_end_to_end_pipeline():
    sample_repo_code = '''
"""User Authentication and Access Management."""

class TokenAuth:
    def __init__(self, key: str):
        self.key = key

    def validate_user_credentials(self, username: str, password_hash: str) -> bool:
        """Validate user username and cryptographic password hash against store."""
        if not username or not password_hash:
            return False
        return username == "admin" and len(password_hash) > 16

class DatabaseConnector:
    def connect_to_postgres(self, connection_url: str):
        """Establish pool connection to remote PostgreSQL database."""
        pass
'''
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        index_dir = Path(tmpdir) / "index"
        repo_dir.mkdir()
        (repo_dir / "auth.py").write_text(sample_repo_code, encoding="utf-8")

        cfg = CodeIntelConfig(project_root=repo_dir, index_dir=index_dir)
        
        # 1. Index
        indexer = HybridIndexer(cfg)
        metrics = indexer.index_codebase(target_dir=repo_dir, force_reindex=True)
        assert metrics["total_files"] == 1
        assert metrics["total_chunks"] >= 2

        # 2. Query Pipeline
        pipeline = HybridRetrievalPipeline(cfg)
        assert pipeline.is_indexed() is True

        # Warm up models to account for cold-start model weights loading
        pipeline.query("warmup search query", top_k=2, use_reranker=True)

        # Steady-state query evaluation
        res = pipeline.query("how to validate user credentials and passwords?", top_k=2, use_reranker=True)
        assert len(res.results) > 0

        top_result = res.results[0]
        assert "auth.py" in top_result.citation
        assert top_result.chunk.symbol_name == "validate_user_credentials"
        assert top_result.chunk.start_line >= 1
        assert top_result.chunk.end_line >= top_result.chunk.start_line

        # Verify citation format
        assert ":L" in top_result.citation

        # Verify sub-second steady-state retrieval (< 1000ms, typically < 100ms)
        assert res.latency.total_ms < 1000.0, f"Expected sub-second retrieval, got {res.latency.total_ms}ms"


def test_citation_formatter():
    from semantic_code_intel.parser.base import CodeChunk, SymbolType

    chunk = CodeChunk(
        chunk_id="test_id",
        file_path="src/security/crypto.py",
        absolute_path="/abs/src/security/crypto.py",
        language="python",
        symbol_name="hash_sha256",
        symbol_type=SymbolType.FUNCTION,
        start_line=15,
        end_line=25,
        content="def hash_sha256(data: bytes) -> str:\n    return hashlib.sha256(data).hexdigest()",
        context_header="def hash_sha256(data: bytes) -> str"
    )

    result = SearchResult.from_chunk(chunk, score=0.98)
    assert result.citation == "src/security/crypto.py:L15-L25"
    assert "src/security/crypto.py:L15-L25" in result.markdown_link

    llm_prompt_ctx = CitationFormatter.format_for_llm_prompt([result])
    assert "src/security/crypto.py:L15-L25" in llm_prompt_ctx
    assert "hash_sha256" in llm_prompt_ctx
