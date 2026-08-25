"""Unit tests for FAISS Dense Index and BM25 Sparse Index."""

import tempfile
from pathlib import Path
import numpy as np
from semantic_code_intel.indexing.bm25_index import BM25SparseIndex, CodeTokenizer
from semantic_code_intel.indexing.faiss_index import FAISSDenseIndex
from semantic_code_intel.parser.base import CodeChunk, SymbolType
from semantic_code_intel.retrieval.query_analysis import expand_code_query, exact_match_boost


def test_code_tokenizer():
    tokens = CodeTokenizer.tokenize("def calculateUserAccountBalance(user_id: int):")
    assert "calculateuseraccountbalance" in tokens
    assert "calculate" in tokens
    assert "user" in tokens
    assert "balance" in tokens
    assert "user_id" in tokens
    assert "id" in tokens


def test_query_expansion_and_exact_symbol_boost():
    expanded = expand_code_query("Where does the application render the UI?")
    assert "frontend" in expanded
    assert "implementation" in expanded

    chunk = CodeChunk(
        chunk_id="serve-ui",
        file_path="semantic_code_intel/api/app.py",
        absolute_path="/repo/semantic_code_intel/api/app.py",
        language="python",
        symbol_name="serve_ui",
        symbol_type=SymbolType.FUNCTION,
        start_line=1,
        end_line=3,
        content="def serve_ui(): pass",
        context_header="def serve_ui()",
    )
    boost, reasons = exact_match_boost("Where is serve_ui implemented?", chunk)
    assert boost >= 2.5
    assert "exact symbol" in reasons


def test_faiss_dense_index():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        dim = 128
        index = FAISSDenseIndex(dimension=dim)

        # Generate mock embeddings
        np.random.seed(42)
        vectors = np.random.randn(10, dim).astype(np.float32)
        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms

        chunk_ids = [f"chunk_{i}" for i in range(10)]
        index.add_vectors(vectors, chunk_ids)

        assert index.total_vectors == 10

        # Query with first vector
        results = index.search(vectors[0], top_k=3)
        assert len(results) == 3
        assert results[0][0] == "chunk_0"
        assert abs(results[0][1] - 1.0) < 1e-4

        # Test save and load
        index.save(tmp_path, "test_vector.faiss")
        loaded_index = FAISSDenseIndex.load(tmp_path, "test_vector.faiss")
        assert loaded_index.total_vectors == 10

        loaded_results = loaded_index.search(vectors[0], top_k=3)
        assert loaded_results[0][0] == "chunk_0"


def test_bm25_sparse_index():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        bm25 = BM25SparseIndex()

        docs = [
            "def authenticate_jwt_token(token: str): return True",
            "class PaymentProcessor: def charge_credit_card(): pass",
            "def query_database_pool(sql_query: str): return None"
        ]
        chunk_ids = ["auth_chunk", "pay_chunk", "db_chunk"]

        bm25.build_index(chunk_ids, docs)
        assert bm25.total_documents == 3

        # Test search
        res = bm25.search("authenticate jwt", top_k=2)
        assert len(res) > 0
        assert res[0][0] == "auth_chunk"

        res_db = bm25.search("sql_query database", top_k=2)
        assert len(res_db) > 0
        assert res_db[0][0] == "db_chunk"

        # Test save and load
        bm25.save(tmp_path, "test_bm25.pkl")
        loaded_bm25 = BM25SparseIndex.load(tmp_path, "test_bm25.pkl")
        assert loaded_bm25.total_documents == 3

        res_loaded = loaded_bm25.search("credit card", top_k=2)
        assert len(res_loaded) > 0
        assert res_loaded[0][0] == "pay_chunk"
