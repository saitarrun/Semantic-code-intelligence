"""
End-to-end hybrid retrieval and reranking pipeline delivering sub-second code intelligence.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from semantic_code_intel.config import CodeIntelConfig, RetrievalConfig
from semantic_code_intel.indexing.bm25_index import BM25SparseIndex
from semantic_code_intel.indexing.embeddings import EmbeddingEngine
from semantic_code_intel.indexing.faiss_index import FAISSDenseIndex
from semantic_code_intel.indexing.metadata_store import MetadataStore
from semantic_code_intel.retrieval.citation import SearchResult
from semantic_code_intel.retrieval.reranker import CrossEncoderReranker
from semantic_code_intel.retrieval.rrf import ReciprocalRankFusion

logger = logging.getLogger(__name__)


class LatencyBreakdown(BaseModel):
    """Timing breakdown in milliseconds for each pipeline phase."""
    dense_search_ms: float = 0.0
    sparse_search_ms: float = 0.0
    rrf_fusion_ms: float = 0.0
    metadata_fetch_ms: float = 0.0
    reranker_ms: float = 0.0
    total_end_to_end_ms: float = 0.0
    
    # Aliases for backwards compatibility with tests and UI
    dense_ms: float = 0.0
    sparse_ms: float = 0.0
    fusion_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0

    def sync_aliases(self) -> None:
        self.dense_ms = self.dense_search_ms
        self.sparse_ms = self.sparse_search_ms
        self.fusion_ms = self.rrf_fusion_ms
        self.rerank_ms = self.reranker_ms
        self.total_ms = self.total_end_to_end_ms

    def to_dict(self) -> Dict[str, float]:
        return {
            "dense_search_ms": round(self.dense_search_ms, 2),
            "sparse_search_ms": round(self.sparse_search_ms, 2),
            "rrf_fusion_ms": round(self.rrf_fusion_ms, 3),
            "metadata_fetch_ms": round(self.metadata_fetch_ms, 2),
            "reranker_ms": round(self.reranker_ms, 2),
            "total_end_to_end_ms": round(self.total_end_to_end_ms, 2),
            "total_ms": round(self.total_end_to_end_ms, 2),
        }


class QueryResponse(BaseModel):
    """Complete response returned by the hybrid retrieval pipeline."""
    query: str
    results: List[SearchResult] = Field(default_factory=list)
    latency: LatencyBreakdown = Field(default_factory=LatencyBreakdown)
    total_candidates_considered: int = 0


class HybridRetrievalPipeline:
    """
    Combines FAISS dense vector search, BM25 sparse search, Reciprocal Rank Fusion,
    and Cross-Encoder reranking to find the most relevant code snippets with exact citations.
    """

    def __init__(self, config: Optional[CodeIntelConfig] = None):
        self.config = config or CodeIntelConfig()
        self.retrieval_config = self.config.retrieval
        self.index_dir = self.config.get_index_dir()

        # Components
        self.embedding_engine = EmbeddingEngine(self.config.embedding)
        self.metadata_store = MetadataStore(self.index_dir / self.config.storage.metadata_db_file)
        self._faiss_index: Optional[FAISSDenseIndex] = None
        self._bm25_index: Optional[BM25SparseIndex] = None
        self._reranker: Optional[CrossEncoderReranker] = None
        self.rrf = ReciprocalRankFusion(
            k=self.retrieval_config.rrf_k,
            dense_weight=self.retrieval_config.dense_weight,
            sparse_weight=self.retrieval_config.sparse_weight
        )

    @property
    def faiss_index(self) -> FAISSDenseIndex:
        if self._faiss_index is None:
            self._faiss_index = FAISSDenseIndex.load(
                self.index_dir, self.config.storage.faiss_index_file
            )
        return self._faiss_index

    @property
    def bm25_index(self) -> BM25SparseIndex:
        if self._bm25_index is None:
            self._bm25_index = BM25SparseIndex.load(
                self.index_dir, self.config.storage.bm25_index_file
            )
        return self._bm25_index

    @property
    def reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(self.config.reranker)
        return self._reranker

    def is_indexed(self) -> bool:
        """Check if required index files exist on disk."""
        faiss_p = self.index_dir / self.config.storage.faiss_index_file
        faiss_npy = self.index_dir / f"{self.config.storage.faiss_index_file}.npy"
        bm25_p = self.index_dir / self.config.storage.bm25_index_file
        meta_p = self.index_dir / self.config.storage.metadata_db_file
        return (faiss_p.exists() or faiss_npy.exists()) and bm25_p.exists() and meta_p.exists()

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        use_reranker: Optional[bool] = None,
        mode: str = "hybrid"  # "hybrid", "dense", "sparse"
    ) -> QueryResponse:
        """
        Execute search across index and rerank candidates.
        Delivers sub-second retrieval with exact line-level citations.
        """
        if not self.is_indexed():
            raise RuntimeError(f"Index not found in {self.index_dir}. Run indexing first.")

        t_start = time.perf_counter()
        k_final = top_k or self.retrieval_config.final_top_k
        should_rerank = self.retrieval_config.use_reranker if use_reranker is None else use_reranker

        dense_top_k = self.retrieval_config.dense_top_k
        sparse_top_k = self.retrieval_config.sparse_top_k

        dense_results: List[Tuple[str, float]] = []
        sparse_results: List[Tuple[str, float]] = []
        timing = LatencyBreakdown()

        # Step 1: Dense Vector Retrieval
        if mode in ["hybrid", "dense"]:
            t0 = time.perf_counter()
            query_vector = self.embedding_engine.encode_query(query_text)
            dense_results = self.faiss_index.search(query_vector, top_k=dense_top_k)
            timing.dense_search_ms = (time.perf_counter() - t0) * 1000.0

        # Step 2: BM25 Sparse Retrieval
        if mode in ["hybrid", "sparse"]:
            t0 = time.perf_counter()
            sparse_results = self.bm25_index.search(query_text, top_k=sparse_top_k)
            timing.sparse_search_ms = (time.perf_counter() - t0) * 1000.0

        # Step 3: Reciprocal Rank Fusion
        t0 = time.perf_counter()
        if mode == "hybrid":
            fused_candidates = self.rrf.fuse(dense_results, sparse_results, top_k=dense_top_k)
        elif mode == "dense":
            fused_candidates = self.rrf.fuse(dense_results, [], top_k=dense_top_k)
        else:
            fused_candidates = self.rrf.fuse([], sparse_results, top_k=sparse_top_k)
        timing.rrf_fusion_ms = (time.perf_counter() - t0) * 1000.0

        # Step 4: Metadata and Chunk Hydration from SQLite
        t0 = time.perf_counter()
        candidate_ids = [c.chunk_id for c in fused_candidates]
        chunks = self.metadata_store.get_chunks_by_ids(candidate_ids)
        chunk_map = {c.chunk_id: c for c in chunks}
        fused_map = {c.chunk_id: c for c in fused_candidates}
        timing.metadata_fetch_ms = (time.perf_counter() - t0) * 1000.0

        # Step 5: Cross-Encoder Reranking
        scored_results: List[SearchResult] = []
        if should_rerank and chunks:
            t0 = time.perf_counter()
            rerank_candidates = [
                (c.chunk_id, c.get_searchable_text())
                for c in chunks
            ]
            reranked_scores = self.reranker.rerank(
                query=query_text,
                candidates=rerank_candidates,
                top_k=k_final
            )
            timing.reranker_ms = (time.perf_counter() - t0) * 1000.0

            for chunk_id, rerank_score in reranked_scores:
                if chunk_id in chunk_map:
                    fc = fused_map.get(chunk_id)
                    scored_results.append(SearchResult.from_chunk(
                        chunk=chunk_map[chunk_id],
                        score=rerank_score,
                        dense_score=fc.dense_score if fc else None,
                        sparse_score=fc.sparse_score if fc else None,
                        rrf_score=fc.rrf_score if fc else None,
                        rerank_score=rerank_score
                    ))
        else:
            for c in fused_candidates[:k_final]:
                if c.chunk_id in chunk_map:
                    scored_results.append(SearchResult.from_chunk(
                        chunk=chunk_map[c.chunk_id],
                        score=c.rrf_score,
                        dense_score=c.dense_score,
                        sparse_score=c.sparse_score,
                        rrf_score=c.rrf_score,
                        rerank_score=None
                    ))

        timing.total_end_to_end_ms = (time.perf_counter() - t_start) * 1000.0
        timing.sync_aliases()

        return QueryResponse(
            query=query_text,
            results=scored_results,
            latency=timing,
            total_candidates_considered=len(candidate_ids)
        )
