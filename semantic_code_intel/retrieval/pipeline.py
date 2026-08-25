"""
End-to-end hybrid retrieval and reranking pipeline delivering sub-second code intelligence.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
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
    dense_ms: float = 0.0
    sparse_ms: float = 0.0
    fusion_ms: float = 0.0
    rerank_ms: float = 0.0
    metadata_fetch_ms: float = 0.0
    total_ms: float = 0.0


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
        bm25_p = self.index_dir / self.config.storage.bm25_index_file
        meta_p = self.index_dir / self.config.storage.metadata_db_file
        return faiss_p.exists() and bm25_p.exists() and meta_p.exists()

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
            t_d0 = time.perf_counter()
            q_emb = self.embedding_engine.encode_query(query_text)
            dense_results = self.faiss_index.search(q_emb, top_k=dense_top_k)
            timing.dense_ms = (time.perf_counter() - t_d0) * 1000.0

        # Step 2: Sparse BM25 Retrieval
        if mode in ["hybrid", "sparse"]:
            t_s0 = time.perf_counter()
            sparse_results = self.bm25_index.search(query_text, top_k=sparse_top_k)
            timing.sparse_ms = (time.perf_counter() - t_s0) * 1000.0

        # Step 3: Reciprocal Rank Fusion
        t_f0 = time.perf_counter()
        if mode == "dense":
            fused_candidates = [
                (cid, score, score, None, score) for cid, score in dense_results
            ]
        elif mode == "sparse":
            fused_candidates = [
                (cid, score, None, score, score) for cid, score in sparse_results
            ]
        else:
            fused = self.rrf.fuse(
                dense_results,
                sparse_results,
                top_k=self.config.reranker.top_candidates_to_rerank
            )
            fused_candidates = [
                (c.chunk_id, c.rrf_score, c.dense_score, c.sparse_score, c.rrf_score)
                for c in fused
            ]
        timing.fusion_ms = (time.perf_counter() - t_f0) * 1000.0

        total_candidates = len(fused_candidates)
        if not fused_candidates:
            timing.total_ms = (time.perf_counter() - t_start) * 1000.0
            return QueryResponse(query=query_text, results=[], latency=timing, total_candidates_considered=0)

        # Step 4: Fetch metadata and candidate chunks from SQLite
        t_m0 = time.perf_counter()
        candidate_ids = [cid for cid, _, _, _, _ in fused_candidates]
        chunks = self.metadata_store.get_chunks_by_ids(candidate_ids)
        chunk_map = {c.chunk_id: c for c in chunks}
        score_map = {cid: (fused_s, d_s, s_s, rrf_s) for cid, fused_s, d_s, s_s, rrf_s in fused_candidates}
        timing.metadata_fetch_ms = (time.perf_counter() - t_m0) * 1000.0

        # Step 5: Cross-Encoder Reranking
        final_results: List[SearchResult] = []
        if should_rerank and len(candidate_ids) > 1:
            t_r0 = time.perf_counter()
            rerank_pairs = [
                (cid, chunk_map[cid].get_searchable_text())
                for cid in candidate_ids
                if cid in chunk_map
            ]
            reranked = self.reranker.rerank(query_text, rerank_pairs, top_k=k_final)
            timing.rerank_ms = (time.perf_counter() - t_r0) * 1000.0

            for cid, r_score in reranked:
                if cid in chunk_map:
                    chunk = chunk_map[cid]
                    fused_s, d_s, s_s, rrf_s = score_map.get(cid, (0.0, None, None, None))
                    final_results.append(
                        SearchResult.from_chunk(
                            chunk=chunk,
                            score=r_score,
                            dense_score=d_s,
                            sparse_score=s_s,
                            rrf_score=rrf_s,
                            rerank_score=r_score
                        )
                    )
        else:
            for cid in candidate_ids[:k_final]:
                if cid in chunk_map:
                    chunk = chunk_map[cid]
                    fused_s, d_s, s_s, rrf_s = score_map.get(cid, (0.0, None, None, None))
                    final_results.append(
                        SearchResult.from_chunk(
                            chunk=chunk,
                            score=fused_s,
                            dense_score=d_s,
                            sparse_score=s_s,
                            rrf_score=rrf_s,
                            rerank_score=None
                        )
                    )

        timing.total_ms = (time.perf_counter() - t_start) * 1000.0

        return QueryResponse(
            query=query_text,
            results=final_results,
            latency=timing,
            total_candidates_considered=total_candidates
        )
