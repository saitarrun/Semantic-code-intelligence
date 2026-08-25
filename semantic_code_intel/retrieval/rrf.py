"""
Reciprocal Rank Fusion (RRF) and score combination for hybrid retrieval.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class FusionCandidate(BaseModel):
    """Container for a candidate retrieved from dense and/or sparse stages."""
    chunk_id: str
    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None
    sparse_rank: Optional[int] = None
    sparse_score: Optional[float] = None
    rrf_score: float = 0.0


class ReciprocalRankFusion:
    """
    Combines ranked lists from multiple retrieval mechanisms (e.g. FAISS Dense + BM25 Sparse).
    
    Formula:
        Score(d) = (w_dense / (k + rank_dense(d))) + (w_sparse / (k + rank_sparse(d)))
    """

    def __init__(self, k: int = 60, dense_weight: float = 0.5, sparse_weight: float = 0.5):
        self.k = k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def fuse(
        self,
        dense_results: List[Tuple[str, float]],
        sparse_results: List[Tuple[str, float]],
        top_k: int = 25
    ) -> List[FusionCandidate]:
        """
        Merge dense and sparse result lists.
        dense_results: list of (chunk_id, similarity_score) sorted descending.
        sparse_results: list of (chunk_id, bm25_score) sorted descending.
        """
        candidates: Dict[str, FusionCandidate] = {}

        # Process Dense results
        for rank, (chunk_id, score) in enumerate(dense_results, start=1):
            if chunk_id not in candidates:
                candidates[chunk_id] = FusionCandidate(chunk_id=chunk_id)
            cand = candidates[chunk_id]
            cand.dense_rank = rank
            cand.dense_score = float(score)
            cand.rrf_score += self.dense_weight * (1.0 / (self.k + rank))

        # Process Sparse results
        for rank, (chunk_id, score) in enumerate(sparse_results, start=1):
            if chunk_id not in candidates:
                candidates[chunk_id] = FusionCandidate(chunk_id=chunk_id)
            cand = candidates[chunk_id]
            cand.sparse_rank = rank
            cand.sparse_score = float(score)
            cand.rrf_score += self.sparse_weight * (1.0 / (self.k + rank))

        # Sort candidates descending by fused RRF score
        sorted_candidates = sorted(
            candidates.values(),
            key=lambda c: c.rrf_score,
            reverse=True
        )

        return sorted_candidates[:top_k]
