"""
Cross-Encoder reranking module for deep query-document relevance scoring.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple
from sentence_transformers import CrossEncoder
from semantic_code_intel.config import RerankerConfig

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks candidate code chunks using a fine-tuned Cross-Encoder model.
    """

    def __init__(self, config: Optional[RerankerConfig] = None):
        self.config = config or RerankerConfig()
        self.model_name = self.config.model_name
        self.device = self.config.device
        self._model: Optional[CrossEncoder] = None

    @property
    def model(self) -> CrossEncoder:
        """Lazy-load the Cross-Encoder model."""
        if self._model is None:
            logger.info(f"Loading Cross-Encoder reranker '{self.model_name}' on '{self.device}'...")
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, str]],  # List of (chunk_id, chunk_text)
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Rerank candidate chunks against the query.
        Returns a list of (chunk_id, rerank_score) sorted descending.
        """
        if not candidates:
            return []

        pairs = [(query, text) for _, text in candidates]
        try:
            scores = self.model.predict(
                pairs,
                batch_size=self.config.batch_size,
                show_progress_bar=False
            )
            # Ensure float scores
            scored_candidates = [
                (candidates[idx][0], float(scores[idx]))
                for idx in range(len(candidates))
            ]
            # Sort descending by cross-encoder score
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            return scored_candidates[:top_k]
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}. Falling back to default order.")
            return [(cid, 1.0 / (idx + 1)) for idx, (cid, _) in enumerate(candidates[:top_k])]
