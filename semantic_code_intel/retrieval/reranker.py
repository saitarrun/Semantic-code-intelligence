"""
Cross-Encoder reranking module using native Transformers with batch classification.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from semantic_code_intel.config import RerankerConfig
from semantic_code_intel.indexing.embeddings import ModelUnavailableError

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks candidate code chunks using a fine-tuned Cross-Encoder model.
    """

    def __init__(self, config: Optional[RerankerConfig] = None):
        self.config = config or RerankerConfig()
        self.model_name = self.config.model_name
        self.device = self.config.device
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModelForSequenceClassification] = None

    def _ensure_loaded(self) -> None:
        """Lazy load tokenizer and model."""
        if self._tokenizer is None or self._model is None:
            logger.info(f"Loading Cross-Encoder reranker '{self.model_name}' on '{self.device}'...")
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, local_files_only=self.config.local_files_only
                )
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name, local_files_only=self.config.local_files_only
                ).to(self.device)
            except (OSError, ValueError) as exc:
                raise ModelUnavailableError(
                    f"Reranker model '{self.model_name}' is not available locally. Set "
                    "CODE_INTEL_ALLOW_MODEL_DOWNLOADS=1 and retry once, or pre-download it."
                ) from exc
            self._model.eval()

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

        self._ensure_loaded()
        pairs = [[query, text[:512]] for _, text in candidates]
        
        try:
            encoded = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                logits = self._model(**encoded).logits
                if logits.dim() > 1 and logits.shape[1] == 1:
                    scores = logits.squeeze(-1).tolist()
                elif logits.dim() == 1:
                    scores = logits.tolist()
                else:
                    scores = logits[:, 0].tolist()

            if isinstance(scores, float):
                scores = [scores]

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
