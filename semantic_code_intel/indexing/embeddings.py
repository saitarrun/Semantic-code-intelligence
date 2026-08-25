"""
Dense embedding generation with native Transformers, mean-pooling, and hardware acceleration.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from semantic_code_intel.config import EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Manages dense vector embedding generation with batching and device acceleration."""

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.model_name = self.config.model_name
        self.device = self.config.device
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModel] = None

    def _ensure_loaded(self) -> None:
        """Lazy load tokenizer and model."""
        if self._tokenizer is None or self._model is None:
            logger.info(f"Loading embedding model '{self.model_name}' on device '{self.device}'...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self._model.eval()

    def _mean_pooling(self, model_output, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling to extract sentence-level embeddings from token embeddings."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress_bar: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> np.ndarray:
        """
        Encode a list of text strings into normalized float32 numpy embeddings.
        Shape: (len(texts), embedding_dim)
        """
        if not texts:
            return np.empty((0, self.config.embedding_dim), dtype=np.float32)

        self._ensure_loaded()
        batch_sz = batch_size or self.config.batch_size
        all_embeddings: List[np.ndarray] = []
        total_texts = len(texts)

        for i in range(0, total_texts, batch_sz):
            batch_texts = texts[i : i + batch_sz]
            encoded = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                out = self._model(**encoded)
                pooled = self._mean_pooling(out, encoded["attention_mask"])
                if self.config.normalize_embeddings:
                    pooled = F.normalize(pooled, p=2, dim=1)

            all_embeddings.append(pooled.cpu().numpy().astype(np.float32))

            if progress_callback:
                processed = min(i + batch_sz, total_texts)
                progress_callback(processed, total_texts)

        return np.vstack(all_embeddings)

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string into a normalized 1D float32 numpy vector.
        Shape: (embedding_dim,)
        """
        self._ensure_loaded()
        encoded = self._tokenizer(
            [query],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            out = self._model(**encoded)
            pooled = self._mean_pooling(out, encoded["attention_mask"])
            if self.config.normalize_embeddings:
                pooled = F.normalize(pooled, p=2, dim=1)

        return pooled.cpu().numpy()[0].astype(np.float32)

    @property
    def dimension(self) -> int:
        return self.config.embedding_dim
