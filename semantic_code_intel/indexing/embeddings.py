"""
Dense embedding generation with SentenceTransformer and hardware acceleration (MPS/CUDA/CPU).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from semantic_code_intel.config import EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Manages dense vector embedding generation with batching and device acceleration."""

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.model_name = self.config.model_name
        self.device = self.config.device
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the sentence transformer model."""
        if self._model is None:
            logger.info(f"Loading embedding model '{self.model_name}' on device '{self.device}'...")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            # Ensure model is in eval mode
            self._model.eval()
        return self._model

    def encode_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Encode a list of text strings into normalized float32 numpy embeddings.
        Shape: (len(texts), embedding_dim)
        """
        if not texts:
            return np.empty((0, self.config.embedding_dim), dtype=np.float32)

        batch_sz = batch_size or self.config.batch_size
        embeddings = self.model.encode(
            texts,
            batch_size=batch_sz,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string into a normalized 1D float32 numpy vector.
        Shape: (embedding_dim,)
        """
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings
        )
        return embedding.astype(np.float32)

    @property
    def dimension(self) -> int:
        """Return the dimension of embeddings produced by the model."""
        return self.model.get_sentence_embedding_dimension() or self.config.embedding_dim
