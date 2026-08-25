"""
FAISS dense vector index for fast sub-millisecond nearest neighbor search.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FAISSDenseIndex:
    """
    Dense vector search index backed by FAISS IndexFlatIP (Cosine Similarity).
    Maintains a bidirectional mapping between FAISS integer IDs and string chunk_ids.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index: faiss.Index = faiss.IndexFlatIP(dimension)
        self.id_to_chunk_id: Dict[int, str] = {}
        self.chunk_id_to_id: Dict[str, int] = {}
        self._next_id: int = 0

    @property
    def total_vectors(self) -> int:
        """Return total number of vectors in the index."""
        return self.index.ntotal

    def add_vectors(self, embeddings: np.ndarray, chunk_ids: List[str]) -> None:
        """
        Add a batch of embeddings and corresponding chunk_ids to the FAISS index.
        """
        if len(embeddings) == 0:
            return

        if len(embeddings) != len(chunk_ids):
            raise ValueError(f"Mismatch: {len(embeddings)} embeddings vs {len(chunk_ids)} chunk_ids")

        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}")

        # Ensure float32 contiguous array
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)

        start_id = self._next_id
        for idx, chunk_id in enumerate(chunk_ids):
            internal_id = start_id + idx
            self.id_to_chunk_id[internal_id] = chunk_id
            self.chunk_id_to_id[chunk_id] = internal_id

        self.index.add(vectors)
        self._next_id += len(chunk_ids)
        logger.debug(f"Added {len(chunk_ids)} vectors to FAISS index. Total: {self.total_vectors}")

    def search(self, query_vector: np.ndarray, top_k: int = 25) -> List[Tuple[str, float]]:
        """
        Search for top_k nearest neighbors given a query embedding vector.
        Returns a list of (chunk_id, similarity_score) tuples sorted descending by score.
        """
        if self.total_vectors == 0:
            return []

        # Ensure 2D float32 array
        if query_vector.ndim == 1:
            q_vec = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        else:
            q_vec = np.ascontiguousarray(query_vector, dtype=np.float32)

        actual_k = min(top_k, self.total_vectors)
        scores, indices = self.index.search(q_vec, actual_k)

        results: List[Tuple[str, float]] = []
        for internal_id, score in zip(indices[0], scores[0]):
            if internal_id != -1 and internal_id in self.id_to_chunk_id:
                chunk_id = self.id_to_chunk_id[internal_id]
                results.append((chunk_id, float(score)))

        return results

    def save(self, index_dir: Path, index_filename: str = "vector_index.faiss") -> None:
        """Persist FAISS index and ID mapping to disk."""
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss_path = index_dir / index_filename
        mapping_path = index_dir / f"{index_filename}.meta.json"

        faiss.write_index(self.index, str(faiss_path))
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump({
                "dimension": self.dimension,
                "next_id": self._next_id,
                "id_to_chunk_id": {str(k): v for k, v in self.id_to_chunk_id.items()}
            }, f, indent=2)

        logger.info(f"FAISS index ({self.total_vectors} vectors) saved to {faiss_path}")

    @classmethod
    def load(cls, index_dir: Path, index_filename: str = "vector_index.faiss") -> FAISSDenseIndex:
        """Load FAISS index and ID mapping from disk."""
        faiss_path = index_dir / index_filename
        mapping_path = index_dir / f"{index_filename}.meta.json"

        if not faiss_path.exists() or not mapping_path.exists():
            raise FileNotFoundError(f"FAISS index files not found in {index_dir}")

        with open(mapping_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        instance = cls(dimension=meta.get("dimension", 384))
        instance.index = faiss.read_index(str(faiss_path))
        instance._next_id = meta.get("next_id", instance.index.ntotal)
        instance.id_to_chunk_id = {int(k): v for k, v in meta.get("id_to_chunk_id", {}).items()}
        instance.chunk_id_to_id = {v: k for k, v in instance.id_to_chunk_id.items()}

        logger.info(f"Loaded FAISS index with {instance.total_vectors} vectors from {faiss_path}")
        return instance
