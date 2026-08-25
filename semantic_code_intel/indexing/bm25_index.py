"""
BM25 sparse lexical search engine with code-aware subword and identifier tokenization.
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from rank_bm25 import BM25Okapi
from semantic_code_intel.config import BM25Config

logger = logging.getLogger(__name__)


class CodeTokenizer:
    """
    Code-aware tokenizer that splits CamelCase, snake_case, screaming snake case,
    dot-notated scopes, and operator symbols while preserving original compound tokens.
    """

    # Matches words, identifiers, or tokens
    TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[^\s\w]")
    CAMEL_SPLIT_PATTERN = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        """Tokenize code or natural language query into decomposed sub-tokens."""
        if not text:
            return []

        tokens: List[str] = []
        raw_tokens = cls.TOKEN_PATTERN.findall(text)

        for token in raw_tokens:
            token_lower = token.lower()
            tokens.append(token_lower)

            # Split snake_case
            if "_" in token:
                parts = [p.lower() for p in token.split("_") if p]
                tokens.extend(parts)

            # Split CamelCase / PascalCase
            camel_parts = cls.CAMEL_SPLIT_PATTERN.split(token)
            if len(camel_parts) > 1:
                for part in camel_parts:
                    p_clean = part.strip("_").lower()
                    if p_clean and p_clean != token_lower:
                        tokens.append(p_clean)

        return tokens


class BM25SparseIndex:
    """
    BM25 lexical search index customized for source code repositories.
    """

    def __init__(self, config: Optional[BM25Config] = None):
        self.config = config or BM25Config()
        self.chunk_ids: List[str] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.tokenizer = CodeTokenizer

    @property
    def total_documents(self) -> int:
        return len(self.chunk_ids)

    def build_index(self, chunk_ids: List[str], documents: List[str]) -> None:
        """Build the BM25 index from a collection of chunk texts."""
        if len(chunk_ids) != len(documents):
            raise ValueError(f"Mismatch: {len(chunk_ids)} chunk_ids vs {len(documents)} documents")

        self.chunk_ids = list(chunk_ids)
        self.corpus_tokens = [self.tokenizer.tokenize(doc) for doc in documents]
        
        if self.corpus_tokens:
            self.bm25 = BM25Okapi(
                self.corpus_tokens,
                k1=self.config.k1,
                b=self.config.b,
                epsilon=self.config.epsilon
            )
        logger.info(f"Built BM25 index with {len(self.chunk_ids)} documents.")

    def search(self, query: str, top_k: int = 25) -> List[Tuple[str, float]]:
        """
        Search for top_k documents matching the query.
        Returns a list of (chunk_id, bm25_score) tuples sorted descending by score.
        """
        if not self.bm25 or self.total_documents == 0:
            return []

        query_tokens = self.tokenizer.tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices with non-zero score
        top_indices = sorted(
            range(len(scores)),
            key=lambda idx: scores[idx],
            reverse=True
        )[:top_k]

        results: List[Tuple[str, float]] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0.0:  # Only return documents with positive lexical match
                results.append((self.chunk_ids[idx], score))

        return results

    def save(self, index_dir: Path, filename: str = "bm25_index.pkl") -> None:
        """Serialize BM25 index and tokens to disk."""
        index_dir.mkdir(parents=True, exist_ok=True)
        target_file = index_dir / filename
        data = {
            "chunk_ids": self.chunk_ids,
            "corpus_tokens": self.corpus_tokens,
            "config": self.config.model_dump(),
        }
        with open(target_file, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"BM25 index saved to {target_file}")

    @classmethod
    def load(cls, index_dir: Path, filename: str = "bm25_index.pkl") -> BM25SparseIndex:
        """Deserialize BM25 index from disk."""
        target_file = index_dir / filename
        if not target_file.exists():
            raise FileNotFoundError(f"BM25 index file not found at {target_file}")

        with open(target_file, "rb") as f:
            data = pickle.load(f)

        config = BM25Config(**data.get("config", {}))
        instance = cls(config=config)
        instance.chunk_ids = data["chunk_ids"]
        instance.corpus_tokens = data["corpus_tokens"]

        if instance.corpus_tokens:
            instance.bm25 = BM25Okapi(
                instance.corpus_tokens,
                k1=instance.config.k1,
                b=instance.config.b,
                epsilon=instance.config.epsilon
            )

        logger.info(f"Loaded BM25 index with {instance.total_documents} documents from {target_file}")
        return instance
