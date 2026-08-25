"""
Configuration settings for the Semantic Code Intelligence Platform.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
import torch


def detect_device() -> str:
    """Detect the fastest available hardware acceleration device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ParserConfig(BaseModel):
    """Configuration for code AST parsing and chunking."""
    min_chunk_lines: int = Field(default=3, description="Minimum lines to form a chunk")
    max_chunk_lines: int = Field(default=75, description="Maximum lines for a single chunk before splitting")
    chunk_overlap_lines: int = Field(default=10, description="Overlap lines when splitting large blocks")
    max_file_size_bytes: int = Field(default=2 * 1024 * 1024, description="Skip files larger than 2MB")
    include_extensions: list[str] = Field(
        default=[
            ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java",
            ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
            ".kt", ".scala", ".sh", ".bash", ".zsh", ".sql", ".html",
            ".css", ".scss", ".json", ".yaml", ".yml", ".md", ".toml"
        ],
        description="Supported file extensions for indexing"
    )
    exclude_patterns: list[str] = Field(
        default=[
            ".git", ".svn", ".hg", "__pycache__", ".pytest_cache", ".ruff_cache",
            ".mypy_cache", "node_modules", "dist", "build", ".venv", "venv",
            "env", ".tox", ".eggs", "*.egg-info", "*.pyc", "*.pyo", "*.pyd",
            "*.so", "*.dylib", "*.dll", "*.a", "*.lib", "*.o", "*.obj",
            "*.lock", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
            ".DS_Store", "Thumbs.db", "*.min.js", "*.min.css", "*.map"
        ],
        description="Glob patterns to ignore"
    )


class EmbeddingConfig(BaseModel):
    """Configuration for dense embedding generation."""
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model ID for dense code/text embeddings"
    )
    embedding_dim: int = Field(default=384, description="Dimension of embedding vectors")
    batch_size: int = Field(default=64, description="Inference batch size")
    normalize_embeddings: bool = Field(default=True, description="L2 normalize embeddings for cosine similarity")
    device: str = Field(default_factory=detect_device, description="Inference compute device")


class BM25Config(BaseModel):
    """Configuration for sparse lexical BM25 indexing."""
    k1: float = Field(default=1.5, description="BM25 term frequency saturation parameter")
    b: float = Field(default=0.75, description="BM25 document length normalization parameter")
    epsilon: float = Field(default=0.25, description="BM25 IDF floor parameter")


class RerankerConfig(BaseModel):
    """Configuration for Cross-Encoder reranking."""
    model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for precision reranking"
    )
    batch_size: int = Field(default=32, description="Reranking batch size")
    device: str = Field(default_factory=detect_device, description="Inference compute device")
    top_candidates_to_rerank: int = Field(default=25, description="Number of candidates to pass to reranker")


class RetrievalConfig(BaseModel):
    """Configuration for the hybrid search and fusion pipeline."""
    dense_top_k: int = Field(default=25, description="Number of dense vector search candidates")
    sparse_top_k: int = Field(default=25, description="Number of sparse BM25 candidates")
    final_top_k: int = Field(default=5, description="Number of final results to return after reranking")
    rrf_k: int = Field(default=60, description="Reciprocal Rank Fusion smoothing constant")
    dense_weight: float = Field(default=0.5, description="Weight for dense retrieval in fusion")
    sparse_weight: float = Field(default=0.5, description="Weight for sparse retrieval in fusion")
    use_reranker: bool = Field(default=True, description="Whether to apply cross-encoder reranking")


class StorageConfig(BaseModel):
    """Configuration for index persistence and metadata storage."""
    index_dir_name: str = Field(default=".code_intel_index", description="Default directory name for storing index")
    faiss_index_file: str = Field(default="vector_index.faiss", description="FAISS index filename")
    bm25_index_file: str = Field(default="bm25_index.pkl", description="BM25 index filename")
    metadata_db_file: str = Field(default="metadata.sqlite3", description="SQLite metadata database filename")
    manifest_file: str = Field(default="manifest.json", description="Index manifest metadata filename")


class CodeIntelConfig(BaseModel):
    """Root configuration object for the Semantic Code Intelligence Platform."""
    project_root: Path = Field(default_factory=Path.cwd, description="Root path of target codebase")
    index_dir: Optional[Path] = Field(default=None, description="Explicit directory for index storage")
    parser: ParserConfig = Field(default_factory=ParserConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    bm25: BM25Config = Field(default_factory=BM25Config)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    def get_index_dir(self) -> Path:
        """Resolve and return the absolute index storage directory."""
        if self.index_dir is not None:
            return self.index_dir.resolve()
        return (self.project_root / self.storage.index_dir_name).resolve()


# Global default configuration instance
DEFAULT_CONFIG = CodeIntelConfig()
