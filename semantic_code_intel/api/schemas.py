"""
Pydantic API schemas for HTTP REST endpoints.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(description="Search query or natural language code question")
    top_k: int = Field(default=5, description="Number of results to return")
    use_reranker: bool = Field(default=True, description="Enable cross-encoder reranking")
    mode: str = Field(default="hybrid", description="Retrieval mode: 'hybrid', 'dense', or 'sparse'")


class IndexRequest(BaseModel):
    target_dir: Optional[str] = Field(default=None, description="Path to codebase to index")
    force_reindex: bool = Field(default=False, description="Wipe and rebuild index from scratch")


class SynthesizeRequest(BaseModel):
    query: str = Field(description="Question to ask about the codebase")
    top_k: int = Field(default=5, description="Number of context snippets to retrieve")
    provider: str = Field(default="extractive", description="LLM provider: 'extractive' or 'ollama'")


class ChunkResponse(BaseModel):
    chunk_id: str
    file_path: str
    absolute_path: str
    language: str
    symbol_name: Optional[str]
    symbol_type: str
    parent_scope: Optional[str]
    start_line: int
    end_line: int
    content: str
    context_header: Optional[str]
    docstring: Optional[str]
    citation: str
    markdown_link: str
    score: float
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None


class SearchApiResponse(BaseModel):
    query: str
    results: List[ChunkResponse]
    latency_ms: Dict[str, float]
    total_candidates: int


class SynthesizeApiResponse(BaseModel):
    query: str
    answer: str
    citations: List[str]
    model_used: str
    search_latency_ms: float
    total_latency_ms: float
