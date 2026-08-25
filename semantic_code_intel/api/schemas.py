"""Canonical Pydantic schemas for the HTTP API."""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str
    repo_path: Optional[str] = None
    index_path: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=50)
    mode: str = Field(default="hybrid", pattern="^(hybrid|dense|sparse)$")
    rerank: bool = True
    use_reranker: Optional[bool] = None


class SearchResultItem(BaseModel):
    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    symbol_name: Optional[str] = None
    symbol_type: Optional[str] = None
    score: float
    citation: str
    code: str
    content: str
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rerank_score: Optional[float] = None
    exact_match_boost: float = 0.0
    match_reasons: List[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    repo_path: str
    total_results: int
    results: List[SearchResultItem]
    latency_ms: Dict[str, float]
    index_status: str = "ready"
    reliability: str = "low"
    reliability_score: float = 0.0
    reliability_reasons: List[str] = Field(default_factory=list)


class SynthesizeRequest(BaseModel):
    query: str
    repo_path: Optional[str] = None
    index_path: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)
    provider: Optional[str] = Field(default=None, pattern="^(ollama|extractive)$")


class SynthesizeResponse(BaseModel):
    query: str
    answer: str
    citations: List[str]
    provider: str
    latency_ms: Dict[str, float]


class PatchGenerateRequest(BaseModel):
    instruction: str
    repo_path: Optional[str] = None
    top_k: int = Field(default=3, ge=1, le=20)


class PatchApplyRequest(BaseModel):
    diff: str
    repo_path: Optional[str] = None


class IndexRequest(BaseModel):
    target_dir: str = Field(default=".")
    index_dir: Optional[str] = None
    force: bool = False


class OpenFileRequest(BaseModel):
    file_path: str
    repo_path: Optional[str] = None
    line: int = Field(default=1, ge=1)
    action: str = Field(default="editor", pattern="^(editor|finder)$")
