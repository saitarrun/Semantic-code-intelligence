"""Deterministic code-query expansion and exact-match analysis."""

from __future__ import annotations

import re
from typing import Iterable

from semantic_code_intel.indexing.bm25_index import CodeTokenizer
from semantic_code_intel.parser.base import CodeChunk


STOP_WORDS = {
    "a", "an", "and", "are", "does", "for", "from", "how", "in", "is",
    "it", "of", "on", "or", "the", "this", "to", "what", "where", "which",
}

INTENT_EXPANSIONS = {
    "ui": ("user interface", "frontend", "html", "css", "javascript", "static", "dashboard", "render"),
    "api": ("endpoint", "route", "request", "response", "fastapi", "handler"),
    "auth": ("authentication", "authorization", "credential", "token", "session"),
    "database": ("db", "sql", "connection", "query", "repository", "pool"),
    "config": ("configuration", "settings", "environment", "option"),
    "error": ("exception", "failure", "raise", "catch", "handling"),
    "render": ("html", "template", "static", "response", "frontend"),
    "serve": ("server", "route", "response", "host", "port"),
}


def meaningful_tokens(text: str) -> set[str]:
    """Return normalized content-bearing query tokens."""
    return {
        token for token in CodeTokenizer.tokenize(text)
        if len(token) > 1 and token not in STOP_WORDS and re.search(r"[a-z0-9]", token)
    }


def expand_code_query(query: str) -> str:
    """Expand common developer intents without requiring a generative model."""
    tokens = meaningful_tokens(query)
    additions: list[str] = []
    for token, expansion in INTENT_EXPANSIONS.items():
        if token in tokens:
            additions.extend(expansion)
    if "where" in CodeTokenizer.tokenize(query):
        additions.extend(("definition", "implementation", "file", "symbol"))
    if "how" in CodeTokenizer.tokenize(query):
        additions.extend(("implementation", "control flow", "function", "calls"))
    unique_additions = [item for item in dict.fromkeys(additions) if item not in query.lower()]
    return f"{query} {' '.join(unique_additions)}".strip()


def exact_match_boost(
    query: str,
    chunk: CodeChunk,
    analysis_query: str | None = None,
) -> tuple[float, list[str]]:
    """Score transparent symbol/path/token agreement for a hydrated candidate."""
    query_lower = query.lower()
    query_tokens = meaningful_tokens(analysis_query or query)
    reasons: list[str] = []
    boost = 0.0

    if chunk.symbol_name:
        symbol_lower = chunk.symbol_name.lower()
        if symbol_lower in query_lower or symbol_lower.replace("_", " ") in query_lower:
            boost += 2.5
            reasons.append("exact symbol")

    path_tokens = meaningful_tokens(chunk.file_path.replace("/", " ").replace(".", " "))
    path_overlap = query_tokens & path_tokens
    if path_overlap:
        boost += min(2.4, 0.9 * len(path_overlap))
        reasons.append("path: " + ", ".join(sorted(path_overlap)))

    chunk_tokens = meaningful_tokens(" ".join(filter(None, (
        chunk.symbol_name, chunk.context_header, chunk.docstring,
    ))))
    overlap = query_tokens & chunk_tokens
    if overlap:
        boost += min(3.6, 0.9 * len(overlap))
        reasons.append("terms: " + ", ".join(sorted(overlap)))

    return boost, reasons
