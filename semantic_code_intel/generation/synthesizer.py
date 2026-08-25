"""
Code synthesizer supporting local extractive summarization and pluggable LLM backends (Ollama/APIs).
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional
import httpx
from pydantic import BaseModel, Field
from semantic_code_intel.generation.prompt_builder import SYSTEM_PROMPT, CodePromptBuilder
from semantic_code_intel.retrieval.citation import SearchResult

logger = logging.getLogger(__name__)


class SynthesisResponse(BaseModel):
    """Answer synthesized from retrieved code chunks."""
    answer: str
    citations: List[str] = Field(default_factory=list)
    model_used: str = "extractive"
    latency_ms: float = 0.0


class CodeSynthesizer:
    """Generates answers backed by exact code citations."""

    def __init__(self, provider: str = "extractive", ollama_base_url: str = "http://localhost:11434"):
        self.provider = provider
        self.ollama_base_url = ollama_base_url

    def synthesize(self, query: str, results: List[SearchResult]) -> SynthesisResponse:
        """Synthesize an answer using either extractive reasoning or an LLM backend."""
        citations = [r.citation for r in results]

        if not results:
            return SynthesisResponse(
                answer="No relevant code was found in the indexed repository to answer this query.",
                citations=[],
                model_used="none"
            )

        if self.provider == "ollama":
            return self._call_ollama(query, results, citations)

        # Default local extractive synthesis
        return self._extractive_synthesis(query, results, citations)

    def _extractive_synthesis(
        self, query: str, results: List[SearchResult], citations: List[str]
    ) -> SynthesisResponse:
        """Construct a structured extractive summary of top matching code snippets."""
        parts = [f"Found {len(results)} relevant code location(s) for query: **'{query}'**\n"]

        for idx, res in enumerate(results, start=1):
            c = res.chunk
            symbol_desc = f"`{c.symbol_type.value} {c.symbol_name}`" if c.symbol_name else "Code Block"
            scope_desc = f" in `{c.parent_scope}`" if c.parent_scope else ""
            
            parts.append(f"### {idx}. {c.citation} ({symbol_desc}{scope_desc})")
            if c.docstring:
                parts.append(f"> **Docstring**: {c.docstring.strip()}")
            if c.context_header:
                parts.append(f"> **Signature**: `{c.context_header}`")
            parts.append(f"```{c.language}\n{c.content}\n```\n")

        answer_text = "\n".join(parts)
        return SynthesisResponse(
            answer=answer_text,
            citations=citations,
            model_used="local_extractive"
        )

    def _call_ollama(
        self, query: str, results: List[SearchResult], citations: List[str]
    ) -> SynthesisResponse:
        """Attempt to invoke local Ollama if running, fallback to extractive on failure."""
        prompt = CodePromptBuilder.build_rag_prompt(query, results)
        try:
            with httpx.Client(timeout=30.0) as client:
                res = client.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b"),
                        "system": SYSTEM_PROMPT,
                        "prompt": prompt,
                        "stream": False
                    }
                )
                if res.status_code == 200:
                    data = res.json()
                    return SynthesisResponse(
                        answer=data.get("response", ""),
                        citations=citations,
                        model_used="ollama"
                    )
        except Exception as e:
            logger.warning(f"Ollama call failed ({e}), falling back to local extractive synthesizer.")

        return self._extractive_synthesis(query, results, citations)
