"""
Generative Code Synthesizer and Explanation Engine.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncGenerator, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from pydantic import BaseModel, Field

from semantic_code_intel.config import CodeIntelConfig
from semantic_code_intel.generation.prompt_builder import CodePromptBuilder
from semantic_code_intel.retrieval.citation import SearchResult

logger = logging.getLogger(__name__)


class SynthesisResponse(BaseModel):
    query: str
    answer: str
    citations: List[str] = Field(default_factory=list)
    provider: str = "extractive"


class CodeSynthesizer:
    """Generates cited answers with a local LLM and an explicit extractive fallback."""

    def __init__(self, config: Optional[CodeIntelConfig] = None, provider: Optional[str] = None):
        self.config = config or CodeIntelConfig()
        self.provider = provider or self.config.generation.provider

    def _extractive_answer(self, query: str, results: List[SearchResult]) -> str:
        top_match = results[0]
        top_symbol = top_match.chunk.symbol_name or top_match.chunk.context_header or top_match.chunk.file_path
        sections = [
            "## Direct answer",
            f"The strongest retrieved implementation for **{query}** is `{top_symbol}` "
            f"at `{top_match.chunk.citation}`. A local reasoning model is not available, so the "
            "evidence is shown without inventing behavior that is not explicit in the code.",
            "\n## Source-backed walkthrough",
        ]
        for index, result in enumerate(results[:6], 1):
            chunk = result.chunk
            symbol = chunk.symbol_name or chunk.context_header or "module-level code"
            scope = f" in `{chunk.parent_scope}`" if chunk.parent_scope else ""
            dependencies = (
                f"Referenced symbols: {', '.join(f'`{item}`' for item in chunk.dependencies[:5])}."
                if chunk.dependencies else ""
            )
            source_lines = chunk.content.splitlines()
            displayed_source = "\n".join(source_lines[:32])
            if len(source_lines) > 32:
                displayed_source += f"\n# … {len(source_lines) - 32} additional lines in {chunk.citation}"
            sections.extend([
                f"\n### {index}. `{symbol}`{scope} — `{chunk.citation}`",
                chunk.symbol_type.value.replace('_', ' ').title() + "." +
                (f" {dependencies}" if dependencies else ""),
                f"```{chunk.language}\n{displayed_source}\n```",
            ])
        sections.extend([
            "\n## Evidence gaps",
            "This is a deterministic evidence view, not a generated explanation. Start Ollama for a "
            "step-by-step answer that connects these snippets; any behavior outside the cited blocks "
            "remains unverified.",
        ])
        return "\n\n".join(sections)

    def _ollama_answer(self, query: str, results: List[SearchResult]) -> str:
        prompt = CodePromptBuilder.build_rag_prompt(query, results)
        payload = json.dumps({
            "model": self.config.generation.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        }).encode("utf-8")
        request = Request(
            f"{self.config.generation.ollama_base_url.rstrip('/')}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.generation.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Local Ollama generation failed: {exc}") from exc
        answer = str(data.get("response", "")).strip()
        if not answer:
            raise RuntimeError("Local Ollama generation returned an empty response")
        return answer

    def synthesize(self, query: str, results: List[SearchResult]) -> SynthesisResponse:
        """
        Synchronously synthesizes a citation-backed technical answer.
        """
        if not results:
            return SynthesisResponse(
                query=query,
                answer="No relevant code implementations found in the index.",
                citations=[],
                provider=self.provider
            )

        citations = [r.chunk.citation for r in results]
        provider_used = self.provider
        if self.provider == "ollama":
            try:
                answer = self._ollama_answer(query, results)
            except RuntimeError:
                if not self.config.generation.fallback_to_extractive:
                    raise
                logger.warning("Ollama unavailable; using deterministic source evidence")
                answer = self._extractive_answer(query, results)
                provider_used = "extractive-fallback"
        elif self.provider == "extractive":
            answer = self._extractive_answer(query, results)
        else:
            raise ValueError(f"Unsupported synthesis provider: {self.provider}")

        return SynthesisResponse(
            query=query,
            answer=answer,
            citations=citations,
            provider=provider_used
        )

    async def stream_synthesis(
        self,
        query: str,
        results: List[SearchResult]
    ) -> AsyncGenerator[str, None]:
        """
        Streams a structured markdown explanation synthesized from the retrieved code chunks.
        """
        if not results:
            yield json.dumps({
                "type": "content",
                "delta": "No matching code implementations found in the index to synthesize an explanation."
            }) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
            return

        # Keep the streaming protocol provider-agnostic. Ollama generation is completed
        # first, then emitted in small deltas so clients receive the same grounded answer
        # as the synchronous endpoint (including explicit fallback behavior).
        response = self.synthesize(query, results)
        # Slice the original answer without tokenizing it. Tokenization previously
        # collapsed newlines and broke headings, lists, and fenced code in the UI.
        for offset in range(0, len(response.answer), 640):
            delta = response.answer[offset:offset + 640]
            yield json.dumps({
                "type": "content",
                "delta": delta,
                "provider": response.provider,
            }) + "\n"
        yield json.dumps({
            "type": "done",
            "provider": response.provider,
            "citations": response.citations,
        }) + "\n"
        return
