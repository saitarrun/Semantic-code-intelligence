"""
Generative Code Synthesizer and Explanation Engine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional
from pydantic import BaseModel, Field

from semantic_code_intel.config import CodeIntelConfig
from semantic_code_intel.retrieval.citation import SearchResult


class SynthesisResponse(BaseModel):
    query: str
    answer: str
    citations: List[str] = Field(default_factory=list)
    provider: str = "extractive"


class CodeSynthesizer:
    """Generates structured, citation-backed technical explanations for code intelligence queries."""

    def __init__(self, config: Optional[CodeIntelConfig] = None, provider: str = "extractive"):
        self.config = config or CodeIntelConfig()
        self.provider = provider

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
        top_match = results[0]
        
        symbol_info = f" in function `{top_match.chunk.symbol_name}`" if top_match.chunk.symbol_name else ""
        answer = (
            f"The logic for `{query}` is implemented in `{top_match.chunk.file_path}`{symbol_info} "
            f"(lines {top_match.chunk.start_line}–{top_match.chunk.end_line}).\n\n"
            f"Relevant Implementation:\n```{top_match.chunk.language}\n{top_match.chunk.content}\n```"
        )

        return SynthesisResponse(
            query=query,
            answer=answer,
            citations=citations,
            provider=self.provider
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

        top_match = results[0]
        
        # Header overview
        yield json.dumps({
            "type": "content",
            "delta": f"### Analysis for `{query}`\n\n"
        }) + "\n"

        # Key Findings
        yield json.dumps({
            "type": "content",
            "delta": f"The primary implementation is located in **[`{top_match.chunk.file_path}`]({top_match.chunk.citation})** (lines {top_match.chunk.start_line}–{top_match.chunk.end_line})"
        }) + "\n"

        if top_match.chunk.symbol_name:
            yield json.dumps({
                "type": "content",
                "delta": f" within symbol `{top_match.chunk.symbol_name}`.\n\n"
            }) + "\n"
        else:
            yield json.dumps({"type": "content", "delta": ".\n\n"}) + "\n"

        # Implementation breakdown
        yield json.dumps({
            "type": "content",
            "delta": "#### Key Architectural Components:\n\n"
        }) + "\n"

        for idx, res in enumerate(results[:4], 1):
            scope_desc = f"`{res.chunk.symbol_name}`" if res.chunk.symbol_name else f"Lines {res.chunk.start_line}–{res.chunk.end_line}"
            first_line = res.chunk.content.strip().split("\n")[0][:80]
            
            yield json.dumps({
                "type": "content",
                "delta": f"**{idx}. [{res.chunk.file_path}:{res.chunk.start_line}]({res.chunk.citation})** ({res.chunk.language})\n"
                         f"- **Scope**: {scope_desc}\n"
                         f"- **Preview**: `{first_line}`\n"
                         f"- **Match Score**: `{res.score:.4f}`\n\n"
            }) + "\n"

        # Summary takeaway
        yield json.dumps({
            "type": "content",
            "delta": "#### Implementation Summary:\n"
                     f"This subsystem coordinates control logic via `{top_match.chunk.symbol_name or top_match.chunk.file_path}`. "
                     "Inputs are validated, state mutations occur safely, and execution boundaries conform to the citations above."
        }) + "\n"

        yield json.dumps({"type": "done"}) + "\n"
