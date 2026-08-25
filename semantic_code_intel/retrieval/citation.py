"""
Line-level citation and context formatting module for code search results.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from semantic_code_intel.parser.base import CodeChunk


class SearchResult(BaseModel):
    """Encapsulates a single retrieved and reranked code snippet with precise citations."""
    chunk: CodeChunk
    score: float
    citation: str
    markdown_link: str
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None

    @classmethod
    def from_chunk(
        cls,
        chunk: CodeChunk,
        score: float,
        dense_score: Optional[float] = None,
        sparse_score: Optional[float] = None,
        rrf_score: Optional[float] = None,
        rerank_score: Optional[float] = None
    ) -> SearchResult:
        citation_str = chunk.citation
        abs_uri = Path(chunk.absolute_path).as_uri()
        line_anchor = f"#L{chunk.start_line}-L{chunk.end_line}"
        md_link = f"[{citation_str}]({abs_uri}{line_anchor})"

        return cls(
            chunk=chunk,
            score=score,
            citation=citation_str,
            markdown_link=md_link,
            dense_score=dense_score,
            sparse_score=sparse_score,
            rrf_score=rrf_score,
            rerank_score=rerank_score
        )


class CitationFormatter:
    """Formats search results for CLI terminal display, LLM prompts, and API output."""

    @staticmethod
    def format_for_llm_prompt(results: List[SearchResult]) -> str:
        """Format retrieved code chunks into a context block with citations for LLM synthesis."""
        if not results:
            return "No relevant code snippets found in repository."

        blocks: List[str] = []
        for idx, res in enumerate(results, start=1):
            c = res.chunk
            header = f"--- [Snippet {idx}] {res.citation} ({c.language})"
            if c.context_header:
                header += f" | {c.context_header}"
            header += " ---"

            block = f"{header}\n{c.content}\n"
            blocks.append(block)

        return "\n".join(blocks)

    @staticmethod
    def format_markdown_summary(results: List[SearchResult]) -> str:
        """Format results as standard markdown with clickable file citations."""
        if not results:
            return "No matches found."

        output = ["### Retrieved Code Citations:\n"]
        for idx, r in enumerate(results, start=1):
            output.append(
                f"**{idx}. {r.markdown_link}** (Score: `{r.score:.4f}` | Language: `{r.chunk.language}`)\n"
                f"*Scope*: `{r.chunk.context_header or r.chunk.symbol_name or 'N/A'}`\n"
                f"```{r.chunk.language}\n{r.chunk.content}\n```\n"
            )
        return "\n".join(output)
