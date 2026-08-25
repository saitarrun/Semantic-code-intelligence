"""
Prompt construction with exact line citations for Code RAG synthesis.
"""

from __future__ import annotations

from typing import List
from semantic_code_intel.retrieval.citation import CitationFormatter, SearchResult

SYSTEM_PROMPT = """You are a precise code-walkthrough assistant.
Use ONLY behavior directly supported by the supplied source snippets. Never infer validation,
state changes, error handling, or side effects that are not visible in those snippets.

Required response structure:
1. `## Direct answer` — answer the user's exact question immediately in 1–3 sentences.
2. `## Execution walkthrough` — ordered runtime steps. Name the concrete symbols involved and cite
   every step with an exact `path:Lx-Ly` citation.
3. `## Inputs, outputs, and edge cases` — include only details proven by the source.
4. `## Evidence gaps` — identify missing callers, implementations, configuration, or runtime context.

Prefer specific identifiers and conditions over general architectural language. Do not dump entire
snippets unless a short excerpt is essential. If the evidence cannot answer the question, say that
plainly in the Direct answer and explain exactly which source is missing.
"""


class CodePromptBuilder:
    """Builds structured prompts for LLM synthesis including exact line-level citations."""

    @staticmethod
    def build_rag_prompt(query: str, search_results: List[SearchResult]) -> str:
        context_str = CitationFormatter.format_for_llm_prompt(search_results)
        
        prompt = f"""{SYSTEM_PROMPT}

### Source evidence:
{context_str}

### Exact user question:
{query}
"""
        return prompt
