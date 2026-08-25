"""
Prompt construction with exact line citations for Code RAG synthesis.
"""

from __future__ import annotations

from typing import List
from semantic_code_intel.retrieval.citation import CitationFormatter, SearchResult

SYSTEM_PROMPT = """You are an expert AI Code Intelligence Assistant.
Answer the user's question accurately using ONLY the provided code snippets and context.
Always cite the exact file path and line numbers when referencing code (e.g. `src/auth.py:L15-L30`).
If the provided snippets do not contain enough information to answer the question, state what is missing.
"""


class CodePromptBuilder:
    """Builds structured prompts for LLM synthesis including exact line-level citations."""

    @staticmethod
    def build_rag_prompt(query: str, search_results: List[SearchResult]) -> str:
        context_str = CitationFormatter.format_for_llm_prompt(search_results)
        
        prompt = f"""### Context Code Snippets with Citations:
{context_str}

### User Question:
{query}

### Instructions:
1. Analyze the context code snippets above.
2. Provide a clear, technically rigorous answer.
3. Cite the exact file names and line ranges (e.g., `file.py:L10-L20`) whenever discussing logic or functions.
"""
        return prompt
