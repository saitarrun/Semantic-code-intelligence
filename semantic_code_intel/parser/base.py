"""
Base models and abstract parser interfaces for code chunking and symbol extraction.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SymbolType(str, Enum):
    MODULE = "module"
    CLASS = "class"
    METHOD = "method"
    FUNCTION = "function"
    ASYNC_FUNCTION = "async_function"
    STRUCT = "struct"
    INTERFACE = "interface"
    ENUM = "enum"
    VARIABLE = "variable"
    CONFIG = "config"
    BLOCK = "block"
    DOCSTRING = "docstring"
    UNKNOWN = "unknown"


class CodeChunk(BaseModel):
    """Represents a discrete semantic chunk of source code with precise citation metadata."""
    chunk_id: str = Field(description="Unique deterministic identifier for the chunk")
    file_path: str = Field(description="Relative path of the source file from repo root")
    absolute_path: str = Field(description="Absolute path of the source file")
    language: str = Field(description="Programming language / file type")
    symbol_name: Optional[str] = Field(default=None, description="Name of the function, class, or symbol")
    symbol_type: SymbolType = Field(default=SymbolType.UNKNOWN, description="Structural type of the symbol")
    parent_scope: Optional[str] = Field(default=None, description="Enclosing class, namespace, or scope")
    start_line: int = Field(description="1-based start line in source file")
    end_line: int = Field(description="1-based end line in source file")
    content: str = Field(description="Exact code content of the chunk")
    context_header: Optional[str] = Field(
        default=None,
        description="Signature, breadcrumbs, or header context (e.g. 'class AuthController > login')"
    )
    docstring: Optional[str] = Field(default=None, description="Extracted docstring or comments")
    dependencies: List[str] = Field(default_factory=list, description="Referenced functions, imports, or symbols")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary extra metadata")

    @property
    def citation(self) -> str:
        """Return a standardized citation string: 'path/to/file.py:L10-L25'."""
        if self.start_line == self.end_line:
            return f"{self.file_path}:L{self.start_line}"
        return f"{self.file_path}:L{self.start_line}-L{self.end_line}"

    @property
    def line_count(self) -> int:
        """Return the number of lines spanned by this chunk."""
        return max(1, self.end_line - self.start_line + 1)

    def get_searchable_text(self) -> str:
        """
        Generate rich contextual representation for dense embedding and sparse BM25 indexing.
        Includes file path, scope, signatures, docstrings, and actual code.
        """
        parts = []
        parts.append(f"File: {self.file_path}")
        if self.language:
            parts.append(f"Language: {self.language}")
        if self.parent_scope:
            parts.append(f"Scope: {self.parent_scope}")
        if self.symbol_name:
            parts.append(f"Symbol: {self.symbol_type.value} {self.symbol_name}")
        if self.context_header:
            parts.append(f"Header: {self.context_header}")
        if self.docstring:
            parts.append(f"Docstring: {self.docstring}")
        parts.append("Code:")
        parts.append(self.content)
        return "\n".join(parts)

    @classmethod
    def generate_id(cls, file_path: str, start_line: int, end_line: int, content: str) -> str:
        """Generate a deterministic chunk ID based on file, line range, and content hash."""
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
        clean_path = file_path.replace("/", "_").replace("\\", "_").replace(".", "_")
        return f"{clean_path}_L{start_line}_L{end_line}_{content_hash}"


class ParseResult(BaseModel):
    """Result of parsing a file into semantic code chunks."""
    file_path: str
    language: str
    chunks: List[CodeChunk] = Field(default_factory=list)
    total_lines: int = 0
    total_bytes: int = 0
    error: Optional[str] = None


class BaseParser:
    """Abstract base class for code parsers."""
    
    def parse_file(self, file_path: Path, repo_root: Path) -> ParseResult:
        """Parse a source file and return extracted code chunks."""
        raise NotImplementedError
