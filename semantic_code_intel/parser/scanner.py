"""
Codebase scanner that walks directories, filters paths, and parses files into semantic chunks.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator, List, Optional, Tuple
from semantic_code_intel.config import CodeIntelConfig, ParserConfig
from semantic_code_intel.parser.base import BaseParser, CodeChunk, ParseResult
from semantic_code_intel.parser.ignore_rules import IgnoreFilter
from semantic_code_intel.parser.polyglot_parser import PolyglotParser
from semantic_code_intel.parser.python_parser import PythonASTParser


class CodebaseScanner:
    """Recursively scans a target directory and parses files into semantic code chunks."""

    def __init__(self, config: Optional[CodeIntelConfig] = None):
        self.config = config or CodeIntelConfig()
        self.parser_config = self.config.parser
        self.python_parser = PythonASTParser(self.parser_config)
        self.polyglot_parser = PolyglotParser(self.parser_config)

    def get_parser_for_file(self, file_path: Path) -> BaseParser:
        """Select the appropriate parser based on file extension."""
        if file_path.suffix.lower() == ".py":
            return self.python_parser
        return self.polyglot_parser

    def discover_files(self, root_dir: Path) -> List[Path]:
        """Discover all eligible source files in root_dir respecting ignore rules."""
        root_path = root_dir.resolve()
        ignore_filter = IgnoreFilter(root_path, self.parser_config)
        discovered: List[Path] = []

        for dirpath, dirnames, filenames in os.walk(root_path):
            current_dir = Path(dirpath)
            
            # Prune ignored directories in-place
            dirnames[:] = [
                d for d in dirnames
                if not ignore_filter.should_ignore(current_dir / d)
                and d != self.config.storage.index_dir_name
            ]

            for filename in filenames:
                file_path = current_dir / filename
                if not ignore_filter.should_ignore(file_path):
                    discovered.append(file_path)

        return sorted(discovered)

    def scan_and_parse(
        self, root_dir: Path
    ) -> Generator[ParseResult, None, Tuple[int, int, int]]:
        """
        Scan and parse all files in root_dir, yielding ParseResult per file.
        Returns total_files, total_lines, total_chunks.
        """
        root_path = root_dir.resolve()
        files = self.discover_files(root_path)
        total_files = len(files)
        total_lines = 0
        total_chunks = 0

        for file_path in files:
            parser = self.get_parser_for_file(file_path)
            res = parser.parse_file(file_path, root_path)
            total_lines += res.total_lines
            total_chunks += len(res.chunks)
            yield res

        return total_files, total_lines, total_chunks
