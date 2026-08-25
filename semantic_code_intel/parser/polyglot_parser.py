"""
Polyglot structural parser for multi-language codebases (JS/TS, Go, Rust, Java, C/C++, SQL, etc.).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Tuple
from semantic_code_intel.config import ParserConfig
from semantic_code_intel.parser.base import BaseParser, CodeChunk, ParseResult, SymbolType

EXTENSION_LANGUAGE_MAP: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".scala": "scala",
    ".cs": "csharp",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".h": "c_header",
    ".hpp": "cpp_header",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
}

# Regex patterns for structural symbols across major languages
LANGUAGE_PATTERNS: Dict[str, List[Tuple[Pattern[str], SymbolType]]] = {
    "typescript": [
        (re.compile(r"^(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+([A-Za-z0-9_$]+)"), SymbolType.CLASS),
        (re.compile(r"^(?:export\s+)?interface\s+([A-Za-z0-9_$]+)"), SymbolType.INTERFACE),
        (re.compile(r"^(?:export\s+)?type\s+([A-Za-z0-9_$]+)\s*="), SymbolType.STRUCT),
        (re.compile(r"^(?:export\s+)?enum\s+([A-Za-z0-9_$]+)"), SymbolType.ENUM),
        (re.compile(r"^(?:export\s+)?(?:async\s+)?function(?:\s+([A-Za-z0-9_$]+)|\s*\()"), SymbolType.FUNCTION),
        (re.compile(r"^(?:export\s+)?(?:const|let|var)\s+([A-Za-z0-9_$]+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[A-Za-z0-9_$]+)\s*=>"), SymbolType.FUNCTION),
    ],
    "javascript": [
        (re.compile(r"^(?:export\s+)?(?:default\s+)?class\s+([A-Za-z0-9_$]+)"), SymbolType.CLASS),
        (re.compile(r"^(?:export\s+)?(?:async\s+)?function(?:\s+([A-Za-z0-9_$]+)|\s*\()"), SymbolType.FUNCTION),
        (re.compile(r"^(?:export\s+)?(?:const|let|var)\s+([A-Za-z0-9_$]+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[A-Za-z0-9_$]+)\s*=>"), SymbolType.FUNCTION),
    ],
    "go": [
        (re.compile(r"^type\s+([A-Za-z0-9_]+)\s+struct\b"), SymbolType.STRUCT),
        (re.compile(r"^type\s+([A-Za-z0-9_]+)\s+interface\b"), SymbolType.INTERFACE),
        (re.compile(r"^func\s+\(\s*[^)]+\s*\)\s*([A-Za-z0-9_]+)\s*\("), SymbolType.METHOD),
        (re.compile(r"^func\s+([A-Za-z0-9_]+)\s*\("), SymbolType.FUNCTION),
    ],
    "rust": [
        (re.compile(r"^(?:pub(?:\([^)]+\))?\s+)?struct\s+([A-Za-z0-9_]+)"), SymbolType.STRUCT),
        (re.compile(r"^(?:pub(?:\([^)]+\))?\s+)?enum\s+([A-Za-z0-9_]+)"), SymbolType.ENUM),
        (re.compile(r"^(?:pub(?:\([^)]+\))?\s+)?trait\s+([A-Za-z0-9_]+)"), SymbolType.INTERFACE),
        (re.compile(r"^impl(?:\s*<[^>]+>)?\s+(?:([A-Za-z0-9_]+)\s+for\s+)?([A-Za-z0-9_]+)"), SymbolType.CLASS),
        (re.compile(r"^(?:pub(?:\([^)]+\))?\s+)?(?:async\s+)?fn\s+([A-Za-z0-9_]+)"), SymbolType.FUNCTION),
    ],
    "java": [
        (re.compile(r"^(?:public|protected|private|static|\s)*class\s+([A-Za-z0-9_]+)"), SymbolType.CLASS),
        (re.compile(r"^(?:public|protected|private|static|\s)*interface\s+([A-Za-z0-9_]+)"), SymbolType.INTERFACE),
        (re.compile(r"^(?:public|protected|private|static|\s)*enum\s+([A-Za-z0-9_]+)"), SymbolType.ENUM),
        (re.compile(r"^(?:public|protected|private|static|final|native|synchronized|abstract|\s)+[\w<>\[\],\s]+\s+([A-Za-z0-9_]+)\s*\([^)]*\)\s*(?:throws\s+[\w,\s]+)?\s*\{"), SymbolType.METHOD),
    ],
    "cpp": [
        (re.compile(r"^(?:class|struct)\s+([A-Za-z0-9_]+)"), SymbolType.CLASS),
        (re.compile(r"^(?:enum(?:\s+class)?)\s+([A-Za-z0-9_]+)"), SymbolType.ENUM),
        (re.compile(r"^template\s*<[^>]+>\s*(?:class|struct)\s+([A-Za-z0-9_]+)"), SymbolType.CLASS),
        (re.compile(r"^(?:[A-Za-z0-9_:<>&*]+\s+)+([A-Za-z0-9_]+)\s*\([^)]*\)\s*(?:const)?\s*\{?"), SymbolType.FUNCTION),
    ],
    "c": [
        (re.compile(r"^(?:typedef\s+)?struct\s+([A-Za-z0-9_]+)?"), SymbolType.STRUCT),
        (re.compile(r"^(?:[A-Za-z0-9_*]+\s+)+([A-Za-z0-9_]+)\s*\([^)]*\)\s*\{?"), SymbolType.FUNCTION),
    ],
    "sql": [
        (re.compile(r"^(?:CREATE\s+(?:OR\s+REPLACE\s+)?TABLE\s+([A-Za-z0-9_.\"]+))", re.IGNORECASE), SymbolType.STRUCT),
        (re.compile(r"^(?:CREATE\s+(?:OR\s+REPLACE\s+)?(?:PROCEDURE|FUNCTION)\s+([A-Za-z0-9_.\"]+))", re.IGNORECASE), SymbolType.FUNCTION),
        (re.compile(r"^(?:CREATE\s+(?:OR\s+REPLACE\s+)?VIEW\s+([A-Za-z0-9_.\"]+))", re.IGNORECASE), SymbolType.CLASS),
    ],
    "markdown": [
        (re.compile(r"^(#{1,6})\s+(.+)"), SymbolType.BLOCK),
    ]
}


class PolyglotParser(BaseParser):
    """
    Language-aware structural parser for polyglot codebases.
    Uses pattern matching, indentation heuristics, and brace/block tracking to isolate symbols.
    """

    def __init__(self, config: Optional[ParserConfig] = None):
        self.config = config or ParserConfig()

    def parse_file(self, file_path: Path, repo_root: Path) -> ParseResult:
        rel_path = str(file_path.relative_to(repo_root)) if file_path.is_relative_to(repo_root) else str(file_path)
        abs_path = str(file_path.resolve())
        ext = file_path.suffix.lower()
        language = EXTENSION_LANGUAGE_MAP.get(ext, "text")

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                source_code = f.read()
        except Exception as e:
            return ParseResult(
                file_path=rel_path,
                language=language,
                error=f"Failed to read file: {e}"
            )

        lines = source_code.splitlines(keepends=True)
        total_lines = len(lines)
        total_bytes = len(source_code.encode("utf-8"))

        if total_lines == 0:
            return ParseResult(file_path=rel_path, language=language, total_lines=0, total_bytes=0)

        # Markdown special section chunker
        if language == "markdown":
            chunks = self._parse_markdown(lines, rel_path, abs_path)
            return ParseResult(
                file_path=rel_path,
                language=language,
                chunks=chunks,
                total_lines=total_lines,
                total_bytes=total_bytes
            )

        # Structural symbol extraction
        chunks = self._parse_structural_code(lines, rel_path, abs_path, language)
        if not chunks:
            # Fallback to sliding window chunking
            chunks = self._sliding_window_chunks(lines, rel_path, abs_path, language)

        chunks.sort(key=lambda c: (c.start_line, c.end_line))

        return ParseResult(
            file_path=rel_path,
            language=language,
            chunks=chunks,
            total_lines=total_lines,
            total_bytes=total_bytes
        )

    def _parse_structural_code(
        self,
        lines: List[str],
        rel_path: str,
        abs_path: str,
        language: str
    ) -> List[CodeChunk]:
        patterns = LANGUAGE_PATTERNS.get(language, [])
        if not patterns:
            return []

        chunks: List[CodeChunk] = []
        covered_lines = set()
        total_lines = len(lines)

        i = 0
        while i < total_lines:
            line_str = lines[i].strip()
            matched_symbol = None
            symbol_name = None
            symbol_type = SymbolType.BLOCK

            for pat, s_type in patterns:
                m = pat.search(line_str)
                if m:
                    matched_symbol = m
                    symbol_type = s_type
                    symbol_name = m.group(1) if m.groups() else None
                    break

            if matched_symbol:
                start_line = i + 1
                # Find matching block end (brace matching or indentation)
                end_line = self._find_block_end(lines, i)
                chunk_lines = lines[start_line - 1 : end_line]
                content = "".join(chunk_lines)

                for l_idx in range(start_line, end_line + 1):
                    covered_lines.add(l_idx)

                chunk_id = CodeChunk.generate_id(rel_path, start_line, end_line, content)
                header = f"{language.capitalize()} {symbol_type.value}: {symbol_name or line_str[:40]}"

                # Split if chunk is too large
                if (end_line - start_line + 1) > self.config.max_chunk_lines:
                    split_chunks = self._split_polyglot_chunk(
                        lines, start_line, end_line, rel_path, abs_path, language, symbol_name, symbol_type, header
                    )
                    chunks.extend(split_chunks)
                else:
                    chunks.append(CodeChunk(
                        chunk_id=chunk_id,
                        file_path=rel_path,
                        absolute_path=abs_path,
                        language=language,
                        symbol_name=symbol_name,
                        symbol_type=symbol_type,
                        start_line=start_line,
                        end_line=end_line,
                        content=content,
                        context_header=header
                    ))

                i = end_line
            else:
                i += 1

        # Fill remaining gaps
        gap_chunks = self._fill_polyglot_gaps(lines, covered_lines, rel_path, abs_path, language)
        chunks.extend(gap_chunks)

        return chunks

    def _find_block_end(self, lines: List[str], start_idx: int) -> int:
        """Find the logical end of a code block using brace balancing or indentation."""
        total_lines = len(lines)
        brace_count = 0
        has_opened = False
        max_lookahead = min(total_lines, start_idx + 150)

        for idx in range(start_idx, max_lookahead):
            line = lines[idx]
            brace_count += line.count("{") - line.count("}")
            if "{" in line:
                has_opened = True

            if has_opened and brace_count <= 0:
                return idx + 1

        # If no braces or unbalanced, look for blank line or indentation break
        for idx in range(start_idx + 1, min(total_lines, start_idx + self.config.max_chunk_lines)):
            if lines[idx].strip() == "":
                return idx
        return min(total_lines, start_idx + self.config.max_chunk_lines)

    def _split_polyglot_chunk(
        self,
        lines: List[str],
        start_line: int,
        end_line: int,
        rel_path: str,
        abs_path: str,
        language: str,
        symbol_name: Optional[str],
        symbol_type: SymbolType,
        context_header: str
    ) -> List[CodeChunk]:
        chunks: List[CodeChunk] = []
        current_start = start_line
        max_lines = self.config.max_chunk_lines
        overlap = self.config.chunk_overlap_lines

        part_idx = 1
        while current_start <= end_line:
            current_end = min(end_line, current_start + max_lines - 1)
            chunk_content = "".join(lines[current_start - 1 : current_end])
            chunk_id = CodeChunk.generate_id(rel_path, current_start, current_end, chunk_content)
            
            sub_header = f"{context_header} [Part {part_idx}]"
            chunks.append(CodeChunk(
                chunk_id=chunk_id,
                file_path=rel_path,
                absolute_path=abs_path,
                language=language,
                symbol_name=symbol_name,
                symbol_type=symbol_type,
                start_line=current_start,
                end_line=current_end,
                content=chunk_content,
                context_header=sub_header,
                metadata={"part": part_idx, "is_split": True}
            ))

            if current_end >= end_line:
                break
            current_start = current_end - overlap + 1
            part_idx += 1

        return chunks

    def _fill_polyglot_gaps(
        self,
        lines: List[str],
        covered_lines: set[int],
        rel_path: str,
        abs_path: str,
        language: str
    ) -> List[CodeChunk]:
        chunks: List[CodeChunk] = []
        total_lines = len(lines)
        in_block = False
        block_start = 1

        for line_no in range(1, total_lines + 1):
            is_covered = line_no in covered_lines
            line_str = lines[line_no - 1].strip()

            if not is_covered and line_str:
                if not in_block:
                    in_block = True
                    block_start = line_no
            else:
                if in_block:
                    block_end = line_no - 1
                    if block_end - block_start + 1 >= self.config.min_chunk_lines:
                        content = "".join(lines[block_start - 1 : block_end])
                        if content.strip():
                            chunk_id = CodeChunk.generate_id(rel_path, block_start, block_end, content)
                            chunks.append(CodeChunk(
                                chunk_id=chunk_id,
                                file_path=rel_path,
                                absolute_path=abs_path,
                                language=language,
                                symbol_name=f"{Path(rel_path).stem}_L{block_start}",
                                symbol_type=SymbolType.BLOCK,
                                start_line=block_start,
                                end_line=block_end,
                                content=content,
                                context_header=f"{rel_path}:L{block_start}-L{block_end}"
                            ))
                    in_block = False

        if in_block:
            block_end = total_lines
            if block_end - block_start + 1 >= self.config.min_chunk_lines:
                content = "".join(lines[block_start - 1 : block_end])
                if content.strip():
                    chunk_id = CodeChunk.generate_id(rel_path, block_start, block_end, content)
                    chunks.append(CodeChunk(
                        chunk_id=chunk_id,
                        file_path=rel_path,
                        absolute_path=abs_path,
                        language=language,
                        symbol_name=f"{Path(rel_path).stem}_L{block_start}",
                        symbol_type=SymbolType.BLOCK,
                        start_line=block_start,
                        end_line=block_end,
                        content=content,
                        context_header=f"{rel_path}:L{block_start}-L{block_end}"
                    ))

        return chunks

    def _parse_markdown(self, lines: List[str], rel_path: str, abs_path: str) -> List[CodeChunk]:
        chunks: List[CodeChunk] = []
        total_lines = len(lines)
        
        section_starts: List[Tuple[int, str]] = []
        for idx, line in enumerate(lines):
            if line.startswith("#"):
                section_starts.append((idx + 1, line.strip()))

        if not section_starts:
            return self._sliding_window_chunks(lines, rel_path, abs_path, "markdown")

        for s_idx, (start_l, title) in enumerate(section_starts):
            end_l = section_starts[s_idx + 1][0] - 1 if s_idx + 1 < len(section_starts) else total_lines
            content = "".join(lines[start_l - 1 : end_l])
            if content.strip():
                chunk_id = CodeChunk.generate_id(rel_path, start_l, end_l, content)
                chunks.append(CodeChunk(
                    chunk_id=chunk_id,
                    file_path=rel_path,
                    absolute_path=abs_path,
                    language="markdown",
                    symbol_name=title.lstrip("#").strip()[:30],
                    symbol_type=SymbolType.BLOCK,
                    start_line=start_l,
                    end_line=end_l,
                    content=content,
                    context_header=f"Markdown Section: {title}"
                ))

        return chunks

    def _sliding_window_chunks(
        self, lines: List[str], rel_path: str, abs_path: str, language: str
    ) -> List[CodeChunk]:
        chunks: List[CodeChunk] = []
        total_lines = len(lines)
        max_lines = self.config.max_chunk_lines
        overlap = self.config.chunk_overlap_lines

        start = 1
        while start <= total_lines:
            end = min(total_lines, start + max_lines - 1)
            content = "".join(lines[start - 1 : end])
            if content.strip():
                chunk_id = CodeChunk.generate_id(rel_path, start, end, content)
                chunks.append(CodeChunk(
                    chunk_id=chunk_id,
                    file_path=rel_path,
                    absolute_path=abs_path,
                    language=language,
                    symbol_name=Path(rel_path).stem,
                    symbol_type=SymbolType.BLOCK,
                    start_line=start,
                    end_line=end,
                    content=content,
                    context_header=f"{rel_path}:L{start}-L{end}"
                ))
            if end >= total_lines:
                break
            start = end - overlap + 1

        return chunks
