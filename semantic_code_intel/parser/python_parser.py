"""
AST-based semantic code parser for Python files with exact line-level resolution.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Optional, Tuple
from semantic_code_intel.config import ParserConfig
from semantic_code_intel.parser.base import BaseParser, CodeChunk, ParseResult, SymbolType


class PythonASTParser(BaseParser):
    """
    Parses Python source code using the native ast module.
    Extracts module docstrings, classes, methods, functions, async functions,
    preserving exact line numbers and generating rich contextual headers.
    """

    def __init__(self, config: Optional[ParserConfig] = None):
        self.config = config or ParserConfig()

    def parse_file(self, file_path: Path, repo_root: Path) -> ParseResult:
        rel_path = str(file_path.relative_to(repo_root)) if file_path.is_relative_to(repo_root) else str(file_path)
        abs_path = str(file_path.resolve())

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                source_code = f.read()
        except Exception as e:
            return ParseResult(
                file_path=rel_path,
                language="python",
                error=f"Failed to read file: {e}"
            )

        lines = source_code.splitlines(keepends=True)
        total_lines = len(lines)
        total_bytes = len(source_code.encode("utf-8"))

        if total_lines == 0:
            return ParseResult(file_path=rel_path, language="python", total_lines=0, total_bytes=0)

        try:
            tree = ast.parse(source_code, filename=str(file_path))
        except SyntaxError:
            # Fallback to line-based chunker if AST parsing fails (e.g. invalid syntax)
            chunks = self._fallback_line_chunk(lines, rel_path, abs_path)
            return ParseResult(
                file_path=rel_path,
                language="python",
                chunks=chunks,
                total_lines=total_lines,
                total_bytes=total_bytes
            )

        chunks: List[CodeChunk] = []
        covered_lines = set()

        # 1. Extract Module Docstring & Top-level Imports/Constants
        module_doc = ast.get_docstring(tree)
        module_header_lines: List[str] = []
        imports: List[str] = []

        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.extend(self._extract_import_names(node))
                if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                    start = node.lineno
                    end = node.end_lineno or node.lineno
                    for line_no in range(start, end + 1):
                        covered_lines.add(line_no)

        if module_doc:
            doc_chunk = self._create_module_doc_chunk(module_doc, rel_path, abs_path, lines)
            if doc_chunk:
                chunks.append(doc_chunk)

        # 2. Traverse AST top-level definitions
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                class_chunks = self._parse_class_node(node, lines, rel_path, abs_path, imports)
                chunks.extend(class_chunks)
                if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                    for l in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                        covered_lines.add(l)

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_chunks = self._parse_function_node(
                    node, lines, rel_path, abs_path, parent_scope=None, imports=imports
                )
                chunks.extend(func_chunks)
                if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                    for l in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                        covered_lines.add(l)

        # 3. Capture uncovered standalone code blocks (scripts, globals, main block)
        uncovered_chunks = self._capture_uncovered_blocks(lines, covered_lines, rel_path, abs_path, imports)
        chunks.extend(uncovered_chunks)

        # If no chunks were created (e.g. small script), create a whole-file chunk
        if not chunks and total_lines > 0:
            content = "".join(lines)
            chunk_id = CodeChunk.generate_id(rel_path, 1, total_lines, content)
            chunks.append(CodeChunk(
                chunk_id=chunk_id,
                file_path=rel_path,
                absolute_path=abs_path,
                language="python",
                symbol_name=Path(rel_path).stem,
                symbol_type=SymbolType.MODULE,
                start_line=1,
                end_line=total_lines,
                content=content,
                context_header=f"Module {Path(rel_path).name}",
                dependencies=imports
            ))

        # Sort chunks by start line
        chunks.sort(key=lambda c: (c.start_line, c.end_line))

        return ParseResult(
            file_path=rel_path,
            language="python",
            chunks=chunks,
            total_lines=total_lines,
            total_bytes=total_bytes
        )

    def _extract_import_names(self, node: ast.AST) -> List[str]:
        names = []
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                names.append(f"{module}.{alias.name}" if module else alias.name)
        return names

    def _create_module_doc_chunk(
        self, docstring: str, rel_path: str, abs_path: str, lines: List[str]
    ) -> Optional[CodeChunk]:
        # Approximate start of docstring
        start_line = 1
        end_line = min(len(lines), len(docstring.splitlines()) + 2)
        content = "".join(lines[start_line - 1 : end_line])
        chunk_id = CodeChunk.generate_id(rel_path, start_line, end_line, content)
        return CodeChunk(
            chunk_id=chunk_id,
            file_path=rel_path,
            absolute_path=abs_path,
            language="python",
            symbol_name=Path(rel_path).stem,
            symbol_type=SymbolType.DOCSTRING,
            start_line=start_line,
            end_line=end_line,
            content=content,
            context_header=f"Module Docstring for {rel_path}",
            docstring=docstring
        )

    def _parse_class_node(
        self,
        node: ast.ClassDef,
        lines: List[str],
        rel_path: str,
        abs_path: str,
        imports: List[str]
    ) -> List[CodeChunk]:
        chunks: List[CodeChunk] = []
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        docstring = ast.get_docstring(node)
        class_name = node.name

        # Extract base classes
        bases = [ast.unparse(b) for b in node.bases if hasattr(ast, "unparse")]
        bases_str = f"({', '.join(bases)})" if bases else ""
        class_signature = f"class {class_name}{bases_str}"

        # Class header chunk (class def up to first method or first few lines)
        header_end_line = start_line
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                header_end_line = max(start_line, child.lineno - 1)
                break
            if hasattr(child, "end_lineno") and child.end_lineno:
                header_end_line = max(header_end_line, child.end_lineno)

        header_content = "".join(lines[start_line - 1 : header_end_line])
        if header_content.strip():
            chunk_id = CodeChunk.generate_id(rel_path, start_line, header_end_line, header_content)
            chunks.append(CodeChunk(
                chunk_id=chunk_id,
                file_path=rel_path,
                absolute_path=abs_path,
                language="python",
                symbol_name=class_name,
                symbol_type=SymbolType.CLASS,
                parent_scope=None,
                start_line=start_line,
                end_line=header_end_line,
                content=header_content,
                context_header=class_signature,
                docstring=docstring,
                dependencies=imports
            ))

        # Parse methods inside the class
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_chunks = self._parse_function_node(
                    child, lines, rel_path, abs_path, parent_scope=class_name, imports=imports
                )
                chunks.extend(method_chunks)
            elif isinstance(child, ast.ClassDef):
                nested_class_chunks = self._parse_class_node(
                    child, lines, rel_path, abs_path, imports=imports
                )
                chunks.extend(nested_class_chunks)

        return chunks

    def _parse_function_node(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        lines: List[str],
        rel_path: str,
        abs_path: str,
        parent_scope: Optional[str],
        imports: List[str]
    ) -> List[CodeChunk]:
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        docstring = ast.get_docstring(node)
        func_name = node.name
        is_async = isinstance(node, ast.AsyncFunctionDef)
        symbol_type = SymbolType.METHOD if parent_scope else (
            SymbolType.ASYNC_FUNCTION if is_async else SymbolType.FUNCTION
        )

        full_scope = f"{parent_scope}.{func_name}" if parent_scope else func_name
        
        # Build decorator string & signature
        decorators = [f"@{ast.unparse(d)}" for d in node.decorator_list if hasattr(ast, "unparse")]
        prefix = "async def" if is_async else "def"
        sig_args = ast.unparse(node.args) if hasattr(ast, "unparse") else "..."
        returns = f" -> {ast.unparse(node.returns)}" if getattr(node, "returns", None) and hasattr(ast, "unparse") else ""
        signature = f"{prefix} {func_name}({sig_args}){returns}"

        if decorators:
            header_context = f"{' '.join(decorators)} {signature}"
        else:
            header_context = signature

        if parent_scope:
            header_context = f"class {parent_scope} > {header_context}"

        content = "".join(lines[start_line - 1 : end_line])
        line_count = end_line - start_line + 1

        # Check if function exceeds max chunk lines. If so, split into sliding sub-chunks with signature context
        if line_count > self.config.max_chunk_lines:
            return self._split_large_block(
                lines=lines,
                start_line=start_line,
                end_line=end_line,
                rel_path=rel_path,
                abs_path=abs_path,
                symbol_name=func_name,
                symbol_type=symbol_type,
                parent_scope=parent_scope,
                context_header=header_context,
                docstring=docstring,
                imports=imports
            )

        chunk_id = CodeChunk.generate_id(rel_path, start_line, end_line, content)
        return [CodeChunk(
            chunk_id=chunk_id,
            file_path=rel_path,
            absolute_path=abs_path,
            language="python",
            symbol_name=func_name,
            symbol_type=symbol_type,
            parent_scope=parent_scope,
            start_line=start_line,
            end_line=end_line,
            content=content,
            context_header=header_context,
            docstring=docstring,
            dependencies=imports
        )]

    def _split_large_block(
        self,
        lines: List[str],
        start_line: int,
        end_line: int,
        rel_path: str,
        abs_path: str,
        symbol_name: str,
        symbol_type: SymbolType,
        parent_scope: Optional[str],
        context_header: str,
        docstring: Optional[str],
        imports: List[str]
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
                language="python",
                symbol_name=symbol_name,
                symbol_type=symbol_type,
                parent_scope=parent_scope,
                start_line=current_start,
                end_line=current_end,
                content=chunk_content,
                context_header=sub_header,
                docstring=docstring,
                dependencies=imports,
                metadata={"part": part_idx, "is_split": True}
            ))

            if current_end >= end_line:
                break
            current_start = current_end - overlap + 1
            part_idx += 1

        return chunks

    def _capture_uncovered_blocks(
        self,
        lines: List[str],
        covered_lines: set[int],
        rel_path: str,
        abs_path: str,
        imports: List[str]
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
                                language="python",
                                symbol_name=f"block_L{block_start}_L{block_end}",
                                symbol_type=SymbolType.BLOCK,
                                start_line=block_start,
                                end_line=block_end,
                                content=content,
                                context_header=f"Module Block {rel_path}:L{block_start}-L{block_end}",
                                dependencies=imports
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
                        language="python",
                        symbol_name=f"block_L{block_start}_L{block_end}",
                        symbol_type=SymbolType.BLOCK,
                        start_line=block_start,
                        end_line=block_end,
                        content=content,
                        context_header=f"Module Block {rel_path}:L{block_start}-L{block_end}",
                        dependencies=imports
                    ))

        return chunks

    def _fallback_line_chunk(self, lines: List[str], rel_path: str, abs_path: str) -> List[CodeChunk]:
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
                    language="python",
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
