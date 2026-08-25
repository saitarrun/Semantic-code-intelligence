"""Unit tests for Python AST and Polyglot structural code parsers."""

import tempfile
from pathlib import Path
from semantic_code_intel.config import ParserConfig
from semantic_code_intel.parser.base import SymbolType
from semantic_code_intel.parser.ignore_rules import IgnoreFilter
from semantic_code_intel.parser.polyglot_parser import PolyglotParser
from semantic_code_intel.parser.python_parser import PythonASTParser
from semantic_code_intel.parser.scanner import CodebaseScanner


def test_python_ast_parser():
    sample_code = '''"""Module docstring for auth service."""

import os
from typing import Optional

class AuthenticationManager:
    """Handles token validation and hashing."""
    def __init__(self, secret_key: str):
        self.secret = secret_key

    def verify_token(self, token: str) -> bool:
        """Verify JWT signature."""
        return len(token) > 10

async def generate_session_id(user_id: int) -> str:
    """Generate async session token."""
    return f"sess_{user_id}"
'''
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        py_file = root / "auth_service.py"
        py_file.write_text(sample_code, encoding="utf-8")

        parser = PythonASTParser()
        result = parser.parse_file(py_file, root)

        assert result.error is None
        assert result.total_lines > 15
        assert len(result.chunks) >= 3

        # Check module docstring chunk
        doc_chunks = [c for c in result.chunks if c.symbol_type == SymbolType.DOCSTRING]
        assert len(doc_chunks) == 1
        assert "auth service" in doc_chunks[0].content

        # Check class and method
        class_chunks = [c for c in result.chunks if c.symbol_type == SymbolType.CLASS]
        assert len(class_chunks) >= 1
        assert class_chunks[0].symbol_name == "AuthenticationManager"

        method_chunks = [c for c in result.chunks if c.symbol_type == SymbolType.METHOD]
        assert any(m.symbol_name == "verify_token" for m in method_chunks)

        # Check async function
        async_chunks = [c for c in result.chunks if c.symbol_type == SymbolType.ASYNC_FUNCTION]
        assert len(async_chunks) == 1
        assert async_chunks[0].symbol_name == "generate_session_id"


def test_polyglot_parser():
    ts_code = '''
export interface UserData {
    id: string;
    email: string;
}

export class UserService {
    private users: Map<string, UserData> = new Map();

    public async findUser(id: string): Promise<UserData | null> {
        return this.users.get(id) || null;
    }
}
'''
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        ts_file = root / "UserService.ts"
        ts_file.write_text(ts_code, encoding="utf-8")

        parser = PolyglotParser()
        result = parser.parse_file(ts_file, root)

        assert result.language == "typescript"
        assert len(result.chunks) >= 1
        symbols = [c.symbol_name for c in result.chunks if c.symbol_name]
        assert "UserData" in symbols or "UserService" in symbols


def test_ignore_filter():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / ".gitignore").write_text("*.secret\nignored_dir/\n", encoding="utf-8")
        
        filter_obj = IgnoreFilter(root)
        
        assert filter_obj.should_ignore(root / ".git" / "config") is True
        assert filter_obj.should_ignore(root / "__pycache__" / "test.pyc") is True
        assert filter_obj.should_ignore(root / "app.py") is False
        assert filter_obj.should_ignore(root / "keys.secret") is True
        assert filter_obj.should_ignore(root / "ignored_dir" / "file.py") is True
