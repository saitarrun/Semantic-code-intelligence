"""
SQLite-backed persistent metadata store for code chunks, file hashes, and indexing manifest.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional
from semantic_code_intel.parser.base import CodeChunk, SymbolType

logger = logging.getLogger(__name__)


class MetadataStore:
    """Manages SQLite persistence for chunk metadata, line mappings, and incremental indexing state."""

    def __init__(self, db_path: Path):
        self.db_path = db_path.resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for high concurrent read performance
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        """Create necessary database tables and indices if they do not exist."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    absolute_path TEXT NOT NULL,
                    language TEXT NOT NULL,
                    symbol_name TEXT,
                    symbol_type TEXT NOT NULL,
                    parent_scope TEXT,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    context_header TEXT,
                    docstring TEXT,
                    dependencies_json TEXT,
                    metadata_json TEXT,
                    created_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_file_path ON chunks(file_path);
                CREATE INDEX IF NOT EXISTS idx_chunks_symbol_name ON chunks(symbol_name);
                CREATE INDEX IF NOT EXISTS idx_chunks_lines ON chunks(file_path, start_line, end_line);

                CREATE TABLE IF NOT EXISTS files (
                    file_path TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    total_lines INTEGER NOT NULL,
                    total_bytes INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    indexed_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS manifest (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
            """)

    def save_chunks(self, chunks: List[CodeChunk]) -> None:
        """Batch insert or replace chunks into the database."""
        if not chunks:
            return

        now = time.time()
        records = []
        for c in chunks:
            records.append((
                c.chunk_id,
                c.file_path,
                c.absolute_path,
                c.language,
                c.symbol_name,
                c.symbol_type.value,
                c.parent_scope,
                c.start_line,
                c.end_line,
                c.content,
                c.context_header,
                c.docstring,
                json.dumps(c.dependencies),
                json.dumps(c.metadata),
                now
            ))

        with self._get_connection() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO chunks (
                    chunk_id, file_path, absolute_path, language, symbol_name,
                    symbol_type, parent_scope, start_line, end_line, content,
                    context_header, docstring, dependencies_json, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
            conn.commit()

        logger.debug(f"Saved {len(chunks)} chunks to SQLite metadata store.")

    def get_chunk(self, chunk_id: str) -> Optional[CodeChunk]:
        """Fetch a single chunk by chunk_id."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
            if row:
                return self._row_to_chunk(row)
        return None

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[CodeChunk]:
        """Batch fetch multiple chunks by their IDs preserving order."""
        if not chunk_ids:
            return []

        placeholders = ",".join("?" * len(chunk_ids))
        with self._get_connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})",
                chunk_ids
            ).fetchall()

        lookup: Dict[str, CodeChunk] = {}
        for row in rows:
            chunk = self._row_to_chunk(row)
            lookup[chunk.chunk_id] = chunk

        # Maintain original order
        return [lookup[cid] for cid in chunk_ids if cid in lookup]

    def get_all_chunks(self) -> List[CodeChunk]:
        """Retrieve all chunks from the database."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM chunks ORDER BY file_path, start_line").fetchall()
            return [self._row_to_chunk(r) for r in rows]

    def record_file(
        self,
        file_path: str,
        file_hash: str,
        total_lines: int,
        total_bytes: int,
        chunk_count: int
    ) -> None:
        """Record indexed file status and hash for incremental indexing."""
        now = time.time()
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO files (
                    file_path, file_hash, total_lines, total_bytes, chunk_count, indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (file_path, file_hash, total_lines, total_bytes, chunk_count, now))
            conn.commit()

    def get_file_hash(self, file_path: str) -> Optional[str]:
        """Get the stored hash of a file."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT file_hash FROM files WHERE file_path = ?", (file_path,)).fetchone()
            if row:
                return row["file_hash"]
        return None

    def get_stats(self) -> Dict[str, int]:
        """Return total indexed file count, chunk count, and line count."""
        with self._get_connection() as conn:
            total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            total_files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            total_lines = conn.execute("SELECT COALESCE(SUM(total_lines), 0) FROM files").fetchone()[0]
            total_bytes = conn.execute("SELECT COALESCE(SUM(total_bytes), 0) FROM files").fetchone()[0]

        return {
            "total_chunks": total_chunks,
            "total_files": total_files,
            "total_lines": total_lines,
            "total_bytes": total_bytes,
        }

    def set_manifest_val(self, key: str, value: Any) -> None:
        """Set a value in the manifest key-value store."""
        val_str = json.dumps(value)
        with self._get_connection() as conn:
            conn.execute("INSERT OR REPLACE INTO manifest (key, value) VALUES (?, ?)", (key, val_str))
            conn.commit()

    def get_manifest_val(self, key: str, default: Any = None) -> Any:
        """Get a value from the manifest key-value store."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT value FROM manifest WHERE key = ?", (key,)).fetchone()
            if row:
                return json.loads(row["value"])
        return default

    def clear(self) -> None:
        """Wipe all indexed data."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM files")
            conn.execute("DELETE FROM manifest")
            conn.commit()

    def _row_to_chunk(self, row: sqlite3.Row) -> CodeChunk:
        return CodeChunk(
            chunk_id=row["chunk_id"],
            file_path=row["file_path"],
            absolute_path=row["absolute_path"],
            language=row["language"],
            symbol_name=row["symbol_name"],
            symbol_type=SymbolType(row["symbol_type"]),
            parent_scope=row["parent_scope"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            content=row["content"],
            context_header=row["context_header"],
            docstring=row["docstring"],
            dependencies=json.loads(row["dependencies_json"] or "[]"),
            metadata=json.loads(row["metadata_json"] or "{}")
        )
