"""
Unified Hybrid Indexing Engine orchestrating Code Parsing, FAISS dense indexing,
BM25 lexical indexing, and SQLite metadata persistence.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional
from semantic_code_intel.config import CodeIntelConfig
from semantic_code_intel.indexing.bm25_index import BM25SparseIndex
from semantic_code_intel.indexing.embeddings import EmbeddingEngine
from semantic_code_intel.indexing.faiss_index import FAISSDenseIndex
from semantic_code_intel.indexing.metadata_store import MetadataStore
from semantic_code_intel.parser.base import CodeChunk
from semantic_code_intel.parser.scanner import CodebaseScanner

logger = logging.getLogger(__name__)


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file for incremental indexing change detection."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


class HybridIndexer:
    """
    Orchestrates end-to-end repository indexing into dense FAISS and sparse BM25 indices.
    """

    def __init__(self, config: Optional[CodeIntelConfig] = None):
        self.config = config or CodeIntelConfig()
        self.index_dir = self.config.get_index_dir()
        self.metadata_store = MetadataStore(self.index_dir / self.config.storage.metadata_db_file)
        self.embedding_engine = EmbeddingEngine(self.config.embedding)
        self.scanner = CodebaseScanner(self.config)

    def index_codebase(
        self,
        target_dir: Optional[Path] = None,
        force_reindex: bool = False,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> Dict[str, float]:
        """
        Index a target codebase directory.
        Returns indexing metrics (elapsed seconds, chunk count, lines count, etc.).
        """
        start_time = time.time()
        repo_root = (target_dir or self.config.project_root).resolve()
        self.index_dir.mkdir(parents=True, exist_ok=True)

        if force_reindex:
            logger.info("Force reindex requested: Clearing existing metadata and indices.")
            self.metadata_store.clear()

        # Step 1: Scan and parse codebase files
        files = self.scanner.discover_files(repo_root)
        total_files = len(files)
        logger.info(f"Discovered {total_files} source files in {repo_root}")

        all_chunks: List[CodeChunk] = []
        total_lines = 0
        total_bytes = 0

        for idx, file_path in enumerate(files):
            if progress_callback:
                progress_callback(f"Parsing: {file_path.name}", idx + 1, total_files)

            file_hash = compute_file_sha256(file_path)
            parser = self.scanner.get_parser_for_file(file_path)
            parse_res = parser.parse_file(file_path, repo_root)

            total_lines += parse_res.total_lines
            total_bytes += parse_res.total_bytes
            all_chunks.extend(parse_res.chunks)

            # Record file in metadata store
            self.metadata_store.record_file(
                file_path=parse_res.file_path,
                file_hash=file_hash,
                total_lines=parse_res.total_lines,
                total_bytes=parse_res.total_bytes,
                chunk_count=len(parse_res.chunks)
            )

        parse_time = time.time() - start_time
        logger.info(
            f"Parsed {len(all_chunks)} chunks across {total_files} files "
            f"({total_lines} lines of code) in {parse_time:.2f}s"
        )

        if not all_chunks:
            logger.warning("No code chunks extracted from repository.")
            return {
                "total_files": total_files,
                "total_lines": total_lines,
                "total_chunks": 0,
                "elapsed_seconds": time.time() - start_time,
            }

        # Step 2: Store chunks in SQLite metadata store
        if progress_callback:
            progress_callback("Persisting metadata into SQLite...", total_files, total_files)
        self.metadata_store.save_chunks(all_chunks)

        # Step 3: Compute dense embeddings and build FAISS index
        if progress_callback:
            progress_callback("Computing dense vector embeddings...", total_files, total_files)

        searchable_texts = [c.get_searchable_text() for c in all_chunks]
        chunk_ids = [c.chunk_id for c in all_chunks]

        embed_start = time.time()
        embeddings = self.embedding_engine.encode_texts(
            searchable_texts,
            batch_size=self.config.embedding.batch_size,
            show_progress_bar=False
        )
        embed_time = time.time() - embed_start

        # Create FAISS Dense Index
        faiss_index = FAISSDenseIndex(dimension=embeddings.shape[1])
        faiss_index.add_vectors(embeddings, chunk_ids)
        faiss_index.save(self.index_dir, self.config.storage.faiss_index_file)

        # Step 4: Build BM25 Sparse Index
        if progress_callback:
            progress_callback("Building BM25 sparse lexical index...", total_files, total_files)

        bm25_start = time.time()
        bm25_index = BM25SparseIndex(self.config.bm25)
        bm25_index.build_index(chunk_ids, searchable_texts)
        bm25_index.save(self.index_dir, self.config.storage.bm25_index_file)
        bm25_time = time.time() - bm25_start

        # Step 5: Save manifest summary
        manifest = {
            "version": "1.0.0",
            "repo_root": str(repo_root),
            "indexed_at": time.time(),
            "total_files": total_files,
            "total_lines": total_lines,
            "total_bytes": total_bytes,
            "total_chunks": len(all_chunks),
            "embedding_model": self.config.embedding.model_name,
            "embedding_dim": embeddings.shape[1],
            "device": self.config.embedding.device
        }
        self.metadata_store.set_manifest_val("index_manifest", manifest)

        total_elapsed = time.time() - start_time
        logger.info(f"Indexing completed in {total_elapsed:.2f}s!")

        return {
            "total_files": float(total_files),
            "total_lines": float(total_lines),
            "total_chunks": float(len(all_chunks)),
            "parse_time_seconds": parse_time,
            "embed_time_seconds": embed_time,
            "bm25_time_seconds": bm25_time,
            "elapsed_seconds": total_elapsed,
        }
