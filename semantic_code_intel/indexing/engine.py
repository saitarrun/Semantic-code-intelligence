"""
Unified Hybrid Indexing Engine orchestrating Code Parsing, FAISS dense indexing,
BM25 lexical indexing, and SQLite metadata persistence with granular progress tracking.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
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

    def is_indexed(self) -> bool:
        """Check if required index files exist on disk."""
        faiss_p = self.index_dir / self.config.storage.faiss_index_file
        faiss_npy = self.index_dir / f"{self.config.storage.faiss_index_file}.npy"
        bm25_p = self.index_dir / self.config.storage.bm25_index_file
        meta_p = self.index_dir / self.config.storage.metadata_db_file
        return (faiss_p.exists() or faiss_npy.exists()) and bm25_p.exists() and meta_p.exists()

    def index_codebase(
        self,
        target_dir: Optional[Path] = None,
        force_reindex: bool = False,
        progress_callback: Optional[Callable[[str, int, int, str, float], None]] = None
    ) -> Dict[str, float]:
        """
        Index a target codebase directory with stage-aware progress reporting and incremental caching.
        """
        start_time = time.time()
        repo_root = (target_dir or self.config.project_root).resolve()
        if target_dir and target_dir != self.config.project_root:
            self.config.project_root = repo_root
            self.index_dir = self.config.get_index_dir()
            self.metadata_store = MetadataStore(self.index_dir / self.config.storage.metadata_db_file)

        self.index_dir.mkdir(parents=True, exist_ok=True)

        if force_reindex:
            logger.info("Force reindex requested: Clearing existing metadata and indices.")
            self.metadata_store.clear()

        # Step 1: Scan and parse codebase files in parallel (0% -> 25%)
        files = self.scanner.discover_files(repo_root)
        total_files = len(files)
        logger.info(f"Discovered {total_files} source files in {repo_root}")

        # Check Cache: If not forced and index exists, verify file hashes
        if not force_reindex and self.is_indexed():
            existing_file_hashes = self.metadata_store.get_all_file_hashes()
            current_files_map = {str(fp.relative_to(repo_root)): fp for fp in files}

            deleted_files = [p for p in existing_file_hashes if p not in current_files_map]
            modified_or_new: List[Path] = []
            for rel_str, fp in current_files_map.items():
                old_hash = existing_file_hashes.get(rel_str)
                if old_hash is None or compute_file_sha256(fp) != old_hash:
                    modified_or_new.append(fp)

            # 100% Cache Hit: No files added, modified, or deleted
            if not deleted_files and not modified_or_new and existing_file_hashes:
                manifest_data = self.metadata_store.get_manifest_val("index_manifest", {})
                t_chunks = manifest_data.get("total_chunks", 0)
                t_lines = manifest_data.get("total_lines", 0)
                logger.info(f"Repository index is up to date (cache hit): {total_files} files, {t_chunks} chunks.")
                if progress_callback:
                    progress_callback("done", total_files, total_files, f"Index is up to date ({total_files} files, cache hit).", 100.0)
                return {
                    "total_files": float(total_files),
                    "total_lines": float(t_lines),
                    "total_chunks": float(t_chunks),
                    "parse_time_seconds": 0.0,
                    "embed_time_seconds": 0.0,
                    "bm25_time_seconds": 0.0,
                    "elapsed_seconds": round(time.time() - start_time, 3),
                    "loc_per_second": 0.0,
                }

        all_chunks: List[CodeChunk] = []
        file_records: List[tuple[str, str, int, int, int]] = []
        total_lines = 0
        total_bytes = 0

        def _parse_single_file(fp: Path):
            f_hash = compute_file_sha256(fp)
            p = self.scanner.get_parser_for_file(fp)
            p_res = p.parse_file(fp, repo_root)
            return fp, f_hash, p_res

        from concurrent.futures import ThreadPoolExecutor, as_completed
        max_workers = min(16, (os.cpu_count() or 4) * 2)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(_parse_single_file, fp): fp for fp in files}
            for idx, future in enumerate(as_completed(future_to_file)):
                fp, f_hash, p_res = future.result()
                total_lines += p_res.total_lines
                total_bytes += p_res.total_bytes
                all_chunks.extend(p_res.chunks)
                file_records.append((
                    p_res.file_path,
                    f_hash,
                    p_res.total_lines,
                    p_res.total_bytes,
                    len(p_res.chunks)
                ))

                if progress_callback and (idx % 5 == 0 or idx == total_files - 1):
                    pct = round(((idx + 1) / max(total_files, 1)) * 25.0, 1)
                    progress_callback("parsing", idx + 1, total_files, f"Parsing AST: {fp.name}", pct)

        # Batch record file statuses into SQLite
        self.metadata_store.record_files_batch(file_records)

        parse_time = time.time() - start_time
        logger.info(
            f"Parsed {len(all_chunks)} chunks across {total_files} files "
            f"({total_lines} lines of code) in {parse_time:.2f}s"
        )

        if not all_chunks:
            logger.warning("No code chunks extracted from repository.")
            if progress_callback:
                progress_callback("done", total_files, total_files, "Completed with 0 chunks.", 100.0)
            return {
                "total_files": total_files,
                "total_lines": total_lines,
                "total_chunks": 0,
                "elapsed_seconds": time.time() - start_time,
            }

        # Step 2: Store chunks in SQLite metadata store (25% -> 30%)
        if progress_callback:
            progress_callback("persisting", len(all_chunks), len(all_chunks), "Hydrating SQLite metadata store...", 27.0)
        self.metadata_store.save_chunks(all_chunks)

        # Step 3: Compute dense embeddings and build FAISS index (30% -> 85%)
        searchable_texts = [c.get_searchable_text() for c in all_chunks]
        chunk_ids = [c.chunk_id for c in all_chunks]
        total_chunks = len(all_chunks)

        def on_embed_progress(current: int, total: int):
            if progress_callback:
                embed_pct = 30.0 + ((current / max(total, 1)) * 55.0)
                progress_callback(
                    "embedding",
                    current,
                    total,
                    f"Generating dense embeddings: {current}/{total} chunks",
                    round(embed_pct, 1)
                )

        embed_start = time.time()
        embeddings = self.embedding_engine.encode_texts(
            searchable_texts,
            batch_size=self.config.embedding.batch_size,
            show_progress_bar=False,
            progress_callback=on_embed_progress
        )
        embed_time = time.time() - embed_start

        # Create FAISS Dense Index (85% -> 90%)
        if progress_callback:
            progress_callback("indexing", total_chunks, total_chunks, "Saving normalized FAISS vector index...", 88.0)
        faiss_index = FAISSDenseIndex(dimension=embeddings.shape[1])
        faiss_index.add_vectors(embeddings, chunk_ids)
        faiss_index.save(self.index_dir, self.config.storage.faiss_index_file)

        # Step 4: Build BM25 Sparse Index (90% -> 97%)
        if progress_callback:
            progress_callback("bm25", total_chunks, total_chunks, "Building BM25 sparse inverted index...", 93.0)

        bm25_start = time.time()
        bm25_index = BM25SparseIndex(self.config.bm25)
        bm25_index.build_index(chunk_ids, searchable_texts)
        bm25_index.save(self.index_dir, self.config.storage.bm25_index_file)
        bm25_time = time.time() - bm25_start

        # Step 5: Save manifest metadata (97% -> 100%)
        manifest_data = {
            "project_root": str(repo_root),
            "total_files": total_files,
            "total_lines": total_lines,
            "total_chunks": total_chunks,
            "embedding_model": self.config.embedding.model_name,
            "embedding_dim": embeddings.shape[1],
            "device": self.config.embedding.device,
            "created_at": time.time(),
            "parse_time_seconds": round(parse_time, 3),
            "embed_time_seconds": round(embed_time, 3),
            "bm25_time_seconds": round(bm25_time, 3),
            "total_indexing_time_seconds": round(time.time() - start_time, 3),
        }
        self.metadata_store.set_manifest_val("index_manifest", manifest_data)

        if progress_callback:
            progress_callback("done", total_chunks, total_chunks, "Indexing complete!", 100.0)

        elapsed = time.time() - start_time
        metrics = {
            "total_files": float(total_files),
            "total_lines": float(total_lines),
            "total_chunks": float(total_chunks),
            "parse_time_seconds": round(parse_time, 2),
            "embed_time_seconds": round(embed_time, 2),
            "bm25_time_seconds": round(bm25_time, 2),
            "elapsed_seconds": round(elapsed, 2),
            "loc_per_second": round(total_lines / max(elapsed, 0.001), 1),
        }

        logger.info(
            f"Indexing completed: {total_chunks} chunks ({total_lines} LOC) "
            f"in {elapsed:.2f}s ({metrics['loc_per_second']} LOC/s)"
        )
        return metrics
