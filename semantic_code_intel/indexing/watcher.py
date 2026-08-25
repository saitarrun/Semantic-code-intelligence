"""
Real-time Background Filesystem Watcher for Zero-Latency Incremental Indexing.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from semantic_code_intel.config import CodeIntelConfig
from semantic_code_intel.indexing.engine import HybridIndexer, compute_file_sha256

logger = logging.getLogger(__name__)


class IncrementalIndexHandler(FileSystemEventHandler):
    """Handles file modification, creation, and deletion events for incremental indexing."""

    def __init__(
        self,
        config: CodeIntelConfig,
        indexer: HybridIndexer,
        on_change_callback: Optional[Callable[[str, str], None]] = None,
        debounce_seconds: float = 0.5
    ):
        super().__init__()
        self.config = config
        self.indexer = indexer
        self.on_change_callback = on_change_callback
        self.debounce_seconds = debounce_seconds
        self.project_root = self.config.project_root.resolve()
        self._pending_files: Set[Path] = set()
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._queue_file(Path(event.src_path))

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._queue_file(Path(event.src_path))

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._queue_file(Path(event.src_path), is_deletion=True)

    def _queue_file(self, file_path: Path, is_deletion: bool = False) -> None:
        # Ignore index storage directory and hidden temp files
        if ".code_intel_index" in file_path.parts or file_path.name.startswith("."):
            return

        # Check if extension is supported
        ext = file_path.suffix.lower()
        if ext not in self.config.supported_extensions and not is_deletion:
            return

        with self._lock:
            self._pending_files.add(file_path)
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(self.debounce_seconds, self._flush_pending)
            self._timer.daemon = True
            self._timer.start()

    def _flush_pending(self) -> None:
        with self._lock:
            files_to_process = list(self._pending_files)
            self._pending_files.clear()

        if not files_to_process:
            return

        logger.info(f"Incremental Watcher processing {len(files_to_process)} changed file(s)...")
        for f in files_to_process:
            try:
                rel_path = f.relative_to(self.project_root).as_posix()
            except ValueError:
                rel_path = f.as_posix()

            if not f.exists():
                # File was deleted: delete from metadata store
                logger.info(f"File deleted on disk: {rel_path}")
                self.indexer.metadata_store.delete_file_chunks(rel_path)
                if self.on_change_callback:
                    self.on_change_callback("deleted", rel_path)
            else:
                # File modified/created: re-parse and record
                parser = self.indexer.scanner.get_parser_for_file(f)
                parse_res = parser.parse_file(f, self.project_root)
                file_hash = compute_file_sha256(f)

                # Update SQLite metadata store
                self.indexer.metadata_store.delete_file_chunks(parse_res.file_path)
                self.indexer.metadata_store.save_chunks(parse_res.chunks)
                self.indexer.metadata_store.record_file(
                    file_path=parse_res.file_path,
                    file_hash=file_hash,
                    total_lines=parse_res.total_lines,
                    total_bytes=parse_res.total_bytes,
                    chunk_count=len(parse_res.chunks)
                )
                logger.info(f"Incrementally updated {rel_path} ({len(parse_res.chunks)} chunks)")

                if self.on_change_callback:
                    self.on_change_callback("updated", rel_path)


class CodebaseWatcher:
    """Manages the background filesystem watcher thread."""

    def __init__(
        self,
        config: Optional[CodeIntelConfig] = None,
        on_change_callback: Optional[Callable[[str, str], None]] = None
    ):
        self.config = config or CodeIntelConfig()
        self.indexer = HybridIndexer(self.config)
        self.on_change_callback = on_change_callback
        self.observer: Optional[Observer] = None
        self._is_running = False

    def start(self) -> None:
        if self._is_running:
            return
        repo_root = self.config.project_root.resolve()
        handler = IncrementalIndexHandler(
            config=self.config,
            indexer=self.indexer,
            on_change_callback=self.on_change_callback
        )
        self.observer = Observer()
        self.observer.schedule(handler, str(repo_root), recursive=True)
        self.observer.daemon = True
        self.observer.start()
        self._is_running = True
        logger.info(f"CodebaseWatcher started watching {repo_root}")

    def stop(self) -> None:
        if self.observer and self._is_running:
            self.observer.stop()
            self.observer.join(timeout=2.0)
            self._is_running = False
            logger.info("CodebaseWatcher stopped")

    @property
    def is_running(self) -> bool:
        return self._is_running
