"""
Ignore rules and file filter for repository scanning (.gitignore and default patterns).
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import List, Optional
from semantic_code_intel.config import ParserConfig


class IgnoreFilter:
    """Evaluates whether paths should be ignored during codebase indexing."""

    def __init__(self, repo_root: Path, config: Optional[ParserConfig] = None):
        self.repo_root = repo_root.resolve()
        self.config = config or ParserConfig()
        self.patterns: List[str] = list(self.config.exclude_patterns)
        self._load_gitignore()

    def _load_gitignore(self) -> None:
        """Load patterns from root .gitignore and .ignore if they exist."""
        for filename in [".gitignore", ".ignore"]:
            ignore_path = self.repo_root / filename
            if ignore_path.is_file():
                try:
                    with open(ignore_path, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#"):
                                self.patterns.append(line)
                except Exception:
                    pass

    def should_ignore(self, path: Path) -> bool:
        """Check if a given file or directory should be ignored."""
        try:
            rel_path = path.relative_to(self.repo_root)
            rel_str = str(rel_path).replace("\\", "/")
        except ValueError:
            rel_str = str(path).replace("\\", "/")

        name = path.name

        # Check default pattern matches
        for pat in self.patterns:
            pat_clean = pat.rstrip("/")
            if fnmatch.fnmatch(name, pat_clean):
                return True
            if fnmatch.fnmatch(rel_str, pat_clean):
                return True
            if fnmatch.fnmatch(rel_str, f"*{pat_clean}*"):
                return True

        # Check path parts
        for part in path.parts:
            if part.startswith(".") and part not in [".", ".."]:
                # Ignore hidden directories by default (.git, .cache, etc.)
                return True
            for pat in self.patterns:
                if fnmatch.fnmatch(part, pat.rstrip("/")):
                    return True

        # Check file extension and size if it's a file
        if path.is_file():
            ext = path.suffix.lower()
            if ext not in self.config.include_extensions and path.name not in ["Dockerfile", "Makefile", "Jenkinsfile"]:
                return True
            try:
                if path.stat().st_size > self.config.max_file_size_bytes:
                    return True
            except OSError:
                return True

        return False
