"""
Semantic Git Commit Message Generator.
Analyzes git diffs and code modifications to generate concise Conventional Commits.
Adheres strictly to zero-attribution rules (no Co-Authored-By or AI tags).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class SemanticCommitGenerator:
    """Generates standard Conventional Commit messages from local git diffs."""

    def __init__(self, repo_path: Optional[Path] = None):
        self.repo_path = (repo_path or Path.cwd()).resolve()

    def get_git_diff(self, staged_only: bool = False) -> Tuple[str, List[str]]:
        """Retrieves unified git diff and list of modified files."""
        try:
            # Check staged diff first
            cmd = ["git", "diff", "--cached"] if staged_only else ["git", "diff", "--cached"]
            res = subprocess.run(
                cmd,
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                check=False
            )
            diff_text = res.stdout

            # If no staged changes and not strictly staged_only, get working tree diff
            if not diff_text.strip() and not staged_only:
                res_work = subprocess.run(
                    ["git", "diff", "HEAD"],
                    cwd=str(self.repo_path),
                    capture_output=True,
                    text=True,
                    check=False
                )
                diff_text = res_work.stdout or ""

                if not diff_text.strip():
                    res_untracked = subprocess.run(
                        ["git", "status", "--short"],
                        cwd=str(self.repo_path),
                        capture_output=True,
                        text=True,
                        check=False
                    )
                    diff_text = res_untracked.stdout or ""

            # Extract modified file paths
            files_res = subprocess.run(
                ["git", "diff", "--name-only", "HEAD"],
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                check=False
            )
            changed_files = [f.strip() for f in files_res.stdout.splitlines() if f.strip()]
            if not changed_files:
                status_res = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=str(self.repo_path),
                    capture_output=True,
                    text=True,
                    check=False
                )
                changed_files = [
                    line[3:].strip() for line in status_res.stdout.splitlines() if len(line) > 3
                ]

            return diff_text, changed_files
        except Exception as e:
            return "", []

    def generate_commit_message(self, staged_only: bool = False) -> Dict[str, Any]:
        """Analyzes code changes and constructs a structured Conventional Commit message."""
        diff_text, changed_files = self.get_git_diff(staged_only=staged_only)

        if not changed_files and not diff_text.strip():
            return {
                "title": "chore: working tree clean",
                "body": "No modified or staged changes found in the repository.",
                "type": "chore",
                "scope": "",
                "full_message": "chore: working tree clean\n\nNo modified or staged changes found.",
                "changed_files": []
            }

        # 1. Determine Scope
        scope = ""
        modules = set()
        for f in changed_files:
            parts = Path(f).parts
            if "tests" in parts:
                modules.add("tests")
            elif "api" in parts:
                modules.add("api")
            elif "graph" in parts:
                modules.add("graph")
            elif "lsp" in parts:
                modules.add("lsp")
            elif "cli" in parts:
                modules.add("cli")
            elif "indexing" in parts:
                modules.add("indexing")
            elif "retrieval" in parts:
                modules.add("retrieval")
            elif "generation" in parts:
                modules.add("generation")
            elif "git_intel" in parts:
                modules.add("git")

        if len(modules) == 1:
            scope = f"({list(modules)[0]})"
        elif len(modules) > 1:
            scope = f"({list(modules)[0]})"

        # 2. Determine Commit Type
        has_tests = any("test" in f.lower() for f in changed_files)
        has_docs = any(f.endswith(".md") or "docs" in f.lower() for f in changed_files)
        has_fixes = bool(re.search(r'\b(fix|bug|issue|error|exception|patch)\b', diff_text, re.IGNORECASE))
        has_perf = bool(re.search(r'\b(perf|optimiz|speed|latency|hnsw|cache)\b', diff_text, re.IGNORECASE))
        has_new_features = bool(re.search(r'^\+\s*def |^\+\s*class |^\+\s*async def ', diff_text, re.MULTILINE))

        if has_tests and len(changed_files) == 1:
            c_type = "test"
        elif has_docs and len(changed_files) == 1:
            c_type = "docs"
        elif has_perf:
            c_type = "perf"
        elif has_fixes:
            c_type = "fix"
        elif has_new_features:
            c_type = "feat"
        else:
            c_type = "refactor" if len(changed_files) > 1 else "chore"

        # 3. Extract touched symbols and summarize
        added_symbols = re.findall(r'^\+\s*(?:def|class|async def)\s+([a-zA-Z0-9_]+)', diff_text, re.MULTILINE)
        unique_symbols = list(dict.fromkeys(added_symbols))[:5]

        # 4. Construct Subject Title
        if unique_symbols:
            subject = f"Add {', '.join(unique_symbols[:2])} implementation"
        elif changed_files:
            main_file = Path(changed_files[0]).name
            subject = f"Update {main_file} logic and dependencies"
        else:
            subject = "Update codebase changes"

        title = f"{c_type}{scope}: {subject.lower()}"

        # 5. Construct Clean Bulleted Body (Strictly no AI tags)
        bullet_points = []
        for f in changed_files[:6]:
            bullet_points.append(f"- Update `{f}`")
        if unique_symbols:
            bullet_points.append(f"- Implement symbols: {', '.join(f'`{s}`' for s in unique_symbols)}")

        body = "\n".join(bullet_points)
        full_message = f"{title}\n\n{body}"

        return {
            "title": title,
            "body": body,
            "type": c_type,
            "scope": scope.strip("()"),
            "full_message": full_message,
            "changed_files": changed_files
        }
