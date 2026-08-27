"""
Security and Sandboxing Module for Semantic Code Intelligence.
Provides path traversal prevention, secret/credential redaction,
git clone argument sanitization, rate limiting, and HTTP security headers.
"""

from __future__ import annotations

import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader, APIKeyQuery

# Common API Key & Credential Regex Patterns
SECRET_PATTERNS = [
    # Anthropic API keys (placed before general sk-)
    (re.compile(r"sk-ant-[a-zA-Z0-9_\-]{20,}", re.IGNORECASE), "[REDACTED_ANTHROPIC_KEY]"),
    # OpenAI API keys
    (re.compile(r"sk-(?:proj-)?[a-zA-Z0-9_\-]{20,}", re.IGNORECASE), "[REDACTED_API_KEY]"),
    # GitHub Personal Access Tokens & OAuth tokens
    (re.compile(r"ghp_[a-zA-Z0-9]{36}"), "[REDACTED_GITHUB_PAT]"),
    (re.compile(r"github_pat_[a-zA-Z0-9_]{40,}"), "[REDACTED_GITHUB_PAT]"),
    (re.compile(r"gho_[a-zA-Z0-9]{36}"), "[REDACTED_GITHUB_OAUTH]"),
    # AWS Access & Secret Keys
    (re.compile(r"(?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}"), "[REDACTED_AWS_KEY]"),
    # Slack Tokens
    (re.compile(r"xox[baprs]-[0-9a-zA-Z]{10,48}"), "[REDACTED_SLACK_TOKEN]"),
    # Private Cryptographic Keys
    (re.compile(r"-----BEGIN [A-Z\s]+PRIVATE KEY-----(?:.|\n)+?-----END [A-Z\s]+PRIVATE KEY-----"), "[REDACTED_PRIVATE_KEY]"),
    # Database Passwords in Connection URIs
    (re.compile(r"((?:postgres|postgresql|mysql|mongodb(?:\+srv)?):\/\/[^\s:]+:)[^\s@]+(@)", re.IGNORECASE), r"\1[REDACTED_PASSWORD]\2"),
]

# Sensitive System Directories that must never be accessed
FORBIDDEN_ROOTS = [
    Path("/etc").resolve(),
    Path("/private/etc").resolve(),
    Path("/root").resolve(),
    (Path.home() / ".ssh").resolve(),
    (Path.home() / ".aws").resolve(),
    (Path.home() / ".gnupg").resolve(),
]


def redact_secrets(content: str) -> str:
    """Scan and redact known API keys, tokens, and private keys from code content."""
    if not content:
        return content
    redacted = content
    for pattern, replacement in SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def validate_safe_path(target_path: Union[str, Path], allowed_roots: Optional[List[Path]] = None) -> Path:
    """
    Validate that a given path does not escape into forbidden system directories
    and resolves strictly within allowed roots.
    """
    resolved = Path(target_path).expanduser().resolve()

    # Check forbidden system folders
    for forbidden in FORBIDDEN_ROOTS:
        try:
            if resolved == forbidden or resolved.is_relative_to(forbidden):
                raise HTTPException(
                    status_code=403,
                    detail=f"Access forbidden: Path '{target_path}' resides within restricted system location."
                )
        except (ValueError, AttributeError):
            pass

    # If allowed_roots are specified, ensure the path resides inside at least one
    if allowed_roots:
        resolved_roots = [r.expanduser().resolve() for r in allowed_roots]
        is_safe = False
        for r in resolved_roots:
            try:
                if resolved == r or resolved.is_relative_to(r):
                    is_safe = True
                    break
            except ValueError:
                continue
        if not is_safe:
            raise HTTPException(
                status_code=403,
                detail=f"Access forbidden: Path '{target_path}' is outside designated workspace roots."
            )

    return resolved


def sanitize_github_url(url: str) -> Tuple[str, str, str]:
    """
    Strictly validate and sanitize a GitHub repository URL or slug.
    Prevents protocol manipulation (e.g. file://, ssh://) and flag injection.
    """
    cleaned = url.strip().rstrip("/")
    if cleaned.endswith(".git"):
        cleaned = cleaned[:-4]

    # Block protocol smuggling
    if cleaned.startswith(("file://", "ssh://", "ftp://", "gopher://")):
        raise HTTPException(
            status_code=400,
            detail="Forbidden URL scheme. Only public GitHub HTTPS repositories are allowed."
        )

    # Strictly parse owner/repo
    if "://" in cleaned:
        match = re.search(r"^https?:\/\/(?:www\.)?github\.com\/([a-zA-Z0-9_\-\.]+)\/([a-zA-Z0-9_\-\.]+)$", cleaned)
    elif cleaned.startswith("github.com/"):
        match = re.search(r"^github\.com\/([a-zA-Z0-9_\-\.]+)\/([a-zA-Z0-9_\-\.]+)$", cleaned)
    else:
        match = re.search(r"^([a-zA-Z0-9_\-\.]+)\/([a-zA-Z0-9_\-\.]+)$", cleaned)

    if not match:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid GitHub repository URL: '{url}'. Expected format: 'owner/repo' or 'https://github.com/owner/repo'"
        )

    owner, repo = match.group(1), match.group(2)
    # Reject flag injection attempts in slugs
    if owner.startswith("-") or repo.startswith("-"):
        raise HTTPException(status_code=400, detail="Invalid repository identifier.")

    clone_url = f"https://github.com/{owner}/{repo}.git"
    return owner, repo, clone_url


class RateLimiter:
    """
    Sliding window in-memory rate limiter per client IP address.
    """
    def __init__(self, requests_per_minute: int = 60):
        self.rpm = requests_per_minute
        self.window = 60.0
        self.clients: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        timestamps = self.clients[client_ip]
        self.clients[client_ip] = [t for t in timestamps if now - t < self.window]
        if len(self.clients[client_ip]) >= self.rpm:
            return False
        self.clients[client_ip].append(now)
        return True


# Global Rate Limiters
search_limiter = RateLimiter(requests_per_minute=120)
index_limiter = RateLimiter(requests_per_minute=10)

# Optional API Key Authentication
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
API_KEY_QUERY = APIKeyQuery(name="api_key", auto_error=False)


def verify_api_access(
    header_key: Optional[str] = Security(API_KEY_HEADER),
    query_key: Optional[str] = Security(API_KEY_QUERY),
    auth_header: Optional[str] = None
) -> bool:
    """
    Verify API key if CODE_INTEL_API_KEY is configured in the environment.
    If no key is configured, allows open access (default for local paired use).
    """
    required_key = os.getenv("CODE_INTEL_API_KEY", "").strip()
    if not required_key:
        return True

    candidate = header_key or query_key
    if not candidate and auth_header and auth_header.startswith("Bearer "):
        candidate = auth_header[7:].strip()

    if not candidate or candidate != required_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key."
        )
    return True
