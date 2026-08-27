"""
Automated tests for the Security and Sandboxing module.
Validates secret redaction, path traversal guards, URL sanitization, rate limiting, and HTTP security headers.
"""

from pathlib import Path
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from semantic_code_intel.api.app import app
from semantic_code_intel.security import (
    RateLimiter, redact_secrets, sanitize_github_url,
    validate_safe_path, FORBIDDEN_ROOTS
)


def test_secret_redaction():
    """Verify that sensitive API keys, tokens, and credentials are redacted."""
    raw_code = """
    # Configuration
    OPENAI_API_KEY = "sk-proj-abcdef12345678901234567890"
    ANTHROPIC_KEY = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz"
    GITHUB_PAT = "ghp_1234567890abcdefghijklmnopqrstuvwxyz"
    AWS_ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"
    DB_URI = "postgresql://admin:supersecretpassword123@db.prod.internal:5432/main"
    SSH_KEY = "-----BEGIN RSA PRIVATE KEY-----\\nMIIEowIBAAKCAQEA0...\\n-----END RSA PRIVATE KEY-----"
    """

    redacted = redact_secrets(raw_code)

    assert "sk-proj-" not in redacted
    assert "[REDACTED_API_KEY]" in redacted
    assert "sk-ant-" not in redacted
    assert "[REDACTED_ANTHROPIC_KEY]" in redacted
    assert "ghp_" not in redacted
    assert "[REDACTED_GITHUB_PAT]" in redacted
    assert "AKIAIOSFODNN7EXAMPLE" not in redacted
    assert "[REDACTED_AWS_KEY]" in redacted
    assert "supersecretpassword123" not in redacted
    assert "[REDACTED_PASSWORD]" in redacted
    assert "BEGIN RSA PRIVATE KEY" not in redacted
    assert "[REDACTED_PRIVATE_KEY]" in redacted


def test_path_traversal_guards():
    """Verify that path traversal and forbidden system paths are blocked."""
    # Forbidden roots
    for forbidden in FORBIDDEN_ROOTS:
        with pytest.raises(HTTPException) as exc_info:
            validate_safe_path(forbidden)
        assert exc_info.value.status_code == 403

    # Traversal escape
    with pytest.raises(HTTPException) as exc_info:
        validate_safe_path("/etc/passwd")
    assert exc_info.value.status_code == 403

    # Safe local path
    safe_path = Path(".").resolve()
    result = validate_safe_path(safe_path)
    assert result == safe_path


def test_github_url_sanitization():
    """Verify URL validation and rejection of dangerous protocols."""
    # Valid HTTPS URL
    owner, repo, clone_url = sanitize_github_url("https://github.com/pallets/flask.git")
    assert owner == "pallets"
    assert repo == "flask"
    assert clone_url == "https://github.com/pallets/flask.git"

    # Valid shorthand slug
    owner, repo, clone_url = sanitize_github_url("torvalds/linux")
    assert owner == "torvalds"
    assert repo == "linux"
    assert clone_url == "https://github.com/torvalds/linux.git"

    # Block file:// protocol smuggling
    with pytest.raises(HTTPException) as exc_info:
        sanitize_github_url("file:///etc/passwd")
    assert exc_info.value.status_code == 400

    # Block flag injection in owner/repo
    with pytest.raises(HTTPException) as exc_info:
        sanitize_github_url("--upload-pack=exploit/repo")
    assert exc_info.value.status_code == 400


def test_rate_limiter():
    """Verify in-memory sliding window rate limiting."""
    limiter = RateLimiter(requests_per_minute=3)
    client_ip = "192.168.1.100"

    assert limiter.is_allowed(client_ip) is True
    assert limiter.is_allowed(client_ip) is True
    assert limiter.is_allowed(client_ip) is True
    # 4th request exceeds limit of 3
    assert limiter.is_allowed(client_ip) is False


def test_http_security_headers():
    """Verify that HTTP security headers are injected into API responses."""
    client = TestClient(app)
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "SAMEORIGIN"
    assert response.headers.get("X-XSS-Protection") == "1; mode=block"
    assert response.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
