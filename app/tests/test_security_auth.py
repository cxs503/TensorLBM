"""Unit tests for auth/RBAC policy helpers."""
from __future__ import annotations

from backend import security
from starlette.requests import Request


def _req(path: str, method: str = "GET", headers: dict[str, str] | None = None) -> Request:
    raw = []
    for k, v in (headers or {}).items():
        raw.append((k.lower().encode("latin-1"), v.encode("latin-1")))
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": method,
        "path": path,
        "headers": raw,
        "query_string": b"",
        "scheme": "http",
        "server": ("testserver", 80),
        "client": ("testclient", 1234),
    }
    return Request(scope)


def test_authorize_disabled_mode(monkeypatch):
    monkeypatch.setenv("TENSORLBM_AUTH_MODE", "disabled")
    ctx = security.authorize_request(_req("/api/jobs/", "GET"))
    assert ctx.role == "admin"


def test_authorize_requires_token_when_enabled(monkeypatch):
    monkeypatch.setenv("TENSORLBM_AUTH_MODE", "header")
    monkeypatch.setenv("TENSORLBM_AUTH_TOKENS", "eng-token:engineer")
    try:
        security.authorize_request(_req("/api/jobs/", "GET"))
    except security.AuthorizationError as exc:
        assert exc.status_code == 401
    else:
        raise AssertionError("Expected AuthorizationError")


def test_authorize_rbac_blocks_delete_for_engineer(monkeypatch):
    monkeypatch.setenv("TENSORLBM_AUTH_MODE", "header")
    monkeypatch.setenv("TENSORLBM_AUTH_TOKENS", "eng-token:engineer")
    try:
        security.authorize_request(
            _req("/api/jobs/abc123", "DELETE", headers={"X-API-Key": "eng-token"}),
        )
    except security.AuthorizationError as exc:
        assert exc.status_code == 403
    else:
        raise AssertionError("Expected AuthorizationError")


def test_authorize_admin_delete_allowed(monkeypatch):
    monkeypatch.setenv("TENSORLBM_AUTH_MODE", "header")
    monkeypatch.setenv("TENSORLBM_AUTH_TOKENS", "adm-token:admin")
    ctx = security.authorize_request(
        _req("/api/jobs/abc123", "DELETE", headers={"X-API-Key": "adm-token"}),
    )
    assert ctx.role == "admin"
