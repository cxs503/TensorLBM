"""Unit tests for auth/RBAC policy helpers."""
from __future__ import annotations

from backend import security
from starlette.requests import Request


def _req(path: str, method: str = "GET", headers: dict[str, str] | None = None, client_ip: str = "testclient") -> Request:
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
        "client": (client_ip, 1234),
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


# ---------------------------------------------------------------------------
# Rate-limiting tests
# ---------------------------------------------------------------------------

def _clear_rl_buckets():
    """Reset the in-process rate-limit state between tests."""
    security._rl_buckets.clear()


def test_rate_limit_disabled_by_default(monkeypatch):
    """Rate limiting must be a no-op when TENSORLBM_RATE_LIMIT_REQUESTS=0."""
    monkeypatch.setenv("TENSORLBM_RATE_LIMIT_REQUESTS", "0")
    _clear_rl_buckets()
    req = _req("/api/jobs/", client_ip="10.0.0.1")
    # Should never raise regardless of how many times called.
    for _ in range(100):
        security.check_rate_limit(req)


def test_rate_limit_allows_up_to_limit(monkeypatch):
    monkeypatch.setenv("TENSORLBM_RATE_LIMIT_REQUESTS", "3")
    monkeypatch.setenv("TENSORLBM_RATE_LIMIT_WINDOW_S", "60")
    _clear_rl_buckets()
    req = _req("/api/jobs/", client_ip="10.0.0.2")
    for _ in range(3):
        security.check_rate_limit(req)  # must not raise


def test_rate_limit_raises_429_when_exceeded(monkeypatch):
    monkeypatch.setenv("TENSORLBM_RATE_LIMIT_REQUESTS", "3")
    monkeypatch.setenv("TENSORLBM_RATE_LIMIT_WINDOW_S", "60")
    _clear_rl_buckets()
    req = _req("/api/jobs/", client_ip="10.0.0.3")
    for _ in range(3):
        security.check_rate_limit(req)
    try:
        security.check_rate_limit(req)
    except security.AuthorizationError as exc:
        assert exc.status_code == 429
    else:
        raise AssertionError("Expected 429 AuthorizationError")


def test_rate_limit_is_per_ip(monkeypatch):
    monkeypatch.setenv("TENSORLBM_RATE_LIMIT_REQUESTS", "2")
    monkeypatch.setenv("TENSORLBM_RATE_LIMIT_WINDOW_S", "60")
    _clear_rl_buckets()
    req_a = _req("/api/jobs/", client_ip="10.1.1.1")
    req_b = _req("/api/jobs/", client_ip="10.1.1.2")
    # Exhaust A
    security.check_rate_limit(req_a)
    security.check_rate_limit(req_a)
    # B should still be fine
    security.check_rate_limit(req_b)
    security.check_rate_limit(req_b)


def test_rate_limit_honours_x_forwarded_for(monkeypatch):
    monkeypatch.setenv("TENSORLBM_RATE_LIMIT_REQUESTS", "2")
    monkeypatch.setenv("TENSORLBM_RATE_LIMIT_WINDOW_S", "60")
    _clear_rl_buckets()
    req = _req("/api/jobs/", headers={"x-forwarded-for": "203.0.113.5"}, client_ip="127.0.0.1")
    security.check_rate_limit(req)
    security.check_rate_limit(req)
    try:
        security.check_rate_limit(req)
    except security.AuthorizationError as exc:
        assert exc.status_code == 429
    else:
        raise AssertionError("Expected 429 AuthorizationError")


def test_rate_limit_window_expires(monkeypatch):
    """Hits outside the window should be evicted and not count."""
    import time

    monkeypatch.setenv("TENSORLBM_RATE_LIMIT_REQUESTS", "2")
    monkeypatch.setenv("TENSORLBM_RATE_LIMIT_WINDOW_S", "1")
    _clear_rl_buckets()
    req = _req("/api/jobs/", client_ip="10.2.0.1")
    # Exhaust the limit.
    security.check_rate_limit(req)
    security.check_rate_limit(req)
    # Wait for window to expire.
    time.sleep(1.05)
    # Should be allowed again.
    security.check_rate_limit(req)


def test_rate_limit_integrated_into_authorize(monkeypatch):
    """authorize_request must enforce the rate limit in disabled auth mode."""
    monkeypatch.setenv("TENSORLBM_AUTH_MODE", "disabled")
    monkeypatch.setenv("TENSORLBM_RATE_LIMIT_REQUESTS", "2")
    monkeypatch.setenv("TENSORLBM_RATE_LIMIT_WINDOW_S", "60")
    _clear_rl_buckets()
    req = _req("/api/jobs/", client_ip="10.3.0.1")
    security.authorize_request(req)
    security.authorize_request(req)
    try:
        security.authorize_request(req)
    except security.AuthorizationError as exc:
        assert exc.status_code == 429
    else:
        raise AssertionError("Expected 429 AuthorizationError")
