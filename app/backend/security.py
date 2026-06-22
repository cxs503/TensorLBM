"""Opt-in API authentication, RBAC, rate-limiting and audit helpers."""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import Request


_ROLE_ORDER = {"viewer": 10, "engineer": 20, "admin": 30}
_DEFAULT_RULES = (
    "GET:/api/*=viewer,"
    "HEAD:/api/*=viewer,"
    "POST:/api/jobs/*/cancel=engineer,"
    "POST:/api/jobs/cleanup=admin,"
    "DELETE:/api/*=admin,"
    "POST:/api/*=engineer,"
    "PUT:/api/*=engineer,"
    "PATCH:/api/*=engineer"
)
_DEFAULT_PUBLIC = "/api/health,/api/status"
_AUDIT_LOGGER = logging.getLogger("tensorlbm.platform.audit")


class AuthorizationError(Exception):
    """Raised when a request is not authorized."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = int(status_code)
        self.detail = detail


@dataclass(frozen=True)
class AuthContext:
    user: str
    role: str
    auth_mode: str


@dataclass(frozen=True)
class AccessRule:
    method: str
    pattern: str
    min_role: str


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

@dataclass
class _RateLimitBucket:
    hits: deque[float] = field(default_factory=deque)
    lock: threading.Lock = field(default_factory=threading.Lock)


_rl_buckets: dict[str, _RateLimitBucket] = {}
_rl_buckets_lock = threading.Lock()


def _rl_config() -> tuple[int, float]:
    """Return (max_requests, window_seconds); (0, …) means disabled."""
    max_req = int(os.environ.get("TENSORLBM_RATE_LIMIT_REQUESTS", "0") or "0")
    window = float(os.environ.get("TENSORLBM_RATE_LIMIT_WINDOW_S", "60") or "60")
    return max(0, max_req), max(1.0, window)


def _client_key(request: Request) -> str:
    """Return a string key that identifies the caller for rate-limiting."""
    # Honour X-Forwarded-For when behind a trusted proxy.
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    client = request.client
    return client.host if client else "unknown"


def check_rate_limit(request: Request) -> None:
    """Raise :class:`AuthorizationError` (429) if the caller has exceeded the
    configured request rate.  Does nothing when rate-limiting is disabled
    (``TENSORLBM_RATE_LIMIT_REQUESTS=0``, the default).
    """
    max_req, window = _rl_config()
    if max_req <= 0:
        return

    key = _client_key(request)
    now = time.monotonic()
    cutoff = now - window

    with _rl_buckets_lock:
        if key not in _rl_buckets:
            _rl_buckets[key] = _RateLimitBucket()
        bucket = _rl_buckets[key]

    with bucket.lock:
        # Evict timestamps outside the current window.
        while bucket.hits and bucket.hits[0] <= cutoff:
            bucket.hits.popleft()
        if len(bucket.hits) >= max_req:
            raise AuthorizationError(429, "Too Many Requests")
        bucket.hits.append(now)


def _auth_mode() -> str:
    return (os.environ.get("TENSORLBM_AUTH_MODE", "disabled").strip().lower() or "disabled")


def _public_paths() -> set[str]:
    raw = os.environ.get("TENSORLBM_AUTH_PUBLIC_PATHS", _DEFAULT_PUBLIC)
    return {p.strip() for p in raw.split(",") if p.strip()}


def _token_map() -> dict[str, AuthContext]:
    raw = os.environ.get("TENSORLBM_AUTH_TOKENS", "").strip()
    out: dict[str, AuthContext] = {}
    for part in (p.strip() for p in raw.split(",") if p.strip()):
        token, _, role = part.partition(":")
        token = token.strip()
        role = (role.strip().lower() or "viewer")
        if token and role in _ROLE_ORDER:
            out[token] = AuthContext(user=f"token:{token[:6]}", role=role, auth_mode="header")
    return out


def _rules() -> list[AccessRule]:
    raw = os.environ.get("TENSORLBM_AUTH_RULES", _DEFAULT_RULES)
    rules: list[AccessRule] = []
    for part in (p.strip() for p in raw.split(",") if p.strip()):
        lhs, _, role = part.partition("=")
        method, _, pattern = lhs.partition(":")
        m = method.strip().upper()
        p = pattern.strip() or "/api/*"
        r = role.strip().lower()
        if m and r in _ROLE_ORDER:
            rules.append(AccessRule(method=m, pattern=p, min_role=r))
    return rules


def _required_role(method: str, path: str) -> str | None:
    if method.upper() == "OPTIONS":
        return None
    if path in _public_paths():
        return None
    m = method.upper()
    for rule in _rules():
        if rule.method == m and fnmatch(path, rule.pattern):
            return rule.min_role
    return "viewer"


def _has_role(actual: str, required: str) -> bool:
    return _ROLE_ORDER.get(actual, 0) >= _ROLE_ORDER.get(required, 0)


def authorize_request(request: Request) -> AuthContext:
    """Authorize request according to auth mode and role policy."""
    required = _required_role(request.method, request.url.path)
    mode = _auth_mode()

    if required is None:
        return AuthContext(user="public", role="viewer", auth_mode=mode)

    if mode == "disabled":
        check_rate_limit(request)
        return AuthContext(user="anonymous", role="admin", auth_mode=mode)

    header_name = os.environ.get("TENSORLBM_AUTH_HEADER", "X-API-Key").strip() or "X-API-Key"
    token = request.headers.get(header_name)
    token_map = _token_map()
    if not token or token not in token_map:
        raise AuthorizationError(401, "Unauthorized")
    ctx = token_map[token]
    if not _has_role(ctx.role, required):
        raise AuthorizationError(403, "Forbidden")
    check_rate_limit(request)
    return ctx


def audit_request(
    request: Request,
    context: AuthContext | None,
    status_code: int,
    duration_s: float,
) -> None:
    """Write one structured audit log line for API requests."""
    if not request.url.path.startswith("/api/"):
        return
    ctx = context or AuthContext(user="unknown", role="unknown", auth_mode=_auth_mode())
    _AUDIT_LOGGER.info(
        "api_audit method=%s path=%s status=%s user=%s role=%s auth_mode=%s duration_ms=%.3f",
        request.method.upper(),
        request.url.path,
        int(status_code),
        ctx.user,
        ctx.role,
        ctx.auth_mode,
        max(0.0, float(duration_s)) * 1000.0,
    )
