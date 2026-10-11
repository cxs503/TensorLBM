"""HTTP middleware for platform access control, audit logging, and compression."""
from __future__ import annotations

from time import perf_counter
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.gzip import GZipMiddleware

from .security import AuthorizationError, audit_request, authorize_request


def install_production_middleware(app: FastAPI) -> None:
    """Install API authorization, rate limiting, audit logging, and gzip."""

    @app.middleware("http")
    async def api_access_middleware(request: Request, call_next: Any) -> Any:
        if not request.url.path.startswith("/api/"):
            return await call_next(request)

        started = perf_counter()
        auth_context = None
        status_code = 500
        try:
            auth_context = authorize_request(request)
            request.state.auth_context = auth_context
            response = await call_next(request)
            status_code = response.status_code
            return response
        except AuthorizationError as exc:
            status_code = exc.status_code
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
        finally:
            audit_request(request, auth_context, status_code, perf_counter() - started)

    app.add_middleware(GZipMiddleware, minimum_size=1000)
