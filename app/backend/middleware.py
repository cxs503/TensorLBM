"""HTTP middleware for platform access control, audit logging, and compression."""

from __future__ import annotations

from time import perf_counter

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.middleware.gzip import GZipMiddleware
from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .security import AuthorizationError, audit_request, authorize_request


class PlatformAccessMiddleware:
    """Apply the configured access policy to API requests and WebSockets."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        scope_type = scope["type"]
        if scope_type == "websocket":
            await self._handle_websocket(scope, receive, send)
        elif scope_type == "http" and scope["path"].startswith("/api/"):
            await self._handle_api(scope, receive, send)
        else:
            await self.app(scope, receive, send)

    async def _handle_api(self, scope: Scope, receive: Receive, send: Send) -> None:
        request = Request(scope, receive)
        started = perf_counter()
        try:
            auth_context = authorize_request(request)
        except AuthorizationError as exc:
            status_code = exc.status_code
            response = JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
            await response(scope, receive, send)
            audit_request(request, None, status_code, perf_counter() - started)
            return

        request.state.auth_context = auth_context
        status_code = 500

        async def send_with_status(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_with_status)
        finally:
            audit_request(request, auth_context, status_code, perf_counter() - started)

    async def _handle_websocket(self, scope: Scope, receive: Receive, send: Send) -> None:
        request_scope = {**scope, "type": "http", "method": "GET"}
        try:
            auth_context = authorize_request(Request(request_scope))
        except AuthorizationError as exc:
            close_code = 4401 if exc.status_code == 401 else 4403
            await send({"type": "websocket.close", "code": close_code})
            return

        scope.setdefault("state", {})["auth_context"] = auth_context
        await self.app(scope, receive, send)


def install_production_middleware(app: FastAPI) -> None:
    """Install API/WebSocket access control, audit logging, and gzip compression."""
    app.add_middleware(PlatformAccessMiddleware)
    app.add_middleware(GZipMiddleware, minimum_size=1000)
