"""TensorLBM Platform – FastAPI application entry point."""
from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

# Ensure tensorlbm src is importable when running from platform/ directory
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from . import job_manager, security  # noqa: E402
from .routers import (  # noqa: E402
    agent,
    ai_governance,
    ai_suboff,  # noqa: E402
    ai_transformer,
    benchmarks,
    cad,
    jobs,
    notifications,
    orchestration,
    postprocess,
    preprocess,
    projects,
    reports,
    solver,
    suboff,
    templates,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

_ws_broadcast_task: asyncio.Task[None] | None = None
_notify_queue: asyncio.Queue[dict] | None = None  # type: ignore[type-arg]


def _cors_origins() -> list[str]:
    raw = os.environ.get("TENSORLBM_CORS_ALLOW_ORIGINS", "*").strip()
    if raw == "*":
        return ["*"]
    vals = [v.strip() for v in raw.split(",")]
    return [v for v in vals if v]


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global _notify_queue, _ws_broadcast_task
    loop = asyncio.get_running_loop()
    _notify_queue = asyncio.Queue()
    job_manager.set_event_loop(loop, _notify_queue)  # type: ignore[arg-type]
    _ws_broadcast_task = asyncio.create_task(_ws_broadcaster(_notify_queue))
    try:
        yield
    finally:
        if _ws_broadcast_task is not None:
            _ws_broadcast_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await _ws_broadcast_task
            _ws_broadcast_task = None
        _notify_queue = None


app = FastAPI(
    title="TensorLBM Platform",
    version="1.0.0",
    description=(
        "TensorLBM – a workflow-centric Lattice Boltzmann Method engineering "
        "simulation platform.  Integrates project management, engineering templates, "
        "pre-processing, solving, post-processing, convergence monitoring, "
        "report generation, and benchmarks."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Security middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def auth_and_audit_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    if not request.url.path.startswith("/api/"):
        return await call_next(request)

    started = asyncio.get_running_loop().time()
    try:
        ctx = security.authorize_request(request)
    except security.AuthorizationError as exc:
        response = JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        security.audit_request(
            request,
            None,
            response.status_code,
            asyncio.get_running_loop().time() - started,
        )
        return response

    request.state.auth_context = ctx
    try:
        response = await call_next(request)
    except Exception:
        security.audit_request(request, ctx, 500, asyncio.get_running_loop().time() - started)
        raise

    security.audit_request(
        request,
        ctx,
        response.status_code,
        asyncio.get_running_loop().time() - started,
    )
    return response


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(jobs.router, prefix="/api/jobs", tags=["Jobs"])
app.include_router(cad.router, prefix="/api/cad", tags=["CAD"])
app.include_router(preprocess.router, prefix="/api/preprocess", tags=["Pre-processing"])
app.include_router(solver.router, prefix="/api/solve", tags=["Solver"])
app.include_router(postprocess.router, prefix="/api/postprocess", tags=["Post-processing"])
app.include_router(benchmarks.router, prefix="/api/benchmarks", tags=["Benchmarks"])
app.include_router(agent.router, prefix="/api/agent", tags=["LLM Agent"])
app.include_router(ai_transformer.router, prefix="/api/ai", tags=["AI Transformer"])
app.include_router(ai_governance.router, prefix="/api/ai/governance", tags=["AI Governance"])
app.include_router(ai_suboff.router, prefix="/api/ai/suboff", tags=["SUBOFF AI"])
app.include_router(suboff.router, prefix="/api/suboff", tags=["SUBOFF Physics"])
app.include_router(orchestration.router, prefix="/api/orchestration", tags=["Orchestration"])
app.include_router(projects.router, prefix="/api/projects", tags=["Projects"])
app.include_router(templates.router, prefix="/api/templates", tags=["Templates"])
app.include_router(reports.router, prefix="/api/reports", tags=["Reports"])
app.include_router(notifications.router, prefix="/api/notifications", tags=["Notifications"])

# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

_ws_connections: set[WebSocket] = set()

# Per-job field-stream WebSocket connections: job_id → set of WebSocket
_field_stream_connections: dict[str, set[WebSocket]] = {}
_field_stream_lock = asyncio.Lock() if False else None  # initialised lazily


async def _ws_broadcaster(queue: asyncio.Queue[dict]) -> None:  # type: ignore[type-arg]
    """Forward job status changes from the queue to all WebSocket clients."""
    while True:
        try:
            msg = await queue.get()
        except asyncio.CancelledError:
            break

        dead: set[WebSocket] = set()
        for ws in tuple(_ws_connections):
            try:
                await ws.send_json({"type": "job_update", "job": msg})
            except (WebSocketDisconnect, RuntimeError):
                dead.add(ws)
            except Exception:
                dead.add(ws)
        for ws in dead:
            with contextlib.suppress(Exception):
                await ws.close()
            _ws_connections.discard(ws)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    _ws_connections.add(ws)
    # Send full job list on first connect so the client can initialise its UI
    with contextlib.suppress(Exception):
        await ws.send_json({"type": "init", "jobs": job_manager.list_jobs()})
    try:
        while True:
            # Keep connection alive; client can send ping messages
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _ws_connections.discard(ws)


@app.websocket("/ws/field-stream/{job_id}")
async def field_stream_endpoint(ws: WebSocket, job_id: str) -> None:
    """WebSocket endpoint for real-time 2-D field slice streaming (P3.3).

    The client connects to ``/ws/field-stream/{job_id}`` and receives JSON
    messages of the form::

        {
            "type": "field_slice",
            "job_id": "...",
            "step": 1234,
            "field": "pressure" | "velocity_x" | ...,
            "data": [[...], [...], ...],   // 2-D list of float32 values
            "shape": [ny, nx]
        }

    Solver workers push field slices by calling
    :func:`push_field_slice` from ``job_manager``.
    """
    await ws.accept()
    if job_id not in _field_stream_connections:
        _field_stream_connections[job_id] = set()
    _field_stream_connections[job_id].add(ws)
    try:
        while True:
            await ws.receive_text()  # keep alive / ping
    except WebSocketDisconnect:
        pass
    finally:
        with contextlib.suppress(KeyError):
            _field_stream_connections.get(job_id, set()).discard(ws)


async def _broadcast_field_slice(job_id: str, payload: dict) -> None:
    """Internal helper: broadcast a field slice to all subscribers."""
    subs = _field_stream_connections.get(job_id, set())
    if not subs:
        return
    dead: set[WebSocket] = set()
    for ws in tuple(subs):
        try:
            await ws.send_json(payload)
        except (WebSocketDisconnect, RuntimeError):
            dead.add(ws)
        except Exception:
            dead.add(ws)
    for ws in dead:
        with contextlib.suppress(Exception):
            await ws.close()
        subs.discard(ws)


@app.post("/api/jobs/{job_id}/field-slice", tags=["Jobs"])
async def push_field_slice_http(
    job_id: str,
    step: int,
    field: str,
    data: list[list[float]],
) -> dict:
    """Push a 2-D field slice from a running solver to WebSocket subscribers.

    Solver code calls this endpoint (or the equivalent Python helper) every
    N steps to stream live field data to the browser.

    Args:
        job_id: Job ID.
        step:   Current simulation step.
        field:  Field name (``"pressure"``, ``"velocity_x"``, etc.).
        data:   2-D array (list of rows) of float values.

    Returns:
        Number of WebSocket subscribers that received the slice.
    """
    if not job_manager.get_job(job_id):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Job not found")

    ny = len(data)
    nx = len(data[0]) if ny > 0 else 0
    payload = {
        "type": "field_slice",
        "job_id": job_id,
        "step": step,
        "field": field,
        "data": data,
        "shape": [ny, nx],
    }
    await _broadcast_field_slice(job_id, payload)
    return {"subscribers": len(_field_stream_connections.get(job_id, set()))}


# ---------------------------------------------------------------------------
# Platform status
# ---------------------------------------------------------------------------

@app.get("/api/health", tags=["Platform"])
async def health() -> dict:
    """Lightweight liveness/readiness probe used by external monitors.

    Returns a minimal JSON payload that does not depend on Torch or the job
    manager and can therefore be served reliably even during startup or when
    the system is under heavy load.
    """
    return {"status": "ok", "service": "tensorlbm-platform"}


@app.get("/api/status", tags=["Platform"])
async def platform_status() -> dict:
    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        n_gpus = torch.cuda.device_count() if cuda_ok else 0
        gpu_names = (
            [torch.cuda.get_device_name(i) for i in range(n_gpus)] if cuda_ok else []
        )
        devices = ["cpu"] + [f"cuda:{i}" for i in range(n_gpus)]
    except Exception:
        cuda_ok = False
        n_gpus = 0
        gpu_names = []
        devices = ["cpu"]

    from . import job_manager as jm
    from .routers.projects import workflow_summary as _wf_summary

    all_jobs = jm.list_jobs()

    try:
        wf = await _wf_summary()
    except Exception:
        wf = {}

    return {
        "version": "1.0.0",
        "platform": "TensorLBM",
        "scheduler_backend": jm.scheduler_backend(),
        "scheduler_profile": jm.scheduler_profile(),
        "cuda_available": cuda_ok,
        "gpu_count": n_gpus,
        "gpu_names": gpu_names,
        "devices": devices,
        "total_jobs": len(all_jobs),
        "running_jobs": sum(1 for j in all_jobs if j["status"] == "running"),
        "completed_jobs": sum(1 for j in all_jobs if j["status"] == "completed"),
        "failed_jobs": sum(1 for j in all_jobs if j["status"] == "failed"),
        "workflow_summary": wf,
    }


# ---------------------------------------------------------------------------
# Frontend static serving
# ---------------------------------------------------------------------------

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
_STATIC_DIR = _FRONTEND_DIR / "static"

# Mount static assets before the SPA fallback so they are served correctly
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    return FileResponse(_FRONTEND_DIR / "index.html")


@app.get("/{full_path:path}", include_in_schema=False)
async def spa_fallback(full_path: str) -> FileResponse:  # noqa: ARG001
    """Return index.html for all non-API client-side routes (SPA fallback)."""
    return FileResponse(_FRONTEND_DIR / "index.html")
