"""TensorLBM Platform – FastAPI application entry point."""
from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Ensure tensorlbm src is importable when running from platform/ directory
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from . import job_manager  # noqa: E402
from .routers import (  # noqa: E402
    agent,
    ai_governance,
    ai_transformer,
    benchmarks,
    cad,
    jobs,
    orchestration,
    postprocess,
    preprocess,
    solver,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

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
        "Browser/Server platform for Lattice Boltzmann Method simulations. "
        "Integrates pre-processing, solving, post-processing and benchmarks."
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
app.include_router(orchestration.router, prefix="/api/orchestration", tags=["Orchestration"])

# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

_ws_connections: set[WebSocket] = set()


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

    all_jobs = jm.list_jobs()
    return {
        "version": "1.0.0",
        "cuda_available": cuda_ok,
        "gpu_count": n_gpus,
        "gpu_names": gpu_names,
        "devices": devices,
        "total_jobs": len(all_jobs),
        "running_jobs": sum(1 for j in all_jobs if j["status"] == "running"),
        "completed_jobs": sum(1 for j in all_jobs if j["status"] == "completed"),
        "failed_jobs": sum(1 for j in all_jobs if j["status"] == "failed"),
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
