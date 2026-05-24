"""TensorLBM Platform – FastAPI application entry point."""
from __future__ import annotations

import asyncio
import contextlib
import sys
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Ensure tensorlbm src is importable when running from platform/ directory
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from . import job_manager  # noqa: E402
from .routers import benchmarks, cad, jobs, postprocess, preprocess, solver  # noqa: E402

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="TensorLBM Platform",
    version="1.0.0",
    description=(
        "Browser/Server platform for Lattice Boltzmann Method simulations. "
        "Integrates pre-processing, solving, post-processing and benchmarks."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

_ws_connections: list[WebSocket] = []
_notify_queue: asyncio.Queue[dict] = asyncio.Queue()  # type: ignore[type-arg]


@app.on_event("startup")
async def _startup() -> None:
    loop = asyncio.get_event_loop()
    job_manager.set_event_loop(loop, _notify_queue)  # type: ignore[arg-type]
    asyncio.create_task(_ws_broadcaster())


async def _ws_broadcaster() -> None:
    """Forward job status changes from the queue to all WebSocket clients."""
    while True:
        msg = await _notify_queue.get()
        dead: list[WebSocket] = []
        for ws in list(_ws_connections):
            try:
                await ws.send_json({"type": "job_update", "job": msg})
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in _ws_connections:
                _ws_connections.remove(ws)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    _ws_connections.append(ws)
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
        if ws in _ws_connections:
            _ws_connections.remove(ws)


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


@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    return FileResponse(_FRONTEND_DIR / "index.html")


@app.get("/{full_path:path}", include_in_schema=False)
async def spa_fallback(full_path: str) -> FileResponse:  # noqa: ARG001
    """Return index.html for all non-API client-side routes (SPA fallback)."""
    return FileResponse(_FRONTEND_DIR / "index.html")
