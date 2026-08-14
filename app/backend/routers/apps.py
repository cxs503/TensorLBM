"""AI4S application management API endpoints.

Thin FastAPI layer over the AI4S application framework
(:mod:`tensorlbm.apps`).  It lists registered applications, runs an
application's full-stack pipeline (optionally dispatching the data-production
step to HPC via :mod:`tensorlbm.apps.hpc`), and queries run status.

Application discovery uses the process-wide
:class:`~tensorlbm.apps.base.ApplicationRegistry` (``tensorlbm.apps.base.registry``);
the built-in applications are imported and registered lazily on first use so
``GET /api/apps`` is populated out of the box.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from tensorlbm.apps.base import AI4SApplication, registry
from tensorlbm.apps.hpc import HpcRunSpec, query_app_hpc, submit_app_hpc

from ..schemas.apps import (
    AppInfo,
    AppListResponse,
    AppRunRequest,
    HpcSubmitResponse,
    RunReportOut,
    RunStatusResponse,
)

router = APIRouter()

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

_DB_DIR = Path(os.environ.get("TENSORLBM_OUTPUT_ROOT", "/tmp/tensorlbm_platform"))

# In-memory run store: platform job_id -> run record (status + optional HPC id).
_RUNS: dict[str, dict[str, Any]] = {}

# Built-in applications auto-registered on first use (idempotent; failures are
# skipped so a missing optional dependency never breaks the router).
_BUILTIN_APPS: tuple[tuple[str, str], ...] = (
    ("tensorlbm.apps.neural_operator_fno", "NeuralOperatorFNO"),
    ("tensorlbm.apps.ai_les_app", "AILesApp"),
    ("tensorlbm.apps.suboff_app", "SuboffSurrogateApp"),
    ("tensorlbm.apps.flow_transformer_app", "FlowTransformerApp"),
    ("tensorlbm.apps.physics_informed_lbm", "PhysicsInformedLBM"),
    ("tensorlbm.apps.inverse_problem", "InverseProblem"),
    ("tensorlbm.apps.mesh_gnn_flow", "MeshGNNFlow"),
)


def _ensure_builtin_apps() -> None:
    """Import and register the built-in applications (idempotent)."""
    registered = set(registry.names())
    for module_name, cls_name in _BUILTIN_APPS:
        if cls_name in registered:
            continue
        try:
            module = importlib.import_module(module_name)
        except Exception:  # pragma: no cover - optional dependency missing
            continue
        cls = getattr(module, cls_name, None)
        if cls is None or cls.name in registered:
            continue
        try:
            registry.register(cls)
        except TypeError:  # pragma: no cover - not an AI4SApplication
            continue
        registered.add(cls.name)


def _get_app_cls(name: str) -> type[AI4SApplication]:
    """Resolve a registered application class, 404 when unknown."""
    _ensure_builtin_apps()
    try:
        return registry.get(name)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=f"unknown application: {name}") from error


def _default_db_path() -> str:
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    return str(_DB_DIR / "apps.db")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=AppListResponse)
async def list_apps() -> AppListResponse:
    """List registered AI4S applications (name / family / version)."""
    _ensure_builtin_apps()
    infos = [
        AppInfo(name=name, family=registry.get(name).family, version=registry.get(name).version)
        for name in registry.names()
    ]
    return AppListResponse(apps=infos, total=len(infos))


@router.post("/{name}/run", response_model=RunReportOut | HpcSubmitResponse)
async def run_app(name: str, body: AppRunRequest) -> RunReportOut | HpcSubmitResponse:
    """Run an application's full-stack pipeline.

    Without ``hpc`` in the request body, the full closed-loop
    (``produce_data`` → register → train → serve → lineage) runs locally and a
    :class:`RunReportOut` is returned.  When ``hpc`` is provided, the
    data-production step is dispatched to the cluster via
    :func:`tensorlbm.apps.hpc.submit_app_hpc` and a submission response is
    returned instead.
    """
    app_cls = _get_app_cls(name)
    app = app_cls()

    if body.hpc is not None:
        spec = HpcRunSpec(
            app_name=name,
            partition=body.hpc.partition,
            nodes=body.hpc.nodes,
            cpus=body.hpc.cpus,
            mem=body.hpc.mem,
            walltime=body.hpc.walltime,
            backend=body.hpc.backend,
            produce_cfg=body.produce_cfg,
        )
        try:
            result = submit_app_hpc(app, spec)
        except (ValueError, RuntimeError, ImportError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        job_id = str(result["job_id"])
        _RUNS[job_id] = {
            "app_name": name,
            "status": "submitted",
            "hpc_job_id": str(result["hpc_job_id"]),
            "backend": str(result.get("backend", "slurm")),
        }
        return HpcSubmitResponse(
            app_name=name,
            job_id=job_id,
            hpc_job_id=str(result["hpc_job_id"]),
            status="submitted",
            backend=str(result.get("backend", "slurm")),
            script_cmd=str(result.get("script_cmd", "")),
        )

    db_path = body.db_path or _default_db_path()
    report = app.run(db_path, body.produce_cfg, body.train_cfg)
    _RUNS[report.job_id] = {
        "app_name": name,
        "status": "completed",
        "report": report,
    }
    return RunReportOut(
        name=report.name,
        family=report.family,
        data_asset_id=report.data_asset_id,
        dataset_asset_id=report.dataset_asset_id,
        job_id=report.job_id,
        model_id=report.model_id,
        metrics=dict(report.metrics),
        lineage_upstream=list(report.lineage_upstream),
    )


@router.get("/{name}/run/{job_id}", response_model=RunStatusResponse)
async def get_run_status(name: str, job_id: str) -> RunStatusResponse:
    """Query the status of a previously submitted run.

    HPC-dispatched runs are queried live from the scheduler; local runs report
    their recorded terminal status.
    """
    record = _RUNS.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown run: {job_id}")

    if record.get("hpc_job_id"):
        status = query_app_hpc(str(record["hpc_job_id"]))
        return RunStatusResponse(
            job_id=job_id,
            app_name=str(record["app_name"]),
            status=str(record["status"]),
            hpc_job_id=str(record["hpc_job_id"]),
            scheduler_state=status.get("state"),
            elapsed=status.get("elapsed"),
        )

    return RunStatusResponse(
        job_id=job_id,
        app_name=str(record["app_name"]),
        status=str(record["status"]),
    )
