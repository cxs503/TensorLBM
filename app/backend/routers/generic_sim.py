# ruff: noqa: TC001 — the request model is needed at runtime by FastAPI
"""Generic simulation API — one endpoint, every case, common modules only.

``POST /api/sim/generic`` runs any registered case (cavity, poiseuille,
couette, shear_wave, cylinder, …) through the *single* generic execution
path in :mod:`backend.services.generic_run`: the solver public API
(``tensorlbm.solver``/``d2q9``/``boundaries``/``lid_driven_cavity``/
``postprocess``) with the whole-step chain routed via
``benchmarks/compile_route`` → ``tensorlbm.compile_utils`` — the same
routing every verified benchmark takes (PR #180).  This is the
generic-run fusion of ``PLATFORM_ANALYSIS.md`` §4.2: no case-specific
branches live in the platform layer, and the ~20 case-specific endpoints
in ``routers/solver.py`` are left untouched (see
``docs/generic_run_api.md`` for the migration order).

Jobs run asynchronously through the shared job manager (same pattern as
the benchmarks router): the POST returns a ``job_id`` plus status/result
URLs; live per-step diagnostics are broadcast over the global
``/ws`` WebSocket.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import job_manager
from ..schemas.generic_run import GenericSimRequest
from ..services import generic_run

router = APIRouter()

_JOB_TYPE = "generic_sim"


@router.get("/generic/cases")
def list_generic_cases() -> dict:
    """List the case registry: grid/physics parameters, defaults, minima."""
    return generic_run.list_cases()


@router.post("/generic")
async def submit_generic_simulation(req: GenericSimRequest) -> dict:
    """Submit a generic LBM simulation job through the common-module path.

    The case type, grid, physics, collision model and compile mode are
    validated against the case registry and the shared compile-mode
    whitelist *before* the job is queued; invalid input is a 422 with the
    registry/shared reason (cudagraph-class compile modes are rejected
    with the structural LBM-feedback-loop reason).
    """
    try:
        grid, physics, steps, collision, _canonical = generic_run.validate_request(
            case=req.case,
            grid=req.grid,
            physics=req.physics,
            steps=req.steps,
            collision=req.collision,
            compile_mode=req.compile_mode,
        )
        try:
            import torch

            torch.device(req.device)
        except (ValueError, RuntimeError) as e:
            raise generic_run.ParamError(f"Invalid device {req.device!r}: {e}") from e
    except ValueError as e:  # ParamError and the shared compile-mode ValueError
        raise HTTPException(status_code=422, detail=str(e)) from e

    def _run(job: job_manager.Job) -> dict:
        return generic_run.run_generic_simulation(
            job,
            case=req.case,
            grid=grid,
            physics=physics,
            steps=steps,
            collision=collision,
            compile_mode=req.compile_mode,
            device=req.device,
            seed=req.seed,
            monitor_interval=req.monitor_interval,
        )

    job_id = job_manager.submit(
        name=f"Generic simulation ({req.case})",
        job_type=_JOB_TYPE,
        config=req.model_dump(),
        fn=_run,
    )
    return {
        "job_id": job_id,
        "case": req.case,
        "steps": steps,
        "collision": collision,
        "compile_mode": req.compile_mode,
        "message": f"Generic {req.case} simulation submitted",
        "status_url": f"/api/sim/generic/{job_id}/status",
        "result_url": f"/api/sim/generic/{job_id}/result",
    }


@router.get("/generic/{job_id}/status")
def generic_simulation_status(job_id: str) -> dict:
    """Return job status plus the latest progress diagnostic."""
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id {job_id!r}")
    payload = job.to_dict()
    latest: dict | None = None
    for diag in reversed(job.diagnostics):
        if isinstance(diag, dict) and diag.get("kind") == "generic_sim_step":
            latest = diag
            break
        if latest is None and isinstance(diag, dict):
            latest = diag
    payload["progress"] = latest
    return payload


@router.get("/generic/{job_id}/result")
def generic_simulation_result(job_id: str) -> dict:
    """Return the final generic-run result (metrics, compile route, modules)."""
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id {job_id!r}")
    status = job.status.value if hasattr(job.status, "value") else job.status
    if status != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job is not completed (current status: {status})",
        )
    if not isinstance(job.result, dict) or not job.result:
        raise HTTPException(status_code=422, detail="Job result payload is missing")
    return job.result
