"""Job management API endpoints."""
from __future__ import annotations

import base64
import json
import logging
import mimetypes
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .. import job_manager

if TYPE_CHECKING:
    from pathlib import Path

router = APIRouter()
logger = logging.getLogger("tensorlbm.platform.audit")


class CleanupJobsRequest(BaseModel):
    retention_hours: int | None = Field(None, ge=0)
    max_completed_jobs: int | None = Field(None, ge=0)
    max_total_mb: int | None = Field(None, ge=0)
    dry_run: bool = False


@router.get("/")
async def list_jobs(
    status: str | None = Query(  # noqa: B008
        None, description="Filter by status (queued/running/completed/failed/cancelled)"
    ),
    limit: int = Query(0, ge=0, description="Max jobs to return (0 = all)"),  # noqa: B008
    offset: int = Query(0, ge=0, description="Number of jobs to skip"),  # noqa: B008
) -> dict:
    """Return jobs sorted newest-first with optional status filter and pagination.

    When ``limit=0`` (the default) all matching jobs are returned so existing
    callers are unaffected.  The response envelope includes a ``total`` count
    of matching jobs before pagination so clients can build pagination controls.
    """
    jobs = job_manager.list_jobs()
    if status:
        jobs = [j for j in jobs if j.get("status") == status]
    total = len(jobs)
    if offset:
        jobs = jobs[offset:]
    if limit:
        jobs = jobs[:limit]
    return {"jobs": jobs, "total": total, "offset": offset, "limit": limit}


@router.get("/compare")
async def compare_jobs(
    ids: list[str] = Query(  # noqa: B008  (FastAPI dependency-injection idiom)
        ..., description="Job IDs to compare (repeat the parameter)"
    ),
) -> dict:
    """Return side-by-side metadata for a small set of jobs.

    Each entry contains the job descriptor plus, when available, the parsed
    ``run_metadata.json`` written by the solver. Unknown job IDs are reported
    in the ``missing`` field. This endpoint powers the frontend "Compare runs"
    panel and is also useful for scripted post-processing.
    """
    if not ids:
        raise HTTPException(status_code=400, detail="Provide at least one job id")
    if len(ids) > 10:
        raise HTTPException(status_code=400, detail="Compare at most 10 jobs at once")

    results: list[dict] = []
    missing: list[str] = []
    for jid in ids:
        job = job_manager.get_job(jid)
        if job is None:
            missing.append(jid)
            continue
        meta: dict = {}
        candidates = list(job.output_dir.rglob("run_metadata.json"))
        if candidates:
            try:
                meta = json.loads(candidates[0].read_text())
            except (OSError, json.JSONDecodeError):
                meta = {}
        results.append(
            {
                "job": job.to_dict(),
                "metadata": meta,
            }
        )
    return {"jobs": results, "missing": missing}


@router.delete("/{job_id}")
async def delete_job(job_id: str) -> dict:
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job.status.value == "running":
        raise HTTPException(status_code=409, detail=f"Job {job_id} is running")
    if not job_manager.delete_job(job_id):
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return {"deleted": job_id}


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict:
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job.status.value in {"completed", "failed", "cancelled"}:
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} already in terminal state: {job.status.value}",
        )
    if not job_manager.cancel_job(job_id):
        raise HTTPException(status_code=409, detail=f"Job {job_id} cannot be cancelled")
    updated = job_manager.get_job(job_id)
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return {
        "job_id": job_id,
        "status": updated.status.value,
        "cancel_requested": bool(updated.cancel_requested),
    }


@router.post("/cleanup")
async def cleanup_jobs(req: CleanupJobsRequest) -> dict:
    retention_seconds = req.retention_hours * 3600 if req.retention_hours is not None else None
    max_total_bytes = req.max_total_mb * 1024 * 1024 if req.max_total_mb is not None else None
    result = job_manager.cleanup_jobs(
        retention_seconds=retention_seconds,
        max_completed=req.max_completed_jobs,
        max_total_bytes=max_total_bytes,
        dry_run=req.dry_run,
    )
    return result


@router.get("/timeline")
async def job_timeline(
    limit: int = Query(default=50, ge=1, le=500),  # noqa: B008
    status: str | None = Query(default=None),  # noqa: B008
) -> dict:
    """Return job timeline data for Gantt-chart rendering.

    Each entry contains ``job_id``, ``name``, ``status``, ``created_at``,
    ``started_at``, ``completed_at``, and ``duration_s`` so the front-end
    can draw a Gantt diagram.  Jobs are returned newest-first.

    Query params:
        limit:  Maximum number of jobs to return (default 50).
        status: Optional status filter (queued/running/completed/failed).
    """
    all_jobs = job_manager.list_jobs()
    if status:
        all_jobs = [j for j in all_jobs if j.get("status") == status]

    # Sort newest first
    all_jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
    page = all_jobs[:limit]

    entries = []
    for j in page:
        created = j.get("created_at")
        started = j.get("started_at")
        completed = j.get("completed_at")
        duration_s = j.get("total_duration_seconds")
        entries.append({
            "job_id": j.get("job_id"),
            "name": j.get("name"),
            "job_type": j.get("job_type"),
            "status": j.get("status"),
            "created_at": created,
            "started_at": started,
            "completed_at": completed,
            "duration_s": duration_s,
            "queue_wait_s": j.get("queue_wait_seconds"),
            "priority": j.get("priority", 5),
            "assigned_resource": j.get("assigned_resource"),
        })

    return {
        "total": len(all_jobs),
        "returned": len(entries),
        "timeline": entries,
    }


@router.get("/{job_id}")
async def get_job(job_id: str) -> dict:
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job.to_dict()


@router.get("/{job_id}/logs")
async def get_logs(job_id: str) -> dict:
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return {"job_id": job_id, "logs": job.logs}


@router.get("/{job_id}/files")
async def list_files(job_id: str) -> dict:
    """List all output files for a job."""
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    out_dir = job.output_dir
    if not out_dir.exists():
        return {"job_id": job_id, "files": []}

    files = []
    for p in sorted(out_dir.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(out_dir))
            mime, _ = mimetypes.guess_type(str(p))
            files.append({
                "path": rel,
                "size": p.stat().st_size,
                "mime": mime or "application/octet-stream",
            })
    return {"job_id": job_id, "files": files}


@router.get("/{job_id}/files/{file_path:path}")
async def get_file(job_id: str, file_path: str) -> FileResponse:
    """Download a specific output file."""
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    target = _safe_job_path(job.output_dir, file_path)
    if target is None:
        logger.warning("job_file_denied job_id=%s path=%s", job_id, file_path)
        raise HTTPException(status_code=403, detail="Forbidden")
    if not target.exists():
        logger.info("job_file_missing job_id=%s path=%s", job_id, file_path)
        raise HTTPException(status_code=404, detail="File not found")
    logger.info("job_file_download job_id=%s path=%s", job_id, file_path)
    return FileResponse(str(target))


@router.get("/{job_id}/images")
async def list_images(job_id: str) -> dict:
    """List PNG images in the job output directory."""
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    out_dir = job.output_dir
    if not out_dir.exists():
        return {"job_id": job_id, "images": []}
    images = sorted(
        str(p.relative_to(out_dir))
        for p in out_dir.rglob("*.png")
        if p.is_file()
    )
    return {"job_id": job_id, "images": images}


@router.get("/{job_id}/images/{image_path:path}")
async def get_image_b64(job_id: str, image_path: str) -> dict:
    """Return a PNG image as base64 data URL."""
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    target = _safe_job_path(job.output_dir, image_path)
    if target is None:
        logger.warning("job_image_denied job_id=%s path=%s", job_id, image_path)
        raise HTTPException(status_code=403, detail="Forbidden")
    if not target.exists():
        logger.info("job_image_missing job_id=%s path=%s", job_id, image_path)
        raise HTTPException(status_code=404, detail="Image not found")
    logger.info("job_image_access job_id=%s path=%s", job_id, image_path)
    data = target.read_bytes()
    b64 = base64.b64encode(data).decode()
    return {"job_id": job_id, "path": image_path, "data": f"data:image/png;base64,{b64}"}


@router.get("/{job_id}/metadata")
async def get_metadata(job_id: str) -> dict:
    """Return run_metadata.json from job output, or empty dict."""
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    candidates = list(job.output_dir.rglob("run_metadata.json"))
    if not candidates:
        return {"job_id": job_id, "metadata": {}}
    return {"job_id": job_id, "metadata": json.loads(candidates[0].read_text())}


# ---------------------------------------------------------------------------
# Live metrics – per-step diagnostics pushed during a running job
# ---------------------------------------------------------------------------

@router.get("/{job_id}/live-metrics")
async def live_metrics(
    job_id: str,
    since_step: int = Query(0, ge=0, description="Return only diagnostics with step > since_step"),
    limit: int = Query(200, ge=1, le=1000, description="Maximum number of records to return"),
) -> dict:
    """Return live per-step diagnostic data for a running or completed job.

    Solvers that support diagnostics (e.g. cumulant cylinder flow, parametric
    study) push force coefficients and other KPIs at each output interval via
    ``job_manager.push_diagnostic``.  This endpoint exposes that data for
    real-time convergence monitoring in the UI — equivalent to the residual
    and force-history plots in PowerFlow's in-situ monitor.

    Args:
        job_id: The job to query.
        since_step: Only return records whose ``step`` field is strictly
            greater than this value.  Use the last ``step`` from a previous
            poll to fetch only new records (incremental polling).
        limit: Maximum number of records per response.

    Returns:
        ``{"job_id": ..., "status": ..., "diagnostics": [...], "has_more": bool}``
    """
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    diags = job.diagnostics  # list of dicts with at least a "step" key

    # Filter by since_step
    if since_step > 0:
        diags = [d for d in diags if d.get("step", 0) > since_step]

    # Apply limit (return most recent `limit` entries)
    has_more = len(diags) > limit
    if has_more:
        diags = diags[-limit:]

    return {
        "job_id": job_id,
        "status": job.status.value,
        "total_diagnostics": len(job.diagnostics),
        "diagnostics": diags,
        "has_more": has_more,
    }


def _safe_job_path(root: Path, rel_path: str) -> Path | None:
    target = (root / rel_path).resolve()
    if not str(target).startswith(str(root.resolve())):
        return None
    return target


# ---------------------------------------------------------------------------
# HPC cluster job submission (new industrial feature)
# ---------------------------------------------------------------------------

class HPCSubmitRequest(BaseModel):
    solver_cmd: str | None = Field(
        default=None,
        description="Shell command to run on the cluster.  Defaults to a no-op echo.",
    )
    partition: str | None = Field(default=None, description="Cluster partition/queue.")
    nodes: int | None = Field(default=None, ge=1, le=512)
    cpus: int | None = Field(default=None, ge=1, le=256)
    mem: str | None = Field(default=None, description="Memory per node, e.g. '8G'.")
    walltime: str | None = Field(default=None, description="Walltime limit, e.g. '02:00:00'.")
    extra_slurm_directives: list[str] | None = Field(
        default=None, description="Extra #SBATCH directives (SLURM only)."
    )


@router.post("/{job_id}/submit-hpc")
async def submit_job_to_hpc(job_id: str, req: HPCSubmitRequest) -> dict:
    """Submit a previously created platform job to an HPC cluster.

    Dispatches the job to the HPC scheduler configured via
    ``TENSORLBM_HPC_MODE`` (slurm | pbs).  Returns the cluster job ID so
    the user can track progress via native scheduler commands.

    Requires ``TENSORLBM_HPC_MODE=slurm`` or ``=pbs`` to be set; returns 400
    if HPC mode is ``none`` (default local execution).
    """
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    try:
        from ..services.hpc_scheduler import submit_hpc_job  # noqa: PLC0415
        result = submit_hpc_job(
            job_id=job_id,
            output_dir=str(job.output_dir),
            solver_cmd=req.solver_cmd,
            partition=req.partition,
            nodes=req.nodes,
            cpus=req.cpus,
            mem=req.mem,
            walltime=req.walltime,
            extra_slurm_directives=req.extra_slurm_directives,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    logger.info("HPC submission: job=%s hpc_id=%s", job_id, result.get("hpc_job_id"))
    return {"job_id": job_id, **result}





# ---------------------------------------------------------------------------
# P3.2 Auto-convergence control
# ---------------------------------------------------------------------------

class AutoStopConfigRequest(BaseModel):
    """Request body for configuring automatic convergence-based job stopping."""
    enabled: bool = True
    residual_key: str = "residual"
    rel_tol: float = Field(default=1e-4, gt=0.0, le=1.0)
    patience: int = Field(default=5, ge=1)
    min_steps: int = Field(default=20, ge=1)


@router.patch("/{job_id}/auto-stop-config")
async def set_auto_stop_config(job_id: str, req: AutoStopConfigRequest) -> dict:
    """Configure automatic convergence-based stopping for a running or queued job.

    Once configured the job manager monitors the ``residual_key`` field of
    every diagnostic pushed via :func:`push_diagnostic`.  When the relative
    change in the residual falls below *rel_tol* for *patience* consecutive
    checks (and at least *min_steps* diagnostics have been collected), the
    job is automatically stopped and marked as converged.

    Args:
        job_id: Target job ID.
        req:    Auto-stop configuration.

    Returns:
        Acknowledgement with the applied configuration.
    """
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    cfg = job_manager.AutoStopConfig(
        enabled=req.enabled,
        residual_key=req.residual_key,
        rel_tol=req.rel_tol,
        patience=req.patience,
        min_steps=req.min_steps,
    )
    ok = job_manager.set_auto_stop_config(job_id, cfg)
    if not ok:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": job_id,
        "message": "Auto-stop configuration applied",
        "config": cfg.to_dict(),
    }
