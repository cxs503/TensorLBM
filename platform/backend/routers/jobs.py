"""Job management API endpoints."""
from __future__ import annotations

import base64
import json
import logging
import mimetypes
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .. import job_manager

router = APIRouter()
logger = logging.getLogger("tensorlbm.platform.audit")


class CleanupJobsRequest(BaseModel):
    retention_hours: int | None = Field(None, ge=0)
    max_completed_jobs: int | None = Field(None, ge=0)
    max_total_mb: int | None = Field(None, ge=0)
    dry_run: bool = False


@router.get("/")
async def list_jobs() -> list[dict]:
    """Return all jobs sorted newest-first."""
    return job_manager.list_jobs()


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
    return {"job_id": job_id, "status": "cancelled"}


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


def _safe_job_path(root: Path, rel_path: str) -> Path | None:
    target = (root / rel_path).resolve()
    if not str(target).startswith(str(root.resolve())):
        return None
    return target
