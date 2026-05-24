"""Job management API endpoints."""
from __future__ import annotations

import base64
import json
import mimetypes

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from .. import job_manager

router = APIRouter()


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


@router.get("/{job_id}")
async def get_job(job_id: str) -> dict:
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job.to_dict()


@router.delete("/{job_id}")
async def delete_job(job_id: str) -> dict:
    if not job_manager.delete_job(job_id):
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return {"deleted": job_id}


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
    target = (job.output_dir / file_path).resolve()
    # Prevent path traversal
    if not str(target).startswith(str(job.output_dir.resolve())):
        raise HTTPException(status_code=403, detail="Forbidden")
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")
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
    target = (job.output_dir / image_path).resolve()
    if not str(target).startswith(str(job.output_dir.resolve())):
        raise HTTPException(status_code=403, detail="Forbidden")
    if not target.exists():
        raise HTTPException(status_code=404, detail="Image not found")
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
