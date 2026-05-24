"""Post-processing API endpoints.

Provides result analysis and metric extraction from completed simulation jobs.
"""
from __future__ import annotations

import csv as csv_mod
import json
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import job_manager

router = APIRouter()


# ---------------------------------------------------------------------------
# Velocity profile extraction
# ---------------------------------------------------------------------------

class VelocityProfileRequest(BaseModel):
    job_id: str
    direction: str = "y"        # "x" or "y"
    position: float = 0.5       # fractional position (0–1) along the other axis


@router.post("/velocity-profile")
async def velocity_profile(req: VelocityProfileRequest) -> dict:
    """Extract a 1-D velocity profile from the latest checkpoint of a job."""
    job = job_manager.get_job(req.job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status.value != "completed":
        raise HTTPException(status_code=409, detail="Job not completed yet")

    try:
        from tensorlbm import load_checkpoint, macroscopic

        # Find the latest checkpoint in the job's output tree
        ckpts = sorted(job.output_dir.rglob("checkpoint_*.pt"), key=lambda p: p.stem)
        if not ckpts:
            raise ValueError("No checkpoint files found in job output")

        f, step = load_checkpoint(ckpts[-1])
        _rho, ux, uy = macroscopic(f)

        ny, nx = ux.shape
        if req.direction == "y":
            idx = int(req.position * nx)
            idx = max(0, min(idx, nx - 1))
            profile_u = ux[:, idx].cpu().tolist()
            profile_v = uy[:, idx].cpu().tolist()
            coords = [i / (ny - 1) for i in range(ny)]
            label_coord = f"y/H (x-slice at x={idx})"
        else:
            idx = int(req.position * ny)
            idx = max(0, min(idx, ny - 1))
            profile_u = ux[idx, :].cpu().tolist()
            profile_v = uy[idx, :].cpu().tolist()
            coords = [i / (nx - 1) for i in range(nx)]
            label_coord = f"x/L (y-slice at y={idx})"

        return {
            "job_id": req.job_id,
            "step": step,
            "direction": req.direction,
            "position": req.position,
            "coords": coords,
            "u": profile_u,
            "v": profile_v,
            "label_coord": label_coord,
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Kinetic energy time series from forces CSV
# ---------------------------------------------------------------------------

@router.get("/csv/{job_id}/{csv_name}")
async def get_csv_data(job_id: str, csv_name: str) -> dict:
    """Parse a CSV from a job's output and return column data."""
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    candidates = list(job.output_dir.rglob(csv_name))
    if not candidates:
        raise HTTPException(status_code=404, detail=f"{csv_name} not found")

    rows: list[dict[str, str]] = []
    with candidates[0].open(newline="", encoding="utf-8") as fh:
        reader = csv_mod.DictReader(fh)
        for row in reader:
            rows.append(dict(row))

    if not rows:
        return {"job_id": job_id, "filename": csv_name, "columns": [], "data": {}}

    columns = list(rows[0].keys())
    data: dict[str, list[float]] = {}
    for col in columns:
        try:
            data[col] = [float(r[col]) for r in rows]
        except ValueError:
            data[col] = []

    return {"job_id": job_id, "filename": csv_name, "columns": columns, "data": data}


# ---------------------------------------------------------------------------
# Snapshot image analysis (vorticity, velocity-magnitude overlay)
# ---------------------------------------------------------------------------

@router.get("/snapshot-analysis/{job_id}")
async def snapshot_analysis(job_id: str) -> dict:
    """List all PNG snapshots and return basic file info."""
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    snapshots = sorted(job.output_dir.rglob("snapshot_*.png"))
    result = []
    for p in snapshots:
        result.append({
            "name": p.name,
            "path": str(p.relative_to(job.output_dir)),
            "size_kb": round(p.stat().st_size / 1024, 1),
        })
    return {"job_id": job_id, "snapshot_count": len(snapshots), "snapshots": result}


# ---------------------------------------------------------------------------
# Run-metadata summary
# ---------------------------------------------------------------------------

@router.get("/summary/{job_id}")
async def job_summary(job_id: str) -> dict:
    """Return a human-readable summary of the job result and metadata."""
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    meta: dict[str, Any] = {}
    candidates = list(job.output_dir.rglob("run_metadata.json"))
    if candidates:
        meta = json.loads(candidates[0].read_text())

    png_count = len(list(job.output_dir.rglob("*.png")))
    csv_count = len(list(job.output_dir.rglob("*.csv")))

    return {
        "job_id": job_id,
        "job_name": job.name,
        "job_type": job.job_type,
        "status": job.status.value,
        "duration_s": _duration(job),
        "png_files": png_count,
        "csv_files": csv_count,
        "metadata": meta,
        "result": job.result,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _duration(job: job_manager.Job) -> float | None:
    from datetime import datetime

    if job.started_at and job.completed_at:
        t0 = datetime.fromisoformat(job.started_at)
        t1 = datetime.fromisoformat(job.completed_at)
        return round((t1 - t0).total_seconds(), 2)
    return None
