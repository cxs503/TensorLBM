"""Post-processing API endpoints.

Provides result analysis and metric extraction from completed simulation jobs,
including interactive field-viewer data (heatmaps, vectors, streamlines).
"""
from __future__ import annotations

import csv as csv_mod
import json
from typing import Any

from fastapi import APIRouter, HTTPException, Query
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
# Field Viewer – list checkpoints
# ---------------------------------------------------------------------------

@router.get("/checkpoints/{job_id}")
async def list_checkpoints(job_id: str) -> dict:
    """Return a sorted list of checkpoint files available for a job."""
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    ckpts = sorted(job.output_dir.rglob("checkpoint_*.pt"), key=lambda p: p.stem)
    return {
        "job_id": job_id,
        "checkpoints": [str(p.relative_to(job.output_dir)) for p in ckpts],
    }


# ---------------------------------------------------------------------------
# Field Viewer – extract 2-D field data from a checkpoint
# ---------------------------------------------------------------------------

_MAX_CELLS = 150  # maximum grid dimension after downsampling for JSON transfer


@router.get("/field-data/{job_id}")
async def field_data(
    job_id: str,
    field: str = Query(
        "velocity_magnitude",
        description="Field to extract: velocity_magnitude | vorticity | density | pressure_coeff | ux | uy",  # noqa: E501
    ),
    checkpoint: str = Query(
        "latest",
        description="Relative path inside job output dir, or 'latest'",
    ),
) -> dict:
    """Extract a 2-D field from a job checkpoint and return it as a JSON array.

    The field is downsampled (nearest-neighbour) to at most ``_MAX_CELLS`` in
    each direction so the JSON payload stays manageable in the browser.

    Returns ``nx``, ``ny``, ``field_min``, ``field_max`` plus:

    * ``data``  – row-major flat list of float values (length ``ny × nx``)
    * ``ux``    – downsampled x-velocity flat list (for vector/streamline overlay)
    * ``uy``    – downsampled y-velocity flat list
    """
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    try:
        import torch

        from tensorlbm import load_checkpoint, macroscopic
        from tensorlbm.postprocess import (
            compute_pressure_coefficient,
            compute_velocity_magnitude,
            compute_vorticity_2d,
        )

        # ---- locate checkpoint ------------------------------------------------
        if checkpoint == "latest":
            ckpts = sorted(job.output_dir.rglob("checkpoint_*.pt"), key=lambda p: p.stem)
            if not ckpts:
                raise ValueError("No checkpoint files found in job output")
            ckpt_path = ckpts[-1]
        else:
            ckpt_path = (job.output_dir / checkpoint).resolve()
            if not str(ckpt_path).startswith(str(job.output_dir.resolve())):
                raise HTTPException(status_code=403, detail="Forbidden")
            if not ckpt_path.exists():
                raise HTTPException(status_code=404, detail="Checkpoint not found")

        f_tensor, step = load_checkpoint(ckpt_path)

        # Only 2-D (D2Q9) supported for the interactive viewer
        if f_tensor.ndim != 3:
            raise ValueError(
                f"Field viewer only supports 2-D checkpoints (D2Q9), got {f_tensor.ndim}-D tensor"
            )

        rho, ux, uy = macroscopic(f_tensor)

        # ---- compute requested field ------------------------------------------
        if field == "velocity_magnitude":
            arr2d = compute_velocity_magnitude(ux, uy)
        elif field == "vorticity":
            arr2d = compute_vorticity_2d(ux, uy)
        elif field == "density":
            arr2d = rho
        elif field == "pressure_coeff":
            u_in_guess = float(ux[:, 0].mean().abs().item()) or 0.04
            arr2d = compute_pressure_coefficient(rho, u_in_guess)
        elif field == "ux":
            arr2d = ux
        elif field == "uy":
            arr2d = uy
        else:
            raise HTTPException(status_code=400, detail=f"Unknown field '{field}'")

        arr2d = arr2d.float()
        ux_f = ux.float()
        uy_f = uy.float()

        # ---- downsample to _MAX_CELLS -----------------------------------------
        ny_orig, nx_orig = arr2d.shape

        def _downsample(t: torch.Tensor, ny_t: int, nx_t: int) -> torch.Tensor:
            if ny_t <= _MAX_CELLS and nx_t <= _MAX_CELLS:
                return t
            scale = max(ny_t / _MAX_CELLS, nx_t / _MAX_CELLS)
            new_ny = max(1, int(ny_t / scale))
            new_nx = max(1, int(nx_t / scale))
            # Use avg_pool2d for speed; shape (1, 1, ny, nx)
            t4d = t.unsqueeze(0).unsqueeze(0)
            ky = ny_t // new_ny
            kx = nx_t // new_nx
            if ky > 1 or kx > 1:
                t4d = torch.nn.functional.avg_pool2d(
                    t4d, kernel_size=(max(1, ky), max(1, kx)),
                    stride=(max(1, ky), max(1, kx)), padding=0
                )
            return t4d.squeeze(0).squeeze(0)

        arr_ds = _downsample(arr2d, ny_orig, nx_orig)
        ux_ds = _downsample(ux_f, ny_orig, nx_orig)
        uy_ds = _downsample(uy_f, ny_orig, nx_orig)

        ny_ds, nx_ds = arr_ds.shape
        f_min = float(arr_ds.min().item())
        f_max = float(arr_ds.max().item())

        # Flatten to Python lists for JSON
        data_list = arr_ds.cpu().reshape(-1).tolist()
        ux_list = ux_ds.cpu().reshape(-1).tolist()
        uy_list = uy_ds.cpu().reshape(-1).tolist()

        return {
            "job_id": job_id,
            "step": step,
            "field": field,
            "nx": nx_ds,
            "ny": ny_ds,
            "nx_orig": nx_orig,
            "ny_orig": ny_orig,
            "field_min": f_min,
            "field_max": f_max,
            "data": data_list,
            "ux": ux_list,
            "uy": uy_list,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


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
