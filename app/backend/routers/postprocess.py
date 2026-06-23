"""Post-processing API endpoints.

Provides result analysis and metric extraction from completed simulation jobs,
including interactive field-viewer data (heatmaps, vectors, streamlines).
"""
from __future__ import annotations

import contextlib
import csv as csv_mod
import io
import json
import zipfile
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .. import job_manager
from ..file_patterns import list_step_images

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
        ckpts = sorted(job.output_dir.rglob("checkpoint_f.pt"), key=lambda p: p.stat().st_mtime)
        if not ckpts:
            raise ValueError("No checkpoint files found in job output")

        f, step, _meta = load_checkpoint(ckpts[-1].parent)
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

    snapshots = list_step_images(job.output_dir)
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
            ckpts = sorted(job.output_dir.rglob("checkpoint_f.pt"), key=lambda p: p.stat().st_mtime)
            if not ckpts:
                raise ValueError("No checkpoint files found in job output")
            ckpt_path = ckpts[-1].parent
        else:
            candidate = (job.output_dir / checkpoint).resolve()
            if not str(candidate).startswith(str(job.output_dir.resolve())):
                raise HTTPException(status_code=403, detail="Forbidden")
            ckpt_path = candidate.parent if candidate.suffix == ".pt" else candidate
            if not (ckpt_path / "checkpoint_f.pt").exists():
                raise HTTPException(status_code=404, detail="Checkpoint not found")

        f_tensor, step, _meta = load_checkpoint(ckpt_path)

        # Support both 2-D (D2Q9) and 3-D (D3Q19/D3Q27) with slice extraction
        if f_tensor.ndim == 4:
            raise HTTPException(
                status_code=422,
                detail=(
                    "3-D checkpoint detected. "
                    "Use /api/postprocess/field-data-3d/{job_id} for 3-D slice extraction."
                ),
            )
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
# Convergence monitor – real-time force / residual history
# ---------------------------------------------------------------------------


@router.get("/convergence/{job_id}")
async def convergence_data(job_id: str) -> dict:
    """Return convergence history for a job (force coefficients vs time step).

    The response contains per-step scalar diagnostics (Cd, Cl, drag, lift,
    density residual, …) extracted from the job's ``diagnostics`` list and,
    if present, from ``forces.csv``.  The frontend can poll this endpoint to
    render live convergence plots.
    """
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # Use in-memory diagnostics (available even while running)
    diag = list(job.diagnostics)

    # Also try to read forces.csv for completed jobs (richer data)
    forces_rows: list[dict[str, Any]] = []
    csv_path = next(job.output_dir.rglob("forces.csv"), None)
    if csv_path:
        try:
            import csv as _csv
            with csv_path.open() as fh:
                reader = _csv.DictReader(fh)
                for row in reader:
                    parsed: dict[str, Any] = {}
                    for k, v in row.items():
                        try:
                            parsed[k] = float(v)
                        except (ValueError, TypeError):
                            parsed[k] = v
                    forces_rows.append(parsed)
        except Exception:
            pass

    # Determine available series from diagnostics
    series: dict[str, list[Any]] = {}
    steps: list[int] = []
    for entry in diag:
        step = entry.get("step") or entry.get("t") or entry.get("iter")
        if step is not None:
            steps.append(int(step))
        for k, v in entry.items():
            if k in ("step", "t", "iter"):
                continue
            if isinstance(v, (int, float)):
                series.setdefault(k, []).append(v)

    return {
        "job_id": job_id,
        "job_status": job.status.value,
        "diagnostic_count": len(diag),
        "steps": steps,
        "series": series,
        "forces_rows": forces_rows[-100:],  # last 100 rows to cap payload
        "has_forces_csv": bool(forces_rows),
    }


# ---------------------------------------------------------------------------
# Probe-point time-history monitor
# ---------------------------------------------------------------------------

class ProbePoint(BaseModel):
    x_frac: float   # fractional x position in [0, 1]
    y_frac: float   # fractional y position in [0, 1]
    label: str = ""


class ProbeHistoryRequest(BaseModel):
    job_id: str
    probes: list[ProbePoint]


@router.post("/probe-history")
async def probe_history(req: ProbeHistoryRequest) -> dict:
    """Extract time history of (ux, uy, |u|, ρ) at user-defined probe locations.

    Each probe is specified as a fractional (x_frac, y_frac) position in the
    domain [0,1]×[0,1].  The endpoint loads every checkpoint in the job output
    directory and samples the nearest grid cell at each probe location, returning
    a time series suitable for plotting temporal evolution at monitoring points.

    This mirrors the *probe points* feature found in PowerFlow / XFlow.
    """
    job = job_manager.get_job(req.job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    ckpts = sorted(job.output_dir.rglob("checkpoint_f.pt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise HTTPException(status_code=404, detail="No checkpoints found for this job")

    try:
        import math

        from tensorlbm import load_checkpoint, macroscopic

        # Initialise per-probe time series lists
        probes = req.probes
        n_probes = len(probes)
        series: list[dict[str, list[float]]] = [
            {"step": [], "ux": [], "uy": [], "speed": [], "rho": []}
            for _ in range(n_probes)
        ]

        for ckpt_pt in ckpts:
            f_tensor, step, _meta = load_checkpoint(ckpt_pt.parent)
            if f_tensor.ndim != 3:
                continue  # skip 3-D checkpoints (not yet supported)

            rho, ux, uy = macroscopic(f_tensor)
            ny, nx = ux.shape

            for pi, probe in enumerate(probes):
                ix = int(max(0, min(nx - 1, round(probe.x_frac * (nx - 1)))))
                iy = int(max(0, min(ny - 1, round(probe.y_frac * (ny - 1)))))
                ux_val = float(ux[iy, ix].item())
                uy_val = float(uy[iy, ix].item())
                speed = math.sqrt(ux_val**2 + uy_val**2)
                rho_val = float(rho[iy, ix].item())
                series[pi]["step"].append(step)
                series[pi]["ux"].append(ux_val)
                series[pi]["uy"].append(uy_val)
                series[pi]["speed"].append(speed)
                series[pi]["rho"].append(rho_val)

        probe_results = []
        for pi, probe in enumerate(probes):
            probe_results.append({
                "label": probe.label or f"P{pi + 1}",
                "x_frac": probe.x_frac,
                "y_frac": probe.y_frac,
                **series[pi],
            })

        return {
            "job_id": req.job_id,
            "checkpoint_count": len(ckpts),
            "probe_count": n_probes,
            "probes": probe_results,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Time-averaged field statistics
# ---------------------------------------------------------------------------

_MAX_CELLS_AVG = 150  # same cap as field viewer


@router.get("/time-average/{job_id}")
async def time_average(
    job_id: str,
    field: str = Query(
        "velocity_magnitude",
        description="Field to average: velocity_magnitude | vorticity | ux | uy | density",
    ),
) -> dict:
    """Compute time-averaged mean and RMS fields from all 2-D checkpoints.

    Loads every checkpoint in the job output and accumulates a running mean and
    a running mean-square so that the RMS fluctuation can be derived without
    storing all snapshots in memory.

    Returns the same JSON structure as ``/field-data`` but with an additional
    ``rms`` list and ``n_snapshots`` count.  Useful for turbulent statistics –
    analogous to the *time-average post-processing* capability in XFlow.
    """
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status.value != "completed":
        raise HTTPException(
            status_code=409,
            detail="Job must be completed to compute time averages",
        )

    ckpts = sorted(job.output_dir.rglob("checkpoint_f.pt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise HTTPException(status_code=404, detail="No checkpoints found")

    try:
        import torch

        from tensorlbm import load_checkpoint, macroscopic
        from tensorlbm.postprocess import (
            compute_velocity_magnitude,
            compute_vorticity_2d,
        )

        mean_acc: torch.Tensor | None = None
        sq_acc: torch.Tensor | None = None
        n = 0
        ny_orig = nx_orig = 0

        for ckpt_pt in ckpts:
            f_tensor, _, _meta = load_checkpoint(ckpt_pt.parent)
            if f_tensor.ndim != 3:
                continue  # skip 3-D

            rho, ux, uy = macroscopic(f_tensor)

            if field == "velocity_magnitude":
                arr2d = compute_velocity_magnitude(ux, uy)
            elif field == "vorticity":
                arr2d = compute_vorticity_2d(ux, uy)
            elif field == "ux":
                arr2d = ux
            elif field == "uy":
                arr2d = uy
            elif field == "density":
                arr2d = rho
            else:
                raise HTTPException(status_code=400, detail=f"Unknown field '{field}'")

            arr2d = arr2d.float()
            ny_orig, nx_orig = arr2d.shape

            if mean_acc is None:
                mean_acc = torch.zeros_like(arr2d)
                sq_acc = torch.zeros_like(arr2d)

            mean_acc = mean_acc + arr2d
            sq_acc = sq_acc + arr2d * arr2d
            n += 1

        if mean_acc is None or n == 0:
            raise HTTPException(status_code=422, detail="No 2-D checkpoints could be loaded")

        mean_field = mean_acc / n
        rms_field = torch.sqrt(torch.clamp(sq_acc / n - mean_field * mean_field, min=0.0))

        # Downsample for JSON transfer
        def _ds(t: torch.Tensor) -> torch.Tensor:
            if ny_orig <= _MAX_CELLS_AVG and nx_orig <= _MAX_CELLS_AVG:
                return t
            scale = max(ny_orig / _MAX_CELLS_AVG, nx_orig / _MAX_CELLS_AVG)
            new_ny = max(1, int(ny_orig / scale))
            new_nx = max(1, int(nx_orig / scale))
            ky = max(1, ny_orig // new_ny)
            kx = max(1, nx_orig // new_nx)
            t4d = t.unsqueeze(0).unsqueeze(0)
            if ky > 1 or kx > 1:
                t4d = torch.nn.functional.avg_pool2d(
                    t4d, kernel_size=(ky, kx), stride=(ky, kx), padding=0
                )
            return t4d.squeeze(0).squeeze(0)

        mean_ds = _ds(mean_field)
        rms_ds = _ds(rms_field)
        ny_ds, nx_ds = mean_ds.shape

        return {
            "job_id": job_id,
            "field": field,
            "n_snapshots": n,
            "nx": nx_ds,
            "ny": ny_ds,
            "nx_orig": nx_orig,
            "ny_orig": ny_orig,
            "field_min": float(mean_ds.min().item()),
            "field_max": float(mean_ds.max().item()),
            "rms_max": float(rms_ds.max().item()),
            "mean": mean_ds.cpu().reshape(-1).tolist(),
            "rms": rms_ds.cpu().reshape(-1).tolist(),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc



# ---------------------------------------------------------------------------
# Result export (VTK / VTS / HDF5 / CSV) – industrial post-processing
# ---------------------------------------------------------------------------

_VALID_EXPORT_FORMATS: set[str] = {"vtk", "vts", "hdf5", "csv"}


@router.get("/export/{job_id}")
async def export_results(
    job_id: str,
    format: str = Query(  # noqa: A002
        "vts",
        description=(
            "Export format: "
            "``vts`` (VTK XML StructuredGrid, ParaView-native), "
            "``vtk`` (legacy ASCII VTK), "
            "``hdf5`` (HDF5 + XDMF, requires h5py), "
            "``csv`` (flat comma-separated text)"
        ),
    ),
    checkpoint: str = Query(
        "latest",
        description="Relative checkpoint path inside job output dir, or ``latest``.",
    ),
) -> StreamingResponse:
    """Export the latest (or a specific) checkpoint to an industrial format.

    Returns a downloadable file wrapped in a ZIP archive so the browser can
    save it directly.  Supported formats:

    * **vts** – VTK XML StructuredGrid with Base64 inline binary.  Open
      directly in ParaView or VisIt.
    * **vtk** – Legacy ASCII VTK STRUCTURED_POINTS.  Compatible with older
      VTK-based tools.
    * **hdf5** – HDF5 dataset + companion XDMF sidecar file (both packaged in
      the ZIP).  Requires ``h5py`` to be installed on the server.
    * **csv** – Flat CSV with columns ``i,j,k,ux,uy,uz,rho`` (3-D) or
      ``i,j,ux,uy,rho`` (2-D).  Useful for post-processing in Python/MATLAB.
    """
    fmt = format.lower()
    if fmt not in _VALID_EXPORT_FORMATS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown format '{format}'. Choose from: {sorted(_VALID_EXPORT_FORMATS)}",
        )

    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    try:
        import tempfile
        from pathlib import Path

        from tensorlbm import load_checkpoint, macroscopic, macroscopic3d
        from tensorlbm.io import save_hdf5, save_vtk, save_vts, save_xdmf

        # ---- locate checkpoint directory --------------------------------------
        # save_checkpoint writes two files: checkpoint_f.pt + checkpoint_meta.json
        # into a run directory.  load_checkpoint expects that *directory* path.
        if checkpoint == "latest":
            ckpt_files = sorted(
                job.output_dir.rglob("checkpoint_f.pt"), key=lambda p: p.stat().st_mtime
            )
            if not ckpt_files:
                raise HTTPException(
                    status_code=404,
                    detail="No checkpoint files found for this job",
                )
            ckpt_dir = ckpt_files[-1].parent
        else:
            # caller may pass either the .pt file path or the directory path
            candidate = (job.output_dir / checkpoint).resolve()
            if not str(candidate).startswith(str(job.output_dir.resolve())):
                raise HTTPException(status_code=403, detail="Forbidden")
            ckpt_dir = candidate.parent if candidate.suffix == ".pt" else candidate
            if not (ckpt_dir / "checkpoint_f.pt").exists():
                raise HTTPException(status_code=404, detail="Checkpoint not found")

        f_tensor, step, _meta = load_checkpoint(ckpt_dir)
        is_3d = f_tensor.ndim == 4  # (Q, nz, ny, nx) for 3-D

        if is_3d:
            rho, ux, uy, uz = macroscopic3d(f_tensor)
        else:
            rho, ux, uy = macroscopic(f_tensor)
            uz = None

        # Sanitise job_id so user-controlled URL segments cannot escape the
        # temporary directory via path-traversal sequences (e.g. "../../..").
        safe_id = Path(job_id).name
        stem = f"{safe_id}_step{step:06d}"

        # ---- write export files to a temporary directory ----------------------
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            export_files: list[Path] = []

            if fmt == "vts":
                out = tmp / f"{stem}.vts"
                save_vts(out, ux, uy, uz, rho=rho)
                export_files = [out]

            elif fmt == "vtk":
                out = tmp / f"{stem}.vtk"
                save_vtk(out, ux, uy, uz, rho=rho)
                export_files = [out]

            elif fmt == "hdf5":
                h5_path = tmp / f"{stem}.h5"
                xdmf_path = tmp / f"{stem}.xdmf"
                save_hdf5(h5_path, step, ux, uy, uz, rho=rho)
                save_xdmf(
                    h5_path, xdmf_path, step,
                    ux.shape, has_uz=(uz is not None), has_rho=True,
                )
                export_files = [h5_path, xdmf_path]

            elif fmt == "csv":
                out = tmp / f"{stem}.csv"
                _write_csv(out, ux, uy, uz, rho)
                export_files = [out]

            # ---- package into an in-memory ZIP --------------------------------
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                for p in export_files:
                    zf.write(p, arcname=p.name)
            zip_bytes = zip_buf.getvalue()

        zip_name = f"tensorlbm_{stem}_{fmt}.zip"
        return StreamingResponse(
            io.BytesIO(zip_bytes),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{zip_name}"'},
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _write_csv(
    path: object,
    ux: object,
    uy: object,
    uz: object,
    rho: object,
) -> None:
    """Write field data to a CSV file."""
    from pathlib import Path as _Path

    path = _Path(path)  # type: ignore[arg-type]

    ux_np = ux.detach().cpu().float().numpy()  # type: ignore[union-attr]
    uy_np = uy.detach().cpu().float().numpy()  # type: ignore[union-attr]
    rho_np = rho.detach().cpu().float().numpy() if rho is not None else None  # type: ignore[union-attr]

    is_3d = ux_np.ndim == 3  # type: ignore[union-attr]

    if is_3d:
        nz, ny, nx = ux_np.shape  # type: ignore[union-attr]
        uz_np = uz.detach().cpu().float().numpy()  # type: ignore[union-attr]
        with path.open("w", newline="", encoding="utf-8") as fh:
            fh.write("i,j,k,ux,uy,uz,rho\n")
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        rho_val = float(rho_np[k, j, i]) if rho_np is not None else 1.0  # type: ignore[index]
                        fh.write(
                            f"{i},{j},{k},"
                            f"{ux_np[k, j, i]:.8g},"  # type: ignore[index]
                            f"{uy_np[k, j, i]:.8g},"  # type: ignore[index]
                            f"{uz_np[k, j, i]:.8g},"  # type: ignore[index]
                            f"{rho_val:.8g}\n"
                        )
    else:
        ny, nx = ux_np.shape  # type: ignore[union-attr]
        with path.open("w", newline="", encoding="utf-8") as fh:
            fh.write("i,j,ux,uy,rho\n")
            for j in range(ny):
                for i in range(nx):
                    rho_val = float(rho_np[j, i]) if rho_np is not None else 1.0  # type: ignore[index]
                    fh.write(
                        f"{i},{j},"
                        f"{ux_np[j, i]:.8g},"  # type: ignore[index]
                        f"{uy_np[j, i]:.8g},"  # type: ignore[index]
                        f"{rho_val:.8g}\n"
                    )






def _duration(job: job_manager.Job) -> float | None:
    from datetime import datetime

    if job.started_at and job.completed_at:
        t0 = datetime.fromisoformat(job.started_at)
        t1 = datetime.fromisoformat(job.completed_at)
        return round((t1 - t0).total_seconds(), 2)
    return None


# ---------------------------------------------------------------------------
# 3-D Field Viewer – slice extraction for D3Q19/D3Q27 checkpoints
# ---------------------------------------------------------------------------

_MAX_CELLS_3D = 100  # max per-axis dimension after downsampling for 3-D slices


@router.get("/field-data-3d/{job_id}")
async def field_data_3d(
    job_id: str,
    field: str = Query(
        "velocity_magnitude",
        description=(
            "Field to extract: velocity_magnitude | density | pressure | "
            "q_criterion | ux | uy | uz"
        ),
    ),
    checkpoint: str = Query("latest", description="Relative path or 'latest'"),
    slice_axis: str = Query(
        "z",
        description="Axis to slice along: 'x', 'y', or 'z'",
    ),
    slice_index: int = Query(
        -1,
        description="Slice index along the chosen axis. -1 = midplane.",
    ),
) -> dict:
    """Extract a 2-D slice from a 3-D LBM checkpoint and return it as a JSON array.

    Supports D3Q19 and D3Q27 velocity sets.  The slice is downsampled to at
    most ``100 × 100`` cells before serialisation.

    Returns the same schema as ``/field-data/{job_id}`` plus ``slice_axis``,
    ``slice_index``, and (for velocity_magnitude) ``uz``.
    """
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    try:
        import torch

        from tensorlbm import load_checkpoint

        # ---- locate checkpoint -----------------------------------------------
        if checkpoint == "latest":
            ckpts = sorted(
                job.output_dir.rglob("checkpoint_f.pt"),
                key=lambda p: p.stat().st_mtime,
            )
            if not ckpts:
                raise ValueError("No checkpoint files found in job output")
            ckpt_path = ckpts[-1].parent
        else:
            candidate = (job.output_dir / checkpoint).resolve()
            if not str(candidate).startswith(str(job.output_dir.resolve())):
                raise HTTPException(status_code=403, detail="Forbidden")
            ckpt_path = candidate.parent if candidate.suffix == ".pt" else candidate
            if not (ckpt_path / "checkpoint_f.pt").exists():
                raise HTTPException(status_code=404, detail="Checkpoint not found")

        f_tensor, step, _meta = load_checkpoint(ckpt_path)

        if f_tensor.ndim != 4:
            raise HTTPException(
                status_code=422,
                detail=(
                    "2-D checkpoint detected. "
                    "Use /api/postprocess/field-data/{job_id} for 2-D fields."
                ),
            )

        q_vel = f_tensor.shape[0]
        if q_vel == 19:
            from tensorlbm.d3q19 import macroscopic3d
            rho, ux, uy, uz = macroscopic3d(f_tensor)
        elif q_vel == 27:
            from tensorlbm.d3q27 import macroscopic27
            rho, ux, uy, uz = macroscopic27(f_tensor)
        else:
            raise ValueError(f"Unsupported velocity set Q={q_vel}")

        nz, ny, nx = rho.shape

        # ---- compute requested field ------------------------------------------
        if field == "velocity_magnitude":
            arr3d: torch.Tensor = torch.sqrt(ux * ux + uy * uy + uz * uz)
        elif field == "density":
            arr3d = rho
        elif field == "pressure":
            arr3d = (rho - 1.0) / 3.0
        elif field == "ux":
            arr3d = ux
        elif field == "uy":
            arr3d = uy
        elif field == "uz":
            arr3d = uz
        elif field == "q_criterion":
            from tensorlbm.vtk_export import _q_criterion_3d
            arr3d = _q_criterion_3d(ux, uy, uz)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown field '{field}'")

        # ---- extract 2-D slice ------------------------------------------------
        axis = slice_axis.lower()
        if axis not in ("x", "y", "z"):
            raise HTTPException(status_code=400, detail="slice_axis must be 'x', 'y', or 'z'")

        axis_size = {"z": nz, "y": ny, "x": nx}[axis]
        idx = slice_index if slice_index >= 0 else axis_size // 2
        idx = max(0, min(idx, axis_size - 1))

        if axis == "z":
            slice2d = arr3d[idx, :, :]
            ux2d, uy2d = ux[idx, :, :], uy[idx, :, :]
            slice_ny, slice_nx = ny, nx
        elif axis == "y":
            slice2d = arr3d[:, idx, :]
            ux2d, uy2d = ux[:, idx, :], uy[:, idx, :]
            slice_ny, slice_nx = nz, nx
        else:
            slice2d = arr3d[:, :, idx]
            ux2d, uy2d = ux[:, :, idx], uy[:, :, idx]
            slice_ny, slice_nx = nz, ny

        # ---- downsample -------------------------------------------------------
        def _ds2d(t: torch.Tensor, ny_t: int, nx_t: int) -> torch.Tensor:
            if ny_t <= _MAX_CELLS_3D and nx_t <= _MAX_CELLS_3D:
                return t
            scale = max(ny_t / _MAX_CELLS_3D, nx_t / _MAX_CELLS_3D)
            new_ny = max(1, int(ny_t / scale))
            new_nx = max(1, int(nx_t / scale))
            t4d = t.float().unsqueeze(0).unsqueeze(0)
            ky, kx = ny_t // new_ny, nx_t // new_nx
            if ky > 1 or kx > 1:
                t4d = torch.nn.functional.avg_pool2d(
                    t4d,
                    kernel_size=(max(1, ky), max(1, kx)),
                    stride=(max(1, ky), max(1, kx)),
                    padding=0,
                )
            return t4d.squeeze(0).squeeze(0)

        s_ds = _ds2d(slice2d.float(), slice_ny, slice_nx)
        ux_ds = _ds2d(ux2d.float(), slice_ny, slice_nx)
        uy_ds = _ds2d(uy2d.float(), slice_ny, slice_nx)

        ny_ds, nx_ds = s_ds.shape

        return {
            "job_id": job_id,
            "step": step,
            "field": field,
            "dimensions": {"nz": nz, "ny": ny, "nx": nx},
            "slice_axis": axis,
            "slice_index": idx,
            "nx": nx_ds,
            "ny": ny_ds,
            "field_min": float(s_ds.min().item()),
            "field_max": float(s_ds.max().item()),
            "data": s_ds.cpu().reshape(-1).tolist(),
            "ux": ux_ds.cpu().reshape(-1).tolist(),
            "uy": uy_ds.cpu().reshape(-1).tolist(),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# VTK export endpoint
# ---------------------------------------------------------------------------

@router.get("/export-vtk/{job_id}")
async def export_vtk(
    job_id: str,
    checkpoint: str = Query("latest", description="Relative path or 'latest'"),
    fields: str = Query(
        "",
        description=(
            "Comma-separated list of fields to include. "
            "Empty = all. "
            "2-D: density,pressure,velocity_magnitude,vorticity,velocity. "
            "3-D: density,pressure,velocity_magnitude,q_criterion,velocity."
        ),
    ),
    spacing: float = Query(1.0, description="Physical grid spacing (lattice units by default)"),
) -> StreamingResponse:
    """Export a job checkpoint to VTK Legacy format for ParaView/VisIt.

    Returns a ``field.vtk`` file as an octet-stream download.
    """
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    try:
        import tempfile
        from pathlib import Path as _Path

        from tensorlbm import load_checkpoint
        from tensorlbm.vtk_export import export_vtk_2d, export_vtk_3d

        # ---- locate checkpoint -----------------------------------------------
        if checkpoint == "latest":
            ckpts = sorted(
                job.output_dir.rglob("checkpoint_f.pt"),
                key=lambda p: p.stat().st_mtime,
            )
            if not ckpts:
                raise ValueError("No checkpoint files found in job output")
            ckpt_path = ckpts[-1].parent
        else:
            candidate = (job.output_dir / checkpoint).resolve()
            if not str(candidate).startswith(str(job.output_dir.resolve())):
                raise HTTPException(status_code=403, detail="Forbidden")
            ckpt_path = candidate.parent if candidate.suffix == ".pt" else candidate
            if not (ckpt_path / "checkpoint_f.pt").exists():
                raise HTTPException(status_code=404, detail="Checkpoint not found")

        f_tensor, step, _meta = load_checkpoint(ckpt_path)
        fields_list: list[str] | None = (
            [f.strip() for f in fields.split(",") if f.strip()] if fields else None
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = _Path(tmpdir) / f"tensorlbm_step{step:06d}.vtk"

            if f_tensor.ndim == 3:
                from tensorlbm.d2q9 import macroscopic
                rho, ux, uy = macroscopic(f_tensor)
                export_vtk_2d(rho, ux, uy, out_path, spacing=spacing, fields=fields_list)
            elif f_tensor.ndim == 4:
                q = f_tensor.shape[0]
                if q == 19:
                    from tensorlbm.d3q19 import macroscopic3d
                    rho, ux, uy, uz = macroscopic3d(f_tensor)
                elif q == 27:
                    from tensorlbm.d3q27 import macroscopic27
                    rho, ux, uy, uz = macroscopic27(f_tensor)
                else:
                    raise ValueError(f"Unsupported velocity set Q={q}")
                export_vtk_3d(rho, ux, uy, uz, out_path, spacing=spacing, fields=fields_list)
            else:
                raise ValueError(f"Unexpected tensor shape {tuple(f_tensor.shape)}")

            vtk_bytes = out_path.read_bytes()

        filename = f"tensorlbm_{job_id}_step{step:06d}.vtk"
        return StreamingResponse(
            io.BytesIO(vtk_bytes),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Aeroacoustics endpoint – FWH far-field SPL from probe data
# ---------------------------------------------------------------------------

class AcousticsRequest(BaseModel):
    """Request body for FWH aeroacoustic analysis."""

    job_id: str
    observer_positions: list[list[float]] = Field(
        default=[[10.0, 0.0, 0.0]],
        description=(
            "List of far-field observer [x, y, z] positions. "
            "For 2-D problems set z=0."
        ),
    )
    surface_sample_fraction: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description=(
            "Fraction of boundary cells to use as FWH source points (0.01–1.0). "
            "Lower values are faster but less accurate."
        ),
    )
    dt_physical: float = Field(
        default=1.0e-5,
        gt=0.0,
        description="Physical time step in seconds per lattice step.",
    )
    c0: float = Field(
        default=343.0,
        gt=0.0,
        description="Speed of sound in the medium (m/s).",
    )
    physical_dx: float = Field(
        default=1.0e-3,
        gt=0.0,
        description="Physical grid spacing (m/lattice unit).",
    )


@router.post("/acoustics")
async def acoustics_analysis(req: AcousticsRequest) -> dict:
    """Compute far-field sound pressure level using the FWH acoustic analogy.

    Loads probe-history CSV data from the job output, builds a porous FWH
    control surface, and computes the far-field acoustic pressure and SPL
    spectrum for each observer.

    The probe history must contain columns: ``step``, ``x``, ``y`` (lattice
    index), and ``pressure`` (or ``rho``).  This data is written by the
    probe-history endpoint or by simulations that record boundary-layer
    probes.

    If no probe CSV is found, a synthetic test signal is used to demonstrate
    the spectral output.
    """
    job = job_manager.get_job(req.job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    try:
        import torch

        from tensorlbm.acoustics import (
            AcousticObserver,
            FWHSurface,
            compute_fwh_result,
        )

        observers = [
            AcousticObserver(
                x=float(pos[0]),
                y=float(pos[1]),
                z=float(pos[2]) if len(pos) > 2 else 0.0,
            )
            for pos in req.observer_positions
        ]

        # ---- Try to load probe CSV data --------------------------------------
        probe_csvs = list(job.output_dir.rglob("probe_history*.csv"))

        pressure_history: torch.Tensor | None = None
        positions_list: list[list[float]] = []
        T_steps = 0

        if probe_csvs:
            import csv as _csv

            with probe_csvs[0].open(newline="", encoding="utf-8") as fh:
                reader = _csv.DictReader(fh)
                rows = list(reader)

            if rows and "pressure" in rows[0]:
                # Group by probe id, build (N, T) pressure array
                probe_ids: dict[str, list[float]] = {}
                probe_pos: dict[str, list[float]] = {}
                for row in rows:
                    pid = row.get("probe_id", row.get("x", "0"))
                    p_val = float(row.get("pressure", row.get("rho", 1.0)))
                    probe_ids.setdefault(pid, []).append(p_val)
                    if pid not in probe_pos:
                        probe_pos[pid] = [
                            float(row.get("x", 0)) * req.physical_dx,
                            float(row.get("y", 0)) * req.physical_dx,
                            0.0,
                        ]
                T_steps = max(len(v) for v in probe_ids.values())
                N_probes = len(probe_ids)
                pressure_history = torch.zeros(N_probes, T_steps)
                for i, (pid, vals) in enumerate(probe_ids.items()):
                    pressure_history[i, : len(vals)] = torch.tensor(vals, dtype=torch.float32)
                    positions_list.append(probe_pos[pid])

        # ---- Fall back to synthetic signal for demo if no probe data ---------
        if pressure_history is None or T_steps < 10:
            # Generate a synthetic 100-step sinusoidal pressure fluctuation
            T_steps = 200
            N_probes = max(4, int(1.0 / req.surface_sample_fraction))
            t = torch.linspace(0, T_steps * req.dt_physical, T_steps)
            freq_s = 1000.0
            pressure_history = torch.zeros(N_probes, T_steps)
            for n_i in range(N_probes):
                phase = n_i * 2.0 * torch.pi / N_probes
                pressure_history[n_i] = (
                    1e-3 * torch.sin(2.0 * torch.pi * freq_s * t + phase)
                    + 5e-4 * torch.sin(2.0 * torch.pi * 2000.0 * t + phase * 0.5)
                )
            positions_list = [
                [float(n_i) * req.physical_dx * 10.0, 0.0, 0.0]
                for n_i in range(N_probes)
            ]

        N = pressure_history.shape[0]
        positions = torch.tensor(positions_list[:N], dtype=torch.float32)
        normals = torch.zeros(N, 3)
        normals[:, 0] = 1.0  # outward normal pointing in +x by default
        areas = torch.full((N,), req.physical_dx, dtype=torch.float32)

        surface = FWHSurface(
            positions=positions,
            normals=normals,
            areas=areas,
            pressure=pressure_history,
            dt=req.dt_physical,
            c0=req.c0,
        )

        result = compute_fwh_result(surface, observers)

        # ---- Format response -----------------------------------------------
        obs_results = []
        for i, obs in enumerate(observers):
            p_rms = float((result.p_prime[i] ** 2).mean().sqrt().item())
            spl_peak = float(result.spl[i].max().item())
            obs_results.append({
                "label": obs.label or f"Observer {i}",
                "position": [obs.x, obs.y, obs.z],
                "oaspl_dB": round(result.oaspl[i], 2),
                "p_rms_Pa": round(p_rms, 8),
                "spl_peak_dB": round(spl_peak, 2),
                "spl_spectrum": [round(v, 2) for v in result.spl[i].tolist()[:200]],
                "frequencies_Hz": [round(v, 2) for v in result.frequencies[:200]],
            })

        return {
            "job_id": req.job_id,
            "n_source_points": N,
            "dt_physical_s": req.dt_physical,
            "c0_m_s": req.c0,
            "physical_dx_m": req.physical_dx,
            "observers": obs_results,
            "note": (
                "Synthetic test signal used (no probe CSV found)."
                if T_steps == 200 and not probe_csvs
                else f"Analysed {T_steps} time steps."
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Study-group multi-case comparison
# ---------------------------------------------------------------------------

_KNOWN_SCALAR_METRICS = {
    "drag_coefficient", "cd", "lift_coefficient", "cl", "strouhal", "st",
    "nusselt", "nu", "final_saturation", "drag", "lift", "side_force",
    "porosity", "permeability", "re",
}


def _extract_scalar_metrics(result: dict[str, Any]) -> dict[str, float]:
    """Pull scalar float metrics from a job result dict."""
    out: dict[str, float] = {}
    for k, v in result.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = float(v)
        elif isinstance(v, list) and v and isinstance(v[0], (int, float)):
            # Summarise time-series as last value
            out[k + "_final"] = float(v[-1])
            out[k + "_mean"] = float(sum(v) / len(v))
    return out


@router.get("/study-compare/{study_group}")
async def study_compare(study_group: str) -> dict:
    """Aggregate and compare all completed jobs belonging to *study_group*.

    Every job whose ``config.study.group`` equals *study_group* is included.
    Parametric study jobs store their design-point values in
    ``config.study.design_point``.

    Returns a table of jobs (sorted by creation time) with their
    design points, statuses, and extracted scalar metrics, plus a
    cross-case metric-range summary and a best-value index for each metric.

    This endpoint enables the study-group comparison dashboard in the
    frontend without duplicating job data.
    """
    jobs = job_manager.list_jobs()

    rows: list[dict[str, Any]] = []
    for jd in jobs:
        cfg = jd.get("config") or {}
        study_meta = cfg.get("study") if isinstance(cfg, dict) else None
        if not isinstance(study_meta, dict):
            continue
        if study_meta.get("group") != study_group:
            continue
        result = jd.get("result") or {}
        metrics = _extract_scalar_metrics(result) if isinstance(result, dict) else {}
        rows.append({
            "job_id": jd["job_id"],
            "name": jd.get("name", ""),
            "status": jd.get("status", ""),
            "created_at": jd.get("created_at", ""),
            "design_point": study_meta.get("design_point", {}),
            "metrics": metrics,
        })

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No jobs found for study group '{study_group}'.",
        )

    # Build per-metric statistics across completed rows
    completed = [r for r in rows if r["status"] == "completed"]
    metric_summary: dict[str, dict[str, float | int]] = {}
    if completed:
        all_keys: set[str] = set()
        for r in completed:
            all_keys.update(r["metrics"].keys())
        for mk in sorted(all_keys):
            vals = [r["metrics"][mk] for r in completed if mk in r["metrics"]]
            if not vals:
                continue
            best_idx = int(min(range(len(vals)), key=lambda i: vals[i]))
            metric_summary[mk] = {
                "min": min(vals),
                "max": max(vals),
                "mean": sum(vals) / len(vals),
                "best_job_id": completed[best_idx]["job_id"],
            }

    return {
        "study_group": study_group,
        "n_total": len(rows),
        "n_completed": len(completed),
        "jobs": rows,
        "metric_summary": metric_summary,
    }


# ---------------------------------------------------------------------------
# Streamline tracing
# ---------------------------------------------------------------------------

class StreamlineRequest(BaseModel):
    job_id: str
    n_seeds_x: int = Field(default=8, ge=1, le=64)
    n_seeds_y: int = Field(default=8, ge=1, le=64)
    step_size: float = Field(default=0.5, gt=0.0, le=10.0)
    max_steps: int = Field(default=500, ge=10, le=5000)
    bidirectional: bool = False
    seed_x_range: list[float] | None = None  # [x_min, x_max] in lattice units
    seed_y_range: list[float] | None = None  # [y_min, y_max] in lattice units
    include_velocity_magnitude: bool = True


@router.post("/streamlines")
async def compute_streamlines(req: StreamlineRequest) -> dict:
    """Compute 2-D streamlines from the latest checkpoint of a completed job.

    Seeds are placed on a uniform grid over the specified region (or the full
    domain if no range is given).  Each streamline is integrated using RK4
    until it leaves the domain, enters a solid cell, or reaches *max_steps*.

    Returns a JSON object with ``n_lines`` and a ``lines`` list; each entry
    has ``points`` (list of [x, y] pairs), optional ``scalars`` (velocity
    magnitude per point), ``length`` (arc-length), and ``steps``.
    """
    import torch  # noqa: PLC0415

    from tensorlbm.checkpoint import load_checkpoint  # noqa: PLC0415
    from tensorlbm.d2q9 import macroscopic  # noqa: PLC0415
    from tensorlbm.streamlines import (  # noqa: PLC0415
        seed_points_uniform_2d,
        streamlines_to_dict,
        trace_streamlines_2d,
    )

    job = job_manager.get_job(req.job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status.value != "completed":
        raise HTTPException(status_code=409, detail="Job is not completed")

    ckpt_files = sorted(job.output_dir.rglob("checkpoint_f.pt"))
    if not ckpt_files:
        raise HTTPException(status_code=404, detail="No checkpoint found")

    f_tensor, step, _meta = load_checkpoint(ckpt_files[-1].parent)
    if f_tensor.ndim != 3:
        raise HTTPException(status_code=422, detail="Streamlines only supported for 2-D jobs")

    _rho, ux, uy = macroscopic(f_tensor)
    ny, nx = ux.shape

    x_range = tuple(req.seed_x_range) if req.seed_x_range and len(req.seed_x_range) == 2 else None
    y_range = tuple(req.seed_y_range) if req.seed_y_range and len(req.seed_y_range) == 2 else None

    seeds = seed_points_uniform_2d(nx, ny, req.n_seeds_x, req.n_seeds_y, x_range, y_range)

    scalar_field = None
    if req.include_velocity_magnitude:
        scalar_field = torch.sqrt(ux * ux + uy * uy)

    lines = trace_streamlines_2d(
        ux, uy, seeds,
        step_size=req.step_size,
        max_steps=req.max_steps,
        bidirectional=req.bidirectional,
        scalar_field=scalar_field,
    )

    result = streamlines_to_dict(lines)
    result["job_id"] = req.job_id
    result["step"] = step
    result["domain"] = {"nx": nx, "ny": ny}
    return result


# ---------------------------------------------------------------------------
# Surface integrals
# ---------------------------------------------------------------------------

class SurfaceIntegralRequest(BaseModel):
    job_id: str
    integral_type: str = Field(
        default="mass_flow",
        description=(
            "Type of integral: 'mass_flow', 'area_average', 'pressure_drop', "
            "'surface_force', 'surface_moment'."
        ),
    )
    x_plane: int | None = None           # for mass_flow / pressure_drop
    x_plane2: int | None = None          # second plane for pressure_drop
    x_range: list[int] | None = None     # [x0, x1] for area_average
    y_range: list[int] | None = None
    pivot_x: float = 0.0                 # for surface_moment
    pivot_y: float = 0.0
    rho_ref: float = Field(default=1.0, gt=0.0)
    u_ref: float = Field(default=0.1, gt=0.0)
    area_ref: float = Field(default=1.0, gt=0.0)


@router.post("/surface-integrals")
async def surface_integrals(req: SurfaceIntegralRequest) -> dict:
    """Compute surface and volume integrals from a completed 2-D job.

    Supported *integral_type* values:

    * ``mass_flow`` – volume/mass flow rate at a cross-section x-plane.
    * ``area_average`` – area-averaged velocity magnitude over a sub-region.
    * ``pressure_drop`` – pressure drop between two x-planes.
    * ``surface_force`` – total hydrodynamic force on solid obstacles.
    * ``surface_moment`` – moment about a pivot from surface forces.

    Returns the computed quantities in lattice units along with
    non-dimensionalised coefficients where applicable.
    """
    import torch  # noqa: PLC0415

    from tensorlbm.checkpoint import load_checkpoint  # noqa: PLC0415
    from tensorlbm.d2q9 import macroscopic  # noqa: PLC0415
    from tensorlbm.surface_integrals import (  # noqa: PLC0415
        area_average_2d,
        force_coefficients,
        mass_flow_rate_2d,
        pressure_drop,
        surface_force_2d,
        surface_moment_2d,
    )

    job = job_manager.get_job(req.job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status.value != "completed":
        raise HTTPException(status_code=409, detail="Job is not completed")

    ckpt_files = sorted(job.output_dir.rglob("checkpoint_f.pt"))
    if not ckpt_files:
        raise HTTPException(status_code=404, detail="No checkpoint found")

    f_tensor, step, _meta = load_checkpoint(ckpt_files[-1].parent)
    if f_tensor.ndim != 3:
        raise HTTPException(status_code=422, detail="Surface integrals only supported for 2-D jobs")

    rho, ux, uy = macroscopic(f_tensor)
    ny, nx = ux.shape

    itype = req.integral_type.lower()

    if itype == "mass_flow":
        x_plane = req.x_plane if req.x_plane is not None else nx // 2
        yr = (req.y_range[0], req.y_range[1]) if req.y_range and len(req.y_range) == 2 else None
        result = mass_flow_rate_2d(ux, rho, x_plane, yr)
        result["integral_type"] = "mass_flow"
        result["x_plane"] = x_plane
        result["step"] = step

    elif itype == "area_average":
        speed = torch.sqrt(ux * ux + uy * uy)
        xr = (req.x_range[0], req.x_range[1]) if req.x_range and len(req.x_range) == 2 else None
        yr = (req.y_range[0], req.y_range[1]) if req.y_range and len(req.y_range) == 2 else None
        result = area_average_2d(speed, xr, yr)
        result["integral_type"] = "area_average"
        result["field"] = "velocity_magnitude"
        result["step"] = step

    elif itype == "pressure_drop":
        x_up = req.x_plane if req.x_plane is not None else nx // 4
        x_dn = req.x_plane2 if req.x_plane2 is not None else 3 * nx // 4
        result = pressure_drop(rho, x_up, x_dn)
        result["integral_type"] = "pressure_drop"
        result["x_upstream"] = x_up
        result["x_downstream"] = x_dn
        result["step"] = step

    elif itype == "surface_force":
        # Build solid mask from the job config if available; fall back to
        # detecting near-stagnant cells as a proxy for solid nodes.
        speed = torch.sqrt(ux * ux + uy * uy)
        mask = speed < 1e-6
        forces = surface_force_2d(f_tensor, mask, req.rho_ref)
        coeffs = force_coefficients(
            forces["fx"], forces["fy"], None,
            req.rho_ref, req.u_ref, req.area_ref,
        )
        result = {**forces, **coeffs}
        result["integral_type"] = "surface_force"
        result["step"] = step

    elif itype == "surface_moment":
        speed = torch.sqrt(ux * ux + uy * uy)
        mask = speed < 1e-6
        moments = surface_moment_2d(f_tensor, mask, req.pivot_x, req.pivot_y)
        result = {**moments}
        result["integral_type"] = "surface_moment"
        result["pivot"] = [req.pivot_x, req.pivot_y]
        result["step"] = step

    else:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown integral_type '{req.integral_type}'. "
                "Choose from: mass_flow, area_average, pressure_drop, "
                "surface_force, surface_moment."
            ),
        )

    return result


# ---------------------------------------------------------------------------
# Inlet-profile preview
# ---------------------------------------------------------------------------

class InletProfileRequest(BaseModel):
    profile_type: str = Field(
        default="log_law",
        description="'log_law', 'power_law', 'parabolic', 'blasius', 'womersley'.",
    )
    n: int = Field(default=64, ge=4, le=1024)
    u_ref: float = Field(default=0.1, gt=0.0)
    re_tau: float = Field(default=200.0, gt=0.0)
    nu: float = Field(default=1.0 / 600.0, gt=0.0)
    exponent: float = Field(default=7.0, gt=0.0)
    womersley_number: float = Field(default=5.0, gt=0.0)
    phase: float = 0.0
    turbulence_intensity: float = Field(default=0.05, ge=0.0, le=1.0)
    add_synthetic_turbulence: bool = False
    seed: int = 42


@router.post("/inlet-profile")
async def inlet_profile_preview(req: InletProfileRequest) -> dict:
    """Generate a turbulent or laminar inlet velocity profile for preview.

    Returns the velocity profile as a list of ``(y, ux)`` pairs in lattice
    units.  The profile can be used to validate inlet conditions before
    starting a simulation.

    Supported *profile_type* values:

    * ``log_law``   – log-law turbulent channel profile.
    * ``power_law`` – 1/n power-law profile.
    * ``parabolic`` – Hagen–Poiseuille laminar profile.
    * ``blasius``   – flat-plate boundary-layer (Blasius) profile.
    * ``womersley`` – oscillatory Womersley pulsatile profile.
    """

    from tensorlbm.inlet_profiles import (  # noqa: PLC0415
        blasius_profile,
        log_law_profile,
        parabolic_profile,
        power_law_profile,
        synthetic_turbulence_2d,
        womersley_profile,
    )

    ptype = req.profile_type.lower()

    if ptype == "log_law":
        profile = log_law_profile(req.n, req.u_ref, req.re_tau, req.nu)
    elif ptype == "power_law":
        profile = power_law_profile(req.n, req.u_ref, req.exponent)
    elif ptype == "parabolic":
        profile = parabolic_profile(req.n, req.u_ref)
    elif ptype == "blasius":
        profile = blasius_profile(req.n, req.u_ref)
    elif ptype == "womersley":
        profile = womersley_profile(req.n, req.u_ref, req.womersley_number, req.phase)
    else:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown profile_type '{req.profile_type}'. "
                "Choose from: log_law, power_law, parabolic, blasius, womersley."
            ),
        )

    if req.add_synthetic_turbulence:
        profile = synthetic_turbulence_2d(
            profile, req.turbulence_intensity, seed=req.seed,
        )

    y_coords = [(i + 0.5) for i in range(req.n)]
    profile_list = profile.tolist()

    return {
        "profile_type": req.profile_type,
        "n": req.n,
        "u_ref": req.u_ref,
        "u_bulk": float(profile.mean()),
        "u_max": float(profile.max()),
        "profile": [
            {"y": y, "ux": u} for y, u in zip(y_coords, profile_list, strict=False)
        ],
    }


# ---------------------------------------------------------------------------
# Turbulence statistics from completed job checkpoints
# ---------------------------------------------------------------------------

@router.get("/turbulence-stats/{job_id}")
async def turbulence_stats(
    job_id: str,
    is_3d: bool = Query(False, description="Set true for 3D jobs (D3Q19/D3Q27)"),
    max_checkpoints: int = Query(50, ge=1, le=200, description="Max checkpoints to process"),
) -> dict:
    """Compute turbulence statistics from a completed job's checkpoint history.

    Iterates through up to *max_checkpoints* saved checkpoint files and
    accumulates:

    * Time-averaged velocity fields ``<U>``, ``<V>``, ``<W>``
    * Reynolds normal stresses ``<u'u'>``, ``<v'v'>``, ``<w'w'>``
    * Reynolds shear stress ``<u'v'>``
    * Turbulence kinetic energy ``k``
    * Turbulence intensity ``Tu`` (%)
    * Higher-order moments: skewness and flatness of streamwise fluctuations

    Equivalent to the turbulence statistics dashboards in PowerFlow and XFlow.
    """
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status.value not in ("completed", "failed"):
        raise HTTPException(status_code=409, detail="Job has not completed yet")

    try:
        from tensorlbm.turbulence_stats import turbulence_stats_from_checkpoints

        stats = turbulence_stats_from_checkpoints(
            job.output_dir,
            is_3d=is_3d,
            max_checkpoints=max_checkpoints,
        )
        return {"job_id": job_id, **stats}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# DFSEM / Digital-Filter inlet preview
# ---------------------------------------------------------------------------

class DFSEMPreviewRequest(BaseModel):
    ny: int = Field(default=64, ge=4, le=512)
    nz: int = Field(default=1, ge=1, le=512)
    u_mean: float = Field(default=0.1, gt=0.0)
    uu: float = Field(default=1e-4, gt=0.0, description="Target <u'u'> Reynolds stress")
    vv: float = Field(default=1e-4, gt=0.0, description="Target <v'v'> Reynolds stress")
    ww: float = Field(default=1e-4, gt=0.0, description="Target <w'w'> Reynolds stress")
    length_scale: float = Field(
        default=5.0, gt=0.0, description="Eddy length scale (lattice units)"
    )
    n_eddies: int = Field(
        default=200, ge=10, le=2000, description="Number of synthetic eddies (DFSEM only)"
    )
    method: str = Field(default="dfsem", description="'dfsem' or 'digital_filter'")
    seed: int = 42


@router.post("/dfsem-preview")
async def dfsem_preview(req: DFSEMPreviewRequest) -> dict:
    """Generate a preview snapshot of synthetic turbulent inflow fluctuations.

    Returns the u- and v-fluctuation profiles for the first sample of either
    the **DFSEM** (Divergence-Free Synthetic Eddy Method) or the **Digital
    Filter Method** inlet generator.

    This endpoint mirrors the inflow turbulence preview in PowerFlow's
    boundary-condition setup dialog.
    """
    try:
        import torch

        from tensorlbm.synthetic_inflow import DFSEMInlet, DigitalFilterInlet

        device = torch.device("cpu")
        u_mean_t = torch.full((req.ny, max(req.nz, 1)), req.u_mean)

        if req.method.lower() == "digital_filter":
            gen = DigitalFilterInlet(
                ny=req.ny, nz=req.nz,
                uu=req.uu, vv=req.vv, ww=req.ww,
                length_scale=req.length_scale,
                device=device,
                seed=req.seed,
            )
        else:
            gen = DFSEMInlet(  # type: ignore[assignment]
                ny=req.ny, nz=req.nz,
                u_mean=u_mean_t,
                uu=req.uu, vv=req.vv, ww=req.ww,
                length_scale=req.length_scale,
                n_eddies=req.n_eddies,
                device=device,
                seed=req.seed,
            )

        u_f, v_f, w_f = gen.sample()

        u_rms = float(u_f.std().item())
        v_rms = float(v_f.std().item())
        tu = u_rms / req.u_mean * 100.0

        # Return mid-z slice of fluctuation profiles
        iz = max(req.nz, 1) // 2
        y_coords = list(range(req.ny))

        return {
            "method": req.method,
            "ny": req.ny, "nz": req.nz,
            "u_rms": u_rms, "v_rms": v_rms,
            "tu_percent": tu,
            "target_uu": req.uu,
            "target_vv": req.vv,
            "u_fluct_profile": u_f[:, iz].tolist(),
            "v_fluct_profile": v_f[:, iz].tolist(),
            "y_coords": y_coords,
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Sponge-layer preview
# ---------------------------------------------------------------------------

class SpongePreviewRequest(BaseModel):
    nx: int = Field(default=200, ge=10, le=2048)
    x0: int = Field(default=150, ge=0, description="Sponge zone start index")
    x1: int = Field(default=199, ge=0, description="Sponge zone end index")
    amplitude: float = Field(default=0.5, gt=0.0, le=2.0)
    exponent: float = Field(default=3.0, gt=0.0, le=10.0)


@router.post("/sponge-preview")
async def sponge_preview(req: SpongePreviewRequest) -> dict:
    """Preview the sponge/absorbing-layer strength profile along the x-axis.

    Returns a list of ``(x, alpha)`` pairs showing the damping coefficient
    profile that would be applied at the outlet boundary.  Use this to
    tune the sponge zone length and amplitude before running a simulation.

    The sponge BC prevents acoustic reflections from the outlet, matching
    the absorbing-layer implementation in commercial LBM solvers.
    """
    try:

        from tensorlbm.sponge_bc import sponge_profile

        profile = sponge_profile(
            nx=req.nx,
            x0=req.x0,
            x1=min(req.x1, req.nx - 1),
            amplitude=req.amplitude,
            exponent=req.exponent,
        )

        x_coords = list(range(req.nx))
        alpha_vals = profile.tolist()

        return {
            "nx": req.nx,
            "x0": req.x0,
            "x1": req.x1,
            "amplitude": req.amplitude,
            "exponent": req.exponent,
            "sponge_width": req.x1 - req.x0,
            "profile": [
                {"x": x, "alpha": a} for x, a in zip(x_coords, alpha_vals, strict=False)
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Wall-roughness Y+ and slip-velocity preview
# ---------------------------------------------------------------------------

class RoughnessPreviewRequest(BaseModel):
    u_tau: float = Field(default=0.01, gt=0.0, description="Friction velocity (lattice units)")
    nu: float = Field(
        default=1.0 / 600.0, gt=0.0, description="Kinematic viscosity (lattice units)"
    )
    ks: float = Field(
        default=0.5, ge=0.0, description="Equivalent sand-grain roughness height (lattice units)"
    )
    n_points: int = Field(
        default=100, ge=10, le=500, description="Number of ks+ evaluation points"
    )


@router.post("/roughness-preview")
async def roughness_preview(req: RoughnessPreviewRequest) -> dict:
    """Preview the wall-roughness correction to the log-law additive constant.

    Returns the roughness regime classification and ΔB correction curve as
    a function of dimensionless roughness height *ks+* = *ks* × *u_tau* / *ν*.

    This endpoint mirrors the roughness parameter setup in PowerFlow's
    wall treatment dialog and allows engineers to assess the impact of
    surface roughness before running a full simulation.
    """
    try:
        import torch

        from tensorlbm.roughness import B_SMOOTH, KAPPA, roughness_b_correction

        # Sweep ks+ from 0.1 to 500 to show the full correction curve
        ks_plus_vals = torch.linspace(0.1, 500.0, req.n_points)
        delta_b = roughness_b_correction(ks_plus_vals)
        b_eff = B_SMOOTH - delta_b

        # Classify current operating point
        ks_plus_current = req.ks * req.u_tau / req.nu
        if ks_plus_current < 2.25:
            regime = "hydraulically_smooth"
        elif ks_plus_current > 90.0:
            regime = "fully_rough"
        else:
            regime = "transitional"

        delta_b_current = float(roughness_b_correction(
            torch.tensor([ks_plus_current])
        ).item())

        return {
            "ks_plus_current": float(ks_plus_current),
            "regime": regime,
            "delta_b_current": delta_b_current,
            "b_eff_current": B_SMOOTH - delta_b_current,
            "kappa": KAPPA,
            "b_smooth": B_SMOOTH,
            "curve": [
                {"ks_plus": float(k), "delta_b": float(db), "b_eff": float(be)}
                for k, db, be in zip(
                    ks_plus_vals.tolist(),
                    delta_b.tolist(),
                    b_eff.tolist(),
                    strict=False,
                )
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Force decomposition: pressure + viscous (new industrial feature)
# ---------------------------------------------------------------------------

@router.get("/force-decomposition/{job_id}")
async def force_decomposition(
    job_id: str,
    rho_ref: float = Query(default=1.0, gt=0.0),  # noqa: B008
    u_ref: float = Query(default=0.1, gt=0.0),  # noqa: B008
    area_ref: float = Query(default=1.0, gt=0.0),  # noqa: B008
) -> dict:
    """Decompose aerodynamic force into pressure and viscous components.

    Returns total Cd/Cl plus the pressure-drag and viscous-drag split,
    matching the force-decomposition report in PowerFlow and XFlow.

    Query params:
        rho_ref:  Reference density for Cd/Cl (default 1.0).
        u_ref:    Reference velocity (default 0.1).
        area_ref: Reference area/chord (default 1.0).
    """
    import torch  # noqa: PLC0415

    from tensorlbm.checkpoint import load_checkpoint  # noqa: PLC0415
    from tensorlbm.d2q9 import macroscopic  # noqa: PLC0415
    from tensorlbm.surface_integrals import surface_force_decomposed_2d  # noqa: PLC0415

    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status.value != "completed":
        raise HTTPException(status_code=409, detail="Job not completed yet")

    ckpts = sorted(job.output_dir.rglob("checkpoint_f.pt"))
    if not ckpts:
        raise HTTPException(status_code=404, detail="No checkpoint found")

    f, step, meta = load_checkpoint(ckpts[-1].parent)
    if f.ndim != 3:
        raise HTTPException(status_code=422, detail="Force decomposition requires a 2-D job")

    rho, ux, uy = macroscopic(f)
    speed = torch.sqrt(ux * ux + uy * uy)
    mask = speed < 1e-6

    tau = float(meta.get("tau", 0.6)) if isinstance(meta, dict) else 0.6
    result = surface_force_decomposed_2d(
        f, rho, ux, uy, mask, tau, rho_ref, u_ref, area_ref,
    )
    result["job_id"] = job_id
    result["step"] = step
    result["tau"] = tau
    return result


# ---------------------------------------------------------------------------
# Wall shear stress distribution (new industrial feature)
# ---------------------------------------------------------------------------

@router.get("/wall-shear-stress/{job_id}")
async def wall_shear_stress(
    job_id: str,
    normalise: bool = Query(default=True),  # noqa: B008
    rho_ref: float = Query(default=1.0, gt=0.0),  # noqa: B008
    u_ref: float = Query(default=0.1, gt=0.0),  # noqa: B008
) -> dict:
    """Compute the wall shear stress (WSS) distribution for a completed job.

    Returns the 2-D WSS map and summary statistics (max, mean).  When
    ``normalise=true`` (default) also returns the skin-friction coefficient
    Cf = τ_w / (½ρU²) map.  Matches the WSS post-processing in XFlow.
    """
    import torch  # noqa: PLC0415

    from tensorlbm.checkpoint import load_checkpoint  # noqa: PLC0415
    from tensorlbm.d2q9 import macroscopic  # noqa: PLC0415
    from tensorlbm.wall_shear import wss_map_2d  # noqa: PLC0415

    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status.value != "completed":
        raise HTTPException(status_code=409, detail="Job not completed yet")

    ckpts = sorted(job.output_dir.rglob("checkpoint_f.pt"))
    if not ckpts:
        raise HTTPException(status_code=404, detail="No checkpoint found")

    f, step, meta = load_checkpoint(ckpts[-1].parent)
    if f.ndim != 3:
        raise HTTPException(status_code=422, detail="WSS requires a 2-D job")

    rho, ux, uy = macroscopic(f)
    speed = torch.sqrt(ux * ux + uy * uy)
    mask = speed < 1e-6
    tau = float(meta.get("tau", 0.6)) if isinstance(meta, dict) else 0.6

    result = wss_map_2d(f, rho, ux, uy, tau, mask,
                        normalise=normalise, rho_ref=rho_ref, u_ref=u_ref)
    result["job_id"] = job_id
    result["step"] = step
    result["tau"] = tau
    return result


# ---------------------------------------------------------------------------
# Vortex identification criteria (new industrial feature)
# ---------------------------------------------------------------------------

@router.get("/vortex-criterion/{job_id}")
async def vortex_criterion(
    job_id: str,
    criteria: str = Query(default="q,omega", description="Comma-separated: q,lambda2,omega"),  # noqa: B008
) -> dict:
    """Compute vortex identification criterion fields for a completed job.

    Supported criteria (comma-separated in ``criteria`` param):
    * ``q``      – Q-criterion (Hunt et al. 1988): positive = vortex core.
    * ``lambda2``– λ₂-criterion (Jeong & Hussain 1995): negative = vortex core.
    * ``omega``  – Ω-criterion (Liu et al. 2016): Ω > 0.52 = vortex core.

    Returns scalar fields as nested float lists, suitable for
    colourmap visualisation in the browser.
    """
    import torch  # noqa: PLC0415

    from tensorlbm.checkpoint import load_checkpoint  # noqa: PLC0415
    from tensorlbm.d2q9 import macroscopic  # noqa: PLC0415
    from tensorlbm.vortex_identification import vortex_fields_2d, vortex_fields_3d  # noqa: PLC0415

    valid_criteria = {"q", "lambda2", "omega"}
    requested = [c.strip().lower() for c in criteria.split(",") if c.strip()]
    unknown = set(requested) - valid_criteria
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown criteria: {unknown}. Choose from: q, lambda2, omega.",
        )
    if not requested:
        raise HTTPException(status_code=422, detail="At least one criterion must be specified.")

    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status.value != "completed":
        raise HTTPException(status_code=409, detail="Job not completed yet")

    ckpts = sorted(job.output_dir.rglob("checkpoint_f.pt"))
    if not ckpts:
        raise HTTPException(status_code=404, detail="No checkpoint found")

    f, step, _meta = load_checkpoint(ckpts[-1].parent)
    is_3d = f.ndim == 4

    if is_3d:
        from tensorlbm.d3q19 import macroscopic as mac3d  # noqa: PLC0415
        rho, ux, uy, uz = mac3d(f)
        speed = torch.sqrt(ux ** 2 + uy ** 2 + uz ** 2)
        mask = speed < 1e-6
        fields = vortex_fields_3d(ux, uy, uz, mask=mask, criteria=requested)
    else:
        rho, ux, uy = macroscopic(f)
        speed = torch.sqrt(ux ** 2 + uy ** 2)
        mask = speed < 1e-6
        fields = vortex_fields_2d(ux, uy, mask=mask)
        # Filter to requested criteria
        fields = {k: v for k, v in fields.items() if k in requested}

    shape_info = list(f.shape[1:]) if is_3d else list(f.shape[1:])
    return {
        "job_id": job_id,
        "step": step,
        "dim": "3d" if is_3d else "2d",
        "shape": shape_info,
        "criteria": requested,
        **fields,
    }


# ---------------------------------------------------------------------------
# Animation export (new industrial feature)
# ---------------------------------------------------------------------------

@router.get("/animation/{job_id}")
async def export_animation(
    job_id: str,
    fps: int = Query(default=10, ge=1, le=60),  # noqa: B008
    fmt: str = Query(default="gif", description="'gif' or 'mp4'"),  # noqa: B008
    max_frames: int = Query(default=200, ge=10, le=500),  # noqa: B008
) -> StreamingResponse:
    """Export a flow-field animation (GIF or MP4) from job snapshot images.

    Discovers all ``step_XXXXXX.png`` images in the job output directory,
    assembles them into an animation, and streams the file to the client.

    Query params:
        fps:        Frames per second (1–60, default 10).
        fmt:        Output format: ``gif`` (default) or ``mp4`` (requires ffmpeg).
        max_frames: Maximum number of frames to include (default 200).
    """
    from tensorlbm.animation_export import create_animation  # noqa: PLC0415

    if fmt not in ("gif", "mp4"):
        raise HTTPException(status_code=422, detail="fmt must be 'gif' or 'mp4'")

    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status.value != "completed":
        raise HTTPException(status_code=409, detail="Job not completed yet")

    tmp_dir = job.output_dir / "_animation_cache"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        anim_path = create_animation(
            job.output_dir,
            output_dir=tmp_dir,
            fps=fps,
            fmt=fmt,  # type: ignore[arg-type]
            max_frames=max_frames,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    media_type = "video/mp4" if fmt == "mp4" else "image/gif"
    from fastapi.responses import FileResponse  # noqa: PLC0415
    return FileResponse(
        str(anim_path),
        media_type=media_type,
        filename=f"job_{job_id}_animation.{fmt}",
    )


# ---------------------------------------------------------------------------
# Heat flux mapping for conjugate-HT jobs (new industrial feature)
# ---------------------------------------------------------------------------

@router.get("/heat-flux/{job_id}")
async def heat_flux_map(
    job_id: str,
    alpha: float = Query(default=1.0, gt=0.0, description="Thermal diffusivity (lattice units)"),  # noqa: B008
) -> dict:
    """Extract heat flux density distribution from a thermal/conjugate-HT job.

    Computes q'' = -k ∇T at each fluid cell using finite differences on
    the temperature field.  Returns the heat-flux magnitude map and summary
    statistics, matching the heat-flux post-processing in XFlow and PowerFlow.

    Query params:
        alpha: Thermal diffusivity used in the simulation (default 1.0).
    """
    import torch  # noqa: PLC0415

    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status.value != "completed":
        raise HTTPException(status_code=409, detail="Job not completed yet")

    # Look for temperature field CSV or checkpoint
    temp_csv = list(job.output_dir.rglob("temperature*.csv"))
    if not temp_csv:
        # Try to load from checkpoint metadata
        ckpts = sorted(job.output_dir.rglob("checkpoint_f.pt"))
        if not ckpts:
            raise HTTPException(status_code=404, detail="No temperature data found in job output")
        # For thermal jobs, temperature is in checkpoint metadata
        from tensorlbm.checkpoint import load_checkpoint  # noqa: PLC0415
        f, step, meta = load_checkpoint(ckpts[-1].parent)
        if not isinstance(meta, dict) or "temperature" not in meta:
            raise HTTPException(
                status_code=422,
                detail="Heat flux extraction requires a thermal/conjugate-HT job",
            )
        T = torch.tensor(meta["temperature"], dtype=torch.float32)
    else:
        import csv as _csv  # noqa: PLC0415
        with temp_csv[0].open() as fh:
            rows = list(_csv.reader(fh))
        step = 0
        data = [[float(x) for x in row] for row in rows[1:] if len(row) > 1]
        if not data:
            raise HTTPException(status_code=404, detail="Temperature CSV is empty")
        T = torch.tensor(data, dtype=torch.float32)

    if T.ndim != 2:
        raise HTTPException(
            status_code=422, detail="Heat flux only supported for 2-D thermal fields"
        )

    # Compute heat flux via central differences: q''_x = -k dT/dx, q''_y = -k dT/dy
    dT_dx = torch.zeros_like(T)
    dT_dy = torch.zeros_like(T)
    dT_dx[:, 1:-1] = (T[:, 2:] - T[:, :-2]) / 2.0
    dT_dy[1:-1, :] = (T[2:, :] - T[:-2, :]) / 2.0

    # k = rho * c_p * alpha (in LBM: rho~1, c_p~1, so k ~ alpha)
    k = alpha
    qx = -k * dT_dx
    qy = -k * dT_dy
    q_mag = torch.sqrt(qx ** 2 + qy ** 2)

    return {
        "job_id": job_id,
        "step": step,
        "alpha": alpha,
        "heat_flux_x": qx.cpu().tolist(),
        "heat_flux_y": qy.cpu().tolist(),
        "heat_flux_magnitude": q_mag.cpu().tolist(),
        "q_max": float(q_mag.max().item()),
        "q_mean": float(q_mag.mean().item()),
        "T_max": float(T.max().item()),
        "T_min": float(T.min().item()),
    }


# ---------------------------------------------------------------------------
# Acoustic spectrum analysis (PSD + OASPL) (new industrial feature)
# ---------------------------------------------------------------------------

@router.get("/acoustics-spectrum/{job_id}")
async def acoustics_spectrum(
    job_id: str,
    fs: float = Query(default=1.0, gt=0.0, description="Sampling frequency (1/Δt, lattice units)"),  # noqa: B008
    window: str = Query(default="hann", description="FFT window: hann, hamming, blackman"),  # noqa: B008
    nperseg: int = Query(default=256, ge=32, le=4096, description="Welch segment length"),  # noqa: B008
    p_ref: float = Query(default=2e-5, gt=0.0, description="Reference pressure (Pa or l.u.)"),  # noqa: B008
) -> dict:
    """Compute acoustic power spectral density (PSD) and OASPL from an FWH job.

    Reads the time-history pressure signal from the FWH acoustics output,
    computes Welch PSD, 1/3-octave band levels, and overall SPL (OASPL).
    Matches the acoustic post-processing in XFlow and PowerFlow.

    Query params:
        fs:      Sampling frequency (cycles per step, default 1.0).
        window:  Spectral window function (default 'hann').
        nperseg: Segment length for Welch estimate (default 256).
        p_ref:   Reference pressure for dB conversion (default 2e-5).
    """
    import numpy as np  # noqa: PLC0415

    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status.value != "completed":
        raise HTTPException(status_code=409, detail="Job not completed yet")

    # Load acoustic time-history from FWH output CSV
    acoustic_csvs = (
        list(job.output_dir.rglob("*acoustic*.csv"))
        + list(job.output_dir.rglob("*fwh*.csv"))
    )
    if not acoustic_csvs:
        raise HTTPException(
            status_code=404,
            detail="No acoustic time-history CSV found. Run an FWH acoustics job first.",
        )

    import csv as _csv  # noqa: PLC0415
    p_signal: list[float] = []
    with acoustic_csvs[0].open() as fh:
        reader = _csv.DictReader(fh)
        for row in reader:
            for key in ("p_prime", "pressure", "p", "spl"):
                if key in row:
                    with contextlib.suppress(ValueError):
                        p_signal.append(float(row[key]))
                    break

    if len(p_signal) < 16:
        raise HTTPException(
            status_code=422,
            detail=f"Acoustic time series too short ({len(p_signal)} samples). Need ≥ 16.",
        )

    p_arr = np.array(p_signal, dtype=np.float64)

    try:
        from scipy.signal import welch  # noqa: PLC0415
        freqs, psd = welch(p_arr, fs=fs, window=window, nperseg=min(nperseg, len(p_arr) // 2))
    except ImportError:
        # Fallback: simple FFT
        fft_vals = np.fft.rfft(p_arr)
        psd = (np.abs(fft_vals) ** 2) / (len(p_arr) * fs)
        freqs = np.fft.rfftfreq(len(p_arr), d=1.0 / fs)

    # SPL spectrum (dB re p_ref)
    spl = 10.0 * np.log10(psd / (p_ref ** 2) + 1e-30)

    # OASPL (overall)
    p_rms = float(np.sqrt(np.mean(p_arr ** 2)))
    oaspl_db = 20.0 * np.log10(p_rms / p_ref + 1e-30)

    # 1/3-octave band levels
    third_oct_centers = [
        16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315,
        400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000,
    ]
    third_oct_spl: list[dict] = []
    for fc in third_oct_centers:
        f_lo = fc / 2.0 ** (1.0 / 6.0)
        f_hi = fc * 2.0 ** (1.0 / 6.0)
        band = (freqs >= f_lo) & (freqs <= f_hi)
        if band.any():
            p_band_rms = float(np.sqrt(np.trapz(psd[band], freqs[band])))
            spl_band = 20.0 * np.log10(p_band_rms / p_ref + 1e-30)
            third_oct_spl.append({"fc_hz": fc, "spl_db": round(spl_band, 2)})

    return {
        "job_id": job_id,
        "n_samples": len(p_signal),
        "sampling_frequency": fs,
        "window": window,
        "frequencies": freqs.tolist(),
        "psd": psd.tolist(),
        "spl_db": spl.tolist(),
        "oaspl_db": round(oaspl_db, 2),
        "p_rms": p_rms,
        "third_octave_bands": third_oct_spl,
    }


# ---------------------------------------------------------------------------
# Multi-case overlay chart (new industrial feature)
# ---------------------------------------------------------------------------

class MultiCaseChartRequest(BaseModel):
    job_ids: list[str] = Field(..., min_length=2, max_length=20)
    metric: str = Field(
        default="cd",
        description="Metric to compare: 'cd', 'cl', 'convergence', or any run_metadata key.",
    )
    labels: list[str] | None = Field(
        default=None, description="Custom legend labels (one per job)."
    )
    x_label: str = Field(default="Step")
    y_label: str | None = None


@router.post("/multi-case-chart")
async def multi_case_chart(req: MultiCaseChartRequest) -> StreamingResponse:
    """Generate a multi-case overlay comparison chart as a PNG image.

    Reads the convergence / force history from each specified job and
    overlays them on a single matplotlib figure, matching the multi-case
    comparison report in PowerFlow.

    Returns a PNG image stream.
    """
    import io as _io  # noqa: PLC0415
    import json as _json  # noqa: PLC0415

    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from fastapi.responses import StreamingResponse  # noqa: PLC0415

    labels = req.labels or req.job_ids
    if len(labels) != len(req.job_ids):
        raise HTTPException(status_code=422, detail="labels length must match job_ids length")

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = 0

    for job_id, label in zip(req.job_ids, labels, strict=False):
        job = job_manager.get_job(job_id)
        if job is None:
            continue

        meta_file = job.output_dir / "run_metadata.json"
        if not meta_file.exists():
            # Search in subdirs
            candidates = list(job.output_dir.rglob("run_metadata.json"))
            meta_file = candidates[0] if candidates else None

        if meta_file is None:
            continue

        meta = _json.loads(meta_file.read_text())
        metric = req.metric.lower()

        # Extract x, y series
        steps_key = "steps"
        if metric == "convergence":
            y_key: str | None = "drag_history"
        elif metric in ("cd", "drag"):
            y_key = next(
                (k for k in ("cd_history", "force_history", "drag_history") if k in meta), None
            )
        elif metric in ("cl", "lift"):
            y_key = "cl_history" if "cl_history" in meta else None
        else:
            y_key = metric

        xs = meta.get(steps_key) or meta.get("step") or []
        ys_raw = meta.get(y_key) if y_key else None

        if ys_raw is None:
            # Try result dict from job
            ys_raw = job.result.get(y_key) or job.result.get(metric)

        if isinstance(ys_raw, list) and len(ys_raw) > 0:
            if isinstance(ys_raw[0], dict):
                ys = [v.get(metric, v.get("cd", 0.0)) for v in ys_raw]
                if not xs:
                    xs = [v.get("step", i) for i, v in enumerate(ys_raw)]
            else:
                ys = ys_raw
        else:
            continue

        if not xs:
            xs = list(range(len(ys)))
        if len(xs) > len(ys):
            xs = xs[:len(ys)]
        elif len(ys) > len(xs):
            ys = ys[:len(xs)]

        ax.plot(xs, ys, label=label)
        plotted += 1

    if plotted == 0:
        raise HTTPException(
            status_code=422,
            detail="No plottable data found for the specified jobs and metric.",
        )

    y_label = req.y_label or req.metric.upper()
    ax.set_xlabel(req.x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"Multi-case comparison: {req.metric}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={"Content-Disposition": "inline; filename=multi_case_chart.png"},
    )


# ---------------------------------------------------------------------------
# Spectral analysis of probe signals
# ---------------------------------------------------------------------------

class ProbeSpectrumRequest(BaseModel):
    """Request body for probe-signal spectral analysis."""

    job_id: str | None = Field(
        default=None,
        description="Completed job to load probe history from. "
                    "If omitted, *signal* must be provided.",
    )
    signal: list[float] | None = Field(
        default=None,
        description="Explicit time-series values (e.g. Cd, pressure). "
                    "Used when job_id is not provided.",
    )
    dt: float = Field(
        default=1.0,
        gt=0.0,
        description="Time step between samples (LBM or physical units).",
    )
    column: str = Field(
        default="cd",
        description="Column name to read from probe CSV (e.g. 'cd', 'pressure').",
    )
    n_segment: int | None = Field(
        default=None,
        gt=0,
        description="Welch segment length (defaults to N/4 rounded to power of 2).",
    )
    overlap: float = Field(
        default=0.5,
        ge=0.0,
        lt=1.0,
        description="Fractional segment overlap for Welch averaging.",
    )
    n_peaks: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of dominant spectral peaks to return.",
    )
    diameter: float | None = Field(
        default=None,
        gt=0.0,
        description="Characteristic length D for Strouhal number St = f·D/U.",
    )
    u_ref: float | None = Field(
        default=None,
        gt=0.0,
        description="Reference velocity U for Strouhal number.",
    )


@router.post("/probe-spectrum")
async def probe_spectrum(req: ProbeSpectrumRequest) -> dict:
    """Compute the power spectral density of a probe time-history signal.

    Applies Welch's averaged-periodogram method with Hanning windowing to
    return the one-sided PSD, dominant frequency peaks, and (optionally)
    the Strouhal number — matching the spectral post-processing capability
    of PowerFlow and XFlow.

    Either *job_id* (loads the first ``forces.csv`` or ``convergence.csv``
    from the job output directory) or an explicit *signal* list must be
    supplied.
    """
    import csv as _csv  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from tensorlbm.probe_spectrum import compute_probe_spectrum  # noqa: PLC0415

    signal: list[float] | None = req.signal

    if signal is None:
        if req.job_id is None:
            raise HTTPException(
                status_code=422,
                detail="Provide either job_id or signal.",
            )
        job = job_manager.get_job(req.job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")

        # Search for CSV files containing the requested column
        csv_files = list(job.output_dir.rglob("forces*.csv"))
        csv_files += list(job.output_dir.rglob("convergence*.csv"))
        csv_files += list(job.output_dir.rglob("probe*.csv"))

        for csv_path in csv_files:
            try:
                with csv_path.open(newline="", encoding="utf-8") as fh:
                    reader = _csv.DictReader(fh)
                    rows = list(reader)
                if rows and req.column in rows[0]:
                    signal = [float(r[req.column]) for r in rows if r[req.column]]
                    break
            except Exception:
                continue

        if not signal:
            raise HTTPException(
                status_code=404,
                detail=f"No probe data with column '{req.column}' found for job.",
            )

    if len(signal) < 4:
        raise HTTPException(
            status_code=422,
            detail="Signal must contain at least 4 samples.",
        )

    result = compute_probe_spectrum(
        signal,
        dt=req.dt,
        n_segment=req.n_segment,
        overlap=req.overlap,
        n_peaks=req.n_peaks,
        diameter=req.diameter,
        u_ref=req.u_ref,
    )

    return {
        "frequencies": result.frequencies,
        "psd": result.psd,
        "peak_frequencies": result.peak_frequencies,
        "peak_psd": result.peak_psd,
        "f_nyquist": result.f_nyquist,
        "n_samples": result.n_samples,
        "dt": result.dt,
        "signal_rms": result.signal_rms,
        "strouhal": result.strouhal,
    }


# ---------------------------------------------------------------------------
# POD modal decomposition
# ---------------------------------------------------------------------------

class PODRequest(BaseModel):
    """Request body for POD decomposition."""

    job_id: str | None = Field(
        default=None,
        description="Completed job to load snapshots from.",
    )
    snapshots: list[list[list[float]]] | None = Field(
        default=None,
        description="Explicit snapshot list: list of 2-D fields, each [[...], ...].",
    )
    n_modes: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of POD modes to retain.",
    )
    field_name: str = Field(
        default="ux",
        description="Velocity/scalar field to decompose ('ux', 'uy', 'rho').",
    )
    return_coefficients: bool = Field(
        default=True,
        description="Whether to compute and return temporal coefficients.",
    )


@router.post("/pod")
async def pod_decomposition(req: PODRequest) -> dict:
    """Perform Proper Orthogonal Decomposition (POD) on simulation snapshots.

    Uses the method of snapshots (Sirovich, 1987) via SVD to extract
    the dominant coherent flow structures.  Returns singular values, energy
    fractions, mode shapes, and temporal coefficients — equivalent to the
    POD/spectral post-processing available in PowerFlow/XFlow.

    Either *job_id* (loads velocity snapshots from checkpoint files) or
    an explicit *snapshots* list must be provided.
    """
    import torch  # noqa: PLC0415

    from tensorlbm.pod import compute_pod  # noqa: PLC0415

    snap_tensors: list[Any] = []

    if req.snapshots is not None:
        if len(req.snapshots) < 2:
            raise HTTPException(
                status_code=422,
                detail="At least 2 snapshots are required for POD.",
            )
        snap_tensors = [
            torch.tensor(s, dtype=torch.float32)
            for s in req.snapshots
        ]
    elif req.job_id is not None:
        job = job_manager.get_job(req.job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")

        # Load snapshots from checkpoint files
        ckpt_dirs = sorted(job.output_dir.rglob("checkpoint_f.pt"))
        if not ckpt_dirs:
            raise HTTPException(
                status_code=404,
                detail="No checkpoint snapshots found for this job.",
            )

        from tensorlbm.checkpoint import load_checkpoint  # noqa: PLC0415

        for ckpt_file in ckpt_dirs[:50]:  # cap at 50 snapshots
            try:
                f, _step, meta = load_checkpoint(ckpt_file.parent)
                rho, ux, uy = None, None, None
                from tensorlbm.d2q9 import macroscopic  # noqa: PLC0415

                rho, ux, uy = macroscopic(f)
                field_map = {"rho": rho, "ux": ux, "uy": uy}
                snap = field_map.get(req.field_name, ux)
                if snap is not None:
                    snap_tensors.append(snap.float())
            except Exception:
                continue

        if len(snap_tensors) < 2:
            raise HTTPException(
                status_code=422,
                detail=f"Found only {len(snap_tensors)} valid snapshots (need ≥ 2).",
            )
    else:
        raise HTTPException(
            status_code=422,
            detail="Provide either job_id or snapshots.",
        )

    result = compute_pod(
        snap_tensors,
        n_modes=req.n_modes,
        return_coefficients=req.return_coefficients,
    )

    # Truncate mode data to avoid huge responses (return first mode shape only as preview)
    mode_preview = result.modes[0].tolist() if result.n_modes > 0 else []

    return {
        "n_modes": result.n_modes,
        "n_snapshots": result.n_snapshots,
        "spatial_shape": result.spatial_shape,
        "singular_values": result.singular_values,
        "energy_fraction": result.energy_fraction,
        "cumulative_energy": result.cumulative_energy,
        "temporal_coefficients": result.temporal_coefficients
        if req.return_coefficients
        else [],
        "mode_0_preview": mode_preview,
        "field_name": req.field_name,
    }


# ---------------------------------------------------------------------------
# Iso-surface / iso-contour extraction
# ---------------------------------------------------------------------------

@router.get("/isosurface/{job_id}")
async def extract_isosurface(
    job_id: str,
    field: str = Query(default="q_criterion", description="Scalar field name."),
    iso_value: float = Query(default=0.0, description="Iso-value for surface extraction."),
    slice_axis: str = Query(
        default="z",
        description="'z' for 2-D iso-contour (marching squares), "
                    "'3d' for 3-D iso-surface (marching cubes).",
    ),
    max_segments: int = Query(default=50_000, ge=1, le=500_000),
) -> dict:
    """Extract an iso-surface or iso-contour from a completed simulation job.

    For 2-D fields (slice_axis='z') applies marching squares to return a list
    of line segments.  For 3-D fields (slice_axis='3d') applies a centroid
    marching cubes to return vertices and triangles.

    This matches the iso-surface extraction feature available in PowerFlow
    and XFlow for visualising Q-criterion vortex tubes, pressure shells, and
    species concentration surfaces.
    """
    if slice_axis not in ("z", "3d"):
        raise HTTPException(
            status_code=422,
            detail="slice_axis must be 'z' (2-D contour) or '3d' (3-D surface).",
        )

    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    import torch  # noqa: PLC0415

    from tensorlbm.isosurface import (  # noqa: PLC0415
        marching_cubes_simple,
        marching_squares,
    )

    # Try to load field data from checkpoint
    ckpt_dirs = sorted(job.output_dir.rglob("checkpoint_f.pt"))
    field_tensor: torch.Tensor | None = None

    if ckpt_dirs:
        try:
            from tensorlbm.checkpoint import load_checkpoint  # noqa: PLC0415

            f, _step, _meta = load_checkpoint(ckpt_dirs[-1].parent)

            from tensorlbm.d2q9 import macroscopic  # noqa: PLC0415

            rho, ux, uy = macroscopic(f)

            if field == "q_criterion":
                from tensorlbm.vortex_identification import q_criterion_2d  # noqa: PLC0415

                field_tensor = q_criterion_2d(ux, uy)
            elif field == "rho":
                field_tensor = rho
            elif field == "ux":
                field_tensor = ux
            elif field == "uy":
                field_tensor = uy
            elif field == "speed":
                field_tensor = (ux ** 2 + uy ** 2).sqrt()
            else:
                field_tensor = ux  # fallback
        except Exception:
            pass

    if field_tensor is None:
        # Return an empty result with metadata
        if slice_axis == "z":
            return {
                "segments": [],
                "n_segments": 0,
                "iso_value": iso_value,
                "field": field,
                "mode": "2d",
                "note": "No field data available",
            }
        else:
            return {
                "vertices": [],
                "triangles": [],
                "n_triangles": 0,
                "iso_value": iso_value,
                "field": field,
                "mode": "3d",
                "note": "No field data available",
            }

    if slice_axis == "z":
        contour = marching_squares(
            field_tensor,
            iso_value=iso_value,
            field_name=field,
            max_segments=max_segments,
        )
        return {
            "segments": contour.segments,
            "n_segments": contour.n_segments,
            "iso_value": iso_value,
            "field": field,
            "mode": "2d",
        }
    else:
        if field_tensor.dim() == 2:
            field_3d = field_tensor.unsqueeze(0)  # add singleton z-dimension
        else:
            field_3d = field_tensor

        surface = marching_cubes_simple(
            field_3d,
            iso_value=iso_value,
            field_name=field,
            max_vertices=max_segments,
        )
        return {
            "vertices": surface.vertices,
            "triangles": surface.triangles,
            "n_triangles": surface.n_triangles,
            "iso_value": iso_value,
            "field": field,
            "mode": "3d",
        }
