"""Post-processing API endpoints.

Provides result analysis and metric extraction from completed simulation jobs,
including interactive field-viewer data (heatmaps, vectors, streamlines).
"""
from __future__ import annotations

import csv as csv_mod
import io
import json
import zipfile
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

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
