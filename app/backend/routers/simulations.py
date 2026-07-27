"""General simulation API router — XFlow-style unified interface."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import torch

from .. import job_manager
from tensorlbm.general_sim import (
    BoundaryType,
    CollisionModel,
    GeneralSimConfig,
    GeneralSimEngine,
    GeometryConfig,
    GeometrySource,
    LatticeModel,
    OutputFormat,
    PhysicsConfig,
    SolverConfig,
    OutputConfig,
    BoundaryCondition,
)

router = APIRouter(prefix="/api/simulations", tags=["simulations"])

# ── Job tracking ────────────────────────────────────────────────────────────

_jobs: dict[str, dict[str, Any]] = {}
_job_lock = threading.Lock()


# ── Request schemas ─────────────────────────────────────────────────────────

class GeometryRequest(BaseModel):
    source: str = Field("none", description="Geometry source: stl_file, parametric_sphere, parametric_cylinder, parametric_hull, polygon_2d, none")
    stl_path: str | None = Field(None, description="STL file path (if source=stl_file)")
    stl_units: str = Field("m", description="STL file units: m, mm, lu")
    sphere_radius: float = Field(0.5, description="Sphere radius in meters")
    sphere_center: list[float] = Field([0, 0, 0], description="Sphere center [x,y,z] in meters")
    cylinder_radius: float = Field(0.5)
    cylinder_length: float = Field(2.0)
    cylinder_axis: str = Field("z")
    hull_type: str = Field("wigley")
    hull_length: float = Field(4.356)
    polygon_vertices: list[list[float]] = Field(default_factory=list)


class PhysicsRequest(BaseModel):
    density: float = Field(1000.0, description="Fluid density kg/m³")
    viscosity: float = Field(1e-6, description="Kinematic viscosity m²/s")
    inlet_velocity: float = Field(1.0, description="Inlet velocity m/s")
    reference_length: float = Field(1.0, description="Characteristic length m")
    gravity: list[float] = Field([0, 0, 0], description="Gravity m/s²")


class SolverRequest(BaseModel):
    lattice: str = Field("d3q19", description="Lattice model: d2q9, d3q19, d3q27")
    collision: str = Field("smagorinsky_mrt", description="Collision model: bgk, mrt, smagorinsky_bgk, smagorinsky_mrt")
    resolution: int = Field(48, description="Grid cells along reference length")
    domain_padding: list[float] = Field([2, 4, 1, 1, 1, 1], description="Domain padding [x_min,x_max,y_min,y_max,z_min,z_max] × ref_length")
    max_steps: int = Field(5000)
    warmup_steps: int = Field(1000)
    snapshot_interval: int = Field(100)
    force_sample_interval: int = Field(10)
    smagorinsky_cs: float = Field(0.1)
    target_mach: float = Field(0.05)
    device: str = Field("cpu")


class SimRunRequest(BaseModel):
    name: str = Field("unnamed", description="Simulation name")
    geometry: GeometryRequest = Field(default_factory=GeometryRequest)
    physics: PhysicsRequest = Field(default_factory=PhysicsRequest)
    solver: SolverRequest = Field(default_factory=SolverRequest)
    output_dir: str = Field("/tmp/tensorlbm_sim")


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.get("/capabilities")
def sim_capabilities():
    """List available simulation capabilities."""
    return {
        "lattice_models": [e.value for e in LatticeModel],
        "collision_models": [e.value for e in CollisionModel],
        "geometry_sources": [e.value for e in GeometrySource],
        "boundary_types": [e.value for e in BoundaryType],
        "output_formats": [e.value for e in OutputFormat],
        "parametric_shapes": ["sphere", "cylinder", "hull", "polygon_2d"],
        "stl_support": True,
        "turbulence_models": ["none", "smagorinsky"],
        "dimensions": ["2D", "3D"],
    }


@router.post("/setup")
def sim_setup(req: SimRunRequest):
    """Phase 1: Setup simulation domain (unit conversion, voxelisation, init)."""
    config = _build_config(req)
    engine = GeneralSimEngine(config)
    try:
        info = engine.setup()
        job_id = f"sim_{int(time.time())}"
        with _job_lock:
            _jobs[job_id] = {"config": config, "engine": engine, "status": "setup", "info": info}
        info["job_id"] = job_id
        return info
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run/{job_id}")
def sim_run(job_id: str, steps: int | None = None):
    """Phase 2: Run simulation (background thread via job_manager)."""
    with _job_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    engine = job["engine"]

    def _run(jm_job: Any) -> dict:
        with _job_lock:
            _jobs[job_id]["status"] = "running"
        result = engine.run(steps)
        with _job_lock:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["progress"] = result
        return result or {}

    # Use job_manager for proper thread pool scheduling
    jm_job_id = job_manager.submit(
        name=f"GeneralSim-{job_id}",
        job_type="general_sim",
        config={"sim_job_id": job_id, "steps": steps},
        fn=_run,
    )
    with _job_lock:
        _jobs[job_id]["jm_job_id"] = jm_job_id
    return {"job_id": job_id, "jm_job_id": jm_job_id, "status": "running", "steps": steps or engine.config.solver.max_steps}


@router.get("/status/{job_id}")
def sim_status(job_id: str):
    """Check simulation job status."""
    with _job_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    engine = job["engine"]
    return {
        "job_id": job_id,
        "status": job.get("status", "unknown"),
        "step_count": engine.step_count,
        "snapshots": len(engine.snapshots),
        "force_samples": len(engine.forces_log),
        "error": job.get("error"),
    }


@router.get("/results/{job_id}")
def sim_results(job_id: str, format: str = "npy"):
    """Phase 3: Export results (VTK/HDF5/NPY)."""
    with _job_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    engine = job["engine"]
    try:
        fmt = OutputFormat(format)
        return engine.results(output_format=fmt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/snapshot/{job_id}/{snap_idx}")
def sim_snapshot(job_id: str, snap_idx: int, channel: str = "ux"):
    """Get a snapshot slice for Plotly visualization."""
    try:
        with _job_lock:
            job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        engine = job["engine"]
        if snap_idx >= len(engine.snapshots):
            raise HTTPException(status_code=404, detail=f"Snapshot {snap_idx} not found (total: {len(engine.snapshots)})")
        snap = engine.snapshots[snap_idx]
        if channel not in snap:
            raise HTTPException(status_code=400, detail=f"Channel {channel} not in snapshot")
        arr = snap[channel]
        if isinstance(arr, torch.Tensor):
            arr = arr.cpu().numpy()
        # Ensure float64 for JSON serialization
        arr = arr.astype(float)
        # Take middle slice for 3D
        if arr.ndim == 3:
            mid = arr.shape[0] // 2
            arr = arr[mid]
        data = arr.tolist()
        return JSONResponse(content={
            "job_id": job_id, "snap_idx": snap_idx, "channel": channel,
            "shape": list(arr.shape), "data": data,
        })
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Snapshot error: {e}")


@router.get("/forces/{job_id}")
def sim_forces(job_id: str):
    """Get force history (Cd/Cl evolution)."""
    try:
        with _job_lock:
            job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        engine = job["engine"]
        forces = engine.forces_log
        # Filter out NaN/Inf values for JSON compliance
        clean_forces = []
        for f in forces:
            cf = {}
            for k, v in f.items():
                if isinstance(v, float) and (v != v or abs(v) == float("inf")):  # NaN or Inf
                    cf[k] = 0.0
                else:
                    cf[k] = v
            clean_forces.append(cf)
        return JSONResponse(content={"job_id": job_id, "forces": clean_forces})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forces error: {e}")


@router.get("/list")
def sim_list():
    """List all simulation jobs."""
    with _job_lock:
        return {"jobs": {k: {"status": v.get("status"), "step_count": v["engine"].step_count} for k, v in _jobs.items()}}


# ── Config builder ──────────────────────────────────────────────────────────

def _build_config(req: SimRunRequest) -> GeneralSimConfig:
    geo = GeometryConfig(
        source=GeometrySource(req.geometry.source),
        stl_path=req.geometry.stl_path,
        stl_units=req.geometry.stl_units,
        sphere_radius=req.geometry.sphere_radius,
        sphere_center=tuple(req.geometry.sphere_center),
        cylinder_radius=req.geometry.cylinder_radius,
        cylinder_length=req.geometry.cylinder_length,
        cylinder_axis=req.geometry.cylinder_axis,
        hull_type=req.geometry.hull_type,
        hull_length=req.geometry.hull_length,
        polygon_vertices=req.geometry.polygon_vertices,
    )
    phys = PhysicsConfig(
        density=req.physics.density,
        viscosity=req.physics.viscosity,
        inlet_velocity=req.physics.inlet_velocity,
        reference_length=req.physics.reference_length,
        gravity=tuple(req.physics.gravity),
    )
    sol = SolverConfig(
        lattice=LatticeModel(req.solver.lattice),
        collision=CollisionModel(req.solver.collision),
        resolution=req.solver.resolution,
        domain_padding=tuple(req.solver.domain_padding),
        max_steps=req.solver.max_steps,
        warmup_steps=req.solver.warmup_steps,
        snapshot_interval=req.solver.snapshot_interval,
        force_sample_interval=req.solver.force_sample_interval,
        smagorinsky_cs=req.solver.smagorinsky_cs,
        target_mach=req.solver.target_mach,
        device=req.solver.device,
    )
    out = OutputConfig(
        directory=req.output_dir,
        formats=[OutputFormat.NPY],
    )
    return GeneralSimConfig(
        name=req.name,
        geometry=geo,
        physics=phys,
        solver=sol,
        output=out,
    )
