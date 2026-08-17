"""General simulation API router — XFlow-style unified interface."""
from __future__ import annotations

import functools
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
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


# ════════════════════════════════════════════════════════════════════════════
# Generic-run API (Phase 2 platform fusion)
# ════════════════════════════════════════════════════════════════════════════
#
# A unified endpoint that builds geometry (STL or parametric), auto-selects
# solver parameters, and runs a simulation using the *common interface*
# modules (lbm_step_correct + drag_pressure + drag_friction).  Results are
# streamed in real time via WebSocket diagnostics (Cd / Cl / St).
#
# SDAA cards 8–11 are used in round-robin fashion for compute.

# ── SDAA card selection (cards 8-11) ────────────────────────────────────────

_GENERIC_SDAA_CARDS = [8, 9, 10, 11]
_generic_card_idx = 0
_generic_card_lock = threading.Lock()


def _next_sdaa_card() -> int:
    """Round-robin SDAA card selection from the dedicated pool (8-11)."""
    global _generic_card_idx
    with _generic_card_lock:
        card = _GENERIC_SDAA_CARDS[_generic_card_idx % len(_GENERIC_SDAA_CARDS)]
        _generic_card_idx += 1
        return card


# ── Request schemas ─────────────────────────────────────────────────────────

class GenericRunGeometry(BaseModel):
    """Geometry specification for the generic-run endpoint."""
    source: str = Field("parametric", description="Geometry source: 'stl' or 'parametric'")
    path: str = Field("", description="STL file path (required when source='stl')")
    shape: str = Field("sphere", description="Parametric shape: 'sphere', 'cylinder', 'ellipsoid'")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Shape parameters (radius, length, axis, center, etc.)",
    )


class GenericRunPhysics(BaseModel):
    """Physical conditions for the generic-run endpoint."""
    Re: float = Field(100.0, description="Reynolds number")
    u_in: float = Field(0.08, description="Inlet / free-stream velocity (lattice units)")
    density: float = Field(1000.0, description="Fluid density (kg/m³)")
    viscosity: float = Field(1e-6, description="Kinematic viscosity (m²/s)")


class GenericRunSolver(BaseModel):
    """Solver configuration for the generic-run endpoint."""
    collision: str = Field("", description="Collision model: 'mrt', 'mrt_smag', 'bgk' (auto-selected if empty)")
    Cs: float = Field(0.0, description="Smagorinsky constant (auto-selected if 0)")
    steps: int = Field(0, description="Total steps (auto-selected if 0)")
    warmup: int = Field(0, description="Warmup steps before force averaging (auto-selected if 0)")
    lattice: str = Field("d3q19", description="Lattice model: 'd3q19' or 'd3q27'")


class GenericRunOutput(BaseModel):
    """Output configuration for the generic-run endpoint."""
    fields: list[str] = Field(
        default_factory=lambda: ["velocity", "pressure"],
        description="Output fields to record",
    )
    forces: bool = Field(True, description="Compute drag (Cd) and lift (Cl) forces")
    strouhal: bool = Field(True, description="Compute Strouhal number (St) from lift history")


class GenericRunRequest(BaseModel):
    """Top-level request body for POST /api/simulations/generic-run."""
    geometry: GenericRunGeometry = Field(default_factory=GenericRunGeometry)
    physics: GenericRunPhysics = Field(default_factory=GenericRunPhysics)
    solver: GenericRunSolver = Field(default_factory=GenericRunSolver)
    output: GenericRunOutput = Field(default_factory=GenericRunOutput)


# ── Auto-parameter selection ────────────────────────────────────────────────

def _auto_select_params(req: GenericRunRequest, shape: str) -> dict[str, Any]:
    """Auto-select solver parameters that were left at their sentinel values.

    Selection heuristics (mirroring the verified _common_interface_worker):
      - collision: MRT+Smagorinsky for Re > 1000, plain MRT otherwise.
      - Cs:        0.10 for Re > 1000, 0.05 for Re ≤ 1000.
      - steps:     5000 for cylinders (longer vortex development),
                   2000 for spheres (faster convergence).
      - warmup:    60 % of total steps.
      - domain:    shape-dependent (see _auto_domain).
    """
    Re = req.physics.Re
    collision = req.solver.collision
    Cs = req.solver.Cs
    steps = req.solver.steps
    warmup = req.solver.warmup

    if not collision:
        collision = "mrt_smag" if Re > 1000 else "mrt"
    if Cs == 0.0 and "smag" in collision:
        Cs = 0.10 if Re > 1000 else 0.05
    if steps == 0:
        if shape == "suboff":
            steps = 10000  # SUBOFF needs long warmup
        elif shape == "cylinder":
            steps = 5000
        else:
            steps = 2000
    if warmup == 0:
        warmup = int(steps * 0.5) if shape == "suboff" else int(steps * 0.6)

    return {
        "collision": collision,
        "Cs": Cs,
        "steps": steps,
        "warmup": warmup,
    }


def _auto_domain(shape: str, params: dict[str, Any]) -> dict[str, Any]:
    """Auto-select domain dimensions from the geometry parameters.

    Returns a dict with keys: nx, ny, nz, cx, cy, cz, R, length, axis.
    """
    if shape == "sphere":
        R = float(params.get("radius", 20.0))
        D = 2.0 * R
        nx = int(params.get("nx", max(120, int(6 * D))))
        ny = int(params.get("ny", max(120, int(6 * D))))
        nz = int(params.get("nz", max(120, int(6 * D))))
        cx = nx * 0.25
        cy = ny * 0.5
        cz = nz * 0.5
        return {"nx": nx, "ny": ny, "nz": nz, "cx": cx, "cy": cy, "cz": cz,
                "R": R, "length": D, "axis": "z"}
    elif shape == "cylinder":
        R = float(params.get("radius", 24.0))
        D = 2.0 * R
        length = float(params.get("length", 4.0 * D))
        axis = str(params.get("axis", "z"))
        nx = int(params.get("nx", max(960, int(10 * D))))
        ny = int(params.get("ny", max(384, int(4 * D))))
        nz = int(params.get("nz", 4))
        cx = nx * 0.25
        cy = ny * 0.5
        cz = nz * 0.5
        return {"nx": nx, "ny": ny, "nz": nz, "cx": cx, "cy": cy, "cz": cz,
                "R": R, "length": length, "axis": axis}
    elif shape == "suboff":
        # SUBOFF parametric submarine hull
        L = float(params.get("length", 80.0))
        R = L / (2 * 8.57)  # L/D = 8.57
        hull_type = str(params.get("hull_type", "bare_hull"))
        # 4L domain for best accuracy (2.7% verified)
        nx = int(params.get("nx", 320))
        ny = int(params.get("ny", 120))
        nz = int(params.get("nz", 120))
        cx = nx * 0.5
        cy = ny * 0.5
        cz = nz * 0.5
        return {"nx": nx, "ny": ny, "nz": nz, "cx": cx, "cy": cy, "cz": cz,
                "R": R, "length": L, "axis": "x", "hull_type": hull_type}

    elif shape == "ellipsoid":
        a = float(params.get("a", 20.0))
        b = float(params.get("b", 12.0))
        c = float(params.get("c", 12.0))
        D = 2.0 * max(a, b, c)
        nx = int(params.get("nx", max(120, int(6 * D))))
        ny = int(params.get("ny", max(120, int(6 * D))))
        nz = int(params.get("nz", max(120, int(6 * D))))
        cx = nx * 0.25
        cy = ny * 0.5
        cz = nz * 0.5
        return {"nx": nx, "ny": ny, "nz": nz, "cx": cx, "cy": cy, "cz": cz,
                "R": max(a, b, c), "length": D, "axis": "z",
                "a": a, "b": b, "c": c}
    else:
        # Generic fallback — treat as sphere with radius from params.
        R = float(params.get("radius", 20.0))
        D = 2.0 * R
        nx = int(params.get("nx", max(120, int(6 * D))))
        ny = int(params.get("ny", max(120, int(6 * D))))
        nz = int(params.get("nz", max(120, int(6 * D))))
        cx = nx * 0.25
        cy = ny * 0.5
        cz = nz * 0.5
        return {"nx": nx, "ny": ny, "nz": nz, "cx": cx, "cy": cy, "cz": cz,
                "R": R, "length": D, "axis": "z"}


# ── Geometry builders ───────────────────────────────────────────────────────

def _build_parametric_solid(shape: str, dom: dict, device: torch.device):
    """Build a boolean solid mask for a parametric shape.

    Returns (solid, mesh_kwargs) where mesh_kwargs are passed to the
    appropriate SurfaceMesh.from_* classmethod.
    """
    nx, ny, nz = dom["nx"], dom["ny"], dom["nz"]
    cx, cy, cz = dom["cx"], dom["cy"], dom["cz"]
    R = dom["R"]

    if shape == "cylinder":
        yy, xx = torch.meshgrid(
            torch.arange(ny, device=device, dtype=torch.float32),
            torch.arange(nx, device=device, dtype=torch.float32),
            indexing="ij",
        )
        circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= R ** 2
        solid = circle.unsqueeze(0).expand(nz, ny, nx).clone()
        return solid, {"cx": cx, "cy": cy, "R": R, "axis": dom.get("axis", "z")}

    elif shape == "ellipsoid":
        a, b, c = dom["a"], dom["b"], dom["c"]
        zz, yy, xx = torch.meshgrid(
            torch.arange(nz, device=device, dtype=torch.float32),
            torch.arange(ny, device=device, dtype=torch.float32),
            torch.arange(nx, device=device, dtype=torch.float32),
            indexing="ij",
        )
        solid = ((xx - cx) ** 2 / a ** 2 + (yy - cy) ** 2 / b ** 2
                 + (zz - cz) ** 2 / c ** 2) < 1.0
        return solid, {"cx": cx, "cy": cy, "cz": cz, "a": a, "b": b, "c": c}

    elif shape == "suboff":
        from tensorlbm.suboff_cad import build_suboff_mask, SuboffConfig
        L = dom["length"]
        hull_type = dom.get("hull_type", "bare_hull")
        config = SuboffConfig()
        solid, info = build_suboff_mask(
            hull_type=hull_type, nx=nx, ny=ny, nz=nz,
            length=L, config=config, device=str(device),
        )
        solid = solid.to(device)
        return solid, {
            "cx": info["cx"], "cy": info["cy"], "cz": info["cz"],
            "length": info["length"], "radius": info["radius"],
            "config": config, "_S_wet": info["wetted_area_lu2"],
        }

    else:  # sphere (default)
        zz, yy, xx = torch.meshgrid(
            torch.arange(nz, device=device, dtype=torch.float32),
            torch.arange(ny, device=device, dtype=torch.float32),
            torch.arange(nx, device=device, dtype=torch.float32),
            indexing="ij",
        )
        solid = ((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) < R ** 2
        return solid, {"cx": cx, "cy": cy, "cz": cz, "R": R}


def _build_stl_solid(
    stl_path: str, dom: dict, device: torch.device,
):
    """Read an STL file, voxelize it, and build a SurfaceMesh.

    Returns (solid, mesh, origin, spacing).
    """
    from tensorlbm.stl_geometry import read_stl, voxelize_stl, SurfaceMesh_from_stl
    from tensorlbm.drag_pressure import get_near_wall_3d, SurfaceMesh

    vertices, faces, face_normals = read_stl(stl_path)

    # Determine bounding box and grid spacing so the STL fits the domain.
    import numpy as np
    v_min = vertices.min(axis=0)
    v_max = vertices.max(axis=0)
    extent = v_max - v_min
    nx, ny, nz = dom["nx"], dom["ny"], dom["nz"]
    # Spacing: fit the largest STL extent into 60 % of the corresponding
    # domain dimension (leaving room for far-field).
    max_extent = float(extent.max())
    target_cells = 0.6 * min(nx, ny, nz)
    spacing_val = max_extent / target_cells if target_cells > 0 else 1.0
    spacing = (spacing_val, spacing_val, spacing_val)

    # Origin: centre the STL in the domain.
    origin = (
        dom["cx"] * spacing_val - (v_max[0] + v_min[0]) / 2.0,
        dom["cy"] * spacing_val - (v_max[1] + v_min[1]) / 2.0,
        dom["cz"] * spacing_val - (v_max[2] + v_min[2]) / 2.0,
    )

    solid = voxelize_stl(vertices, faces, (nx, ny, nz), origin=origin, spacing=spacing)
    solid = solid.to(device)
    near = get_near_wall_3d(solid)
    mesh = SurfaceMesh_from_stl(
        solid, near, vertices, faces, face_normals, origin, spacing,
    )
    return solid, mesh, origin, spacing


# ── Job runner ──────────────────────────────────────────────────────────────

def _generic_run_job(job: "job_manager.Job", req: GenericRunRequest) -> dict[str, Any]:
    """Background job function: build geometry, run LBM, compute forces.

    Uses ONLY the common-interface modules:
      - lbm_step_correct  (main loop: NoDynamics + half-way BB + far-field)
      - drag_pressure     (get_near_wall_3d, SurfaceMesh, drag_pressure_integration,
                           drag_friction_integration)
      - boundaries3d      (far_field_bc_3d)
      - postprocess        (detect_strouhal)
      - stl_geometry       (read_stl, voxelize_stl, SurfaceMesh_from_stl)
    """
    import math

    job_id = job.job_id
    geo = req.geometry
    phys = req.physics
    sol = req.solver
    out_cfg = req.output

    shape = geo.shape if geo.source == "parametric" else "stl"

    # ── Auto-select parameters ──
    auto = _auto_select_params(req, shape)
    collision = auto["collision"]
    Cs = auto["Cs"]
    n_steps = auto["steps"]
    warmup = auto["warmup"]

    # ── Auto-select domain ──
    dom = _auto_domain(shape, geo.params)

    # ── Device selection (SDAA cards 8-11) ──
    card = _next_sdaa_card()
    device = torch.device(f"sdaa:{card}")
    try:
        torch.sdaa.set_device(device)
    except Exception:
        device = torch.device("cpu")

    tag = f"[generic-run {job_id}]"

    # ── Physical parameters ──
    Re = phys.Re
    u_in = phys.u_in
    D = 2.0 * dom["R"]
    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5

    # Frontal area → dynamic-pressure × area (dpS)
    if shape == "cylinder":
        A_frontal = D * dom["nz"]
    else:
        A_frontal = math.pi * dom["R"] ** 2
    dpS = 0.5 * u_in ** 2 * A_frontal

    # ── Collision operator ──
    from tensorlbm.solver3d import collide_mrt3d, correct_mass3d
    if "smag" in collision:
        from tensorlbm.turbulence import collide_smagorinsky_mrt3d
        collide_fn = collide_smagorinsky_mrt3d
        collide_kwargs: dict[str, Any] = {"C_s": Cs}
    else:
        collide_fn = collide_mrt3d
        collide_kwargs = {}

    # ── Far-field BC ──
    from tensorlbm.boundaries3d import far_field_bc_3d
    if shape == "cylinder":
        bc_config = {"far_field_faces": ["y-", "y+"], "periodic_faces": ["z-", "z+"]}
    else:
        bc_config = {"far_field_faces": ["y-", "y+", "z-", "z+"], "periodic_faces": []}
    far_field_fn = functools.partial(far_field_bc_3d, bc_config=bc_config)

    # ── Common interface imports ──
    from tensorlbm.d3q19 import equilibrium3d
    from tensorlbm.lbm_step_correct import lbm_step_correct
    from tensorlbm.drag_pressure import (
        get_near_wall_3d,
        SurfaceMesh,
        drag_pressure_integration,
        drag_friction_integration,
    )
    from tensorlbm.postprocess import detect_strouhal

    job_manager.raise_if_cancelled(job_id)

    # ═══ Step 1: Build geometry ═══
    t0 = time.time()
    if geo.source == "stl":
        solid, mesh, origin, spacing = _build_stl_solid(geo.path, dom, device)
    else:
        solid, mesh_kwargs = _build_parametric_solid(shape, dom, device)
        near = get_near_wall_3d(solid)
        if shape == "cylinder":
            mesh = SurfaceMesh.from_cylinder(solid, near, **mesh_kwargs)
        elif shape == "ellipsoid":
            mesh = SurfaceMesh.from_ellipsoid(solid, near, **mesh_kwargs)
        elif shape == "suboff":
            S_wet = mesh_kwargs.pop("_S_wet", float(near.sum().item()))
            mesh = SurfaceMesh.from_suboff(solid, near, **mesh_kwargs)
            dpS = 0.5 * u_in ** 2 * S_wet
        else:
            mesh = SurfaceMesh.from_sphere(solid, near, **mesh_kwargs)

    n_solid = int(solid.sum().item())
    n_near = int(mesh.near.sum().item())

    job_manager.push_diagnostic(job_id, {
        "kind": "generic_run_setup",
        "step": 0,
        "shape": shape,
        "device": str(device),
        "grid": f"{dom['nx']}x{dom['ny']}x{dom['nz']}",
        "n_solid": n_solid,
        "n_near": n_near,
        "Re": Re,
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "collision": collision,
        "Cs": Cs,
        "n_steps": n_steps,
        "elapsed_s": time.time() - t0,
    })

    # ═══ Step 2: Initialise flow field ═══
    nz, ny, nx = solid.shape
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(
        rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device,
    )
    initial_mass = float(rho0.sum().item())

    # ── History accumulators ──
    cd_p_hist: list[float] = []
    cd_f_hist: list[float] = []
    cd_tot_hist: list[float] = []
    cl_hist: list[float] = []

    # ═══ Step 3: Main loop via lbm_step_correct ═══
    for step in range(1, n_steps + 1):
        job_manager.raise_if_cancelled(job_id)

        f = lbm_step_correct(
            f,
            collide_fn,
            tau,
            solid,
            u_in,
            far_field_fn,
            correct_mass_fn=correct_mass3d,
            target_mass=initial_mass,
            step=step,
            mass_interval=200,
            **collide_kwargs,
        )

        # ── Force computation (common interface) ──
        if out_cfg.forces:
            fx_p, fy_p, _ = drag_pressure_integration(f, mesh, dpS, extrap="none")
            fx_f, fy_f, _ = drag_friction_integration(
                f, mesh, dpS, nu, q_wall=None, formula="standard",
            )
            cd_p = fx_p
            cd_f = fx_f
            cd_tot = cd_p + cd_f
            cl = fy_p + fy_f
        else:
            cd_p = cd_f = cd_tot = cl = 0.0

        cd_p_hist.append(cd_p)
        cd_f_hist.append(cd_f)
        cd_tot_hist.append(cd_tot)
        cl_hist.append(cl)

        # Divergence guard
        if not torch.isfinite(f).all():
            job_manager.push_diagnostic(job_id, {
                "kind": "generic_run_diverged", "step": step,
                "Cd_total": cd_tot, "Cl": cl,
            })
            raise RuntimeError(f"Generic-run diverged at step {step}")

        # ── Push real-time diagnostics (WebSocket) ──
        if step % 10 == 0 or step == n_steps:
            n_avg = min(100, len(cd_tot_hist))
            cd_p_avg = sum(cd_p_hist[-n_avg:]) / n_avg
            cd_f_avg = sum(cd_f_hist[-n_avg:]) / n_avg
            cd_tot_avg = sum(cd_tot_hist[-n_avg:]) / n_avg
            cl_avg = sum(cl_hist[-n_avg:]) / n_avg

            # Live Strouhal estimate (from recent Cl history)
            st_live = 0.0
            if out_cfg.strouhal and len(cl_hist) >= 100:
                st_live = detect_strouhal(
                    cl_hist, sample_rate=1.0, u_ref=u_in,
                    length_ref=D, min_cycles=3,
                )

            job_manager.push_diagnostic(job_id, {
                "kind": "generic_run_step",
                "step": step,
                "total_steps": n_steps,
                "Cd_pressure": cd_p_avg,
                "Cd_friction": cd_f_avg,
                "Cd_total": cd_tot_avg,
                "Cl": cl_avg,
                "St": st_live,
                "elapsed_s": time.time() - t0,
            })

        if step % 500 == 0:
            n_avg = min(500, len(cd_tot_hist))
            cd_tot_avg = sum(cd_tot_hist[-n_avg:]) / n_avg
            elapsed = time.time() - t0
            job.logs.append(
                f"{tag} step={step}/{n_steps} Cd_tot={cd_tot_avg:.4f} ({elapsed:.0f}s)"
            )

    elapsed = time.time() - t0

    # ═══ Step 4: Final time-averaged results ═══
    avg_window = max(1, min(n_steps - warmup, len(cd_tot_hist)))
    cd_p_final = sum(cd_p_hist[-avg_window:]) / avg_window if cd_p_hist else 0.0
    cd_f_final = sum(cd_f_hist[-avg_window:]) / avg_window if cd_f_hist else 0.0
    cd_tot_final = cd_p_final + cd_f_final
    cl_final = sum(cl_hist[-avg_window:]) / avg_window if cl_hist else 0.0

    # ── Final Strouhal number ──
    st_final = 0.0
    if out_cfg.strouhal and len(cl_hist) >= 100:
        st_final = detect_strouhal(
            cl_hist, sample_rate=1.0, u_ref=u_in,
            length_ref=D, min_cycles=3,
        )

    # ── Output fields (snapshots) ──
    fields: dict[str, Any] = {}
    if out_cfg.fields:
        from tensorlbm.d3q19 import macroscopic3d
        rho, ux, uy, uz = macroscopic3d(f)
        if "velocity" in out_cfg.fields:
            fields["velocity_magnitude"] = torch.sqrt(ux ** 2 + uy ** 2 + uz ** 2).cpu().numpy()
        if "pressure" in out_cfg.fields:
            fields["pressure"] = ((rho - 1.0) / 3.0).cpu().numpy()
        if "density" in out_cfg.fields:
            fields["density"] = rho.cpu().numpy()
        if "ux" in out_cfg.fields:
            fields["ux"] = ux.cpu().numpy()

    result = {
        "job_id": job_id,
        "shape": shape,
        "device": str(device),
        "grid": f"{dom['nx']}x{dom['ny']}x{dom['nz']}",
        "Re": Re,
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "collision": collision,
        "Cs": Cs,
        "n_steps": n_steps,
        "n_solid": n_solid,
        "n_near": n_near,
        "Cd_pressure": cd_p_final,
        "Cd_friction": cd_f_final,
        "Cd_total": cd_tot_final,
        "Cl": cl_final,
        "St": st_final,
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
        "forces_history": {
            "Cd_pressure": cd_p_hist[-500:],
            "Cd_friction": cd_f_hist[-500:],
            "Cd_total": cd_tot_hist[-500:],
            "Cl": cl_hist[-500:],
        },
        "fields_data": fields,
        "fields": list(fields.keys()),
        "modules_used": [
            "drag_pressure.get_near_wall_3d",
            "drag_pressure.SurfaceMesh",
            "drag_pressure.drag_pressure_integration",
            "drag_pressure.drag_friction_integration",
            "boundaries3d.far_field_bc_3d",
            "lbm_step_correct.lbm_step_correct",
            "postprocess.detect_strouhal",
            "stl_geometry.read_stl" if geo.source == "stl" else "parametric",
        ],
    }

    # Save results to job output directory
    import json
    try:
        results_path = job.output_dir / "generic_run_results.json"
        serializable = {k: v for k, v in result.items()
                        if k != "fields_data" and k != "forces_history"}
        results_path.write_text(json.dumps(serializable, indent=2, default=str))
    except Exception:
        pass

    return result


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/generic-run")
async def generic_run(request: Request) -> dict:
    """Submit a generic LBM simulation job.

    Builds geometry (STL or parametric), auto-selects solver parameters,
    and runs the simulation using the common interface (lbm_step_correct +
    drag_pressure + drag_friction).  Returns a job_id for async monitoring.

    Accepts either:
      - JSON body (``GenericRunRequest``) — for programmatic use with
        server-side STL paths.
      - Multipart form-data (``stl_file`` + ``params`` JSON string) — for
        browser-based STL upload.

    Real-time Cd/Cl/St updates are streamed via:
      - The global WebSocket ``/ws`` (job diagnostics).
      - The dedicated WebSocket ``/api/simulations/generic-run/{job_id}/ws``.
    """
    import json as _json
    import tempfile
    import os

    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        # ── Multipart form-data: browser STL upload ──
        form = await request.form()
        stl_file = form.get("stl_file")
        params_str = form.get("params", "{}")

        try:
            params = _json.loads(params_str) if isinstance(params_str, str) else params_str
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid params JSON")

        # Build GenericRunRequest from the frontend params
        physics_raw = params.get("physics", {})
        solver_raw = params.get("solver", {})
        geometry_raw = params.get("geometry", {})
        output_raw = params.get("output", {})

        # Save uploaded STL to a temp file
        stl_path = ""
        if stl_file is not None and hasattr(stl_file, "read"):
            suffix = os.path.splitext(getattr(stl_file, "filename", "") or "")[1] or ".stl"
            fd, stl_path = tempfile.mkstemp(suffix=suffix, prefix="generic_run_")
            try:
                content = await stl_file.read()
                with os.fdopen(fd, "wb") as fh:
                    fh.write(content)
            except Exception:
                os.close(fd)
                raise

        req = GenericRunRequest(
            geometry=GenericRunGeometry(
                source="stl" if stl_path else geometry_raw.get("source", "parametric"),
                path=stl_path,
                shape=geometry_raw.get("shape", "sphere"),
                params=geometry_raw.get("params", {}),
            ),
            physics=GenericRunPhysics(
                Re=physics_raw.get("re", 100.0),
                u_in=physics_raw.get("u_in", 0.08),
                density=physics_raw.get("density", 1000.0),
                viscosity=physics_raw.get("viscosity", 1e-6),
            ),
            solver=GenericRunSolver(
                collision=solver_raw.get("collision", ""),
                Cs=solver_raw.get("cs", 0.0),
                steps=solver_raw.get("steps", 0),
                warmup=solver_raw.get("warmup", 0),
                lattice=solver_raw.get("lattice", "d3q19"),
            ),
            output=GenericRunOutput(
                fields=output_raw.get("fields", ["velocity", "pressure"]),
                forces=output_raw.get("forces", True),
                strouhal=output_raw.get("st", True),
            ),
        )
    else:
        # ── JSON body: existing programmatic API ──
        body = await request.json()
        req = GenericRunRequest(**body)

    # Validate STL path if source is 'stl'
    if req.geometry.source == "stl" and not req.geometry.path:
        raise HTTPException(
            status_code=400,
            detail="geometry.path is required when geometry.source='stl'",
        )
    if req.geometry.source == "stl" and not Path(req.geometry.path).exists():
        raise HTTPException(
            status_code=400,
            detail=f"STL file not found: {req.geometry.path}",
        )

    # Pre-compute auto-selected params for the response
    shape = req.geometry.shape if req.geometry.source == "parametric" else "stl"
    auto = _auto_select_params(req, shape)

    def _run(job: "job_manager.Job") -> dict[str, Any]:
        return _generic_run_job(job, req)

    job_id = job_manager.submit(
        name=f"GenericRun-{shape}-Re{req.physics.Re}",
        job_type="generic_run",
        config={
            **req.model_dump(),
            "shape": shape,
            "auto_selected": auto,
        },
        fn=_run,
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "shape": shape,
        "auto_selected": auto,
        "message": (
            f"Generic-run job submitted. Monitor via "
            f"GET /api/simulations/generic-run/{job_id}/status or "
            f"WebSocket /api/simulations/generic-run/{job_id}/ws"
        ),
    }


@router.get("/generic-run/{job_id}/status")
def generic_run_status(job_id: str) -> dict:
    """Get the status of a generic-run simulation job."""
    jm_job = job_manager.get_job(job_id)
    if jm_job is None:
        raise HTTPException(status_code=404, detail="Generic-run job not found")

    result: dict[str, Any] = {
        "job_id": job_id,
        "status": jm_job.status.value if hasattr(jm_job.status, "value") else str(jm_job.status),
        "shape": jm_job.config.get("shape"),
        "auto_selected": jm_job.config.get("auto_selected"),
        "logs": jm_job.logs[-20:],
        "jm_status": jm_job.to_dict(),
    }
    if jm_job.diagnostics:
        latest = jm_job.diagnostics[-1]
        if latest.get("kind", "").startswith("generic_run"):
            for key in ("step", "Cd_pressure", "Cd_friction", "Cd_total", "Cl", "St", "elapsed_s"):
                if key in latest:
                    result[key] = latest[key]

    return result


@router.get("/generic-run/{job_id}/results")
def generic_run_results(job_id: str) -> dict:
    """Get the final results of a completed generic-run simulation job."""
    jm_job = job_manager.get_job(job_id)
    if jm_job is None:
        raise HTTPException(status_code=404, detail="Generic-run job not found")
    if jm_job.status.value != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job not completed (status={jm_job.status.value})",
        )
    result = jm_job.result or {}
    # Include force history but exclude large field arrays
    serializable = {
        k: v for k, v in result.items()
        if k != "fields_data"
    }
    return serializable


@router.get("/generic-run/{job_id}/fields/{field_name}")
def generic_run_field(job_id: str, field_name: str, slice_axis: int = 0):
    """Get a 2-D slice of a recorded output field from a completed job."""
    jm_job = job_manager.get_job(job_id)
    if jm_job is None:
        raise HTTPException(status_code=404, detail="Generic-run job not found")
    if jm_job.status.value != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job not completed (status={jm_job.status.value})",
        )
    fields = (jm_job.result or {}).get("fields_data", {})
    if field_name not in fields:
        available = list(fields.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Field '{field_name}' not found. Available: {available}",
        )
    import numpy as np
    arr = np.asarray(fields[field_name])
    if arr.ndim == 3:
        mid = arr.shape[slice_axis] // 2
        if slice_axis == 0:
            arr = arr[mid]
        elif slice_axis == 1:
            arr = arr[:, mid]
        else:
            arr = arr[:, :, mid]
    data = arr.astype(float).tolist()
    return JSONResponse(content={
        "job_id": job_id,
        "field": field_name,
        "shape": list(arr.shape),
        "data": data,
    })


@router.websocket("/generic-run/{job_id}/ws")
async def generic_run_ws(ws: WebSocket, job_id: str):
    """WebSocket for real-time Cd/Cl/St updates during a generic-run job.

    Protocol:
      Server → Client: {"type":"status", ...}  — initial job state
      Server → Client: {"type":"update", "step":N, "Cd_total":..., "Cl":..., "St":...}
      Client → Server: {"action":"close"}  — graceful disconnect
    """
    await ws.accept()

    jm_job = job_manager.get_job(job_id)

    if jm_job is None:
        await ws.send_json({"type": "error", "message": "Job not found"})
        await ws.close()
        return

    # Send initial status
    initial: dict[str, Any] = {"type": "status", "job_id": job_id}
    if jm_job:
        initial["shape"] = jm_job.config.get("shape")
        initial["auto_selected"] = jm_job.config.get("auto_selected")
        initial["jm_status"] = jm_job.status.value if hasattr(jm_job.status, "value") else str(jm_job.status)
    await ws.send_json(initial)

    # Track last diagnostic count to detect new updates
    last_diag_count = len(jm_job.diagnostics) if jm_job else 0

    import asyncio

    try:
        while True:
            # Check for new diagnostics
            jm_job = job_manager.get_job(job_id)
            if jm_job is None:
                await ws.send_json({"type": "closed", "message": "Job no longer exists"})
                break

            current_count = len(jm_job.diagnostics)
            if current_count > last_diag_count:
                # Send new diagnostics
                for diag in jm_job.diagnostics[last_diag_count:]:
                    if diag.get("kind", "").startswith("generic_run"):
                        await ws.send_json({"type": "update", **diag})
                last_diag_count = current_count

            # Check if job is finished
            jm_status = jm_job.status.value if hasattr(jm_job.status, "value") else str(jm_job.status)
            if jm_status in ("completed", "failed", "cancelled"):
                final: dict[str, Any] = {"type": "final", "job_id": job_id, "status": jm_status}
                result = jm_job.result or {}
                for key in ("Cd_total", "Cd_pressure", "Cd_friction", "Cl", "St"):
                    if key in result:
                        final[key] = result[key]
                if jm_job.error:
                    final["error"] = jm_job.error
                await ws.send_json(final)
                break

            # Wait for client ping or timeout
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=1.0)
            except (TimeoutError, asyncio.TimeoutError):
                continue  # No client message; loop and check for new diagnostics
    except WebSocketDisconnect:
        pass
    except Exception:
        import traceback
        traceback.print_exc()
        import contextlib
        with contextlib.suppress(Exception):
            await ws.close()
