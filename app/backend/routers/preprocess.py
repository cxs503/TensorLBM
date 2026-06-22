"""Pre-processing API endpoints.

Provides geometry generation and unit conversion utilities from the
tensorlbm library exposed as REST endpoints.
"""
from __future__ import annotations

import base64
import io
import os
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

router = APIRouter()
_MAX_UPLOAD_MB = max(1, int(os.environ.get("TENSORLBM_MAX_UPLOAD_MB", "50")))
_MAX_UPLOAD_BYTES = _MAX_UPLOAD_MB * 1024 * 1024


# ---------------------------------------------------------------------------
# Polygon → 2-D mask
# ---------------------------------------------------------------------------

class PolygonMaskRequest(BaseModel):
    nx: int = 200
    ny: int = 100
    vertices: list[list[float]]  # [[x0,y0],[x1,y1],...]


@router.post("/polygon-mask")
async def polygon_mask(req: PolygonMaskRequest) -> dict:
    """Convert a polygon (list of [x,y] vertices in *pixel* coordinates)
    to a 2-D boolean obstacle mask.  Returns a base64-encoded PNG preview."""
    try:
        import torch

        from tensorlbm import poly_to_mask_2d

        verts = [tuple(v) for v in req.vertices]
        mask_t = poly_to_mask_2d(verts, req.ny, req.nx, torch.device("cpu"))
        mask = mask_t.cpu().numpy()
        img_b64 = _mask_to_b64(mask)
        ones = int(mask.sum())
        return {
            "nx": req.nx,
            "ny": req.ny,
            "obstacle_cells": ones,
            "fluid_cells": req.nx * req.ny - ones,
            "image": img_b64,
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Random porosity mask (2-D)
# ---------------------------------------------------------------------------

class RandomPorosityRequest(BaseModel):
    nx: int = 128
    ny: int = 128
    porosity: float = 0.4
    sigma: float = 0.0  # smoothing length (0 → uncorrelated)
    seed: int = 0


@router.post("/random-porosity-2d")
async def random_porosity_2d(req: RandomPorosityRequest) -> dict:
    try:
        import torch

        from tensorlbm import random_porosity_mask_2d

        mask_t = random_porosity_mask_2d(
            req.ny, req.nx,
            porosity=req.porosity,
            device=torch.device("cpu"),
            seed=req.seed,
            sigma=req.sigma,
        )
        mask = mask_t.cpu().numpy()
        actual_porosity = float(1.0 - mask.mean())
        img_b64 = _mask_to_b64(mask)
        return {
            "nx": req.nx,
            "ny": req.ny,
            "requested_porosity": req.porosity,
            "actual_porosity": round(actual_porosity, 4),
            "image": img_b64,
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# STL voxelisation (3-D)
# ---------------------------------------------------------------------------

@router.post("/voxelize-stl")
async def voxelize_stl(
    file: Annotated[UploadFile, File()],
    nx: int = 64,
    ny: int = 64,
    nz: int = 64,
) -> dict:
    """Upload an STL file and return voxel statistics."""
    try:
        import tempfile
        from pathlib import Path

        import torch

        from tensorlbm import voxelize_stl_3d

        if not (file.filename or "").lower().endswith(".stl"):
            raise HTTPException(status_code=422, detail="Only .stl uploads are supported")

        content = await _read_upload_limited(file, _MAX_UPLOAD_BYTES)
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        mask = voxelize_stl_3d(str(tmp_path), nx, ny, nz, torch.device("cpu"))
        tmp_path.unlink(missing_ok=True)

        solid = int(mask.sum().item())
        total = nx * ny * nz
        return {
            "nx": nx, "ny": ny, "nz": nz,
            "solid_cells": solid,
            "fluid_cells": total - solid,
            "solid_fraction": round(solid / total, 4),
        }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Fluid material database
# ---------------------------------------------------------------------------

# Common engineering fluids at standard / reference conditions.
# Properties at ~20 °C (293.15 K) unless otherwise noted.
_FLUID_DB: list[dict] = [
    {
        "id": "water_20c",
        "name": "Water (20 °C)",
        "name_zh": "水（20 °C）",
        "category": "liquid",
        "density_kg_m3": 998.2,
        "dynamic_viscosity_pa_s": 1.002e-3,
        "kinematic_viscosity_m2_s": 1.004e-6,
        "surface_tension_n_m": 0.0728,
        "bulk_modulus_pa": 2.18e9,
        "ref_temp_c": 20.0,
        "notes": "Fresh water, standard reference (ISO 5167)",
    },
    {
        "id": "seawater_20c",
        "name": "Seawater (20 °C, 35 ppt)",
        "name_zh": "海水（20 °C，盐度 35 ppt）",
        "category": "liquid",
        "density_kg_m3": 1025.0,
        "dynamic_viscosity_pa_s": 1.08e-3,
        "kinematic_viscosity_m2_s": 1.054e-6,
        "surface_tension_n_m": 0.0725,
        "bulk_modulus_pa": 2.34e9,
        "ref_temp_c": 20.0,
        "notes": "ITTC standard seawater (2011)",
    },
    {
        "id": "air_20c",
        "name": "Air (20 °C, 1 atm)",
        "name_zh": "空气（20 °C，1 atm）",
        "category": "gas",
        "density_kg_m3": 1.204,
        "dynamic_viscosity_pa_s": 1.825e-5,
        "kinematic_viscosity_m2_s": 1.516e-5,
        "surface_tension_n_m": None,
        "bulk_modulus_pa": 1.42e5,
        "ref_temp_c": 20.0,
        "notes": "Dry air at sea level (NIST)",
    },
    {
        "id": "oil_hydraulic",
        "name": "Hydraulic Oil (ISO VG 46)",
        "name_zh": "液压油（ISO VG 46）",
        "category": "liquid",
        "density_kg_m3": 875.0,
        "dynamic_viscosity_pa_s": 0.046,
        "kinematic_viscosity_m2_s": 5.26e-5,
        "surface_tension_n_m": 0.032,
        "bulk_modulus_pa": 1.6e9,
        "ref_temp_c": 40.0,
        "notes": "Typical ISO VG 46 mineral oil at 40 °C",
    },
    {
        "id": "glycerin_25c",
        "name": "Glycerin (25 °C)",
        "name_zh": "甘油（25 °C）",
        "category": "liquid",
        "density_kg_m3": 1261.0,
        "dynamic_viscosity_pa_s": 0.934,
        "kinematic_viscosity_m2_s": 7.41e-4,
        "surface_tension_n_m": 0.0634,
        "bulk_modulus_pa": 4.35e9,
        "ref_temp_c": 25.0,
        "notes": "Pure glycerin; often used in multiphase benchmark studies",
    },
    {
        "id": "mercury_25c",
        "name": "Mercury (25 °C)",
        "name_zh": "汞（25 °C）",
        "category": "liquid",
        "density_kg_m3": 13534.0,
        "dynamic_viscosity_pa_s": 1.526e-3,
        "kinematic_viscosity_m2_s": 1.13e-7,
        "surface_tension_n_m": 0.485,
        "bulk_modulus_pa": 2.85e10,
        "ref_temp_c": 25.0,
        "notes": "Liquid mercury; high surface tension / density ratio",
    },
]


@router.get("/materials")
async def list_materials(category: str | None = None) -> dict:
    """Return the built-in fluid material database.

    Optional ``category`` filter: ``liquid`` or ``gas``.
    """
    fluids = _FLUID_DB
    if category is not None:
        fluids = [f for f in fluids if f["category"] == category]
    return {"count": len(fluids), "materials": fluids}


@router.get("/materials/{material_id}")
async def get_material(material_id: str) -> dict:
    """Return properties of a single material by ID."""
    for f in _FLUID_DB:
        if f["id"] == material_id:
            return f
    raise HTTPException(status_code=404, detail=f"Material '{material_id}' not found")



# ---------------------------------------------------------------------------
# Y+ wall distance calculator
# ---------------------------------------------------------------------------

# Geometry-specific skin-friction correlations (flat-plate turbulent BL)
_CF_MODELS = {
    "flat_plate": "Schlichting turbulent flat plate: Cf = 0.0576·Re^(-0.2)",
    "cylinder":   "Achenbach (1968) approximation: Cf = 0.079·Re^(-0.25) (pipe Blasius form)",
    "channel":    "Blasius pipe/channel: Cf = 0.079·Re^(-0.25)",
}


class YPlusRequest(BaseModel):
    re: float = 1e5                    # Reynolds number (= U·L / ν)
    u_ms: float = 1.0                  # freestream / bulk velocity [m/s]
    l_m: float = 1.0                   # characteristic length [m]
    nu_m2s: float = 1e-5              # kinematic viscosity [m²/s]
    target_yplus: float = 1.0         # desired y+ at first cell centre
    n_cells: int = 100                 # number of cells along L (for dx)
    geometry: str = "flat_plate"      # flat_plate | cylinder | channel


@router.post("/yplus")
async def yplus_calculator(req: YPlusRequest) -> dict:
    """Estimate the required first-cell height Δy to hit a target y⁺.

    Uses skin-friction correlations to compute the wall friction velocity
    u_τ, then solves Δy = y⁺ · ν / u_τ.  Also reports the equivalent
    LBM lattice-unit cell height given the user's resolution (n_cells).

    Geometry options and their Cf correlations
    ------------------------------------------
    * ``flat_plate`` – Schlichting turbulent flat plate: ``Cf = 0.0576·Re^(-0.2)``
    * ``cylinder``   – Blasius / Achenbach form:         ``Cf = 0.079·Re^(-0.25)``
    * ``channel``    – Same Blasius form as cylinder
    """
    import math

    re = req.re
    geom = req.geometry.lower()
    if geom not in _CF_MODELS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown geometry '{req.geometry}'. Choose: {list(_CF_MODELS)}",
        )

    # ---- skin-friction coefficient ----------------------------------------
    if geom == "flat_plate":
        c_f = 0.0576 * re ** (-0.2)
    else:  # cylinder / channel → Blasius
        c_f = 0.079 * re ** (-0.25)

    # ---- friction velocity u_τ = U · sqrt(Cf/2) ---------------------------
    u_tau = req.u_ms * math.sqrt(c_f / 2.0)
    if u_tau <= 0:
        raise HTTPException(status_code=422, detail="Friction velocity is zero or negative.")

    # ---- first-cell height in physical units --------------------------------
    delta_y_m = req.target_yplus * req.nu_m2s / u_tau

    # ---- convert to LBM lattice units (dx = L/N_cells) --------------------
    dx_m = req.l_m / max(req.n_cells, 1)
    delta_y_lbm = delta_y_m / dx_m

    # ---- recommended minimum cells across BL --------------------------------
    # Viscous sub-layer ends at y+ ≈ 5, recommend ≥5 cells inside y+<5
    bl_thickness_m = 0.37 * req.l_m * re ** (-0.2)  # Prandtl 1/5 law
    n_cells_bl = max(1, int(bl_thickness_m / delta_y_m))

    return {
        "reynolds_number": round(re, 4),
        "geometry": geom,
        "c_f": round(c_f, 6),
        "u_tau_ms": round(u_tau, 6),
        "target_yplus": req.target_yplus,
        "delta_y_m": delta_y_m,
        "delta_y_lbm": round(delta_y_lbm, 4),
        "dx_m": round(dx_m, 8),
        "n_cells_along_L": req.n_cells,
        "bl_thickness_m": round(bl_thickness_m, 6),
        "cells_inside_bl": n_cells_bl,
        "cf_model": _CF_MODELS[geom],
        "note": (
            "delta_y_lbm < 0.5 means sub-cell resolution – increase n_cells or "
            "accept y+ > 1.  Values delta_y_lbm ≈ 1–2 are typical for wall-resolved LES."
        ),
    }


class UnitConvertRequest(BaseModel):
    # Physical quantities
    phys_length_m: float = 1.0          # characteristic length [m]
    phys_velocity_ms: float = 1.0       # characteristic velocity [m/s]
    phys_nu_m2s: float = 1e-6           # kinematic viscosity [m²/s]
    # LBM target
    lbm_length: float = 100.0           # characteristic length in lattice units
    lbm_velocity: float = 0.1           # characteristic velocity in lattice units


@router.post("/units")
async def convert_units(req: UnitConvertRequest) -> dict:
    try:
        from tensorlbm import LBMUnitConverter

        re = req.phys_velocity_ms * req.phys_length_m / req.phys_nu_m2s
        conv = LBMUnitConverter(
            re=re,
            l_phys=req.phys_length_m,
            u_phys=req.phys_velocity_ms,
            nu_phys=req.phys_nu_m2s,
            nx=int(req.lbm_length),
            u_lb=req.lbm_velocity,
        )
        return {
            "reynolds_number": round(re, 4),
            "lbm_nu": round(conv.nu_lb, 6),
            "lbm_tau": round(conv.tau, 6),
            "dx_m": round(conv.dx, 6),
            "dt_s": round(conv.dt, 10),
            "mach_number": round(conv.ma, 4),
            "stable": bool(conv.tau > 0.5),
            "note": "tau > 0.5 is required for BGK stability",
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Pre-flight engineering validation wizard
# ---------------------------------------------------------------------------

_PREFLIGHT_TAU_MIN = 0.51
_PREFLIGHT_MA_MAX = 0.30
_BYTES_PER_FLOAT = 4

# Velocities per cell for each lattice type (upper bound)
_VELOCITIES_PER_CELL = {
    "d2q9": 9,
    "d2q5": 5,
    "d3q19": 19,
    "d3q27": 27,
}

# Minimum n_steps heuristic: Re × domain_length / u_in  (∝ flow-through times)
_MIN_FLOW_THROUGH_TIMES = 5


class PreflightRequest(BaseModel):
    """Input for the pre-flight validation wizard.

    Most fields are optional; only the checks for which all required
    inputs are present will be executed.
    """
    solver_type: str = "cylinder_flow"
    # Grid dimensions (lu = lattice units)
    nx: int | None = None
    ny: int | None = None
    nz: int | None = None
    # Physics (lattice units)
    re: float | None = None
    u_in: float | None = None
    u_lid: float | None = None
    radius: float | None = None       # characteristic length for Re (2D/3D sphere)
    n_steps: int | None = None
    output_interval: int | None = None
    # Physical-units input for y+ estimation (optional)
    phys_length_m: float | None = None   # characteristic length [m]
    phys_velocity_ms: float | None = None
    phys_nu_m2s: float | None = None
    target_yplus: float | None = None    # desired y+; default 1.0 for wall-resolved

    model_config = {"extra": "allow"}


class PreflightCheck(BaseModel):
    name: str
    status: str  # "ok" | "warning" | "error"
    message: str


class PreflightResponse(BaseModel):
    solver_type: str
    checks: list[PreflightCheck]
    recommendations: list[str]
    memory_mb: float | None
    suggested_n_steps: int | None
    suggested_output_interval: int | None
    yplus_first_cell_m: float | None


@router.post("/preflight", response_model=PreflightResponse)
async def preflight(req: PreflightRequest) -> PreflightResponse:
    """Run the pre-flight engineering validation wizard.

    Performs a sequence of checks on the submitted solver parameters and
    returns a structured checklist, recommendations, and simple resource
    estimates — without actually submitting a job.

    Checks performed (when sufficient inputs are provided):
    - **tau**: BGK relaxation time stability (τ > 0.51).
    - **Mach**: lattice Mach number Ma < 0.30.
    - **Grid size**: maximum dimension vs. platform limits.
    - **n_steps / output_interval**: consistency and step-count adequacy.
    - **Memory estimate**: grid × velocities × float32.
    - **y+ first cell**: estimates near-wall cell height for wall resolution.
    """
    checks: list[PreflightCheck] = []
    recommendations: list[str] = []

    is_3d = req.nz is not None
    _MAX_GRID_2D = 1024
    _MAX_GRID_3D = 256

    # ---- Grid checks -------------------------------------------------------
    if req.nx is not None and req.ny is not None:
        max_g = _MAX_GRID_3D if is_3d else _MAX_GRID_2D
        dims = (req.nx, req.ny, req.nz) if is_3d else (req.nx, req.ny)
        over = any(d > max_g for d in dims if d is not None)  # type: ignore[operator]
        if over:
            checks.append(PreflightCheck(
                name="grid_size",
                status="error",
                message=f"Grid dimension exceeds platform limit of {max_g} for "
                        f"{'3D' if is_3d else '2D'} simulations.",
            ))
        elif req.nx < 10 or req.ny < 10 or (is_3d and (req.nz or 0) < 10):
            checks.append(PreflightCheck(
                name="grid_size",
                status="warning",
                message="Very small grid (<10 cells on a side); accuracy may be poor.",
            ))
        else:
            checks.append(PreflightCheck(
                name="grid_size",
                status="ok",
                message=f"Grid {'×'.join(str(d) for d in dims if d is not None)} is within limits.",
            ))

    # ---- Stability (tau, Ma) -----------------------------------------------
    u = req.u_in or req.u_lid
    char_len = req.radius * 2.0 if req.radius else (req.nx or None)
    tau: float | None = None
    ma: float | None = None
    if u is not None and req.re is not None and char_len is not None:
        nu_lb = u * char_len / req.re
        tau = 3.0 * nu_lb + 0.5
        cs = 1.0 / (3.0 ** 0.5)
        ma = u / cs

        if tau < _PREFLIGHT_TAU_MIN:
            checks.append(PreflightCheck(
                name="tau_stability",
                status="error",
                message=(
                    f"τ = {tau:.4f} < {_PREFLIGHT_TAU_MIN}: BGK will diverge. "
                    "Reduce u_in / Re, increase characteristic length, or increase nx."
                ),
            ))
        elif tau < 0.60:
            checks.append(PreflightCheck(
                name="tau_stability",
                status="warning",
                message=f"τ = {tau:.4f} is close to the stability limit. Consider TRT/MRT.",
            ))
        else:
            checks.append(PreflightCheck(
                name="tau_stability",
                status="ok",
                message=f"τ = {tau:.4f} — stable.",
            ))

        if ma > _PREFLIGHT_MA_MAX:
            checks.append(PreflightCheck(
                name="mach_number",
                status="warning",
                message=(
                    f"Ma = {ma:.4f} > {_PREFLIGHT_MA_MAX}: noticeable compressibility error. "
                    "Lower u_in for better incompressible accuracy."
                ),
            ))
        else:
            checks.append(PreflightCheck(
                name="mach_number",
                status="ok",
                message=f"Ma = {ma:.4f} — within incompressible limit.",
            ))

    # ---- n_steps adequacy -------------------------------------------------
    suggested_n_steps: int | None = None
    if req.n_steps is not None:
        if req.n_steps > 200_000:
            checks.append(PreflightCheck(
                name="n_steps",
                status="error",
                message=f"n_steps={req.n_steps} exceeds platform limit of 200 000.",
            ))
        elif req.n_steps < 100:
            checks.append(PreflightCheck(
                name="n_steps",
                status="warning",
                message="n_steps < 100 — result unlikely to be physically meaningful.",
            ))
        else:
            checks.append(PreflightCheck(
                name="n_steps",
                status="ok",
                message=f"n_steps = {req.n_steps}.",
            ))
    elif u is not None and req.re is not None and char_len is not None:
        flow_through = max(20, int(char_len / u) if u > 0 else 1000)
        suggested_n_steps = min(200_000, _MIN_FLOW_THROUGH_TIMES * flow_through)
        recommendations.append(
            f"Suggested n_steps ≈ {suggested_n_steps} "
            f"({_MIN_FLOW_THROUGH_TIMES} flow-through times at Re={req.re:.0f})."
        )

    # ---- output_interval consistency --------------------------------------
    suggested_output_interval: int | None = None
    if req.output_interval is not None and req.n_steps is not None:
        if req.output_interval > req.n_steps:
            checks.append(PreflightCheck(
                name="output_interval",
                status="warning",
                message="output_interval > n_steps — no checkpoint will be written.",
            ))
        else:
            n_snaps = req.n_steps // req.output_interval
            checks.append(PreflightCheck(
                name="output_interval",
                status="ok",
                message=f"output_interval = {req.output_interval} → {n_snaps} snapshots.",
            ))
    elif suggested_n_steps:
        suggested_output_interval = max(1, suggested_n_steps // 20)
        recommendations.append(
            f"Suggested output_interval ≈ {suggested_output_interval} (≈20 snapshots total)."
        )

    # ---- Memory estimate --------------------------------------------------
    memory_mb: float | None = None
    if req.nx is not None and req.ny is not None:
        lattice_key = "d3q27" if "d3q27" in req.solver_type else ("d3q19" if is_3d else "d2q9")
        vpn = _VELOCITIES_PER_CELL.get(lattice_key, 9)
        nz_ = req.nz if is_3d else 1
        total_cells = req.nx * req.ny * nz_
        # f arrays (2× double buffer) + moments + mask
        n_arrays = vpn * 2 + 4
        memory_mb = total_cells * n_arrays * _BYTES_PER_FLOAT / (1024 ** 2)
        if memory_mb > 8192:
            checks.append(PreflightCheck(
                name="memory",
                status="warning",
                message=f"Estimated memory ≈ {memory_mb:.1f} MB — may exceed typical device RAM.",
            ))
        else:
            checks.append(PreflightCheck(
                name="memory",
                status="ok",
                message=f"Estimated memory ≈ {memory_mb:.1f} MB.",
            ))

    # ---- y+ first-cell estimate -------------------------------------------
    yplus_first_cell_m: float | None = None
    if (
        req.phys_length_m is not None
        and req.phys_velocity_ms is not None
        and req.phys_nu_m2s is not None
    ):
        import math
        re_phys = req.phys_velocity_ms * req.phys_length_m / req.phys_nu_m2s
        # Schlichting flat-plate friction-coefficient approximation
        cf = 0.026 / (re_phys ** (1.0 / 7.0))
        tau_w = 0.5 * cf * req.phys_velocity_ms ** 2  # /ρ (ρ=1)
        u_tau = math.sqrt(tau_w)
        target_yp = req.target_yplus if req.target_yplus is not None else 1.0
        y1 = target_yp * req.phys_nu_m2s / u_tau
        yplus_first_cell_m = y1
        if req.nx is not None and req.phys_length_m > 0:
            dx_phys = req.phys_length_m / req.nx
            actual_yp = u_tau * dx_phys / req.phys_nu_m2s
            if actual_yp > 5.0:
                checks.append(PreflightCheck(
                    name="yplus",
                    status="warning",
                    message=(
                        f"Wall y+ ≈ {actual_yp:.1f} with current grid. "
                        f"For y+ ≤ {target_yp:.1f} use first cell height {y1*1000:.4f} mm."
                    ),
                ))
            else:
                checks.append(PreflightCheck(
                    name="yplus",
                    status="ok",
                    message=f"Wall y+ ≈ {actual_yp:.1f}.",
                ))
        else:
            recommendations.append(
                f"Required first-cell height for y+={target_yp:.1f}: {y1*1000:.4f} mm "
                f"(Re={re_phys:.2e})."
            )

    # ---- General recommendations ------------------------------------------
    if tau is not None and tau < 0.65 and not any(c.status == "error" for c in checks):
        recommendations.append(
            "Consider switching to TRT collision (magic parameter Λ=3/16) "
            "to improve stability near the limit."
        )
    if ma is not None and ma > 0.15:
        recommendations.append(
            "Lattice Mach number is moderate. Halving u_in and doubling nx "
            "improves incompressible accuracy with similar Re."
        )

    return PreflightResponse(
        solver_type=req.solver_type,
        checks=checks,
        recommendations=recommendations,
        memory_mb=round(memory_mb, 2) if memory_mb is not None else None,
        suggested_n_steps=suggested_n_steps,
        suggested_output_interval=suggested_output_interval,
        yplus_first_cell_m=round(yplus_first_cell_m, 8) if yplus_first_cell_m is not None else None,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mask_to_b64(mask: object) -> str:
    """Render a boolean mask as a base64-encoded PNG data URL.

    Accepts either a numpy array or a torch tensor.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Support torch tensors transparently
    if hasattr(mask, "cpu") and hasattr(mask, "numpy"):
        mask = mask.cpu().numpy()  # type: ignore[union-attr]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(np.asarray(mask).astype(np.uint8) * 255, cmap="gray_r",
              origin="lower", vmin=0, vmax=255)
    ax.set_title("Obstacle mask (black = solid)")
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


async def _read_upload_limited(file: UploadFile, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise ValueError(
                f"Upload exceeds limit ({_MAX_UPLOAD_MB} MB). "
                "Set TENSORLBM_MAX_UPLOAD_MB to adjust."
            )
        chunks.append(chunk)
    return b"".join(chunks)
