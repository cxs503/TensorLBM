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
