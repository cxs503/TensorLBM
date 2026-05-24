"""Pre-processing API endpoints.

Provides geometry generation and unit conversion utilities from the
tensorlbm library exposed as REST endpoints.
"""
from __future__ import annotations

import base64
import io
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

router = APIRouter()


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
        import numpy as np  # noqa: I001
        from tensorlbm import poly_to_mask_2d

        verts = np.array(req.vertices, dtype=np.float32)
        mask = poly_to_mask_2d(req.ny, req.nx, verts)  # bool ndarray (ny, nx)
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
    corr_length: float = 5.0
    seed: int = 0


@router.post("/random-porosity-2d")
async def random_porosity_2d(req: RandomPorosityRequest) -> dict:
    try:
        from tensorlbm import random_porosity_mask_2d

        mask = random_porosity_mask_2d(
            req.ny, req.nx,
            porosity=req.porosity,
            corr_length=req.corr_length,
            seed=req.seed,
        )
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

        from tensorlbm import voxelize_stl_3d

        content = await file.read()
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        mask = voxelize_stl_3d(str(tmp_path), nz, ny, nx)
        tmp_path.unlink(missing_ok=True)

        solid = int(mask.sum())
        total = nx * ny * nz
        return {
            "nx": nx, "ny": ny, "nz": nz,
            "solid_cells": solid,
            "fluid_cells": total - solid,
            "solid_fraction": round(solid / total, 4),
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

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

        conv = LBMUnitConverter(
            phys_length=req.phys_length_m,
            phys_velocity=req.phys_velocity_ms,
            phys_nu=req.phys_nu_m2s,
            lbm_length=req.lbm_length,
            lbm_velocity=req.lbm_velocity,
        )
        re = req.phys_velocity_ms * req.phys_length_m / req.phys_nu_m2s
        lbm_nu = conv.lbm_nu
        tau = 3.0 * lbm_nu + 0.5
        return {
            "reynolds_number": round(re, 4),
            "lbm_nu": round(lbm_nu, 6),
            "lbm_tau": round(tau, 6),
            "dx_m": round(conv.dx, 6),
            "dt_s": round(conv.dt, 10),
            "mach_number": round(req.lbm_velocity / (1.0 / 3.0 ** 0.5), 4),
            "stable": tau > 0.5,
            "note": "tau > 0.5 is required for BGK stability",
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mask_to_b64(mask: object) -> str:
    """Render a boolean mask as a base64-encoded PNG data URL."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(mask.astype(np.uint8) * 255, cmap="gray_r", origin="lower", vmin=0, vmax=255)
    ax.set_title("Obstacle mask (black = solid)")
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()
