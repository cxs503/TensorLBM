"""CAD modelling API endpoints for ship hull geometry.

Exposes the :mod:`tensorlbm.ship_cad` module as REST endpoints so that the
browser-based platform can:

1. Generate parametric hull previews (body-plan / waterplane / side-profile).
2. Retrieve hull form statistics (Cb, Cwp, Cm, Cp, …).
3. Compute LBM parameters from physical ship dimensions.
4. Launch a solver job directly from CAD parameters (CAD → solver shortcut).
5. Export a hull STL file.
"""
from __future__ import annotations

import base64
import io
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .. import job_manager

router = APIRouter()

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class HullPreviewRequest(BaseModel):
    hull_type: Literal["wigley", "series60", "kcs"] = "series60"
    length: float = Field(100.0, gt=0, description="Hull length (lattice units)")
    beam: float = Field(16.0, gt=0, description="Hull beam (lattice units)")
    draft: float = Field(8.0, gt=0, description="Hull draft (lattice units)")
    n_stations: int = Field(11, ge=3, le=41, description="Number of body-plan stations")


class HullMaskRequest(BaseModel):
    hull_type: Literal["wigley", "series60", "kcs"] = "series60"
    nx: int = Field(160, ge=20, description="Grid x-size")
    ny: int = Field(60, ge=10, description="Grid y-size")
    nz: int = Field(40, ge=10, description="Grid z-size")
    length: float = Field(80.0, gt=0)
    beam: float = Field(8.0, gt=0)
    draft: float = Field(12.0, gt=0)
    cx: float | None = Field(None, description="Midship x (default: nx/2)")
    cy: float | None = Field(None, description="Centreline y (default: ny/2)")
    cz_keel: float | None = Field(None, description="Keel z (default: nz/4)")
    device: str = "cpu"


class LBMParametersRequest(BaseModel):
    length_m: float = Field(100.0, gt=0, description="Ship length [m]")
    speed_ms: float = Field(5.0, gt=0, description="Ship speed [m/s]")
    nu_m2s: float = Field(1.139e-6, gt=0, description="Kinematic viscosity [m²/s]")
    lbm_length: float = Field(100.0, gt=0, description="LBM hull length (cells)")
    lbm_speed: float = Field(0.05, gt=0, description="LBM inlet velocity (lu/step)")
    froude_target: float | None = Field(
        None, ge=0, description="Target Froude number (overrides speed_ms)"
    )


class HullSolverRequest(BaseModel):
    """Launch a ship-hull LBM solver job from CAD parameters."""

    hull_type: Literal["wigley", "series60", "kcs"] = "wigley"
    nx: int = Field(160, ge=20)
    ny: int = Field(60, ge=10)
    nz: int = Field(40, ge=10)
    hull_length: float = Field(80.0, gt=0)
    hull_beam: float = Field(8.0, gt=0)
    hull_draft: float = Field(12.0, gt=0)
    u_in: float = Field(0.05, gt=0)
    re: float = Field(200.0, gt=0)
    smagorinsky_cs: float = Field(0.1, ge=0)
    wave_amp: float = Field(0.0, ge=0)
    wave_period: float = Field(200.0, gt=0)
    n_steps: int = Field(2000, ge=1)
    output_interval: int = Field(200, ge=1)
    device: str = "cpu"
    seed: int = 0


class HullSTLRequest(BaseModel):
    hull_type: Literal["wigley", "series60", "kcs"] = "series60"
    length: float = Field(100.0, gt=0)
    beam: float = Field(16.0, gt=0)
    draft: float = Field(8.0, gt=0)
    n_long: int = Field(60, ge=4, le=200)
    n_vert: int = Field(30, ge=4, le=100)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/preview")
async def hull_preview(req: HullPreviewRequest) -> dict:
    """Generate a multi-view hull preview (body-plan / waterplane / side profile).

    Returns a base64-encoded PNG of the three-panel figure plus hull form
    statistics (Cb, Cwp, Cm, Cp, L/B, B/T).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        from tensorlbm.ship_cad import (
            ShipHullType,
            generate_hull_previews,
            hull_statistics,
        )

        ht = ShipHullType(req.hull_type)
        fig = generate_hull_previews(
            ht,
            length=req.length,
            beam=req.beam,
            draft=req.draft,
            n_stations=req.n_stations,
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        buf.seek(0)
        img_b64 = "data:image/png;base64," + base64.b64encode(buf.read()).decode()

        stats = hull_statistics(ht, req.length, req.beam, req.draft)
        return {"image": img_b64, "stats": stats}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/hull-mask")
async def hull_mask(req: HullMaskRequest) -> dict:
    """Build a 3-D hull voxel mask and return solid/fluid statistics.

    Also returns a 2-D top-view preview PNG (waterplane projection).
    """
    try:
        import matplotlib  # noqa: I001
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: I001
        import numpy as np  # noqa: I001
        from tensorlbm.ship_cad import ShipHullType, build_hull_mask  # noqa: I001

        mask_tensor, stats = build_hull_mask(
            hull_type=req.hull_type,
            nx=req.nx,
            ny=req.ny,
            nz=req.nz,
            cx=req.cx,
            cy=req.cy,
            cz_keel=req.cz_keel,
            length=req.length,
            beam=req.beam,
            draft=req.draft,
            device=req.device,
        )

        # Generate a top-view (z-projection) preview image
        top_view = mask_tensor.any(dim=0).cpu().numpy().astype(np.uint8) * 255

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.imshow(top_view, cmap="Blues", origin="lower", vmin=0, vmax=255)
        ax.set_title(
            f"{ShipHullType(req.hull_type).value.upper()} hull – top view "
            f"(Cb={stats['Cb_numerical']:.3f})"
        )
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        img_b64 = "data:image/png;base64," + base64.b64encode(buf.read()).decode()

        return {"image": img_b64, "stats": stats}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/lbm-parameters")
async def lbm_parameters(req: LBMParametersRequest) -> dict:
    """Compute LBM dimensionless parameters from physical ship dimensions."""
    try:
        from tensorlbm.ship_cad import ship_lbm_parameters

        result = ship_lbm_parameters(
            length_m=req.length_m,
            speed_ms=req.speed_ms,
            nu_m2s=req.nu_m2s,
            lbm_length=req.lbm_length,
            lbm_speed=req.lbm_speed,
            froude_target=req.froude_target,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/send-to-solver")
async def send_to_solver(req: HullSolverRequest) -> dict:
    """Submit a ship-hull LBM solver job directly from CAD parameters.

    For Wigley hull, the existing :func:`run_ship_hull_flow` runner is used
    unchanged.  For Series 60 and KCS, a custom job is submitted that builds
    the appropriate hull mask and runs the 3-D LBM solver.
    """
    try:
        if req.hull_type == "wigley":
            # Use the existing Wigley runner
            def _run_wigley(job: job_manager.Job) -> dict:
                from tensorlbm import ShipHullFlowConfig, run_ship_hull_flow

                cfg = ShipHullFlowConfig(
                    nx=req.nx,
                    ny=req.ny,
                    nz=req.nz,
                    u_in=req.u_in,
                    re=req.re,
                    hull_length=req.hull_length,
                    hull_beam=req.hull_beam,
                    hull_draft=req.hull_draft,
                    smagorinsky_cs=req.smagorinsky_cs,
                    wave_amp=req.wave_amp,
                    wave_period=req.wave_period,
                    wave_k=0.05,
                    water_depth=0.0,
                    n_steps=req.n_steps,
                    output_interval=req.output_interval,
                    output_root=job.output_dir,
                    overwrite=True,
                    device=req.device,
                    seed=req.seed,
                )
                run_dir = run_ship_hull_flow(cfg)
                return {"run_dir": str(run_dir)}

            job_id = job_manager.submit(
                name=f"CAD→Solver: Wigley Re={req.re}",
                job_type="ship_hull",
                config=req.model_dump(),
                fn=_run_wigley,
            )
        else:
            # Series 60 / KCS: inject custom hull mask into the ship flow runner
            req_snapshot = req.model_copy()

            def _run_custom(job: job_manager.Job) -> dict:
                import torch  # noqa: I001, TC002
                import tensorlbm.obstacles as _obs
                from tensorlbm.ship_cad import build_hull_mask
                from tensorlbm.ship_flow import ShipHullFlowConfig, run_ship_hull_flow

                nx, ny, nz = req_snapshot.nx, req_snapshot.ny, req_snapshot.nz
                L = req_snapshot.hull_length
                B = req_snapshot.hull_beam
                T = req_snapshot.hull_draft
                cx = nx / 2.0
                cy = ny / 2.0
                cz_keel = nz / 4.0

                hull_mask_tensor, _stats = build_hull_mask(
                    hull_type=req_snapshot.hull_type,
                    nx=nx, ny=ny, nz=nz,
                    cx=cx, cy=cy, cz_keel=cz_keel,
                    length=L, beam=B, draft=T,
                    device=req_snapshot.device,
                )

                # Build a Wigley config with matching dims; we override the mask
                # inside run_ship_hull_flow by monkey-patching wigley_hull_mask
                _orig = _obs.wigley_hull_mask

                def _patched_mask(*_args: object, **_kwargs: object) -> torch.Tensor:
                    return hull_mask_tensor

                _obs.wigley_hull_mask = _patched_mask  # type: ignore[assignment]
                try:
                    cfg = ShipHullFlowConfig(
                        nx=nx, ny=ny, nz=nz,
                        u_in=req_snapshot.u_in,
                        re=req_snapshot.re,
                        hull_length=L,
                        hull_beam=B,
                        hull_draft=T,
                        smagorinsky_cs=req_snapshot.smagorinsky_cs,
                        wave_amp=req_snapshot.wave_amp,
                        wave_period=req_snapshot.wave_period,
                        wave_k=0.05,
                        water_depth=0.0,
                        n_steps=req_snapshot.n_steps,
                        output_interval=req_snapshot.output_interval,
                        output_root=job.output_dir,
                        overwrite=True,
                        device=req_snapshot.device,
                        seed=req_snapshot.seed,
                    )
                    run_dir = run_ship_hull_flow(cfg)
                finally:
                    _obs.wigley_hull_mask = _orig  # type: ignore[assignment]

                return {"run_dir": str(run_dir)}

            job_id = job_manager.submit(
                name=(
                    f"CAD→Solver: {req.hull_type.upper()} Re={req.re}"
                ),
                job_type="ship_hull",
                config=req.model_dump(),
                fn=_run_custom,
            )

        return {"job_id": job_id, "message": "Ship hull CAD job submitted"}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/export-stl")
async def export_stl(req: HullSTLRequest) -> Response:
    """Generate and download an ASCII STL file for the requested hull form."""
    try:
        import tempfile
        from pathlib import Path

        from tensorlbm.ship_cad import export_hull_stl

        with tempfile.TemporaryDirectory() as td:
            # Use a fixed filename within the temp dir to avoid path injection
            stl_path = export_hull_stl(
                hull_type=req.hull_type,
                length=req.length,
                beam=req.beam,
                draft=req.draft,
                n_long=req.n_long,
                n_vert=req.n_vert,
                output_path=Path(td) / "hull.stl",
            )
            content = stl_path.read_bytes()

        return Response(
            content=content,
            media_type="model/stl",
            headers={
                "Content-Disposition": f'attachment; filename="{req.hull_type}_hull.stl"'
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/hull-types")
async def list_hull_types() -> dict:
    """Return the list of supported hull types with descriptions."""
    return {
        "hull_types": [
            {
                "value": "wigley",
                "label": "Wigley Parabolic",
                "description": (
                    "Classic ITTC benchmark hull with parabolic cross-sections. "
                    "Analytical block coefficient Cb = 4/9 ≈ 0.444."
                ),
                "Cb": 0.4444,
            },
            {
                "value": "series60",
                "label": "Series 60 (Cb=0.60)",
                "description": (
                    "DTMB Series 60 polynomial approximation. "
                    "Representative of a standard merchant ship hull. Cb = 0.60."
                ),
                "Cb": 0.600,
            },
            {
                "value": "kcs",
                "label": "KCS Approximation (Cb≈0.651)",
                "description": (
                    "KRISO Container Ship approximation. "
                    "Modern container ship hull form. Cb ≈ 0.651."
                ),
                "Cb": 0.651,
            },
        ]
    }
