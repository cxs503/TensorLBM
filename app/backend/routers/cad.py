"""CAD modelling API endpoints for ship hull geometry.

Exposes the :mod:`tensorlbm.ship_cad` module as REST endpoints so that the
browser-based platform can:

1. Generate parametric hull previews (body-plan / waterplane / side-profile).
2. Retrieve hull form statistics (Cb, Cwp, Cm, Cp, …).
3. Compute LBM parameters from physical ship dimensions.
4. Launch a solver job directly from CAD parameters (CAD → solver shortcut).
5. Export a hull STL file.
"""
# ruff: noqa: TC001
from __future__ import annotations

import base64
import contextlib
import io
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from .. import job_manager
from ..cad3d_service import cad3d_service
from ..schemas.cad import (
    CAD3DCreateRequest,
    CAD3DExportRequest,
    CAD3DMaskBridgeRequest,
    CAD3DUpdateRequest,
    HullMaskRequest,
    HullPreviewRequest,
    HullSolverRequest,
    HullSTLRequest,
    LBMParametersRequest,
    OffshorePreviewRequest,
    OffshoreSTLRequest,
    PropellerCurveRequest,
    PropellerDesignRequest,
    PropellerOpenWaterRequest,
    ResistanceEstimateRequest,
    SuboffMaskRequest,
    SuboffPreviewRequest,
    SuboffSTLRequest,
)
from ..services.cad import figure_to_png_data_url

router = APIRouter()

# ---------------------------------------------------------------------------
# Request / response models
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

        img_b64 = figure_to_png_data_url(fig, dpi=110)
        import matplotlib.pyplot as plt
        plt.close(fig)

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
        img_b64 = figure_to_png_data_url(fig, dpi=100)
        plt.close(fig)

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


@router.post("/resistance-estimate")
async def resistance_estimate(req: ResistanceEstimateRequest) -> dict:
    """Estimate calm-water ship resistance for quick design screening."""
    try:
        from tensorlbm.ship_cad import ship_resistance_estimate

        return ship_resistance_estimate(
            hull_type=req.hull_type,
            length_m=req.length_m,
            beam_m=req.beam_m,
            draft_m=req.draft_m,
            speed_ms=req.speed_ms,
            nu_m2s=req.nu_m2s,
            rho_kgm3=req.rho_kgm3,
            residual_ratio=req.residual_ratio,
        )
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
        req_snapshot = req.model_copy()

        def _run_ship(job: job_manager.Job) -> dict:
            from tensorlbm import ShipHullFlowConfig, run_ship_hull_flow

            cfg = ShipHullFlowConfig(
                hull_type=req_snapshot.hull_type,
                nx=req_snapshot.nx,
                ny=req_snapshot.ny,
                nz=req_snapshot.nz,
                u_in=req_snapshot.u_in,
                re=req_snapshot.re,
                hull_length=req_snapshot.hull_length,
                hull_beam=req_snapshot.hull_beam,
                hull_draft=req_snapshot.hull_draft,
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
            return {"run_dir": str(run_dir)}

        job_id = job_manager.submit(
            name=f"CAD→Solver: {req.hull_type.upper()} Re={req.re}",
            job_type="ship_hull",
            config=req.model_dump(),
            fn=_run_ship,
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
            {
                "value": "kvlcc2",
                "label": "KVLCC2 Tanker (Cb≈0.810)",
                "description": (
                    "KRISO Very Large Crude Carrier 2. "
                    "Standard CFD benchmark VLCC tanker with U-shaped midship sections. "
                    "Cb ≈ 0.810."
                ),
                "Cb": 0.810,
            },
            {
                "value": "npl",
                "label": "NPL High-Speed (Cb≈0.397)",
                "description": (
                    "National Physical Laboratory high-speed displacement hull "
                    "(Bailey 1976 series). Fine V-sections, raked stern. Cb ≈ 0.397."
                ),
                "Cb": 0.397,
            },
        ]
    }


# ---------------------------------------------------------------------------
# SUBOFF submarine endpoints
# ---------------------------------------------------------------------------


@router.post("/suboff/preview")
async def suboff_preview(req: SuboffPreviewRequest) -> dict:
    """Generate a multi-view SUBOFF submarine preview figure.

    Returns a base64-encoded PNG plus hull form statistics.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from tensorlbm.suboff_cad import (
            SuboffConfig,
            SuboffHullType,
            generate_suboff_previews,
            suboff_statistics,
        )

        config = SuboffConfig(
            bow_fraction=req.bow_fraction,
            stern_fraction=req.stern_fraction,
            stern_exponent=req.stern_exponent,
        )
        ht = SuboffHullType(req.hull_type)
        radius = req.radius if req.radius > 0 else None

        fig = generate_suboff_previews(ht, length=req.length, radius=radius, config=config)

        import io as _io
        buf = _io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        img_b64 = "data:image/png;base64," + base64.b64encode(buf.read()).decode()

        r_val = (req.radius if req.radius > 0 else config.r_over_l * req.length)
        stats = suboff_statistics(ht, req.length, r_val, config)
        return {"image": img_b64, "stats": stats}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/suboff/hull-mask")
async def suboff_hull_mask_endpoint(req: SuboffMaskRequest) -> dict:
    """Build a 3-D SUBOFF voxel mask and return statistics + top-view preview."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        from tensorlbm.suboff_cad import SuboffHullType, build_suboff_mask

        mask, stats = build_suboff_mask(
            hull_type=req.hull_type,
            nx=req.nx,
            ny=req.ny,
            nz=req.nz,
            cx=req.cx,
            cy=req.cy,
            cz=req.cz,
            length=req.length,
            radius=req.radius if req.radius > 0 else None,
            device=req.device,
        )

        top_view = mask.any(dim=0).cpu().numpy().astype(np.uint8) * 255
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.imshow(top_view, cmap="Blues", origin="lower", vmin=0, vmax=255)
        ax.set_title(
            f"SUBOFF {SuboffHullType(req.hull_type).value.upper()} – top view "
            f"(L/D={stats['L_D_ratio']:.2f})"
        )
        ax.axis("off")
        import io as _io
        buf = _io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        img_b64 = "data:image/png;base64," + base64.b64encode(buf.read()).decode()

        return {"image": img_b64, "stats": stats}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/suboff/export-stl")
async def suboff_export_stl(req: SuboffSTLRequest) -> Response:
    """Generate and download an ASCII STL for the requested SUBOFF variant."""
    try:
        import tempfile as _tempfile

        from tensorlbm.suboff_cad import export_suboff_stl

        radius = req.radius if req.radius > 0 else None
        with _tempfile.TemporaryDirectory() as td:
            stl_path = export_suboff_stl(
                hull_type=req.hull_type,
                length=req.length,
                radius=radius,
                n_axial=req.n_axial,
                n_circ=req.n_circ,
                output_path=Path(td) / "suboff.stl",
            )
            content = stl_path.read_bytes()

        return Response(
            content=content,
            media_type="model/stl",
            headers={
                "Content-Disposition": f'attachment; filename="suboff_{req.hull_type}.stl"'
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/suboff/model-types")
async def list_suboff_model_types() -> dict:
    """Return the list of supported SUBOFF model variants."""
    return {
        "model_types": [
            {
                "value": "bare_hull",
                "label": "SUBOFF Bare Hull (AFF-1)",
                "description": (
                    "Axisymmetric body of revolution only. "
                    "Ellipsoidal bow, cylindrical parallel midbody, polynomial stern. "
                    "L/D ≈ 8.57."
                ),
            },
            {
                "value": "with_sail",
                "label": "SUBOFF + Conning Tower (AFF-3)",
                "description": (
                    "Bare hull plus a conning-tower sail (fairwater). "
                    "Suitable for studying sail-induced vortex shedding."
                ),
            },
            {
                "value": "full",
                "label": "SUBOFF Full Appendage (AFF-8)",
                "description": (
                    "Bare hull, conning-tower sail, and four cruciform stern "
                    "control-surface fins. Full-configuration drag benchmark."
                ),
            },
        ]
    }


@router.post("/3d/models")
async def cad3d_create_model(req: CAD3DCreateRequest) -> dict:
    """Create a 3-D CAD model (parametric/STL/STEP)."""
    try:
        payload: dict[str, object]
        if req.source_type == "parametric":
            payload = {
                "hull_type": req.hull_type,
                "length": req.length,
                "beam": req.beam,
                "draft": req.draft,
                "n_long": req.n_long,
                "n_vert": req.n_vert,
            }
        else:
            if not req.file_b64:
                raise ValueError("file_b64 required for stl/step model import")
            suffix = ".stl" if req.source_type == "stl" else ".step"
            root = Path(tempfile.gettempdir()) / "tensorlbm_platform" / "cad3d_uploads"
            root.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(prefix="cad3d_import_", suffix=suffix, dir=root)
            os.close(fd)
            target = Path(tmp_name)
            target.write_bytes(base64.b64decode(req.file_b64))
            payload = {"file_path": str(target)}

        model = cad3d_service.create_model(
            source_type=req.source_type,
            payload=payload,
            units=req.units,
        )
        stats = cad3d_service.model_stats(model.model_id)
        return {"model_id": model.model_id, "stats": stats}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.put("/3d/models/{model_id}")
async def cad3d_update_model(model_id: str, req: CAD3DUpdateRequest) -> dict:
    """Update parametric 3-D CAD model parameters and append a new version."""
    try:
        model = cad3d_service.get_model(model_id)
        if model.source_type != "parametric":
            raise ValueError("only parametric models support direct parameter updates")
        payload = {
            "hull_type": req.hull_type,
            "length": req.length,
            "beam": req.beam,
            "draft": req.draft,
            "n_long": req.n_long,
            "n_vert": req.n_vert,
        }
        cad3d_service.update_model(model_id, payload)
        return {"model_id": model_id, "stats": cad3d_service.model_stats(model_id)}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/3d/models/{model_id}")
async def cad3d_get_model(model_id: str) -> dict:
    """Get a 3-D CAD model summary."""
    try:
        model = cad3d_service.get_model(model_id)
        return {
            "model_id": model.model_id,
            "source_type": model.source_type,
            "units": model.units,
            "payload": model.payload,
            "version_count": len(model.versions),
        }
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/3d/models/{model_id}/stats")
async def cad3d_get_stats(model_id: str) -> dict:
    """Get mesh statistics for a CAD model."""
    try:
        return cad3d_service.model_stats(model_id)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/3d/models/{model_id}/mesh")
async def cad3d_get_mesh(model_id: str) -> dict:
    """Get mesh vertices/faces for frontend rendering."""
    try:
        mesh = cad3d_service.model_mesh(model_id)
        return {
            "model_id": model_id,
            "vertices": mesh.vertices.tolist(),
            "faces": mesh.faces.tolist(),
            "stats": mesh.stats(),
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/3d/models/{model_id}/versions")
async def cad3d_get_versions(model_id: str) -> dict:
    """List model versions for restore/reproducibility."""
    try:
        model = cad3d_service.get_model(model_id)
        return {
            "model_id": model.model_id,
            "versions": [
                {
                    "version": v.version,
                    "source_type": v.source_type,
                    "payload": v.payload,
                }
                for v in model.versions
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/3d/models/{model_id}/versions/{version}/restore")
async def cad3d_restore_version(model_id: str, version: int) -> dict:
    """Restore a previous model version as a new head version."""
    try:
        model = cad3d_service.restore_version(model_id, version)
        return {"model_id": model.model_id, "version_count": len(model.versions)}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/3d/models/{model_id}/export")
async def cad3d_export(model_id: str, req: CAD3DExportRequest) -> Response:
    """Export a CAD model as glTF/STL/STEP."""
    try:
        out, mime = cad3d_service.export_model(model_id, req.fmt)
        content = out.read_bytes()
        ext = "gltf" if req.fmt == "gltf" else req.fmt
        return Response(
            content=content,
            media_type=mime,
            headers={"Content-Disposition": f'attachment; filename="{model_id}.{ext}"'},
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/3d/models/{model_id}/lbm-mask")
async def cad3d_build_lbm_mask(model_id: str, req: CAD3DMaskBridgeRequest) -> dict:
    """Build LBM mask from CAD model through stable bridge interface."""
    try:
        return cad3d_service.build_lbm_mask(
            model_id,
            nx=req.nx,
            ny=req.ny,
            nz=req.nz,
            device=req.device,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ===========================================================================
# Offshore structure endpoints
# ===========================================================================


def _offshore_kwargs(req: object) -> dict:
    """Extract non-None geometry overrides from an offshore request model."""
    fields = [
        "diameter", "leg_diameter", "foot_spread", "head_spread",
        "hull_diameter", "keel_diameter", "column_diameter",
        "pontoon_length", "pontoon_width", "pontoon_height", "column_height",
    ]
    return {k: getattr(req, k) for k in fields if getattr(req, k, None) is not None}


@router.post("/offshore/preview")
async def offshore_preview(req: OffshorePreviewRequest) -> dict:
    """Generate top/side/front projection previews for an offshore structure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from tensorlbm.offshore_cad import generate_offshore_previews
        kwargs = _offshore_kwargs(req)
        fig = generate_offshore_previews(
            req.struct_type, nx=req.nx, ny=req.ny, nz=req.nz, **kwargs
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
        plt.close(fig)
        img_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        return {"image": img_b64, "struct_type": req.struct_type}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/offshore/hull-mask")
async def offshore_hull_mask(req: OffshorePreviewRequest) -> dict:
    """Build a 3-D LBM voxel mask for an offshore structure."""
    try:
        from tensorlbm.offshore_cad import build_offshore_mask
        kwargs = _offshore_kwargs(req)
        result = build_offshore_mask(
            req.struct_type, req.nx, req.ny, req.nz, device="cpu", **kwargs
        )
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        mask_np = result["mask"].cpu().numpy()
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        ax.imshow(mask_np.max(axis=2).T, origin="lower", aspect="equal", cmap="Blues")
        ax.set_title(f"{req.struct_type} – top view")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
        plt.close(fig)
        img_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        return {"image": img_b64, "stats": result["stats"]}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/offshore/export-stl")
async def offshore_export_stl(req: OffshoreSTLRequest) -> Response:
    """Export an offshore structure as ASCII STL."""
    try:
        from tensorlbm.offshore_cad import OffshoreStructureType, export_offshore_stl
        # Validate struct_type against the allowed enum before using in path
        safe_type = OffshoreStructureType(req.struct_type).value
        kwargs = _offshore_kwargs(req)
        with tempfile.NamedTemporaryFile(
            suffix=f"_{safe_type}.stl", delete=False
        ) as tmp:
            tmp_path = tmp.name
        export_offshore_stl(
            safe_type, tmp_path, req.nx, req.ny, req.nz, **kwargs
        )
        content = Path(tmp_path).read_bytes()
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        return Response(
            content=content,
            media_type="model/stl",
            headers={
                "Content-Disposition": f'attachment; filename="{safe_type}.stl"'
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/offshore/structure-types")
async def list_offshore_structure_types() -> dict:
    """List available offshore structure types."""
    from tensorlbm.offshore_cad import _STRUCTURE_LABELS, OffshoreStructureType
    return {
        "structure_types": [
            {"value": t.value, "label": _STRUCTURE_LABELS[t]}
            for t in OffshoreStructureType
        ]
    }


# ===========================================================================
# Propeller performance endpoints (Wageningen B-series)
# ===========================================================================




@router.post("/propeller/open-water")
async def propeller_open_water(req: PropellerOpenWaterRequest) -> dict:
    """Compute Wageningen B-series open-water coefficients at a single advance ratio."""
    try:
        from tensorlbm.propeller_cad import wageningen_b_series
        return wageningen_b_series(
            req.J, req.P_D, req.Ae_A0, req.Z,
            rho=req.rho,
            n=req.n_rps,
            D=req.D_m,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/propeller/curves")
async def propeller_curves(req: PropellerCurveRequest) -> dict:
    """Return open-water diagram (image + data) over a J sweep."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        from tensorlbm.propeller_cad import plot_b_series_curves, wageningen_b_series
        fig = plot_b_series_curves(
            req.P_D, req.Ae_A0, req.Z,
            J_range=(req.J_min, req.J_max),
            n_points=req.n_points,
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
        plt.close(fig)
        img_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        J_arr = np.linspace(req.J_min, req.J_max, req.n_points)
        rows = [
            {
                "J": round(float(Jv), 4),
                **wageningen_b_series(float(Jv), req.P_D, req.Ae_A0, req.Z),
            }
            for Jv in J_arr
        ]
        return {"image": img_b64, "data": rows}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/propeller/design")
async def propeller_design_endpoint(req: PropellerDesignRequest) -> dict:
    """Size a propeller for a required thrust at a given advance speed."""
    try:
        from tensorlbm.propeller_cad import propeller_design
        return propeller_design(
            req.thrust_n, req.Va_ms, req.P_D, req.Ae_A0, req.Z, req.n_rps,
            rho=req.rho,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
