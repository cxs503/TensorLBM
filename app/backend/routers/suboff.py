"""SubOff-specific API endpoints — full physical-quantity comparison platform.

Provides a dedicated router for DARPA SUBOFF simulations, exposing all
quantitative output quantities needed to benchmark TensorLBM against
PowerFlow and XFlow:

  POST /api/suboff/solve
      Submit a SUBOFF resistance simulation (all hull types).

  GET  /api/suboff/resistance-report/{job_id}
      Total resistance CT, broken down into Cf (viscous) and Cp (pressure/form),
      with ITTC-57 and DTMB experimental comparison.

  GET  /api/suboff/cp-hull/{job_id}
      Pressure-coefficient distribution Cp(x/L) along the hull surface.

  GET  /api/suboff/skin-friction/{job_id}
      Skin-friction coefficient Cf(x/L) distribution along the hull surface.

  GET  /api/suboff/boundary-layer/{job_id}
      Boundary-layer parameters δ, δ*, θ, H, y+ at standard x/L stations.

  GET  /api/suboff/wake-profile/{job_id}
      Propeller-plane axial wake profile U(r)/U∞ and nominal wake fraction.

  GET  /api/suboff/cross-sections/{job_id}
      Axial-velocity cross-section data at specified x/L planes.

  GET  /api/suboff/yplus/{job_id}
      y+ distribution along the hull surface.

  GET  /api/suboff/compare/{job_id}
      Quantitative comparison table: TensorLBM vs DTMB experiments vs
      PowerFlow and XFlow reference values.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from .. import job_manager
from ..services.benchmarks import submit_benchmark

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_3d_checkpoint(job_id: str):
    """Load the latest 3-D checkpoint for a completed job.

    Returns ``(f, rho, ux, uy, uz, step, meta)``.
    """
    import torch  # noqa: PLC0415
    from tensorlbm.checkpoint import load_checkpoint  # noqa: PLC0415
    from tensorlbm.d3q19 import macroscopic3d  # noqa: PLC0415

    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status.value != "completed":
        raise HTTPException(status_code=409, detail="Job not yet completed")

    ckpts = sorted(job.output_dir.rglob("checkpoint_f.pt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise HTTPException(status_code=404, detail="No checkpoint found for this job")

    f, step, meta = load_checkpoint(ckpts[-1].parent)
    if f.ndim != 4:
        raise HTTPException(
            status_code=422,
            detail="SubOff post-processing requires a 3-D (D3Q19) checkpoint",
        )

    rho, ux, uy, uz = macroscopic3d(f)
    return f, rho, ux, uy, uz, step, meta


def _solid_mask_from_field(ux):
    """Approximate solid mask: cells where speed ≈ 0."""
    import torch  # noqa: PLC0415
    return ux.abs() < 1e-6


# ---------------------------------------------------------------------------
# 1. Solver endpoint — submit a SubOff simulation
# ---------------------------------------------------------------------------

class SuboffSolveParams(BaseModel):
    """Parameters for a dedicated SUBOFF solver job."""

    hull_type: str = Field(
        "bare_hull",
        description="Hull variant: 'bare_hull' (AFF-1), 'with_sail' (AFF-3), 'full' (AFF-8)",
    )
    length_m: float = Field(4.356, gt=0.0, description="Physical hull length [m]")
    speed_ms: float = Field(2.5, gt=0.0, description="Inflow speed [m/s]")
    nu_m2s: float = Field(1.0e-6, gt=0.0, description="Kinematic viscosity [m²/s]")
    rho_kgm3: float = Field(1000.0, gt=0.0, description="Fluid density [kg/m³]")
    base_length_lu: float = Field(
        48.0, ge=20.0, description="Lattice hull length (controls mesh resolution)"
    )
    max_iterations: int = Field(3, ge=1, le=6, description="Richardson extrapolation iterations")
    lbm_steps: int = Field(200, ge=10, description="LBM time steps per iteration")
    lbm_warmup_steps: int = Field(50, ge=0, description="Warm-up steps before sampling")
    use_rans_ke: bool = Field(False, description="Enable k-ε RANS turbulence model")
    use_wall_model: bool = Field(False, description="Enable wall-model bounce-back BC")
    use_adaptive_mesh: bool = Field(False, description="Enable adaptive mesh refinement")
    device: str = Field("cpu", description="Compute device: 'cpu' or 'cuda'")
    save_snapshots: bool = Field(False, description="Save flow-field snapshots for ML")


@router.post("/solve")
async def solve_suboff(params: SuboffSolveParams) -> dict:
    """Submit a DARPA SUBOFF resistance simulation.

    Runs the TensorLBM D3Q19 LBM solver on the parametric SUBOFF geometry with
    optional turbulence modelling and adaptive mesh refinement.  Returns a
    job_id for status polling.  Equivalent to submitting a SUBOFF run in
    PowerFlow or XFlow.
    """
    params_dict = params.model_dump()

    def _run(job: job_manager.Job) -> dict:
        from tensorlbm import SuboffResistanceBenchmarkConfig, run_suboff_resistance_benchmark  # noqa: PLC0415

        cfg = SuboffResistanceBenchmarkConfig(
            hull_type=params_dict["hull_type"],
            length_m=params_dict["length_m"],
            speed_ms=params_dict["speed_ms"],
            nu_m2s=params_dict["nu_m2s"],
            rho_kgm3=params_dict["rho_kgm3"],
            base_length_lu=params_dict["base_length_lu"],
            max_iterations=params_dict["max_iterations"],
            lbm_steps=params_dict["lbm_steps"],
            lbm_warmup_steps=params_dict["lbm_warmup_steps"],
            use_rans_ke=params_dict["use_rans_ke"],
            use_wall_model=params_dict["use_wall_model"],
            use_adaptive_mesh=params_dict["use_adaptive_mesh"],
            device=params_dict["device"],
            save_snapshots=params_dict["save_snapshots"],
            snapshot_dir=str(job.output_dir / "snapshots"),
        )
        return run_suboff_resistance_benchmark(cfg)

    job_id = job_manager.submit(
        name=f"SUBOFF {params.hull_type} Re={params.length_m * params.speed_ms / params.nu_m2s:.2e}",
        job_type="suboff_solve",
        config=params_dict,
        fn=_run,
    )
    return {
        "job_id": job_id,
        "message": "SUBOFF simulation submitted",
        "hull_type": params.hull_type,
        "Re_L": params.length_m * params.speed_ms / params.nu_m2s,
    }


# ---------------------------------------------------------------------------
# 2. Resistance report — CT = Cf + Cp decomposition
# ---------------------------------------------------------------------------

@router.get("/resistance-report/{job_id}")
async def resistance_report(
    job_id: str,
    rho_ref: float = Query(default=1.0, gt=0.0),  # noqa: B008
    u_ref: float = Query(default=0.1, gt=0.0),  # noqa: B008
    tau: float = Query(default=0.6, gt=0.5),  # noqa: B008
    length_m: float = Query(default=4.356, gt=0.0),  # noqa: B008
    speed_ms: float = Query(default=2.5, gt=0.0),  # noqa: B008
    nu_m2s: float = Query(default=1.0e-6, gt=0.0),  # noqa: B008
    hull_type: str = Query(default="bare_hull"),  # noqa: B008
) -> dict:
    """Full resistance breakdown: CT, Cf (viscous), Cp (pressure/form).

    Computes the force decomposition on the completed SUBOFF job's final
    checkpoint and compares against:
    - ITTC-57 friction coefficient
    - DTMB experimental data (Liu & Huang 1998)
    - PowerFlow and XFlow published validation values

    Matches the *Resistance Report* panel in PowerFlow and XFlow.
    """
    from tensorlbm.suboff_postprocess import (  # noqa: PLC0415
        DTMB_REFERENCE,
        POWERFLOW_XFLOW_BENCHMARK,
        build_comparison_table,
        resistance_breakdown_3d,
        scale_lattice_to_physical,
    )
    from tensorlbm.suboff_resistance import _ittc57_friction_coefficient  # noqa: PLC0415

    f, rho, ux, uy, uz, step, meta = _load_3d_checkpoint(job_id)
    mask = _solid_mask_from_field(ux)

    # Get tau from metadata if available
    tau_eff = float(meta.get("tau", tau)) if isinstance(meta, dict) else tau

    breakdown = resistance_breakdown_3d(
        f, rho, ux, uy, uz, mask,
        tau=tau_eff, rho_ref=rho_ref, u_ref=u_ref,
    )

    # Physical Reynolds number
    re_L = speed_ms * length_m / nu_m2s

    # ITTC-57 reference
    try:
        cf_ittc57 = _ittc57_friction_coefficient(re_L)
    except ValueError:
        cf_ittc57 = None

    # Comparison table
    compare = build_comparison_table(
        CT_sim=breakdown["CT"],
        Cf_sim=breakdown["Cf"],
        Cp_sim=breakdown["Cp"],
        re_L=re_L,
        hull_type=hull_type,
    )

    return {
        "job_id": job_id,
        "step": step,
        "hull_type": hull_type,
        "re_L": re_L,
        "resistance_breakdown": breakdown,
        "cf_ittc57": cf_ittc57,
        "dtmb_reference": DTMB_REFERENCE,
        "powerflow_xflow_reference": POWERFLOW_XFLOW_BENCHMARK,
        "comparison": compare,
    }


# ---------------------------------------------------------------------------
# 3. Pressure-coefficient hull distribution
# ---------------------------------------------------------------------------

@router.get("/cp-hull/{job_id}")
async def cp_hull(
    job_id: str,
    rho_ref: float = Query(default=1.0, gt=0.0),  # noqa: B008
    u_ref: float = Query(default=0.1, gt=0.0),  # noqa: B008
    n_sections: int = Query(default=50, ge=5, le=200),  # noqa: B008
) -> dict:
    """Pressure coefficient Cp(x/L) distribution along the hull surface.

    Returns the longitudinal Cp distribution (mean, top-meridian,
    bottom-meridian) matching the hull-surface Cp plots in PowerFlow and
    XFlow.

    Cp = (p - p_∞) / (½ ρ U²)
    """
    from tensorlbm.suboff_postprocess import pressure_coefficient_hull_3d  # noqa: PLC0415

    _f, rho, ux, _uy, _uz, step, _meta = _load_3d_checkpoint(job_id)
    mask = _solid_mask_from_field(ux)

    result = pressure_coefficient_hull_3d(
        rho, mask, rho_ref=rho_ref, u_ref=u_ref, n_sections=n_sections,
    )
    result["job_id"] = job_id
    result["step"] = step
    return result


# ---------------------------------------------------------------------------
# 4. Skin-friction coefficient distribution
# ---------------------------------------------------------------------------

@router.get("/skin-friction/{job_id}")
async def skin_friction(
    job_id: str,
    rho_ref: float = Query(default=1.0, gt=0.0),  # noqa: B008
    u_ref: float = Query(default=0.1, gt=0.0),  # noqa: B008
    tau: float = Query(default=0.6, gt=0.5),  # noqa: B008
    n_sections: int = Query(default=50, ge=5, le=200),  # noqa: B008
) -> dict:
    """Skin-friction coefficient Cf(x/L) distribution along the hull surface.

    Uses the f_neq stress-tensor method to compute wall shear stress at
    each hull-surface cell, then returns the axial Cf distribution.
    Matches the skin-friction map in XFlow and PowerFlow.
    """
    from tensorlbm.suboff_postprocess import skin_friction_hull_3d  # noqa: PLC0415

    f, rho, ux, uy, uz, step, meta = _load_3d_checkpoint(job_id)
    tau_eff = float(meta.get("tau", tau)) if isinstance(meta, dict) else tau
    mask = _solid_mask_from_field(ux)

    result = skin_friction_hull_3d(
        f, rho, ux, uy, uz, mask,
        tau=tau_eff, rho_ref=rho_ref, u_ref=u_ref, n_sections=n_sections,
    )
    result["job_id"] = job_id
    result["step"] = step
    result["tau"] = tau_eff
    return result


# ---------------------------------------------------------------------------
# 5. Boundary-layer parameters
# ---------------------------------------------------------------------------

_DEFAULT_BL_STATIONS = [0.2, 0.4, 0.6, 0.75, 0.85, 0.95]


@router.get("/boundary-layer/{job_id}")
async def boundary_layer(
    job_id: str,
    stations: str = Query(  # noqa: B008
        default="0.2,0.4,0.6,0.75,0.85,0.95",
        description="Comma-separated x/L stations",
    ),
    u_inf: float = Query(default=0.1, gt=0.0),  # noqa: B008
    tau: float = Query(default=0.6, gt=0.5),  # noqa: B008
) -> dict:
    """Boundary-layer parameters δ, δ*, θ, H, y+ at specified x/L stations.

    Returns the same integral boundary-layer quantities available in the
    PowerFlow and XFlow boundary-layer panel, computed via profile integration
    along the hull surface at each requested axial station.
    """
    from tensorlbm.suboff_postprocess import (  # noqa: PLC0415
        boundary_layer_at_station,
        yplus_hull_3d,
    )

    # Parse stations
    try:
        x_stations = [float(s.strip()) for s in stations.split(",") if s.strip()]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid stations: {exc}") from exc
    if not x_stations:
        x_stations = _DEFAULT_BL_STATIONS

    f, rho, ux, uy, uz, step, meta = _load_3d_checkpoint(job_id)
    tau_eff = float(meta.get("tau", tau)) if isinstance(meta, dict) else tau
    nu_lu = (tau_eff - 0.5) / 3.0
    mask = _solid_mask_from_field(ux)

    # Compute y+ for tau_w at each station
    yp_data = yplus_hull_3d(
        f, rho, ux, uy, uz, mask, tau_lbm=tau_eff, nu_lu=nu_lu, n_sections=len(x_stations),
    )

    station_results = []
    for i, x_frac in enumerate(x_stations):
        x_frac = max(0.0, min(float(x_frac), 1.0))
        # Get approximate tau_w at this station (from y+ data)
        tau_w_lu: float | None = None
        if yp_data["y_plus_mean"] and i < len(yp_data["y_plus_mean"]):
            yp = yp_data["y_plus_mean"][i]
            # u_tau = y+ * nu / Δy (Δy=1)
            tau_w_lu = (yp * nu_lu) ** 2  # τ_w = ρ u_τ² = u_τ² (ρ=1)

        bl = boundary_layer_at_station(
            ux, mask,
            x_over_L=x_frac,
            u_inf=u_inf,
            nu_lu=nu_lu,
            tau_w_lu=tau_w_lu,
        )
        station_results.append(bl)

    return {
        "job_id": job_id,
        "step": step,
        "tau": tau_eff,
        "nu_lu": nu_lu,
        "stations": station_results,
        "y_plus": yp_data,
    }


# ---------------------------------------------------------------------------
# 6. Wake / propeller-plane profile
# ---------------------------------------------------------------------------

@router.get("/wake-profile/{job_id}")
async def wake_profile(
    job_id: str,
    x_over_L: float = Query(default=0.978, ge=0.0, le=1.0),  # noqa: B008
    u_inf: float = Query(default=0.1, gt=0.0),  # noqa: B008
    n_radial: int = Query(default=32, ge=4, le=128),  # noqa: B008
) -> dict:
    """Propeller-plane axial velocity wake profile U(r/R) / U∞.

    Samples the axial velocity at the propeller plane (default x/L = 0.978)
    as a function of the normalised radius r/R, returning the nominal wake
    fraction.  Matches the wake-plane visualisation in PowerFlow and XFlow.

    The DARPA SUBOFF experimental reference (Groves et al. 1989) gives
    U/U∞ ≈ 0.82 on the hull axis at the propeller plane.
    """
    from tensorlbm.suboff_postprocess import DTMB_REFERENCE, wake_profile_3d  # noqa: PLC0415

    _f, _rho, ux, _uy, _uz, step, _meta = _load_3d_checkpoint(job_id)
    mask = _solid_mask_from_field(ux)

    result = wake_profile_3d(ux, mask, x_over_L=x_over_L, u_inf=u_inf, n_radial=n_radial)
    result["job_id"] = job_id
    result["step"] = step
    # Attach DTMB reference value for wake centre
    result["dtmb_reference_U_centre"] = DTMB_REFERENCE["wake_u_over_U_center"]["value"]
    result["dtmb_reference_source"] = DTMB_REFERENCE["wake_u_over_U_center"]["source"]
    return result


# ---------------------------------------------------------------------------
# 7. Axial cross-section velocity data
# ---------------------------------------------------------------------------

@router.get("/cross-sections/{job_id}")
async def cross_sections(
    job_id: str,
    stations: str = Query(  # noqa: B008
        default="0.2,0.5,0.8,0.978",
        description="Comma-separated x/L stations",
    ),
    u_inf: float = Query(default=0.1, gt=0.0),  # noqa: B008
    max_grid: int = Query(default=64, ge=8, le=256),  # noqa: B008
) -> dict:
    """Axial-velocity cross-sectional data at specified x/L planes.

    Returns 2-D slices of normalised streamwise velocity U/U∞ at each
    requested axial station, matching the cross-section plot panel in
    XFlow and PowerFlow.
    """
    from tensorlbm.suboff_postprocess import axial_cross_section_3d  # noqa: PLC0415

    try:
        x_list = [float(s.strip()) for s in stations.split(",") if s.strip()]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid stations: {exc}") from exc
    if not x_list:
        x_list = [0.2, 0.5, 0.8, 0.978]

    _f, _rho, ux, uy, uz, step, _meta = _load_3d_checkpoint(job_id)
    mask = _solid_mask_from_field(ux)

    sections = axial_cross_section_3d(
        ux, uy, uz, mask,
        x_over_L_stations=x_list,
        u_inf=u_inf,
        max_grid=max_grid,
    )
    return {
        "job_id": job_id,
        "step": step,
        "sections": sections,
    }


# ---------------------------------------------------------------------------
# 8. y+ distribution
# ---------------------------------------------------------------------------

@router.get("/yplus/{job_id}")
async def yplus_distribution(
    job_id: str,
    tau: float = Query(default=0.6, gt=0.5),  # noqa: B008
    n_sections: int = Query(default=50, ge=5, le=200),  # noqa: B008
) -> dict:
    """y+ distribution over the hull surface (axial distribution).

    Returns the mean y+ at each axial station, and the global max/mean y+.
    The y+ map is a key quality indicator in PowerFlow and XFlow — values
    should be in the range 1–100 for wall-resolved LES/LBM.
    """
    from tensorlbm.suboff_postprocess import yplus_hull_3d  # noqa: PLC0415

    f, rho, ux, uy, uz, step, meta = _load_3d_checkpoint(job_id)
    tau_eff = float(meta.get("tau", tau)) if isinstance(meta, dict) else tau
    nu_lu = (tau_eff - 0.5) / 3.0
    mask = _solid_mask_from_field(ux)

    result = yplus_hull_3d(
        f, rho, ux, uy, uz, mask,
        tau_lbm=tau_eff, nu_lu=nu_lu, n_sections=n_sections,
    )
    result["job_id"] = job_id
    result["step"] = step
    result["tau"] = tau_eff
    result["nu_lu"] = nu_lu
    return result


# ---------------------------------------------------------------------------
# 9. Quantitative comparison table
# ---------------------------------------------------------------------------

@router.get("/compare/{job_id}")
async def quantitative_compare(
    job_id: str,
    rho_ref: float = Query(default=1.0, gt=0.0),  # noqa: B008
    u_ref: float = Query(default=0.1, gt=0.0),  # noqa: B008
    tau: float = Query(default=0.6, gt=0.5),  # noqa: B008
    length_m: float = Query(default=4.356, gt=0.0),  # noqa: B008
    speed_ms: float = Query(default=2.5, gt=0.0),  # noqa: B008
    nu_m2s: float = Query(default=1.0e-6, gt=0.0),  # noqa: B008
    hull_type: str = Query(default="bare_hull"),  # noqa: B008
) -> dict:
    """Full quantitative comparison: TensorLBM vs PowerFlow / XFlow vs DTMB experiments.

    Computes CT, Cf, Cp from the completed job's checkpoint and builds a
    side-by-side comparison table against:

    * DTMB Model 5470 experimental data (Liu & Huang 1998, Re_L = 1.2×10⁷)
    * PowerFlow 5.x validation results for SUBOFF AFF-1
    * XFlow validation results for SUBOFF AFF-1
    * ITTC-57 friction-line formula

    Returns relative errors and a pass/fail assessment for each metric.
    """
    from tensorlbm.suboff_postprocess import (  # noqa: PLC0415
        build_comparison_table,
        resistance_breakdown_3d,
        scale_lattice_to_physical,
    )

    f, rho, ux, uy, uz, step, meta = _load_3d_checkpoint(job_id)
    tau_eff = float(meta.get("tau", tau)) if isinstance(meta, dict) else tau
    mask = _solid_mask_from_field(ux)

    breakdown = resistance_breakdown_3d(
        f, rho, ux, uy, uz, mask,
        tau=tau_eff, rho_ref=rho_ref, u_ref=u_ref,
    )

    re_L = speed_ms * length_m / nu_m2s
    comparison = build_comparison_table(
        CT_sim=breakdown["CT"],
        Cf_sim=breakdown["Cf"],
        Cp_sim=breakdown["Cp"],
        re_L=re_L,
        hull_type=hull_type,
    )

    # Physical force scaling
    nz_len = f.shape[1]  # approximate length in lu from grid
    length_lu = float(nz_len) * 0.6  # rough estimate; overridden by user params
    scale = scale_lattice_to_physical(
        length_m=length_m,
        length_lu=max(length_lu, 1.0),
        speed_ms=speed_ms,
        u_lbm=u_ref,
    )

    return {
        "job_id": job_id,
        "step": step,
        "hull_type": hull_type,
        "re_L": re_L,
        "resistance_breakdown": breakdown,
        "comparison_table": comparison,
        "physical_scaling": scale,
        "summary": {
            "CT": breakdown["CT"],
            "Cf": breakdown["Cf"],
            "Cp": breakdown["Cp"],
            "overall_error_pct": comparison["overall_error_pct"],
            "pass": comparison["pass"],
        },
    }


# ---------------------------------------------------------------------------
# 10. Reference data endpoint (no job required)
# ---------------------------------------------------------------------------

@router.get("/reference-data")
async def reference_data() -> dict:
    """Return the full DTMB experimental reference data and PowerFlow/XFlow
    comparison values used by all SubOff quantitative comparison endpoints.

    No job required — returns static reference tables for display in the UI.
    """
    from tensorlbm.suboff_postprocess import DTMB_REFERENCE, POWERFLOW_XFLOW_BENCHMARK  # noqa: PLC0415

    return {
        "dtmb_experimental": DTMB_REFERENCE,
        "powerflow_xflow": POWERFLOW_XFLOW_BENCHMARK,
        "notes": (
            "DTMB: Liu & Huang (1998) CRDKNSWC/HD-1298-11. "
            "Geometry: Groves et al. (1989) DTRC/SHD-1298-01."
        ),
    }
