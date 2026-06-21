"""Solver API endpoints – submit simulation jobs.

Each endpoint accepts a Pydantic config model, creates a Job, and runs the
corresponding tensorlbm simulation function in a background thread.
"""
# ruff: noqa: TC001
from __future__ import annotations

from typing import Any
from uuid import uuid4

from fastapi import APIRouter
from pydantic import BaseModel, Field

from .. import job_manager
from ..schemas.solver import (
    BackwardFacingStepParams,
    CylinderFlowParams,
    CylinderFlowScanParams,
    DamBreakParams,
    LidDrivenCavityParams,
    PipelineFlowParams,
    PorousDrainageParams,
    ShipHullFlowParams,
    SloshingTankParams,
    SphereFlowParams,
    TurbulentChannelParams,
)
from ..services.solver import overwrite_output_root, prepare_solver_configs

router = APIRouter()


# ---------------------------------------------------------------------------
# 1. Cylinder Flow (2D)
# ---------------------------------------------------------------------------



@router.post("/cylinder-flow")
async def start_cylinder_flow(params: CylinderFlowParams) -> dict:
    """Start a 2D cylinder flow simulation."""
    run_config, submit_config = prepare_solver_configs("cylinder_flow", params)

    def _run(job: job_manager.Job) -> dict:
        from tensorlbm import CylinderFlowConfig, run_cylinder_flow

        cfg = CylinderFlowConfig(
            **overwrite_output_root(run_config, job),
        )
        run_dir = run_cylinder_flow(cfg)
        return {"run_dir": str(run_dir)}

    job_id = job_manager.submit(
        name=f"Cylinder Flow Re={params.re}",
        job_type="cylinder_flow",
        config=submit_config,
        fn=_run,
    )
    return {"job_id": job_id, "message": "Cylinder flow job submitted"}


@router.post("/cylinder-flow/scan")
async def start_cylinder_flow_scan(params: CylinderFlowScanParams) -> dict:
    """Submit multiple cylinder-flow jobs for Reynolds-number sweep."""
    scan_group = uuid4().hex[:12]
    job_ids: list[str] = []
    values = [float(v) for v in params.re_values]
    total = len(values)

    for idx, re_value in enumerate(values, start=1):
        single_params = CylinderFlowParams(
            nx=params.nx,
            ny=params.ny,
            u_in=params.u_in,
            re=re_value,
            radius=params.radius,
            n_steps=params.n_steps,
            output_interval=params.output_interval,
            device=params.device,
            seed=params.seed,
            physics=params.physics,
        )
        run_config, submit_config = prepare_solver_configs("cylinder_flow", single_params)

        def _run(job: job_manager.Job, rc: dict[str, Any] = run_config) -> dict:
            from tensorlbm import CylinderFlowConfig, run_cylinder_flow

            cfg = CylinderFlowConfig(
                **overwrite_output_root(rc, job),
            )
            run_dir = run_cylinder_flow(cfg)
            return {"run_dir": str(run_dir)}

        submit_cfg = dict(submit_config)
        submit_cfg["scan"] = {
            "group": scan_group,
            "parameter": "re",
            "index": idx,
            "total": total,
            "value": re_value,
        }
        job_id = job_manager.submit(
            name=f"Cylinder Flow Scan [{idx}/{total}] Re={re_value}",
            job_type="cylinder_flow",
            config=submit_cfg,
            fn=_run,
        )
        job_ids.append(job_id)

    return {
        "message": "Cylinder flow parameter scan jobs submitted",
        "scan_group": scan_group,
        "parameter": "re",
        "values": values,
        "job_ids": job_ids,
    }


# ---------------------------------------------------------------------------
# 2. Lid-Driven Cavity (2D)
# ---------------------------------------------------------------------------



@router.post("/lid-driven-cavity")
async def start_lid_driven_cavity(params: LidDrivenCavityParams) -> dict:
    run_config, submit_config = prepare_solver_configs("lid_driven_cavity", params)

    def _run(job: job_manager.Job) -> dict:
        from tensorlbm import LidDrivenCavityConfig, run_lid_driven_cavity

        cfg = LidDrivenCavityConfig(
            **overwrite_output_root(run_config, job),
        )
        run_dir = run_lid_driven_cavity(cfg)
        return {"run_dir": str(run_dir)}

    job_id = job_manager.submit(
        name=f"Lid-Driven Cavity Re={params.re}",
        job_type="lid_driven_cavity",
        config=submit_config,
        fn=_run,
    )
    return {"job_id": job_id, "message": "Lid-driven cavity job submitted"}


# ---------------------------------------------------------------------------
# 3. Backward-Facing Step (2D)
# ---------------------------------------------------------------------------



@router.post("/backward-facing-step")
async def start_bfs(params: BackwardFacingStepParams) -> dict:
    run_config, submit_config = prepare_solver_configs("backward_facing_step", params)

    def _run(job: job_manager.Job) -> dict:
        from tensorlbm import BackwardFacingStepConfig, run_backward_facing_step

        cfg = BackwardFacingStepConfig(
            **overwrite_output_root(run_config, job),
        )
        run_dir = run_backward_facing_step(cfg)
        return {"run_dir": str(run_dir)}

    job_id = job_manager.submit(
        name=f"Backward-Facing Step Re={params.re}",
        job_type="backward_facing_step",
        config=submit_config,
        fn=_run,
    )
    return {"job_id": job_id, "message": "Backward-facing step job submitted"}


# ---------------------------------------------------------------------------
# 4. Turbulent Channel (2D LES)
# ---------------------------------------------------------------------------



@router.post("/turbulent-channel")
async def start_turbulent_channel(params: TurbulentChannelParams) -> dict:
    run_config, submit_config = prepare_solver_configs("turbulent_channel", params)

    def _run(job: job_manager.Job) -> dict:
        from tensorlbm import TurbulentChannelConfig, run_turbulent_channel

        cfg = TurbulentChannelConfig(
            **overwrite_output_root(run_config, job),
        )
        run_dir = run_turbulent_channel(cfg)
        return {"run_dir": str(run_dir)}

    job_id = job_manager.submit(
        name=f"Turbulent Channel Re_τ={params.re_tau}",
        job_type="turbulent_channel",
        config=submit_config,
        fn=_run,
    )
    return {"job_id": job_id, "message": "Turbulent channel job submitted"}


# ---------------------------------------------------------------------------
# 5. Near-bed Pipeline Flow (2D)
# ---------------------------------------------------------------------------



@router.post("/pipeline-flow")
async def start_pipeline_flow(params: PipelineFlowParams) -> dict:
    run_config, submit_config = prepare_solver_configs("pipeline_flow", params)

    def _run(job: job_manager.Job) -> dict:
        from tensorlbm import PipelineFlowConfig, run_pipeline_flow

        cfg = PipelineFlowConfig(
            **overwrite_output_root(run_config, job),
        )
        run_dir = run_pipeline_flow(cfg)
        return {"run_dir": str(run_dir)}

    job_id = job_manager.submit(
        name=f"Pipeline Flow Re={params.re} e/D={params.gap_ratio}",
        job_type="pipeline_flow",
        config=submit_config,
        fn=_run,
    )
    return {"job_id": job_id, "message": "Pipeline flow job submitted"}


# ---------------------------------------------------------------------------
# 6. Dam Break (2D multiphase)
# ---------------------------------------------------------------------------



@router.post("/dam-break")
async def start_dam_break(params: DamBreakParams) -> dict:
    run_config, submit_config = prepare_solver_configs("dam_break", params)

    def _run(job: job_manager.Job) -> dict:
        from tensorlbm import DamBreakConfig, run_dam_break

        cfg = DamBreakConfig(
            **overwrite_output_root(run_config, job),
        )
        run_dir = run_dam_break(cfg)
        return {"run_dir": str(run_dir)}

    job_id = job_manager.submit(
        name=f"Dam Break [{params.model.upper()}]",
        job_type="dam_break",
        config=submit_config,
        fn=_run,
    )
    return {"job_id": job_id, "message": "Dam break job submitted"}


# ---------------------------------------------------------------------------
# 7. Sloshing Tank (2D multiphase)
# ---------------------------------------------------------------------------



@router.post("/sloshing-tank")
async def start_sloshing_tank(params: SloshingTankParams) -> dict:
    run_config, submit_config = prepare_solver_configs("sloshing_tank", params)

    def _run(job: job_manager.Job) -> dict:
        from tensorlbm import SloshingTankConfig, run_sloshing_tank

        cfg = SloshingTankConfig(
            **overwrite_output_root(run_config, job),
        )
        run_dir = run_sloshing_tank(cfg)
        return {"run_dir": str(run_dir)}

    job_id = job_manager.submit(
        name="Sloshing Tank",
        job_type="sloshing_tank",
        config=submit_config,
        fn=_run,
    )
    return {"job_id": job_id, "message": "Sloshing tank job submitted"}


# ---------------------------------------------------------------------------
# 8. Sphere Flow 3D (D3Q19)
# ---------------------------------------------------------------------------



@router.post("/sphere-flow")
async def start_sphere_flow(params: SphereFlowParams) -> dict:
    run_config, submit_config = prepare_solver_configs("sphere_flow", params)

    def _run(job: job_manager.Job) -> dict:
        from tensorlbm import SphereFlowConfig, run_sphere_flow

        cfg = SphereFlowConfig(
            **overwrite_output_root(run_config, job),
        )
        run_dir = run_sphere_flow(cfg)
        return {"run_dir": str(run_dir)}

    job_id = job_manager.submit(
        name=f"Sphere Flow Re={params.re} (3D)",
        job_type="sphere_flow",
        config=submit_config,
        fn=_run,
    )
    return {"job_id": job_id, "message": "Sphere flow job submitted"}


# ---------------------------------------------------------------------------
# 9. Ship Hull Flow 3D (Wigley)
# ---------------------------------------------------------------------------



@router.post("/ship-hull")
async def start_ship_hull(params: ShipHullFlowParams) -> dict:
    run_config, submit_config = prepare_solver_configs("ship_hull", params)

    def _run(job: job_manager.Job) -> dict:
        from tensorlbm import ShipHullFlowConfig, run_ship_hull_flow

        p = dict(run_config)
        # wave_k and water_depth are required by the config but not in params
        p.setdefault("wave_k", 0.05)
        p.setdefault("water_depth", 0.0)
        cfg = ShipHullFlowConfig(
            **overwrite_output_root(p, job),
        )
        run_dir = run_ship_hull_flow(cfg)
        return {"run_dir": str(run_dir)}

    job_id = job_manager.submit(
        name=f"Ship Hull (Wigley) Re={params.re}",
        job_type="ship_hull",
        config=submit_config,
        fn=_run,
    )
    return {"job_id": job_id, "message": "Ship hull flow job submitted"}


# ---------------------------------------------------------------------------
# 10. Porous Drainage (2D)
# ---------------------------------------------------------------------------



@router.post("/porous-drainage")
async def start_porous_drainage(params: PorousDrainageParams) -> dict:
    run_config, submit_config = prepare_solver_configs("porous_drainage", params)

    def _run(job: job_manager.Job) -> dict:
        from tensorlbm import PorousDrainageConfig, run_porous_drainage

        cfg = PorousDrainageConfig(
            nx=run_config["nx"],
            ny=run_config["ny"],
            medium=run_config["medium"],
            model=run_config["model"],
            porosity=run_config["porosity"],
            n_steps=run_config["n_steps"],
            output_interval=run_config["output_interval"],
            device=run_config["device"],
            seed=run_config["seed"],
            output_root=job.output_dir,
            overwrite=True,
        )
        run_dir = run_porous_drainage(cfg)
        return {"run_dir": str(run_dir)}

    job_id = job_manager.submit(
        name=f"Porous Drainage [{params.medium}]",
        job_type="porous_drainage",
        config=submit_config,
        fn=_run,
    )
    return {"job_id": job_id, "message": "Porous drainage job submitted"}


# ---------------------------------------------------------------------------
# Parameter validation endpoint
# ---------------------------------------------------------------------------

# Grid size safety limits (same as job_manager guards)
_MAX_GRID_2D = 1024
_MAX_GRID_3D = 256
_MAX_STEPS = 200_000

# Minimum tau for BGK/TRT stability
_TAU_MIN = 0.51
_MA_MAX = 0.3  # Mach number limit for compressibility errors


class ValidateParamsRequest(BaseModel):
    """Generic parameter validation request.

    Pass any subset of solver configuration fields; the validator will
    check what it can without requiring all fields to be present.
    """
    solver_type: str = Field(..., description="Solver type key, e.g. 'cylinder_flow'")
    # Grid
    nx: int | None = Field(None, ge=1)
    ny: int | None = Field(None, ge=1)
    nz: int | None = Field(None, ge=1)
    # Physics
    re: float | None = Field(None, gt=0)
    u_in: float | None = Field(None, gt=0)
    u_lid: float | None = Field(None, gt=0)
    # Time
    n_steps: int | None = Field(None, ge=1)
    output_interval: int | None = Field(None, ge=1)


@router.post("/validate")
async def validate_params(req: ValidateParamsRequest) -> dict:
    """Validate simulation parameters before submission.

    Returns a list of warnings and errors.  An empty ``errors`` list means
    the parameters passed all checks.  ``warnings`` are non-blocking advisories.
    """
    errors: list[str] = []
    warnings: list[str] = []
    info: list[str] = []

    # ---- Grid checks -------------------------------------------------------
    is_3d = req.nz is not None
    if req.nx is not None and req.ny is not None:
        if is_3d and req.nz is not None:
            if req.nx > _MAX_GRID_3D or req.ny > _MAX_GRID_3D or req.nz > _MAX_GRID_3D:
                errors.append(
                    f"3D grid dimensions must not exceed {_MAX_GRID_3D} per axis "
                    f"(got {req.nx}×{req.ny}×{req.nz})."
                )
        else:
            if req.nx > _MAX_GRID_2D or req.ny > _MAX_GRID_2D:
                errors.append(
                    f"2D grid dimensions must not exceed {_MAX_GRID_2D} per axis "
                    f"(got {req.nx}×{req.ny})."
                )
        if req.nx < 10 or req.ny < 10:
            warnings.append("Very small grid; numerical accuracy may be poor.")

    # ---- Step count checks -------------------------------------------------
    if req.n_steps is not None:
        if req.n_steps > _MAX_STEPS:
            errors.append(
                f"n_steps={req.n_steps} exceeds platform limit of {_MAX_STEPS}."
            )
        if req.n_steps < 100:
            warnings.append("n_steps < 100 – result may not be physically meaningful.")

    # ---- Output interval checks --------------------------------------------
    if req.n_steps is not None and req.output_interval is not None:
        if req.output_interval > req.n_steps:
            warnings.append(
                "output_interval > n_steps – no snapshot will be saved."
            )

    # ---- LBM stability checks (u_in / Re → tau / Ma) ----------------------
    u = req.u_in or req.u_lid
    if u is not None and req.re is not None and req.nx is not None:
        # Derive lattice viscosity and tau
        nu_lb = u * req.nx / req.re
        tau = 3.0 * nu_lb + 0.5
        ma = u / (1.0 / 3.0 ** 0.5)  # cs = 1/√3

        info.append(f"Estimated τ = {tau:.4f}, Ma = {ma:.4f}, ν_lb = {nu_lb:.6f}")

        if tau < _TAU_MIN:
            errors.append(
                f"Estimated τ={tau:.4f} < {_TAU_MIN}: BGK scheme will be unstable. "
                "Reduce Re, increase nx, or lower u_in."
            )
        elif tau < 0.6:
            warnings.append(
                f"τ={tau:.4f} is close to the stability limit (0.51). "
                "Consider using TRT or MRT collision."
            )

        if ma > _MA_MAX:
            warnings.append(
                f"Mach number Ma={ma:.4f} > {_MA_MAX}: significant compressibility "
                "errors expected in incompressible flow. Lower u_in."
            )

    valid = len(errors) == 0
    return {
        "valid": valid,
        "solver_type": req.solver_type,
        "errors": errors,
        "warnings": warnings,
        "info": info,
    }

