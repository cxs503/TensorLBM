"""Solver API endpoints – submit simulation jobs.

Each endpoint accepts a Pydantic config model, creates a Job, and runs the
corresponding tensorlbm simulation function in a background thread.
"""
from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .. import job_manager

router = APIRouter()

TurbulenceModel = Literal["none", "smagorinsky_les", "dynamic_smagorinsky_les"]
MultiphaseModel = Literal["none", "sc", "scmp", "cg", "fe"]
FlowType = Literal["single_phase", "multiphase", "free_surface"]
BoundaryCondition = Literal["standard_bounce_back", "zou_he", "periodic"]
NumericalScheme = Literal["bgk", "trt", "mrt"]


class PhysicsSelection(BaseModel):
    flow_type: FlowType = "single_phase"
    turbulence_model: TurbulenceModel = "none"
    turbulence_params: dict[str, float] = Field(default_factory=dict)
    multiphase_model: MultiphaseModel = "none"
    multiphase_params: dict[str, float] = Field(default_factory=dict)
    boundary_condition: BoundaryCondition = "standard_bounce_back"
    numerical_scheme: NumericalScheme = "bgk"
    preset: str | None = None


_PHYSICS_DEFAULTS: dict[str, dict[str, Any]] = {
    "cylinder_flow": {"flow_type": "single_phase"},
    "lid_driven_cavity": {"flow_type": "single_phase"},
    "backward_facing_step": {"flow_type": "single_phase"},
    "turbulent_channel": {
        "flow_type": "single_phase",
        "turbulence_model": "smagorinsky_les",
        "turbulence_params": {"smagorinsky_cs": 0.1},
    },
    "pipeline_flow": {"flow_type": "single_phase"},
    "dam_break": {"flow_type": "multiphase", "multiphase_model": "cg"},
    "sloshing_tank": {"flow_type": "multiphase", "multiphase_model": "cg"},
    "sphere_flow": {"flow_type": "single_phase"},
    "ship_hull": {
        "flow_type": "free_surface",
        "turbulence_model": "smagorinsky_les",
        "turbulence_params": {"smagorinsky_cs": 0.1},
    },
    "porous_drainage": {"flow_type": "multiphase", "multiphase_model": "cg"},
}

_CAPABILITY_MATRIX: dict[str, dict[str, list[str]]] = {
    "cylinder_flow": {
        "flow_types": ["single_phase"],
        "turbulence_models": ["none", "smagorinsky_les"],
        "multiphase_models": ["none"],
    },
    "lid_driven_cavity": {
        "flow_types": ["single_phase"],
        "turbulence_models": ["none"],
        "multiphase_models": ["none"],
    },
    "backward_facing_step": {
        "flow_types": ["single_phase"],
        "turbulence_models": ["none", "smagorinsky_les"],
        "multiphase_models": ["none"],
    },
    "turbulent_channel": {
        "flow_types": ["single_phase"],
        "turbulence_models": ["none", "smagorinsky_les", "dynamic_smagorinsky_les"],
        "multiphase_models": ["none"],
    },
    "pipeline_flow": {
        "flow_types": ["single_phase"],
        "turbulence_models": ["none", "smagorinsky_les"],
        "multiphase_models": ["none"],
    },
    "dam_break": {
        "flow_types": ["multiphase", "free_surface"],
        "turbulence_models": ["none"],
        "multiphase_models": ["sc", "scmp", "cg", "fe"],
    },
    "sloshing_tank": {
        "flow_types": ["multiphase", "free_surface"],
        "turbulence_models": ["none"],
        "multiphase_models": ["cg"],
    },
    "sphere_flow": {
        "flow_types": ["single_phase"],
        "turbulence_models": ["none", "smagorinsky_les"],
        "multiphase_models": ["none"],
    },
    "ship_hull": {
        "flow_types": ["single_phase", "free_surface"],
        "turbulence_models": ["none", "smagorinsky_les", "dynamic_smagorinsky_les"],
        "multiphase_models": ["none"],
    },
    "porous_drainage": {
        "flow_types": ["multiphase"],
        "turbulence_models": ["none"],
        "multiphase_models": ["sc", "cg"],
    },
}

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _overwrite_output_root(config_dict: dict, job: job_manager.Job) -> dict:
    """Replace output_root with the job's dedicated temp directory."""
    d = dict(config_dict)
    d["output_root"] = str(job.output_dir)
    d["overwrite"] = True
    d.pop("run_name", None)
    return d


def _merge_physics(job_type: str, physics: PhysicsSelection | None) -> PhysicsSelection:
    defaults = _PHYSICS_DEFAULTS[job_type]
    merged = dict(defaults)
    merged["turbulence_params"] = dict(defaults.get("turbulence_params", {}))
    merged["multiphase_params"] = dict(defaults.get("multiphase_params", {}))
    if physics is not None:
        p = physics.model_dump(exclude_none=True)
        merged.update(
            {
                k: v
                for k, v in p.items()
                if k not in ("turbulence_params", "multiphase_params")
            }
        )
        merged["turbulence_params"].update(p.get("turbulence_params", {}))
        merged["multiphase_params"].update(p.get("multiphase_params", {}))
    return PhysicsSelection(**merged)


def _validate_physics(job_type: str, physics: PhysicsSelection) -> None:
    caps = _CAPABILITY_MATRIX[job_type]
    if physics.flow_type not in caps["flow_types"]:
        raise HTTPException(
            status_code=422,
            detail=f"Flow type '{physics.flow_type}' is not supported by {job_type}",
        )
    if physics.turbulence_model not in caps["turbulence_models"]:
        raise HTTPException(
            status_code=422,
            detail=f"Turbulence model '{physics.turbulence_model}' is not supported by {job_type}",
        )
    if physics.multiphase_model not in caps["multiphase_models"]:
        raise HTTPException(
            status_code=422,
            detail=f"Multiphase model '{physics.multiphase_model}' is not supported by {job_type}",
        )


def _prepare_solver_configs(
    job_type: str, params: BaseModel
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_config = params.model_dump(exclude={"physics"})
    physics = _merge_physics(job_type, getattr(params, "physics", None))
    if "model" in run_config and physics.multiphase_model == "none":
        physics.multiphase_model = str(run_config["model"])
    _validate_physics(job_type, physics)

    if "model" in run_config and physics.multiphase_model != "none":
        run_config["model"] = physics.multiphase_model
    if "smagorinsky_cs" in run_config:
        if physics.turbulence_model == "none":
            run_config["smagorinsky_cs"] = 0.0
        else:
            cs = physics.turbulence_params.get("smagorinsky_cs")
            if cs is not None:
                run_config["smagorinsky_cs"] = float(cs)

    submit_config = dict(run_config)
    submit_config["physics"] = physics.model_dump()
    return run_config, submit_config


# ---------------------------------------------------------------------------
# 1. Cylinder Flow (2D)
# ---------------------------------------------------------------------------

class CylinderFlowParams(BaseModel):
    nx: int = Field(320, ge=20, description="Grid width")
    ny: int = Field(100, ge=10, description="Grid height")
    u_in: float = Field(0.08, gt=0, description="Inlet velocity (lattice units)")
    re: float = Field(100.0, gt=0, description="Reynolds number")
    radius: float = Field(12.0, gt=0, description="Cylinder radius (cells)")
    n_steps: int = Field(1200, ge=1, description="Total time steps")
    output_interval: int = Field(200, ge=1, description="Output every N steps")
    device: str = Field("cpu", description="Torch device (cpu / cuda:0 …)")
    seed: int = 0
    physics: PhysicsSelection | None = None


@router.post("/cylinder-flow")
async def start_cylinder_flow(params: CylinderFlowParams) -> dict:
    """Start a 2D cylinder flow simulation."""
    run_config, submit_config = _prepare_solver_configs("cylinder_flow", params)

    def _run(job: job_manager.Job) -> dict:
        from tensorlbm import CylinderFlowConfig, run_cylinder_flow

        cfg = CylinderFlowConfig(
            **_overwrite_output_root(run_config, job),
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


# ---------------------------------------------------------------------------
# 2. Lid-Driven Cavity (2D)
# ---------------------------------------------------------------------------

class LidDrivenCavityParams(BaseModel):
    nx: int = Field(128, ge=8, description="Grid size (square, ny = nx)")
    u_lid: float = Field(0.1, gt=0, description="Lid velocity")
    re: float = Field(100.0, gt=0, description="Reynolds number")
    n_steps: int = Field(10000, ge=1)
    output_interval: int = Field(2000, ge=1)
    device: str = "cpu"
    seed: int = 0
    physics: PhysicsSelection | None = None


@router.post("/lid-driven-cavity")
async def start_lid_driven_cavity(params: LidDrivenCavityParams) -> dict:
    run_config, submit_config = _prepare_solver_configs("lid_driven_cavity", params)

    def _run(job: job_manager.Job) -> dict:
        from tensorlbm import LidDrivenCavityConfig, run_lid_driven_cavity

        cfg = LidDrivenCavityConfig(
            **_overwrite_output_root(run_config, job),
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

class BackwardFacingStepParams(BaseModel):
    nx: int = Field(400, ge=20)
    ny: int = Field(80, ge=6)
    step_h: int = Field(40, ge=1, description="Step height (cells)")
    x_step: int = Field(80, ge=1, description="Pre-step solid length (cells)")
    u_in: float = Field(0.05, gt=0)
    re: float = Field(100.0, gt=0)
    n_steps: int = Field(30000, ge=1)
    output_interval: int = Field(5000, ge=1)
    device: str = "cpu"
    seed: int = 0
    physics: PhysicsSelection | None = None


@router.post("/backward-facing-step")
async def start_bfs(params: BackwardFacingStepParams) -> dict:
    run_config, submit_config = _prepare_solver_configs("backward_facing_step", params)

    def _run(job: job_manager.Job) -> dict:
        from tensorlbm import BackwardFacingStepConfig, run_backward_facing_step

        cfg = BackwardFacingStepConfig(
            **_overwrite_output_root(run_config, job),
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

class TurbulentChannelParams(BaseModel):
    nx: int = Field(256, ge=16)
    ny: int = Field(64, ge=8)
    re_tau: float = Field(100.0, gt=0, description="Friction Reynolds number Re_τ")
    u_tau: float = Field(0.005, gt=0, description="Friction velocity (lattice units)")
    smagorinsky_cs: float = Field(0.1, gt=0, description="Smagorinsky constant C_s")
    n_steps: int = Field(50000, ge=1)
    averaging_start: int = Field(20000, ge=0)
    output_interval: int = Field(5000, ge=1)
    device: str = "cpu"
    seed: int = 0
    physics: PhysicsSelection | None = None


@router.post("/turbulent-channel")
async def start_turbulent_channel(params: TurbulentChannelParams) -> dict:
    run_config, submit_config = _prepare_solver_configs("turbulent_channel", params)

    def _run(job: job_manager.Job) -> dict:
        from tensorlbm import TurbulentChannelConfig, run_turbulent_channel

        cfg = TurbulentChannelConfig(
            **_overwrite_output_root(run_config, job),
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

class PipelineFlowParams(BaseModel):
    nx: int = Field(400, ge=20)
    ny: int = Field(160, ge=10)
    diameter: float = Field(20.0, gt=0, description="Cylinder diameter (cells)")
    gap_ratio: float = Field(0.5, ge=0, description="Gap e/D")
    u_in: float = Field(0.05, gt=0)
    re: float = Field(200.0, gt=0)
    n_steps: int = Field(30000, ge=1)
    output_interval: int = Field(5000, ge=1)
    device: str = "cpu"
    seed: int = 0
    physics: PhysicsSelection | None = None


@router.post("/pipeline-flow")
async def start_pipeline_flow(params: PipelineFlowParams) -> dict:
    run_config, submit_config = _prepare_solver_configs("pipeline_flow", params)

    def _run(job: job_manager.Job) -> dict:
        from tensorlbm import PipelineFlowConfig, run_pipeline_flow

        cfg = PipelineFlowConfig(
            **_overwrite_output_root(run_config, job),
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

class DamBreakParams(BaseModel):
    nx: int = Field(400, ge=20)
    ny: int = Field(200, ge=10)
    dam_width: int = Field(100, ge=1)
    model: Literal["sc", "scmp", "cg", "fe"] = "cg"
    rho_heavy: float = Field(0.8, gt=0)
    rho_light: float = Field(0.4, gt=0)
    G: float = Field(0.9, description="Coupling constant")
    tau: float = Field(1.0, gt=0.5)
    g: float = Field(5e-5, gt=0, description="Gravity (lattice units)")
    n_steps: int = Field(4000, ge=1)
    output_interval: int = Field(400, ge=1)
    device: str = "cpu"
    physics: PhysicsSelection | None = None


@router.post("/dam-break")
async def start_dam_break(params: DamBreakParams) -> dict:
    run_config, submit_config = _prepare_solver_configs("dam_break", params)

    def _run(job: job_manager.Job) -> dict:
        from tensorlbm import DamBreakConfig, run_dam_break

        cfg = DamBreakConfig(
            **_overwrite_output_root(run_config, job),
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

class SloshingTankParams(BaseModel):
    nx: int = Field(200, ge=16)
    ny: int = Field(160, ge=16)
    water_level: int = Field(80, ge=1)
    rho_water: float = Field(0.8, gt=0)
    rho_air: float = Field(0.4, gt=0)
    G: float = Field(0.9, description="Color-gradient surface-tension coefficient")
    tau: float = Field(1.0, gt=0.5)
    g: float = Field(2e-5, gt=0)
    forcing_amp: float = Field(3e-5, ge=0)
    forcing_omega: float = Field(0.0, ge=0, description="0 = use natural frequency")
    n_steps: int = Field(6000, ge=1)
    output_interval: int = Field(600, ge=1)
    device: str = "cpu"
    physics: PhysicsSelection | None = None


@router.post("/sloshing-tank")
async def start_sloshing_tank(params: SloshingTankParams) -> dict:
    run_config, submit_config = _prepare_solver_configs("sloshing_tank", params)

    def _run(job: job_manager.Job) -> dict:
        from tensorlbm import SloshingTankConfig, run_sloshing_tank

        cfg = SloshingTankConfig(
            **_overwrite_output_root(run_config, job),
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

class SphereFlowParams(BaseModel):
    nx: int = Field(120, ge=20)
    ny: int = Field(60, ge=10)
    nz: int = Field(60, ge=10)
    u_in: float = Field(0.06, gt=0)
    re: float = Field(50.0, gt=0)
    radius: float = Field(8.0, gt=0)
    n_steps: int = Field(500, ge=1)
    output_interval: int = Field(100, ge=1)
    device: str = "cpu"
    seed: int = 0
    physics: PhysicsSelection | None = None


@router.post("/sphere-flow")
async def start_sphere_flow(params: SphereFlowParams) -> dict:
    run_config, submit_config = _prepare_solver_configs("sphere_flow", params)

    def _run(job: job_manager.Job) -> dict:
        from tensorlbm import SphereFlowConfig, run_sphere_flow

        cfg = SphereFlowConfig(
            **_overwrite_output_root(run_config, job),
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

class ShipHullFlowParams(BaseModel):
    nx: int = Field(160, ge=20)
    ny: int = Field(60, ge=10)
    nz: int = Field(40, ge=10)
    u_in: float = Field(0.05, gt=0)
    re: float = Field(200.0, gt=0)
    hull_length: float = Field(80.0, gt=0)
    hull_beam: float = Field(8.0, gt=0)
    hull_draft: float = Field(12.0, gt=0)
    smagorinsky_cs: float = Field(0.1, gt=0)
    wave_amp: float = Field(0.0, ge=0)
    wave_period: float = Field(200.0, gt=0)
    n_steps: int = Field(2000, ge=1)
    output_interval: int = Field(200, ge=1)
    device: str = "cpu"
    seed: int = 0
    physics: PhysicsSelection | None = None


@router.post("/ship-hull")
async def start_ship_hull(params: ShipHullFlowParams) -> dict:
    run_config, submit_config = _prepare_solver_configs("ship_hull", params)

    def _run(job: job_manager.Job) -> dict:
        from tensorlbm import ShipHullFlowConfig, run_ship_hull_flow

        p = dict(run_config)
        # wave_k and water_depth are required by the config but not in params
        p.setdefault("wave_k", 0.05)
        p.setdefault("water_depth", 0.0)
        cfg = ShipHullFlowConfig(
            **_overwrite_output_root(p, job),
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

class PorousDrainageParams(BaseModel):
    nx: int = Field(160, ge=20)
    ny: int = Field(80, ge=10)
    medium: Literal["random_cylinders", "tube_array"] = "random_cylinders"
    model: Literal["sc", "cg"] = "cg"
    porosity: float = Field(0.6, gt=0, lt=1)
    n_steps: int = Field(5000, ge=1)
    output_interval: int = Field(1000, ge=1)
    device: str = "cpu"
    seed: int = 0
    physics: PhysicsSelection | None = None


@router.post("/porous-drainage")
async def start_porous_drainage(params: PorousDrainageParams) -> dict:
    run_config, submit_config = _prepare_solver_configs("porous_drainage", params)

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
