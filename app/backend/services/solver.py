"""Service-layer helpers for solver endpoint configuration handling."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

from ..schemas.solver import PhysicsSelection

if TYPE_CHECKING:
    from pydantic import BaseModel

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
    "rotating_cylinder": {
        "flow_type": "single_phase",
        "turbulence_model": "smagorinsky_les",
        "turbulence_params": {"smagorinsky_cs": 0.1},
    },
    "actuator_disk": {
        "flow_type": "single_phase",
        "turbulence_model": "smagorinsky_les",
        "turbulence_params": {"smagorinsky_cs": 0.1},
    },
    "propeller_open_water": {"flow_type": "single_phase"},
    "ibm_propeller": {
        "flow_type": "single_phase",
        "turbulence_model": "smagorinsky_les",
        "turbulence_params": {"smagorinsky_cs": 0.1},
    },
}

_CAPABILITY_MATRIX: dict[str, dict[str, list[str]]] = {
    "cylinder_flow": {
        "flow_types": ["single_phase"],
        "turbulence_models": ["none", "smagorinsky_les"],
        "multiphase_models": ["none"],
        "enhancements": ["synthetic_inflow", "sponge_layer", "turbulence_statistics"],
    },
    "lid_driven_cavity": {
        "flow_types": ["single_phase"],
        "turbulence_models": ["none"],
        "multiphase_models": ["none"],
        "enhancements": [],
    },
    "backward_facing_step": {
        "flow_types": ["single_phase"],
        "turbulence_models": ["none", "smagorinsky_les"],
        "multiphase_models": ["none"],
        "enhancements": [],
    },
    "turbulent_channel": {
        "flow_types": ["single_phase"],
        "turbulence_models": ["none", "smagorinsky_les", "dynamic_smagorinsky_les"],
        "multiphase_models": ["none"],
        "enhancements": ["rough_wall", "turbulence_statistics"],
    },
    "pipeline_flow": {
        "flow_types": ["single_phase"],
        "turbulence_models": ["none", "smagorinsky_les"],
        "multiphase_models": ["none"],
        "enhancements": [],
    },
    "dam_break": {
        "flow_types": ["multiphase", "free_surface"],
        "turbulence_models": ["none"],
        "multiphase_models": ["sc", "scmp", "cg", "fe"],
        "enhancements": [],
    },
    "sloshing_tank": {
        "flow_types": ["multiphase", "free_surface"],
        "turbulence_models": ["none"],
        "multiphase_models": ["cg"],
        "enhancements": [],
    },
    "sphere_flow": {
        "flow_types": ["single_phase"],
        "turbulence_models": ["none", "smagorinsky_les"],
        "multiphase_models": ["none"],
        "enhancements": [],
    },
    "ship_hull": {
        "flow_types": ["single_phase", "free_surface"],
        "turbulence_models": ["none", "smagorinsky_les", "dynamic_smagorinsky_les"],
        "multiphase_models": ["none"],
        "enhancements": [],
    },
    "porous_drainage": {
        "flow_types": ["multiphase"],
        "turbulence_models": ["none"],
        "multiphase_models": ["sc", "cg"],
        "enhancements": [],
    },
    "rotating_cylinder": {
        "flow_types": ["single_phase"],
        "turbulence_models": ["none", "smagorinsky_les"],
        "multiphase_models": ["none"],
        "enhancements": [],
    },
    "actuator_disk": {
        "flow_types": ["single_phase"],
        "turbulence_models": ["none", "smagorinsky_les"],
        "multiphase_models": ["none"],
        "enhancements": [],
    },
    "propeller_open_water": {
        "flow_types": ["single_phase"],
        "turbulence_models": ["none", "smagorinsky_les"],
        "multiphase_models": ["none"],
        "enhancements": [],
    },
    "ibm_propeller": {
        "flow_types": ["single_phase"],
        "turbulence_models": ["none", "smagorinsky_les"],
        "multiphase_models": ["none"],
        "enhancements": [],
    },
}


def overwrite_output_root(config_dict: dict, job: object) -> dict:
    """Replace output_root with the job's dedicated temp directory."""
    d = dict(config_dict)
    d["output_root"] = str(job.output_dir)
    d["overwrite"] = True
    d.pop("run_name", None)
    return d


def merge_physics(job_type: str, physics: PhysicsSelection | None) -> PhysicsSelection:
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


def validate_physics(job_type: str, physics: PhysicsSelection) -> None:
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
    enhancements = {
        "synthetic_inflow": physics.synthetic_inflow,
        "sponge_layer": physics.sponge_layer,
        "rough_wall": physics.rough_wall,
        "turbulence_statistics": physics.turbulence_statistics,
    }
    unsupported = [
        name
        for name, cfg in enhancements.items()
        if cfg is not None and cfg.enabled and name not in caps.get("enhancements", [])
    ]
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise HTTPException(
            status_code=422,
            detail=f"Enhancement '{names}' is not supported by {job_type}",
        )


def prepare_solver_configs(
    job_type: str, params: BaseModel
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_config = params.model_dump(exclude={"physics"})
    physics = merge_physics(job_type, getattr(params, "physics", None))
    if "model" in run_config and physics.multiphase_model == "none":
        physics.multiphase_model = str(run_config["model"])
    validate_physics(job_type, physics)

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
    submit_config["physics"] = physics.model_dump(exclude_none=True)
    return run_config, submit_config
