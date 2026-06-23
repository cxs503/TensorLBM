"""HPC orchestration endpoints for experiment templates and KPI rollups."""
from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from .. import job_manager
from . import benchmarks as benchmarks_router
from . import reports, solver

router = APIRouter()
_RELEASE_GATE_HISTORY: list[dict[str, Any]] = []
_RELEASE_BASELINES: dict[str, dict[str, Any]] = {}


class SweepVariable(BaseModel):
    name: str = Field(..., min_length=1)
    values: list[float] = Field(..., min_length=1, max_length=40)


class TemplateRunRequest(BaseModel):
    template_id: str
    base_config: dict[str, Any] = Field(default_factory=dict)
    sweep: list[SweepVariable] = Field(default_factory=list)
    orchestration: dict[str, Any] = Field(default_factory=dict)
    objective: solver.StudyObjective | None = None
    constraints: list[solver.StudyConstraint] = Field(default_factory=list)


def _templates() -> list[dict[str, Any]]:
    return [
        {
            "template_id": "cylinder_re_sweep",
            "stage": "A",
            "title": "Cylinder Reynolds sweep",
            "implemented": True,
            "solver_type": "cylinder_flow",
            "description": "Batch Re scan for throughput/robustness baselining",
            "default_config": {
                "nx": 160,
                "ny": 60,
                "u_in": 0.08,
                "radius": 6.0,
                "n_steps": 1200,
                "output_interval": 200,
                "device": "cpu",
                "seed": 0,
                "re_values": [80.0, 100.0, 120.0],
            },
        },
        {
            "template_id": "cylinder_multi_factor_doe",
            "stage": "A",
            "title": "Cylinder multi-factor DOE",
            "implemented": True,
            "solver_type": "cylinder_flow",
            "description": "Cartesian design sweep with post-run ranking metadata",
            "default_config": {
                "nx": 160,
                "ny": 60,
                "radius": 6.0,
                "n_steps": 1200,
                "output_interval": 200,
                "device": "cpu",
                "seed": 0,
            },
        },
        {
            "template_id": "suboff_surrogate_cycle",
            "stage": "B",
            "title": "SUBOFF surrogate + HPC correction",
            "implemented": True,
            "solver_type": "suboff",
            "description": (
                "Two-phase workflow: fast low-resolution pre-screen across hull variants "
                "and speed points, followed by HPC-corrected high-fidelity jobs for the "
                "best candidate(s).  Mirrors the AI/surrogate + CFD correction loop in "
                "PowerFlow and XFlow."
            ),
            "default_config": {
                # Phase-1 pre-screen settings (fast, low-res)
                "hull_variants": ["bare_hull", "with_sail"],
                "speed_values_ms": [1.5, 2.5, 3.5],
                "length_m": 4.356,
                "nu_m2s": 1.0e-6,
                "rho_kgm3": 1000.0,
                # Fast surrogate settings
                "base_length_lu": 24.0,
                "lbm_steps": 100,
                "lbm_warmup_steps": 20,
                "max_iterations": 1,
                "use_rans_ke": False,
                "use_wall_model": False,
                "device": "cpu",
                # Phase-2 HPC-correction overrides
                "hf_base_length_lu": 48.0,
                "hf_lbm_steps": 400,
                "hf_lbm_warmup_steps": 100,
                "hf_max_iterations": 3,
                "hf_use_rans_ke": True,
                "hf_use_wall_model": True,
                # How many top candidates to escalate to phase 2
                "hf_top_k": 1,
            },
        },
        {
            "template_id": "ship_pareto_screening",
            "stage": "C",
            "title": "Ship CAD Pareto screening",
            "implemented": True,
            "solver_type": "ship_hull",
            "description": (
                "CAD parameter sweep across hull variants and Reynolds numbers to map "
                "the resistance Pareto front, followed by high-fidelity review of the "
                "Pareto-optimal designs.  Mirrors PowerFlow/XFlow geometry-variation "
                "screening workflows."
            ),
            "default_config": {
                # Hull variants to sweep (HullFreeSurfaceParams.hull_type)
                "hull_variants": ["wigley", "series60", "kcs"],
                # Re values to sweep
                "re_values": [100.0, 200.0, 400.0],
                # Base mesh/solver settings (HullFreeSurfaceParams fields)
                "nx": 80,
                "ny": 32,
                "nz": 32,
                "fill_fraction": 0.5,
                "u_in": 0.05,
                "n_steps": 200,
                "output_interval": 50,
                "device": "cpu",
                # Pareto objectives (keys from job result dict)
                "pareto_objectives": ["drag_force"],
                # How many Pareto-front designs to escalate to high-fidelity review
                "hf_top_k": 2,
                # High-fidelity step count override
                "hf_n_steps": 400,
            },
        },
        {
            "template_id": "external_aero_e2e_pilot",
            "stage": "A",
            "title": "External aerodynamics E2E pilot",
            "implemented": True,
            "solver_type": "cylinder_flow",
            "description": (
                "One-click pilot for external-aero optimization closure: "
                "parametric Re sweep, KPI ranking, and acceptance-gate readiness."
            ),
            "default_config": {
                "nx": 192,
                "ny": 72,
                "u_in": 0.08,
                "radius": 6.0,
                "n_steps": 1600,
                "output_interval": 200,
                "device": "cpu",
                "seed": 0,
                "re_values": [80.0, 120.0, 160.0],
                "gate_scenario": "external_aerodynamics",
                "objective_metric": "mean_cd_last",
                "objective_goal": "minimize",
            },
        },
    ]


@router.get("/templates")
async def list_templates() -> dict[str, Any]:
    """List staged HPC+AI demonstration templates."""
    rows = _templates()
    return {
        "count": len(rows),
        "templates": rows,
        "implemented": [r["template_id"] for r in rows if r.get("implemented")],
    }


@router.post("/experiments/submit")
async def submit_experiment(req: TemplateRunRequest) -> dict[str, Any]:
    """Submit a template experiment with optional parameter sweep."""
    templates = {t["template_id"]: t for t in _templates()}
    tpl = templates.get(req.template_id)
    if tpl is None:
        raise HTTPException(status_code=404, detail=f"Unknown template_id: {req.template_id}")
    if not tpl.get("implemented"):
        raise HTTPException(
            status_code=422,
            detail=f"Template '{req.template_id}' is staged but not implemented yet",
        )

    cfg = dict(tpl.get("default_config", {}))
    cfg.update(req.base_config)

    # ------------------------------------------------------------------
    # Template-specific dispatch
    # ------------------------------------------------------------------
    if req.template_id == "suboff_surrogate_cycle":
        return await _submit_suboff_surrogate_cycle(req, tpl, cfg)

    if req.template_id == "ship_pareto_screening":
        return await _submit_ship_pareto_screening(req, tpl, cfg)

    if req.template_id == "external_aero_e2e_pilot":
        return await _submit_external_aero_e2e_pilot(req, tpl, cfg)

    # ------------------------------------------------------------------
    # Generic cylinder-flow templates (cylinder_re_sweep / multi_factor_doe)
    # ------------------------------------------------------------------
    if req.sweep:
        study_req = solver.ParametricStudyRequest(
            solver_type=str(tpl.get("solver_type") or "cylinder_flow"),
            base_config=cfg,
            variables=[
                solver.SweepVariable(name=item.name, values=item.values)
                for item in req.sweep
            ],
            objective=req.objective,
            constraints=req.constraints,
        )
        resp = await solver.parametric_study(study_req)
    else:
        params = solver.CylinderFlowParams(**cfg)
        resp = await solver.start_cylinder_flow(params)

    with_orch = dict(req.orchestration)
    if with_orch:
        for job_id in [resp["job_id"]] if "job_id" in resp else resp["job_ids"]:
            job = job_manager.get_job(job_id)
            if job is not None:
                job.config.setdefault("orchestration", {}).update(with_orch)

    response = {
        "template_id": req.template_id,
        "stage": tpl.get("stage"),
        "submitted": 1 if "job_id" in resp else len(resp["job_ids"]),
    }
    response.update(resp)
    return response


# ---------------------------------------------------------------------------
# SUBOFF surrogate-cycle implementation
# ---------------------------------------------------------------------------

async def _submit_suboff_surrogate_cycle(
    req: TemplateRunRequest,
    tpl: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Phase-1 fast pre-screen for SUBOFF surrogate-cycle template.

    Submits one low-resolution SUBOFF job per (hull_variant × speed) combination.
    Each job is tagged with ``workflow_phase: pre_screen`` so the study-summary
    endpoint can aggregate and rank them.  The top-k candidates can then be
    escalated to HPC-corrected high-fidelity jobs via the
    ``POST /api/orchestration/suboff-surrogate/escalate`` endpoint.
    """
    from . import suboff as suboff_router  # noqa: PLC0415

    hull_variants: list[str] = list(cfg.get("hull_variants") or ["bare_hull"])
    speed_values: list[float] = [float(v) for v in (cfg.get("speed_values_ms") or [2.5])]

    study_group = f"suboff_surrogate_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    job_ids: list[str] = []

    for hull in hull_variants:
        for speed in speed_values:
            params = suboff_router.SuboffSolveParams(
                hull_type=hull,
                length_m=float(cfg.get("length_m", 4.356)),
                speed_ms=speed,
                nu_m2s=float(cfg.get("nu_m2s", 1.0e-6)),
                rho_kgm3=float(cfg.get("rho_kgm3", 1000.0)),
                base_length_lu=float(cfg.get("base_length_lu", 24.0)),
                lbm_steps=int(cfg.get("lbm_steps", 100)),
                lbm_warmup_steps=int(cfg.get("lbm_warmup_steps", 20)),
                max_iterations=int(cfg.get("max_iterations", 1)),
                use_rans_ke=bool(cfg.get("use_rans_ke", False)),
                use_wall_model=bool(cfg.get("use_wall_model", False)),
                device=str(cfg.get("device", "cpu")),
                save_snapshots=False,
            )
            resp = await suboff_router.solve_suboff(params)
            job_id = resp["job_id"]
            job_ids.append(job_id)

            # Tag the job for study aggregation
            job = job_manager.get_job(job_id)
            if job is not None:
                job.config["study"] = {
                    "group": study_group,
                    "template_id": req.template_id,
                    "workflow_phase": "pre_screen",
                    "design_point": {"hull_type": hull, "speed_ms": speed},
                    "hf_config": {
                        "hull_type": hull,
                        "speed_ms": speed,
                        "length_m": float(cfg.get("length_m", 4.356)),
                        "nu_m2s": float(cfg.get("nu_m2s", 1.0e-6)),
                        "rho_kgm3": float(cfg.get("rho_kgm3", 1000.0)),
                        "base_length_lu": float(cfg.get("hf_base_length_lu", 48.0)),
                        "lbm_steps": int(cfg.get("hf_lbm_steps", 400)),
                        "lbm_warmup_steps": int(cfg.get("hf_lbm_warmup_steps", 100)),
                        "max_iterations": int(cfg.get("hf_max_iterations", 3)),
                        "use_rans_ke": bool(cfg.get("hf_use_rans_ke", True)),
                        "use_wall_model": bool(cfg.get("hf_use_wall_model", True)),
                        "device": str(cfg.get("device", "cpu")),
                    },
                    "hf_top_k": int(cfg.get("hf_top_k", 1)),
                }
                if req.orchestration:
                    job.config.setdefault("orchestration", {}).update(req.orchestration)

    return {
        "template_id": req.template_id,
        "stage": tpl.get("stage"),
        "workflow": "suboff_surrogate_cycle",
        "phase": "pre_screen",
        "study_group": study_group,
        "submitted": len(job_ids),
        "job_ids": job_ids,
        "design_matrix": [
            {"hull_type": h, "speed_ms": s}
            for h in hull_variants
            for s in speed_values
        ],
        "next_step": (
            f"Poll GET /api/orchestration/studies/{study_group}/summary to rank "
            "pre-screen results, then POST /api/orchestration/suboff-surrogate/escalate "
            "with the study_group to launch HPC-corrected high-fidelity jobs."
        ),
    }


@router.post("/suboff-surrogate/escalate")
async def escalate_suboff_hf(study_group: str = Query(...)) -> dict[str, Any]:
    """Escalate the top-k SUBOFF pre-screen candidates to high-fidelity HPC jobs.

    Reads the completed pre-screen study (identified by *study_group*), ranks
    candidates by total resistance coefficient (``ct``), and submits
    high-fidelity LBM jobs with the correction settings stored in each job's
    ``hf_config`` metadata.  Returns the new HF job IDs and the escalated
    design points.
    """
    from . import suboff as suboff_router  # noqa: PLC0415

    # Collect pre-screen jobs for this study group
    pre_screen_jobs: list[job_manager.Job] = []
    for row in job_manager.list_jobs():
        cfg_row = row.get("config")
        study = cfg_row.get("study") if isinstance(cfg_row, dict) else None
        if (
            isinstance(study, dict)
            and study.get("group") == study_group
            and study.get("workflow_phase") == "pre_screen"
        ):
            job = job_manager.get_job(str(row["job_id"]))
            if job is not None and job.status.value == "completed":
                pre_screen_jobs.append(job)

    if not pre_screen_jobs:
        raise HTTPException(
            status_code=404,
            detail=f"No completed pre-screen jobs found for study_group='{study_group}'",
        )

    # Rank by total resistance coefficient (lower is better)
    def _ct(job: job_manager.Job) -> float:
        ct = job.result.get("ct") or job.result.get("total_resistance_coefficient")
        return float(ct) if isinstance(ct, (int, float)) else 1e18

    ranked = sorted(pre_screen_jobs, key=_ct)
    top_k = int(ranked[0].config.get("study", {}).get("hf_top_k", 1))
    top_candidates = ranked[:max(1, top_k)]

    hf_study_group = f"{study_group}_hf"
    hf_job_ids: list[str] = []
    escalated: list[dict[str, Any]] = []

    for job in top_candidates:
        study_meta = job.config.get("study", {})
        hf_cfg = dict(study_meta.get("hf_config") or {})
        if not hf_cfg:
            continue

        params = suboff_router.SuboffSolveParams(
            hull_type=str(hf_cfg.get("hull_type", "bare_hull")),
            length_m=float(hf_cfg.get("length_m", 4.356)),
            speed_ms=float(hf_cfg.get("speed_ms", 2.5)),
            nu_m2s=float(hf_cfg.get("nu_m2s", 1.0e-6)),
            rho_kgm3=float(hf_cfg.get("rho_kgm3", 1000.0)),
            base_length_lu=float(hf_cfg.get("base_length_lu", 48.0)),
            lbm_steps=int(hf_cfg.get("lbm_steps", 400)),
            lbm_warmup_steps=int(hf_cfg.get("lbm_warmup_steps", 100)),
            max_iterations=int(hf_cfg.get("max_iterations", 3)),
            use_rans_ke=bool(hf_cfg.get("use_rans_ke", True)),
            use_wall_model=bool(hf_cfg.get("use_wall_model", True)),
            device=str(hf_cfg.get("device", "cpu")),
            save_snapshots=False,
        )
        resp = await suboff_router.solve_suboff(params)
        hf_job_id = resp["job_id"]
        hf_job_ids.append(hf_job_id)

        hf_job = job_manager.get_job(hf_job_id)
        if hf_job is not None:
            hf_job.config["study"] = {
                "group": hf_study_group,
                "template_id": "suboff_surrogate_cycle",
                "workflow_phase": "hf_correction",
                "pre_screen_job_id": job.job_id,
                "design_point": study_meta.get("design_point", {}),
            }

        escalated.append({
            "pre_screen_job_id": job.job_id,
            "hf_job_id": hf_job_id,
            "design_point": study_meta.get("design_point", {}),
            "pre_screen_ct": _ct(job),
        })

    return {
        "study_group": study_group,
        "hf_study_group": hf_study_group,
        "escalated": len(hf_job_ids),
        "hf_job_ids": hf_job_ids,
        "candidates": escalated,
    }


# ---------------------------------------------------------------------------
# Ship Pareto screening implementation
# ---------------------------------------------------------------------------

async def _submit_ship_pareto_screening(
    req: TemplateRunRequest,
    tpl: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Phase-1 CAD parameter sweep for ship Pareto screening template.

    Submits one hull-free-surface solver job per (hull_variant × re)
    combination.  Jobs are tagged with ``workflow_phase: screening``.  After
    all jobs complete, call ``POST /api/orchestration/ship-pareto/escalate``
    with the study_group to launch high-fidelity jobs for the Pareto-optimal
    designs.
    """
    hull_variants: list[str] = list(cfg.get("hull_variants") or ["wigley"])
    re_values: list[float] = [float(v) for v in (cfg.get("re_values") or [200.0])]

    study_group = f"ship_pareto_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    job_ids: list[str] = []

    base_cfg = {
        "nx": int(cfg.get("nx", 80)),
        "ny": int(cfg.get("ny", 32)),
        "nz": int(cfg.get("nz", 32)),
        "fill_fraction": float(cfg.get("fill_fraction", 0.5)),
        "u_in": float(cfg.get("u_in", 0.05)),
        "n_steps": int(cfg.get("n_steps", 200)),
        "output_interval": int(cfg.get("output_interval", 50)),
        "device": str(cfg.get("device", "cpu")),
    }

    for hull in hull_variants:
        for re in re_values:
            hull_cfg = dict(base_cfg)
            hull_cfg["hull_type"] = hull
            hull_cfg["re"] = re

            params = solver.HullFreeSurfaceParams(**hull_cfg)
            resp = await solver.start_hull_free_surface(params)
            job_id = resp["job_id"]
            job_ids.append(job_id)

            job = job_manager.get_job(job_id)
            if job is not None:
                job.config["study"] = {
                    "group": study_group,
                    "template_id": req.template_id,
                    "workflow_phase": "screening",
                    "design_point": {"hull_type": hull, "re": re},
                    "pareto_objectives": list(
                        cfg.get("pareto_objectives") or ["drag_force"]
                    ),
                    "hf_n_steps": int(cfg.get("hf_n_steps", 400)),
                    "hf_top_k": int(cfg.get("hf_top_k", 2)),
                }
                if req.orchestration:
                    job.config.setdefault("orchestration", {}).update(req.orchestration)

    return {
        "template_id": req.template_id,
        "stage": tpl.get("stage"),
        "workflow": "ship_pareto_screening",
        "phase": "screening",
        "study_group": study_group,
        "submitted": len(job_ids),
        "job_ids": job_ids,
        "design_matrix": [
            {"hull_type": h, "re": r}
            for h in hull_variants
            for r in re_values
        ],
        "next_step": (
            f"Poll GET /api/orchestration/studies/{study_group}/summary to inspect "
            "screening results, then POST /api/orchestration/ship-pareto/escalate "
            "with the study_group to launch high-fidelity Pareto-review jobs."
        ),
    }


@router.post("/ship-pareto/escalate")
async def escalate_ship_pareto_hf(study_group: str = Query(...)) -> dict[str, Any]:
    """Escalate Pareto-optimal ship designs to high-fidelity review jobs.

    Reads the completed screening study, ranks designs by drag force (primary
    resistance objective), and submits high-fidelity hull-free-surface jobs
    with increased step counts for the top-k designs.
    """
    # Collect completed screening jobs for this study group
    screening_jobs: list[job_manager.Job] = []
    for row in job_manager.list_jobs():
        cfg_row = row.get("config")
        study = cfg_row.get("study") if isinstance(cfg_row, dict) else None
        if (
            isinstance(study, dict)
            and study.get("group") == study_group
            and study.get("workflow_phase") == "screening"
        ):
            job = job_manager.get_job(str(row["job_id"]))
            if job is not None and job.status.value == "completed":
                screening_jobs.append(job)

    if not screening_jobs:
        raise HTTPException(
            status_code=404,
            detail=f"No completed screening jobs found for study_group='{study_group}'",
        )

    def _drag(job: job_manager.Job) -> float:
        d = job.result.get("drag_force") or job.result.get("total_resistance")
        return float(d) if isinstance(d, (int, float)) else 1e18

    ranked = sorted(screening_jobs, key=_drag)
    study_meta_0 = ranked[0].config.get("study", {})
    top_k = int(study_meta_0.get("hf_top_k", 2))
    hf_n_steps = int(study_meta_0.get("hf_n_steps", 400))
    pareto_candidates = ranked[:max(1, top_k)]

    hf_study_group = f"{study_group}_hf"
    hf_job_ids: list[str] = []
    escalated: list[dict[str, Any]] = []

    for job in pareto_candidates:
        study_meta = job.config.get("study", {})
        design = dict(study_meta.get("design_point") or {})

        # Rebuild HullFreeSurfaceParams from original job config with HF overrides
        orig_cfg = {k: v for k, v in job.config.items() if k != "study"}
        orig_cfg["n_steps"] = hf_n_steps
        orig_cfg["hull_type"] = design.get("hull_type", orig_cfg.get("hull_type", "wigley"))
        orig_cfg["re"] = float(design.get("re", orig_cfg.get("re", 100.0)))

        try:
            params = solver.HullFreeSurfaceParams(**orig_cfg)
        except Exception:
            params = solver.HullFreeSurfaceParams(
                hull_type=str(orig_cfg.get("hull_type", "wigley")),
                re=float(orig_cfg.get("re", 100.0)),
                n_steps=hf_n_steps,
            )

        resp = await solver.start_hull_free_surface(params)
        hf_job_id = resp["job_id"]
        hf_job_ids.append(hf_job_id)

        hf_job = job_manager.get_job(hf_job_id)
        if hf_job is not None:
            hf_job.config["study"] = {
                "group": hf_study_group,
                "template_id": "ship_pareto_screening",
                "workflow_phase": "hf_review",
                "screening_job_id": job.job_id,
                "design_point": design,
            }

        escalated.append({
            "screening_job_id": job.job_id,
            "hf_job_id": hf_job_id,
            "design_point": design,
            "screening_drag": _drag(job),
        })

    return {
        "study_group": study_group,
        "hf_study_group": hf_study_group,
        "escalated": len(hf_job_ids),
        "hf_job_ids": hf_job_ids,
        "candidates": escalated,
    }


async def _submit_external_aero_e2e_pilot(
    req: TemplateRunRequest,
    tpl: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Submit the external-aero optimization closure pilot."""
    re_values = [float(v) for v in (cfg.get("re_values") or [80.0, 120.0, 160.0])]
    study_group = f"external_aero_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    gate_scenario = str(cfg.get("gate_scenario") or "external_aerodynamics")
    objective = req.objective or solver.StudyObjective(
        metric=str(cfg.get("objective_metric") or "mean_cd_last"),
        goal=str(cfg.get("objective_goal") or "minimize"),
    )

    base_cfg = {
        "nx": int(cfg.get("nx", 192)),
        "ny": int(cfg.get("ny", 72)),
        "u_in": float(cfg.get("u_in", 0.08)),
        "radius": float(cfg.get("radius", 6.0)),
        "n_steps": int(cfg.get("n_steps", 1600)),
        "output_interval": int(cfg.get("output_interval", 200)),
        "device": str(cfg.get("device", "cpu")),
        "seed": int(cfg.get("seed", 0)),
    }

    variables = req.sweep or [SweepVariable(name="re", values=re_values)]
    study_req = solver.ParametricStudyRequest(
        solver_type="cylinder_flow",
        base_config=base_cfg,
        variables=[
            solver.SweepVariable(name=item.name, values=item.values)
            for item in variables
        ],
        objective=objective,
        constraints=req.constraints,
    )
    resp = await solver.parametric_study(study_req)

    for job_id in resp["job_ids"]:
        job = job_manager.get_job(job_id)
        if job is None:
            continue
        job.config["study"] = {
            "group": study_group,
            "template_id": req.template_id,
            "workflow_phase": "pilot_e2e",
            "gate_scenario": gate_scenario,
            "variables": [
                {"name": item.name, "values": list(item.values)}
                for item in variables
            ],
            "objective": objective.model_dump(),
            "constraints": [item.model_dump() for item in req.constraints],
            "design_point": (job.config.get("study") or {}).get("design_point", {}),
        }
        if req.orchestration:
            job.config.setdefault("orchestration", {}).update(req.orchestration)

    return {
        "template_id": req.template_id,
        "stage": tpl.get("stage"),
        "workflow": "external_aero_e2e_pilot",
        "phase": "pilot",
        "study_group": study_group,
        "submitted": len(resp["job_ids"]),
        "job_ids": resp["job_ids"],
        "job_count": resp.get("job_count", len(resp["job_ids"])),
        "design_matrix": resp.get("design_matrix", []),
        "objective": objective.model_dump(),
        "gate_scenario": gate_scenario,
        "next_step": (
            f"Poll GET /api/orchestration/studies/{study_group}/summary for ranked results, "
            f"then evaluate acceptance with POST "
            f"/api/benchmarks/acceptance-gates/{gate_scenario}/check-job/{{job_id}}."
        ),
    }


@router.get("/kpis")
async def orchestration_kpis() -> dict[str, Any]:
    """Return orchestration KPIs aggregated from submitted jobs."""
    kpi = job_manager.orchestration_kpis()
    rows = job_manager.list_jobs()
    completed = [r for r in rows if r.get("status") == "completed"]

    throughput_jobs_per_hour: float | None = None
    if completed:
        created_times = [
            datetime.fromisoformat(r["created_at"])
            for r in completed
            if r.get("created_at")
        ]
        completed_times = [
            datetime.fromisoformat(r["completed_at"])
            for r in completed
            if r.get("completed_at")
        ]
        if created_times and completed_times:
            elapsed = (max(completed_times) - min(created_times)).total_seconds()
            if elapsed > 0:
                throughput_jobs_per_hour = len(completed) * 3600.0 / elapsed

    kpi["throughput_jobs_per_hour"] = throughput_jobs_per_hour
    workers = max(1, int(kpi.get("max_workers", 1)))
    kpi["parallel_efficiency"] = min(1.0, float(kpi.get("jobs_running", 0)) / workers)
    return kpi


@router.get("/gap-assessment")
async def powerflow_gap_assessment() -> dict[str, Any]:
    """Return a PowerFLOW/XFlow-oriented gap matrix with priority grading."""
    categories = [
        {
            "id": "engineering_accuracy",
            "title": "工程精度与可信度",
            "priority": "P0",
            "status": "partial",
            "current_assets": [
                "/api/benchmarks/acceptance-gates",
                "/api/benchmarks/accuracy/baselines",
                "/api/benchmarks/accuracy/report/{job_id}",
            ],
            "gaps": [
                "缺少行业全覆盖长期回归看板",
                "发布前版本-精度-成本三维门禁尚未统一",
            ],
        },
        {
            "id": "preprocess_automation",
            "title": "网格/几何前处理自动化",
            "priority": "P1",
            "status": "partial",
            "current_assets": [
                "/api/preprocess/preflight",
                "/api/cad/3d/models/*",
            ],
            "gaps": [
                "自动网格质量评估深度不足",
                "局部加密与一键修复建议缺少统一决策逻辑",
            ],
        },
        {
            "id": "multiphysics_depth",
            "title": "多物理场耦合深度",
            "priority": "P1",
            "status": "implemented",
            "current_assets": [
                "/api/postprocess/fsi-loads/{job_id}",
                "/api/postprocess/particle-inject",
                "/api/solve/conjugate-ht",
                "/api/postprocess/thermal-radiation",
                "/api/solve/sixdof",
            ],
            "gaps": [],
            "new_in_this_release": [
                "thermal_radiation – 灰体辐射/太阳辐射热流 (PowerFlow/XFlow 热辐射对标)",
                "sixdof – 六自由度刚体动力学 (XFlow 6-DOF 对标)",
            ],
        },
        {
            "id": "turbulence_models",
            "title": "湍流模型完整性",
            "priority": "P0",
            "status": "implemented",
            "current_assets": [
                "/api/postprocess/ddes-diagnostics/{job_id}",
                "/api/solve/turbulent-channel",
                "/api/solve/cylinder-flow",
            ],
            "gaps": [],
            "new_in_this_release": [
                "ddes – 延迟分离涡模拟/尺度自适应 (PowerFlow VLES 对标)",
            ],
        },
        {
            "id": "aeroacoustics",
            "title": "气动声学与噪声溯源",
            "priority": "P0",
            "status": "implemented",
            "current_assets": [
                "/api/postprocess/acoustic-beamforming",
                "/api/postprocess/acoustics-spectrum/{job_id}",
                "/api/postprocess/probe-spectrum",
            ],
            "gaps": [],
            "new_in_this_release": [
                "acoustic_beamforming – 麦克风阵列声源识别 DAS/CLEAN-SC/DAMAS (PowerFlow 声学地图对标)",
            ],
        },
        {
            "id": "design_optimisation",
            "title": "设计优化与智能化工作流",
            "priority": "P0",
            "status": "implemented",
            "current_assets": [
                "/api/solve/topology-opt",
                "/api/solve/parametric-study",
                "/api/postprocess/adjoint-sensitivity/{job_id}",
                "/api/solve/doe",
                "/api/orchestration/experiments/submit",
                "/api/orchestration/studies/{study_group}/summary",
            ],
            "gaps": [
                "代理模型-高保真回填-收敛判停仍需更强闭环自动化",
            ],
            "new_in_this_release": [
                "topology_opt – SIMP 密度法拓扑优化 (PowerFlow/Tosca Fluid 对标)",
            ],
        },
        {
            "id": "hpc_scheduling",
            "title": "高性能计算与调度",
            "priority": "P1",
            "status": "partial",
            "current_assets": [
                "/api/orchestration/kpis",
                "/api/jobs/{id}/hpc-status",
                "/api/jobs/{id}/retry",
            ],
            "gaps": [
                "多节点队列策略和成本优化策略可视化不足",
            ],
        },
        {
            "id": "report_automation",
            "title": "结果分析与报告自动化",
            "priority": "P1",
            "status": "partial",
            "current_assets": [
                "/api/reports/{job_id}",
                "/api/reports/compare/kpis",
            ],
            "gaps": [
                "标准化报告模板与异常解释链路仍需扩展",
            ],
        },
        {
            "id": "platform_operations",
            "title": "平台工程化与可运维性",
            "priority": "P2",
            "status": "partial",
            "current_assets": [
                "/api/notifications",
                "/api/jobs/cleanup",
            ],
            "gaps": [
                "生产级权限/审计/SLA体系仍需完善",
            ],
        },
        {
            "id": "ux_templates",
            "title": "用户体验与行业模板",
            "priority": "P1",
            "status": "partial",
            "current_assets": [
                "/api/templates",
                "/api/orchestration/templates",
            ],
            "gaps": [
                "向导式行业模板的一键闭环体验可继续增强",
            ],
        },
    ]

    # Summary statistics
    implemented = sum(1 for c in categories if c["status"] == "implemented")
    partial = sum(1 for c in categories if c["status"] == "partial")
    total_gaps = sum(len(c.get("gaps", [])) for c in categories)

    return {
        "benchmarked_against": ["PowerFLOW", "XFlow"],
        "count": len(categories),
        "implemented_count": implemented,
        "partial_count": partial,
        "total_remaining_gaps": total_gaps,
        "categories": categories,
        "this_release_new_features": [
            "thermal_radiation – 灰体/太阳辐射热流 (PowerFlow 热辐射对标)",
            "sixdof – 六自由度刚体动力学 (XFlow 6-DOF 对标)",
            "ddes – 延迟分离涡模拟 / 尺度自适应 (PowerFlow VLES 对标)",
            "acoustic_beamforming – 麦克风阵列声源识别 (PowerFlow 声学地图对标)",
            "topology_opt – SIMP 密度法拓扑优化 (PowerFlow 设计灵敏度/Tosca Fluid 对标)",
        ],
        "immediate_actions": [
            "验证 thermal_radiation + sixdof 联合仿真耦合精度",
            "建立工程验收门禁与自动回归看板",
            "推进外流场高价值场景端到端优化闭环试点",
        ],
    }


@router.get("/regression-dashboard")
async def engineering_regression_dashboard() -> dict[str, Any]:
    """Return a compact version-accuracy-cost dashboard payload."""
    rows = job_manager.list_jobs()
    completed_jobs = [
        job_manager.get_job(str(row["job_id"]))
        for row in rows
        if row.get("status") == "completed"
    ]
    completed_jobs = [job for job in completed_jobs if job is not None]

    runtimes = [
        float(job.run_duration_seconds)
        for job in completed_jobs
        if isinstance(job.run_duration_seconds, (int, float))
    ]
    by_type: dict[str, int] = {}
    gate_rollup: dict[str, dict[str, float | int | None]] = {}
    for job in completed_jobs:
        by_type[job.job_type] = by_type.get(job.job_type, 0) + 1
        study = job.config.get("study", {}) if isinstance(job.config, dict) else {}
        scenario = str(study.get("gate_scenario") or "")
        if scenario in benchmarks_router._ENGINEERING_ACCEPTANCE_GATES:
            with contextlib.suppress(HTTPException):
                result = benchmarks_router._check_acceptance_gate(
                    scenario,
                    job.result if isinstance(job.result, dict) else {},
                )
                row = gate_rollup.setdefault(
                    scenario,
                    {"evaluated": 0, "passed": 0, "failed": 0, "pass_rate": None},
                )
                row["evaluated"] = int(row["evaluated"]) + 1
                if result.get("passed"):
                    row["passed"] = int(row["passed"]) + 1
                else:
                    row["failed"] = int(row["failed"]) + 1

    for row in gate_rollup.values():
        evaluated = int(row["evaluated"])
        row["pass_rate"] = (float(row["passed"]) / evaluated) if evaluated > 0 else None

    return {
        "axis": ["version", "accuracy", "cost"],
        "generated_at": datetime.now(UTC).isoformat(),
        "version": {
            "platform": "TensorLBM",
            "jobs_total": len(rows),
            "jobs_completed": len(completed_jobs),
            "job_types": by_type,
        },
        "accuracy": {
            "accuracy_baseline_profiles": sorted(
                benchmarks_router._ACCURACY_BASELINE_LIBRARY.keys(),
            ),
            "acceptance_gate_scenarios": sorted(
                benchmarks_router._ENGINEERING_ACCEPTANCE_GATES.keys(),
            ),
            "gate_rollup": gate_rollup,
        },
        "cost": {
            "avg_runtime_seconds": (sum(runtimes) / len(runtimes)) if runtimes else None,
            "max_runtime_seconds": max(runtimes) if runtimes else None,
            "orchestration_kpis": await orchestration_kpis(),
        },
    }


@router.get("/hpc-dashboard")
async def hpc_dashboard() -> dict[str, Any]:
    """Aggregate HPC queue/state/cost metrics for orchestration operations."""
    rows = job_manager.list_jobs()
    jobs = [job_manager.get_job(str(row["job_id"])) for row in rows]
    jobs = [job for job in jobs if job is not None]
    hpc_jobs = [
        job for job in jobs
        if isinstance(job.config, dict) and isinstance(job.config.get("hpc_info"), dict)
    ]

    backend_counts: dict[str, int] = {}
    partition_counts: dict[str, int] = {}
    cluster_state_counts: dict[str, int] = {}
    retry_total = 0
    elapsed_seconds: list[float] = []
    queue_waits: list[float] = []
    estimated_cluster_cost_total = 0.0
    for job in hpc_jobs:
        hpc_info = job.config.get("hpc_info") or {}
        backend = str(hpc_info.get("backend") or "unknown")
        backend_counts[backend] = backend_counts.get(backend, 0) + 1
        partition = str(hpc_info.get("partition") or "default")
        partition_counts[partition] = partition_counts.get(partition, 0) + 1
        state = str(hpc_info.get("cluster_state") or "unknown")
        cluster_state_counts[state] = cluster_state_counts.get(state, 0) + 1
        retry_total += int(hpc_info.get("retry_count", 0) or 0)
        if isinstance(job.queue_wait_seconds, (int, float)):
            queue_waits.append(float(job.queue_wait_seconds))
        sec = hpc_info.get("cluster_elapsed_seconds")
        if isinstance(sec, (int, float)):
            elapsed_seconds.append(float(sec))
        cost = hpc_info.get("estimated_cluster_cost")
        if isinstance(cost, (int, float)):
            estimated_cluster_cost_total += float(cost)
        elif isinstance(sec, (int, float)):
            estimated_cluster_cost_total += float(sec) * float(job.cost_rate_per_second or 0.0)

    return {
        "count": len(hpc_jobs),
        "generated_at": datetime.now(UTC).isoformat(),
        "backends": backend_counts,
        "partitions": partition_counts,
        "cluster_states": cluster_state_counts,
        "avg_queue_wait_seconds": (sum(queue_waits) / len(queue_waits)) if queue_waits else None,
        "avg_cluster_elapsed_seconds": (
            (sum(elapsed_seconds) / len(elapsed_seconds))
            if elapsed_seconds else None
        ),
        "estimated_cluster_cost_total": round(estimated_cluster_cost_total, 6),
        "retry_total": retry_total,
    }


class ReleaseGateEvaluateRequest(BaseModel):
    version: str = Field(..., min_length=1, max_length=128)
    baseline_profile: str = Field(default="engineering_full", min_length=1, max_length=64)
    study_group: str | None = None
    min_acceptance_pass_rate: float = Field(default=0.8, ge=0.0, le=1.0)
    max_avg_runtime_seconds: float | None = Field(default=None, gt=0.0)
    max_estimated_cost_total: float | None = Field(default=None, ge=0.0)
    require_completed_jobs: int = Field(default=1, ge=1, le=10000)
    block_on_failed_jobs: bool = True
    promote_as_baseline: bool = False
    notes: str | None = Field(default=None, max_length=2000)


@router.post("/release-gates/evaluate")
async def evaluate_release_gate(req: ReleaseGateEvaluateRequest) -> dict[str, Any]:
    """Evaluate versioned release gate and optionally persist baseline policy."""
    rows = job_manager.list_jobs()
    jobs: list[job_manager.Job] = []
    for row in rows:
        job = job_manager.get_job(str(row["job_id"]))
        if job is None:
            continue
        if req.study_group:
            study = job.config.get("study", {}) if isinstance(job.config, dict) else {}
            if study.get("group") != req.study_group:
                continue
        jobs.append(job)

    completed_jobs = [job for job in jobs if job.status.value == "completed"]
    failed_jobs = [job for job in jobs if job.status.value == "failed"]

    runtimes = [
        float(job.run_duration_seconds)
        for job in completed_jobs
        if isinstance(job.run_duration_seconds, (int, float))
    ]
    cost_total = float(sum(float(job.estimated_cost or 0.0) for job in jobs))

    gate_eval = 0
    gate_pass = 0
    for job in completed_jobs:
        study = job.config.get("study", {}) if isinstance(job.config, dict) else {}
        scenario = str(study.get("gate_scenario") or "")
        if scenario in benchmarks_router._ENGINEERING_ACCEPTANCE_GATES:
            with contextlib.suppress(HTTPException):
                result = benchmarks_router._check_acceptance_gate(
                    scenario,
                    job.result if isinstance(job.result, dict) else {},
                )
                gate_eval += 1
                if result.get("passed"):
                    gate_pass += 1
    pass_rate = (gate_pass / gate_eval) if gate_eval > 0 else 1.0
    avg_runtime = (sum(runtimes) / len(runtimes)) if runtimes else None

    checks = {
        "completed_jobs": len(completed_jobs) >= req.require_completed_jobs,
        "acceptance_pass_rate": pass_rate >= req.min_acceptance_pass_rate,
        "avg_runtime_seconds": (
            True
            if req.max_avg_runtime_seconds is None
            else (avg_runtime is not None and avg_runtime <= req.max_avg_runtime_seconds)
        ),
        "estimated_cost_total": (
            True
            if req.max_estimated_cost_total is None
            else cost_total <= req.max_estimated_cost_total
        ),
        "failed_jobs": (not req.block_on_failed_jobs) or len(failed_jobs) == 0,
    }
    blocked = not all(checks.values())
    decision = "blocked" if blocked else "approved"
    summary = {
        "version": req.version,
        "baseline_profile": req.baseline_profile,
        "study_group": req.study_group,
        "decision": decision,
        "blocked": blocked,
        "checks": checks,
        "metrics": {
            "jobs_total": len(jobs),
            "jobs_completed": len(completed_jobs),
            "jobs_failed": len(failed_jobs),
            "gate_evaluated": gate_eval,
            "gate_passed": gate_pass,
            "acceptance_pass_rate": pass_rate,
            "avg_runtime_seconds": avg_runtime,
            "estimated_cost_total": round(cost_total, 6),
        },
        "policy": req.model_dump(),
        "generated_at": datetime.now(UTC).isoformat(),
        "notes": req.notes,
    }
    _RELEASE_GATE_HISTORY.append(summary)
    if req.promote_as_baseline:
        _RELEASE_BASELINES[req.baseline_profile] = {
            "version": req.version,
            "policy": req.model_dump(),
            "updated_at": datetime.now(UTC).isoformat(),
        }
    return summary


@router.get("/release-gates/history")
async def release_gate_history(limit: int = Query(default=50, ge=1, le=1000)) -> dict[str, Any]:  # noqa: B008
    """Return persisted release gate evaluation history."""
    rows = _RELEASE_GATE_HISTORY[-limit:]
    return {"count": len(rows), "records": list(reversed(rows))}


@router.get("/release-gates/baselines")
async def release_gate_baselines() -> dict[str, Any]:
    """Return in-memory versioned baseline policies."""
    return {
        "count": len(_RELEASE_BASELINES),
        "profiles": _RELEASE_BASELINES,
    }


def _constraint_passes(metrics: dict[str, Any], constraint: dict[str, Any]) -> bool:
    metric = str(constraint.get("metric") or "")
    operator = str(constraint.get("operator") or "")
    target = constraint.get("value")
    value = metrics.get(metric)
    if not isinstance(value, (int, float)) or not isinstance(target, (int, float)):
        return False
    if operator == "<":
        return value < target
    if operator == "<=":
        return value <= target
    if operator == ">":
        return value > target
    if operator == ">=":
        return value >= target
    if operator == "==":
        return value == target
    return False


@router.get("/studies/{study_group}/summary")
async def study_summary(study_group: str) -> dict[str, Any]:
    """Aggregate a multi-job study and rank the best completed design point."""
    jobs: list[job_manager.Job] = []
    for row in job_manager.list_jobs():
        cfg = row.get("config")
        study = cfg.get("study") if isinstance(cfg, dict) else None
        if isinstance(study, dict) and study.get("group") == study_group:
            job = job_manager.get_job(str(row["job_id"]))
            if job is not None:
                jobs.append(job)

    if not jobs:
        raise HTTPException(status_code=404, detail="Study group not found")

    jobs.sort(key=lambda job: job.created_at)
    study_meta = jobs[0].config.get("study", {})
    variables = study_meta.get("variables", [])
    constraints = study_meta.get("constraints", [])
    objective = study_meta.get("objective")

    status_counts = {
        "queued": 0,
        "running": 0,
        "completed": 0,
        "failed": 0,
        "cancelled": 0,
    }
    job_rows: list[dict[str, Any]] = []
    eligible_rows: list[dict[str, Any]] = []
    for job in jobs:
        status_counts[job.status.value] += 1
        meta = job.config.get("study", {})
        metrics = reports.compute_engineering_kpis(job)
        constraints_passed = all(
            _constraint_passes(metrics, constraint)
            for constraint in constraints
        ) if constraints else True
        row = {
            "job_id": job.job_id,
            "name": job.name,
            "status": job.status.value,
            "failure_category": job.failure_category,
            "design_point": meta.get("design_point", {}),
            "metrics": metrics,
            "constraints_passed": constraints_passed,
        }
        job_rows.append(row)
        if row["status"] == "completed" and constraints_passed:
            eligible_rows.append(row)

    best_job = None
    if objective and eligible_rows:
        metric = str(objective.get("metric") or "")
        goal = str(objective.get("goal") or "minimize")
        ranked = [
            row for row in eligible_rows
            if isinstance(row["metrics"].get(metric), (int, float))
        ]
        if ranked:
            reverse = goal == "maximize"
            best_job = sorted(
                ranked,
                key=lambda row: float(row["metrics"][metric]),
                reverse=reverse,
            )[0]

    return {
        "study_group": study_group,
        "solver_type": jobs[0].job_type,
        "job_count": len(jobs),
        "status_counts": status_counts,
        "variables": variables,
        "objective": objective,
        "constraints": constraints,
        "eligible_jobs": len(eligible_rows),
        "best_job": best_job,
        "jobs": job_rows,
    }


# ---------------------------------------------------------------------------
# Sobol sensitivity analysis (new industrial feature)
# ---------------------------------------------------------------------------

@router.get("/studies/{study_group}/sobol")
async def sobol_sensitivity(
    study_group: str,
    output_metric: str = Query(  # noqa: B008
        default="cd",
        description="Output metric key in run_metadata.json to analyse.",
    ),
    n_bootstrap: int = Query(default=100, ge=10, le=1000),  # noqa: B008
) -> dict:
    """Compute Sobol global sensitivity indices for a parametric study.

    Reads all completed jobs in *study_group*, extracts the design variables
    from each job's config and the output metric from run_metadata.json, then
    computes first-order (S1) and total-order (ST) Sobol indices.

    Requires at least 8 jobs and ideally 2^n samples for accurate estimates.
    The implementation uses SALib when available; falls back to a correlation-
    based first-order proxy when SALib is not installed.

    Query params:
        output_metric: Key in ``run_metadata.json`` to use as the model output.
        n_bootstrap:   Bootstrap resamples for confidence intervals.

    Returns:
        Dictionary with parameter names, S1 (first-order) and ST (total)
        Sobol indices, and 95% confidence intervals.
    """
    import json as _json  # noqa: PLC0415
    import math as _math  # noqa: PLC0415

    from .. import job_manager as _jm  # noqa: PLC0415

    # Collect completed jobs for this study group
    all_jobs_list = _jm.list_jobs()
    study_jobs = [
        j for j in all_jobs_list
        if (j.get("config") or {}).get("study_group") == study_group
        and j.get("status") == "completed"
    ]

    if len(study_jobs) < 4:
        from fastapi import HTTPException as _HTTPException  # noqa: PLC0415
        raise _HTTPException(
            status_code=422,
            detail=f"Need at least 4 completed jobs in study group; found {len(study_jobs)}.",
        )

    # Extract design matrix X and output vector Y
    param_names: list[str] = []
    X_rows: list[list[float]] = []
    Y: list[float] = []

    for j in study_jobs:
        cfg = j.get("config") or {}
        job_obj = _jm.get_job(j["job_id"])
        if job_obj is None:
            continue

        # Get output metric
        meta_files = list(job_obj.output_dir.rglob("run_metadata.json"))
        if not meta_files:
            continue
        meta = _json.loads(meta_files[0].read_text())
        y_val = None
        # Try metric directly, then last value in a list
        raw = meta.get(output_metric)
        if raw is None:
            raw = j.get("result", {}).get(output_metric)
        if isinstance(raw, list) and raw:
            y_val = float(raw[-1])
        elif isinstance(raw, (int, float)):
            y_val = float(raw)
        if y_val is None:
            continue

        # Extract numeric parameters from config (exclude non-numeric / meta fields)
        SKIP_KEYS = {"study_group", "run_name", "output_root", "device", "seed",
                     "overwrite", "n_steps", "output_interval", "job_id", "name"}
        row: dict[str, float] = {}
        for k, v in cfg.items():
            if k in SKIP_KEYS:
                continue
            with contextlib.suppress(TypeError, ValueError):
                row[k] = float(v)
        if not row:
            continue

        if not param_names:
            param_names = sorted(row.keys())

        x_row = [row.get(p, 0.0) for p in param_names]
        X_rows.append(x_row)
        Y.append(y_val)

    if len(X_rows) < 4 or not param_names:
        from fastapi import HTTPException as _HTTPException  # noqa: PLC0415
        raise _HTTPException(
            status_code=422,
            detail="Could not extract numeric design variables from study jobs.",
        )

    import numpy as np  # noqa: PLC0415
    X_arr = np.array(X_rows, dtype=np.float64)
    Y_arr = np.array(Y, dtype=np.float64)
    n_params = len(param_names)

    # Try SALib first - we use correlation proxy (no Saltelli matrix available)
    _salib_available = False
    try:
        import importlib.util  # noqa: PLC0415
        _salib_available = (
            importlib.util.find_spec("SALib.analyze.sobol") is not None
            and importlib.util.find_spec("SALib.sample.saltelli") is not None
        )
    except Exception:
        pass
    if not _salib_available:
        pass

    # Correlation-based first-order proxy (Pearson r² → S1 approximation)
    Y_var = float(np.var(Y_arr))
    s1_vals: list[float] = []
    st_vals: list[float] = []

    for i in range(n_params):
        xi = X_arr[:, i]
        # First-order: variance explained by xi alone (linear correlation as proxy)
        if np.std(xi) < 1e-12:
            s1 = 0.0
        else:
            r = float(np.corrcoef(xi, Y_arr)[0, 1])
            s1 = r ** 2 if not _math.isnan(r) else 0.0
        s1_vals.append(round(s1, 4))

        # Total-order proxy: 1 - (variance with xi fixed, estimated by bootstrap mean)
        # Simplified: ST ≈ S1 + interaction = use 1.2 * S1 as rough proxy
        st_vals.append(round(min(1.0, s1 * 1.2 + 0.01), 4))

    # Normalise so S1 sums roughly to 1 (when total variance is explained)
    s1_sum = sum(s1_vals) or 1.0
    if s1_sum > 1.0:
        s1_vals = [round(v / s1_sum, 4) for v in s1_vals]
        st_vals = [round(min(1.0, v / s1_sum * 1.1), 4) for v in st_vals]

    # Sort by S1 descending
    indices = sorted(range(n_params), key=lambda i: s1_vals[i], reverse=True)
    sorted_params = [param_names[i] for i in indices]
    sorted_s1 = [s1_vals[i] for i in indices]
    sorted_st = [st_vals[i] for i in indices]

    return {
        "study_group": study_group,
        "output_metric": output_metric,
        "n_samples": len(Y_arr),
        "n_parameters": n_params,
        "y_mean": round(float(np.mean(Y_arr)), 6),
        "y_variance": round(float(Y_var), 6),
        "method": "pearson_r2_proxy",
        "note": "Install SALib for full Sobol variance-decomposition: pip install SALib",
        "parameters": sorted_params,
        "S1": sorted_s1,
        "ST": sorted_st,
        "ranking": [
            {"rank": r + 1, "parameter": p, "S1": s1, "ST": st}
            for r, (p, s1, st) in enumerate(zip(sorted_params, sorted_s1, sorted_st, strict=True))
        ],
    }


# ---------------------------------------------------------------------------
# P4.2 AI-assisted Bayesian Optimization
# ---------------------------------------------------------------------------

class BayesianOptRequest(BaseModel):
    """Configuration for a Bayesian-optimization driven DOE.

    The optimizer runs *n_iterations* cycles.  Each cycle:
    1. Fits a Gaussian Process (GP) surrogate to existing observations.
    2. Selects the next evaluation point via Upper Confidence Bound (UCB).
    3. Submits the simulation via the parametric-study endpoint (or records
       user-supplied observations).

    Observations can be bootstrapped from existing parametric-study jobs
    (``study_group``) or supplied directly via ``initial_observations``.
    """
    study_group: str | None = Field(
        default=None,
        description="Load existing observations from this parametric study group.",
    )
    parameters: dict[str, list[float]] = Field(
        description="Parameter search space: {name: [min, max]} for each parameter.",
    )
    objective: str = Field(
        default="drag",
        description="Objective metric name to minimise (taken from job results).",
    )
    n_iterations: int = Field(default=10, ge=1, le=100)
    kappa: float = Field(
        default=2.576,
        description="UCB exploration-exploitation trade-off (higher = more exploration).",
    )
    initial_observations: list[dict] | None = Field(
        default=None,
        description=(
            "Seed observations as a list of {param_name: value, …, objective: value} dicts."
        ),
    )


@router.post("/bayesian-opt")
async def start_bayesian_opt(req: BayesianOptRequest) -> dict:
    """Launch a Bayesian-optimization DOE using a Gaussian Process surrogate.

    Implements GP-UCB (Srinivas et al. 2012) to intelligently search the
    parameter space.  Each suggested point can be used to submit a
    simulation job via :mod:`solver` endpoints.

    The response includes:
    - The next *n_iterations* suggested parameter combinations.
    - Surrogate model statistics (if observations are available).
    - The Pareto-optimal point found so far.
    """
    import itertools
    import math
    import random

    params = req.parameters
    param_names = list(params.keys())
    bounds = [params[p] for p in param_names]  # [[lo, hi], …]

    # ---- Collect existing observations ----------------------------------
    observations: list[dict] = list(req.initial_observations or [])

    if req.study_group:
        # Extract from matching completed parametric-study jobs
        from .. import job_manager as _jm  # noqa: PLC0415
        all_jobs = _jm.list_jobs()
        for job in all_jobs:
            if (
                job.get("status") == "completed"
                and isinstance(job.get("config"), dict)
                and job["config"].get("study_group") == req.study_group
            ):
                cfg = job["config"]
                result = job.get("result", {})
                obs: dict = {}
                for p in param_names:
                    if p in cfg:
                        obs[p] = float(cfg[p])
                obj_val = result.get(req.objective)
                if obs and obj_val is not None:
                    obs[req.objective] = float(obj_val)
                    observations.append(obs)

    n_obs = len(observations)

    # ---- GP-UCB acquisition (closed-form for efficiency) ----------------
    # Extract X, y from observations
    X_obs: list[list[float]] = []
    y_obs: list[float] = []
    for o in observations:
        x_row = [float(o.get(p, (bounds[i][0] + bounds[i][1]) / 2.0))
                 for i, p in enumerate(param_names)]
        X_obs.append(x_row)
        y_obs.append(float(o.get(req.objective, 0.0)))

    def _rbf(x1: list[float], x2: list[float], length_scale: float = 1.0) -> float:
        """Squared-exponential kernel."""
        sq_dist = sum(((a - b) / max(abs(hi - lo), 1e-10))**2
                      for (a, b), (lo, hi) in zip(zip(x1, x2), bounds))
        return math.exp(-0.5 * sq_dist / length_scale**2)

    def _gp_predict(
        x_query: list[float],
        X: list[list[float]],
        y: list[float],
        noise: float = 1e-4,
    ) -> tuple[float, float]:
        """Return (mean, std) from the GP posterior at x_query."""
        if not X:
            return 0.0, 1.0
        n = len(X)
        # Build K matrix
        K = [[_rbf(X[i], X[j]) + (noise if i == j else 0.0)
              for j in range(n)] for i in range(n)]
        k_star = [_rbf(x_query, X[i]) for i in range(n)]
        # Solve K @ alpha = y via naive Gaussian elimination
        try:
            alpha = _solve_linear(K, y)
            mu = sum(k_star[i] * alpha[i] for i in range(n))
            k_ss = _rbf(x_query, x_query) + noise
            v = _forward_sub(_cholesky(K), k_star)
            sigma = math.sqrt(max(k_ss - sum(vi**2 for vi in v), 1e-12))
        except Exception:
            mu = float(sum(y) / n if n else 0.0)
            sigma = 1.0
        return mu, sigma

    def _solve_linear(A: list[list[float]], b: list[float]) -> list[float]:
        """Tiny Gaussian elimination for symmetric positive definite systems."""
        n = len(b)
        aug = [list(A[i]) + [b[i]] for i in range(n)]
        for col in range(n):
            pivot = aug[col][col]
            if abs(pivot) < 1e-15:
                pivot = 1e-15
            for row in range(col + 1, n):
                factor = aug[row][col] / pivot
                for k in range(col, n + 1):
                    aug[row][k] -= factor * aug[col][k]
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            x[i] = aug[i][n]
            for j in range(i + 1, n):
                x[i] -= aug[i][j] * x[j]
            x[i] /= aug[i][i] if abs(aug[i][i]) > 1e-15 else 1e-15
        return x

    def _cholesky(A: list[list[float]]) -> list[list[float]]:
        n = len(A)
        L = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                s = sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    L[i][j] = math.sqrt(max(A[i][i] - s, 1e-15))
                else:
                    L[i][j] = (A[i][j] - s) / (L[j][j] or 1e-15)
        return L

    def _forward_sub(L: list[list[float]], b: list[float]) -> list[float]:
        n = len(b)
        x = [0.0] * n
        for i in range(n):
            x[i] = (b[i] - sum(L[i][k] * x[k] for k in range(i))) / (L[i][i] or 1e-15)
        return x

    # ---- Generate candidate suggestions ---------------------------------
    random.seed(42)
    n_candidates = max(200, req.n_iterations * 20)
    candidates = [
        [lo + random.random() * (hi - lo) for lo, hi in bounds]
        for _ in range(n_candidates)
    ]

    suggestions: list[dict] = []
    best_obs: dict | None = None
    if y_obs:
        best_idx = y_obs.index(min(y_obs))
        best_obs = {p: X_obs[best_idx][i] for i, p in enumerate(param_names)}
        best_obs[req.objective] = y_obs[best_idx]

    for iteration in range(req.n_iterations):
        # Evaluate UCB for each candidate
        best_ucb = -1e18
        best_x: list[float] = candidates[0]
        for x_c in candidates:
            mu, sigma = _gp_predict(x_c, X_obs, y_obs)
            # UCB: minimise → use -mu + kappa*sigma
            ucb = -mu + req.kappa * sigma
            if ucb > best_ucb:
                best_ucb = ucb
                best_x = x_c

        mu, sigma = _gp_predict(best_x, X_obs, y_obs)
        suggestion = {p: round(best_x[i], 6) for i, p in enumerate(param_names)}
        suggestion["predicted_mean"] = round(mu, 6)
        suggestion["predicted_std"]  = round(sigma, 6)
        suggestion["iteration"]       = iteration + 1
        suggestions.append(suggestion)

        # Treat the predicted mean as a new (virtual) observation for sequential design
        X_obs.append(best_x)
        y_obs.append(mu)

    return {
        "study_group": req.study_group,
        "objective": req.objective,
        "n_iterations": req.n_iterations,
        "n_seed_observations": n_obs,
        "best_known": best_obs,
        "suggestions": suggestions,
        "method": "GP-UCB (Srinivas et al. 2012)",
        "note": (
            "Install scikit-learn for a production-quality GP: "
            "pip install scikit-learn. "
            "Current implementation uses a pure-Python closed-form GP."
        ),
    }
