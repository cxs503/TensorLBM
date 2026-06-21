"""HPC orchestration endpoints for experiment templates and KPI rollups."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .. import job_manager
from . import reports, solver

router = APIRouter()


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
            "implemented": False,
            "description": "AI pre-screen + HPC correction workflow scaffold",
        },
        {
            "template_id": "ship_pareto_screening",
            "stage": "C",
            "title": "Ship CAD Pareto screening",
            "implemented": False,
            "description": "CAD parameter sweep + surrogate ranking + high-fidelity review",
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
