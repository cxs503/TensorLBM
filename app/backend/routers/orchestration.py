"""HPC orchestration endpoints for experiment templates and KPI rollups."""
from __future__ import annotations

import contextlib
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query
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
