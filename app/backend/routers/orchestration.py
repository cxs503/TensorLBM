"""HPC orchestration endpoints for experiment templates and KPI rollups."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .. import job_manager
from . import solver

router = APIRouter()


class SweepVariable(BaseModel):
    name: str = Field(..., min_length=1)
    values: list[float] = Field(..., min_length=1, max_length=40)


class TemplateRunRequest(BaseModel):
    template_id: str
    base_config: dict[str, Any] = Field(default_factory=dict)
    sweep: list[SweepVariable] = Field(default_factory=list)
    orchestration: dict[str, Any] = Field(default_factory=dict)


def _templates() -> list[dict[str, Any]]:
    return [
        {
            "template_id": "cylinder_re_sweep",
            "stage": "A",
            "title": "Cylinder Reynolds sweep",
            "implemented": True,
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
        sweep_map = {item.name: item.values for item in req.sweep}
        if set(sweep_map) != {"re"}:
            raise HTTPException(
                status_code=422,
                detail="Current implementation supports only 're' sweep variable",
            )
        cfg["re_values"] = [float(v) for v in sweep_map["re"]]

    params = solver.CylinderFlowScanParams(**cfg)
    resp = await solver.start_cylinder_flow_scan(params)

    with_orch = dict(req.orchestration)
    if with_orch:
        for job_id in resp["job_ids"]:
            job = job_manager.get_job(job_id)
            if job is not None:
                job.config.setdefault("orchestration", {}).update(with_orch)

    return {
        "template_id": req.template_id,
        "stage": tpl.get("stage"),
        "submitted": len(resp["job_ids"]),
        "scan_group": resp["scan_group"],
        "job_ids": resp["job_ids"],
        "parameter": resp["parameter"],
        "values": resp["values"],
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
