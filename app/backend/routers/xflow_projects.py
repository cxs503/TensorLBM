"""XFlow-style Project/Case/Run hierarchy.

Models the workflow:
    Project (e.g. "DTMB-5415 resistance study")
        └── Case (e.g. "Fr=0.28 baseline")
                └── Run (a single simulation execution)

Persists to a JSON file under TENSORLBM_PROJECTS_DIR (default: /root/tensorlbm_projects).
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/projects", tags=["Projects"])

_PROJECTS_DIR = Path(os.environ.get("TENSORLBM_PROJECTS_DIR", "/root/tensorlbm_projects"))
_PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _load() -> dict[str, Any]:
    f = _PROJECTS_DIR / "index.json"
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            return {"projects": {}, "cases": {}, "runs": {}}
    return {"projects": {}, "cases": {}, "runs": {}}


def _save(idx: dict[str, Any]) -> None:
    f = _PROJECTS_DIR / "index.json"
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(idx, indent=2, ensure_ascii=False))
    tmp.replace(f)


# ── Schemas ─────────────────────────────────────────────────────────────────

class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = ""
    tags: list[str] = Field(default_factory=list)


class CaseCreate(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    reference_image: str | None = None


class RunCreate(BaseModel):
    name: str = Field(..., min_length=1)
    job_id: str | None = None  # bind to an existing simulation job
    config: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.get("")
def list_projects():
    with _LOCK:
        idx = _load()
    return {"projects": list(idx["projects"].values())}


@router.post("")
def create_project(req: ProjectCreate):
    with _LOCK:
        idx = _load()
        pid = f"proj_{uuid.uuid4().hex[:8]}"
        idx["projects"][pid] = {
            "id": pid, "name": req.name, "description": req.description,
            "tags": req.tags, "created_at": _now(), "updated_at": _now(),
            "case_ids": [],
        }
        _save(idx)
    return idx["projects"][pid]


@router.get("/{project_id}")
def get_project(project_id: str):
    with _LOCK:
        idx = _load()
    if project_id not in idx["projects"]:
        raise HTTPException(404, "Project not found")
    proj = idx["projects"][project_id]
    cases = [idx["cases"][c] for c in proj["case_ids"] if c in idx["cases"]]
    return {"project": proj, "cases": cases}


@router.delete("/{project_id}")
def delete_project(project_id: str):
    with _LOCK:
        idx = _load()
        if project_id not in idx["projects"]:
            raise HTTPException(404, "Project not found")
        for case_id in idx["projects"][project_id].get("case_ids", []):
            for run_id in idx["cases"].get(case_id, {}).get("run_ids", []):
                idx["runs"].pop(run_id, None)
            idx["cases"].pop(case_id, None)
        idx["projects"].pop(project_id)
        _save(idx)
    return {"deleted": project_id}


@router.post("/{project_id}/cases")
def create_case(project_id: str, req: CaseCreate):
    with _LOCK:
        idx = _load()
        if project_id not in idx["projects"]:
            raise HTTPException(404, "Project not found")
        cid = f"case_{uuid.uuid4().hex[:8]}"
        idx["cases"][cid] = {
            "id": cid, "project_id": project_id, "name": req.name,
            "description": req.description, "parameters": req.parameters,
            "reference_image": req.reference_image, "created_at": _now(),
            "updated_at": _now(), "run_ids": [],
        }
        idx["projects"][project_id]["case_ids"].append(cid)
        idx["projects"][project_id]["updated_at"] = _now()
        _save(idx)
    return idx["cases"][cid]


@router.get("/{project_id}/cases")
def list_cases(project_id: str):
    with _LOCK:
        idx = _load()
    if project_id not in idx["projects"]:
        raise HTTPException(404, "Project not found")
    case_ids = idx["projects"][project_id].get("case_ids", [])
    return {"cases": [idx["cases"][c] for c in case_ids if c in idx["cases"]]}


@router.get("/cases/{case_id}")
def get_case(case_id: str):
    with _LOCK:
        idx = _load()
    if case_id not in idx["cases"]:
        raise HTTPException(404, "Case not found")
    case = idx["cases"][case_id]
    runs = [idx["runs"][r] for r in case["run_ids"] if r in idx["runs"]]
    return {"case": case, "runs": runs}


@router.post("/cases/{case_id}/runs")
def create_run(case_id: str, req: RunCreate):
    with _LOCK:
        idx = _load()
        if case_id not in idx["cases"]:
            raise HTTPException(404, "Case not found")
        rid = f"run_{uuid.uuid4().hex[:8]}"
        idx["runs"][rid] = {
            "id": rid, "case_id": case_id, "name": req.name,
            "job_id": req.job_id, "config": req.config,
            "notes": req.notes, "status": "submitted",
            "created_at": _now(),
        }
        idx["cases"][case_id]["run_ids"].append(rid)
        idx["cases"][case_id]["updated_at"] = _now()
        _save(idx)
    return idx["runs"][rid]


@router.get("/runs/{run_id}")
def get_run(run_id: str):
    with _LOCK:
        idx = _load()
    if run_id not in idx["runs"]:
        raise HTTPException(404, "Run not found")
    return idx["runs"][run_id]


@router.post("/compare")
def compare_runs(run_ids: list[str]):
    """Side-by-side comparison of multiple runs (for design-space exploration)."""
    with _LOCK:
        idx = _load()
    rows = [idx["runs"][r] for r in run_ids if r in idx["runs"]]
    return {"runs": rows}