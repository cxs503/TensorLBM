"""Project and simulation-case management for the PowerFlow platform.

Provides a project → case hierarchy that mirrors the workflow-centric
organisation used by commercial CFD tools (PowerFLOW / XFlow).

Storage is an in-process SQLite database (via the stdlib ``sqlite3`` module)
so no extra dependencies are needed.  The database is persisted in the
``TENSORLBM_OUTPUT_ROOT`` directory (or ``/tmp/tensorlbm_platform`` by
default) as ``projects.db``.
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

_OUTPUT_ROOT = Path(os.environ.get("TENSORLBM_OUTPUT_ROOT", "/tmp/tensorlbm_platform"))
_DB_PATH = _OUTPUT_ROOT / "projects.db"


def _get_conn() -> sqlite3.Connection:
    _OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS projects (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            owner       TEXT NOT NULL DEFAULT '',
            tags        TEXT NOT NULL DEFAULT '[]',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS cases (
            id          TEXT PRIMARY KEY,
            project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            name        TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            scenario    TEXT NOT NULL DEFAULT 'custom',
            status      TEXT NOT NULL DEFAULT 'draft',
            config      TEXT NOT NULL DEFAULT '{}',
            job_id      TEXT,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );
        """
    )
    conn.commit()


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    for key in ("tags", "config"):
        if key in d and isinstance(d[key], str):
            try:
                d[key] = json.loads(d[key])
            except (json.JSONDecodeError, TypeError):
                pass
    return d


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    description: str = Field("", max_length=1000)
    owner: str = Field("", max_length=120)
    tags: list[str] = Field(default_factory=list)


class ProjectUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=120)
    description: str | None = Field(None, max_length=1000)
    owner: str | None = Field(None, max_length=120)
    tags: list[str] | None = None


class CaseCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    description: str = Field("", max_length=1000)
    scenario: str = Field("custom", max_length=80)
    config: dict[str, Any] = Field(default_factory=dict)


class CaseUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=120)
    description: str | None = Field(None, max_length=1000)
    scenario: str | None = Field(None, max_length=80)
    status: str | None = Field(None, max_length=40)
    config: dict[str, Any] | None = None
    job_id: str | None = None


# ---------------------------------------------------------------------------
# Project endpoints
# ---------------------------------------------------------------------------


@router.get("/")
async def list_projects() -> list[dict]:
    """Return all projects ordered by creation time (newest first)."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM projects ORDER BY created_at DESC"
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


@router.post("/", status_code=201)
async def create_project(body: ProjectCreate) -> dict:
    """Create a new project."""
    now = datetime.now(UTC).isoformat()
    pid = uuid.uuid4().hex
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO projects VALUES (?,?,?,?,?,?,?)",
            (
                pid,
                body.name,
                body.description,
                body.owner,
                json.dumps(body.tags),
                now,
                now,
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
    return _row_to_dict(row)


@router.get("/{project_id}")
async def get_project(project_id: str) -> dict:
    """Return a single project by ID."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM projects WHERE id=?", (project_id,)
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return _row_to_dict(row)


@router.put("/{project_id}")
async def update_project(project_id: str, body: ProjectUpdate) -> dict:
    """Patch project metadata."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM projects WHERE id=?", (project_id,)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Project not found")
        d = _row_to_dict(row)
        if body.name is not None:
            d["name"] = body.name
        if body.description is not None:
            d["description"] = body.description
        if body.owner is not None:
            d["owner"] = body.owner
        if body.tags is not None:
            d["tags"] = body.tags
        d["updated_at"] = datetime.now(UTC).isoformat()
        conn.execute(
            "UPDATE projects SET name=?, description=?, owner=?, tags=?, updated_at=? WHERE id=?",
            (d["name"], d["description"], d["owner"], json.dumps(d["tags"]), d["updated_at"], project_id),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
    return _row_to_dict(row)


@router.delete("/{project_id}", status_code=204)
async def delete_project(project_id: str) -> None:
    """Delete a project and all its cases."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM projects WHERE id=?", (project_id,)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Project not found")
        conn.execute("DELETE FROM projects WHERE id=?", (project_id,))
        conn.commit()


# ---------------------------------------------------------------------------
# Case endpoints (nested under a project)
# ---------------------------------------------------------------------------


@router.get("/{project_id}/cases")
async def list_cases(project_id: str) -> list[dict]:
    """List simulation cases for a project."""
    with _get_conn() as conn:
        if conn.execute("SELECT id FROM projects WHERE id=?", (project_id,)).fetchone() is None:
            raise HTTPException(status_code=404, detail="Project not found")
        rows = conn.execute(
            "SELECT * FROM cases WHERE project_id=? ORDER BY created_at DESC", (project_id,)
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


@router.post("/{project_id}/cases", status_code=201)
async def create_case(project_id: str, body: CaseCreate) -> dict:
    """Create a simulation case inside a project."""
    now = datetime.now(UTC).isoformat()
    cid = uuid.uuid4().hex
    with _get_conn() as conn:
        if conn.execute("SELECT id FROM projects WHERE id=?", (project_id,)).fetchone() is None:
            raise HTTPException(status_code=404, detail="Project not found")
        conn.execute(
            "INSERT INTO cases VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                cid,
                project_id,
                body.name,
                body.description,
                body.scenario,
                "draft",
                json.dumps(body.config),
                None,
                now,
                now,
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM cases WHERE id=?", (cid,)).fetchone()
    return _row_to_dict(row)


@router.get("/{project_id}/cases/{case_id}")
async def get_case(project_id: str, case_id: str) -> dict:
    """Get a single simulation case."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM cases WHERE id=? AND project_id=?", (case_id, project_id)
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Case not found")
    return _row_to_dict(row)


@router.put("/{project_id}/cases/{case_id}")
async def update_case(project_id: str, case_id: str, body: CaseUpdate) -> dict:
    """Update a simulation case (patch fields)."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM cases WHERE id=? AND project_id=?", (case_id, project_id)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Case not found")
        d = _row_to_dict(row)
        if body.name is not None:
            d["name"] = body.name
        if body.description is not None:
            d["description"] = body.description
        if body.scenario is not None:
            d["scenario"] = body.scenario
        if body.status is not None:
            d["status"] = body.status
        if body.config is not None:
            d["config"] = body.config
        if body.job_id is not None:
            d["job_id"] = body.job_id
        d["updated_at"] = datetime.now(UTC).isoformat()
        conn.execute(
            "UPDATE cases SET name=?, description=?, scenario=?, status=?, config=?, job_id=?, updated_at=? WHERE id=?",
            (
                d["name"],
                d["description"],
                d["scenario"],
                d["status"],
                json.dumps(d["config"]),
                d.get("job_id"),
                d["updated_at"],
                case_id,
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
    return _row_to_dict(row)


@router.delete("/{project_id}/cases/{case_id}", status_code=204)
async def delete_case(project_id: str, case_id: str) -> None:
    """Delete a simulation case."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM cases WHERE id=? AND project_id=?", (case_id, project_id)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Case not found")
        conn.execute("DELETE FROM cases WHERE id=?", (case_id,))
        conn.commit()
