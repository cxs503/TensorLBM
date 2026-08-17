from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from backend.routers import simulations


@dataclass
class _FakeStatus:
    value: str


class _FakeJob:
    def __init__(
        self,
        *,
        status: str = "queued",
        config: dict[str, Any] | None = None,
        diagnostics: list[dict[str, Any]] | None = None,
        result: dict[str, Any] | None = None,
        logs: list[str] | None = None,
        error: str | None = None,
    ) -> None:
        self.status = _FakeStatus(status)
        self.config = config or {}
        self.diagnostics = diagnostics or []
        self.result = result or {}
        self.logs = logs or []
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "config": self.config,
        }


def test_generic_run_submission_includes_shape_and_auto_selection(client, monkeypatch):
    captured: dict[str, Any] = {}

    def _fake_submit(*, name: str, job_type: str, config: dict[str, Any], fn):  # noqa: ANN001
        captured["name"] = name
        captured["job_type"] = job_type
        captured["config"] = config
        return "job12345"

    monkeypatch.setattr(simulations.job_manager, "submit", _fake_submit)

    r = client.post(
        "/api/simulations/generic-run",
        json={
            "geometry": {"source": "parametric", "shape": "sphere", "params": {"radius": 6}},
            "physics": {"Re": 200.0, "u_in": 0.08},
            "solver": {"collision": "", "steps": 0, "warmup": 0},
            "output": {"fields": ["pressure"], "forces": True, "strouhal": False},
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["job_id"] == "job12345"
    assert captured["job_type"] == "generic_run"
    assert captured["config"]["shape"] == "sphere"
    assert "auto_selected" in captured["config"]


def test_generic_run_status_reads_from_job_manager_only(client, monkeypatch):
    fake_job = _FakeJob(
        status="running",
        config={"shape": "stl", "auto_selected": {"collision": "mrt", "steps": 500}},
        diagnostics=[
            {
                "kind": "generic_run_step",
                "step": 120,
                "Cd_total": 0.41,
                "Cl": 0.02,
                "St": 0.19,
            }
        ],
        logs=["line1", "line2"],
    )
    monkeypatch.setattr(simulations.job_manager, "get_job", lambda _job_id: fake_job)

    r = client.get("/api/simulations/generic-run/job123/status")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "running"
    assert body["shape"] == "stl"
    assert body["step"] == 120
    assert body["Cd_total"] == 0.41


def test_generic_run_results_excludes_large_field_payload(client, monkeypatch):
    fake_job = _FakeJob(
        status="completed",
        result={
            "Cd_total": 0.4,
            "fields_data": {"pressure": np.ones((4, 4, 4), dtype=np.float32)},
        },
    )
    monkeypatch.setattr(simulations.job_manager, "get_job", lambda _job_id: fake_job)

    r = client.get("/api/simulations/generic-run/jobxyz/results")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["Cd_total"] == 0.4
    assert "fields_data" not in body


def test_generic_run_field_returns_2d_slice(client, monkeypatch):
    field = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    fake_job = _FakeJob(
        status="completed",
        result={"fields_data": {"pressure": field}},
    )
    monkeypatch.setattr(simulations.job_manager, "get_job", lambda _job_id: fake_job)

    r = client.get("/api/simulations/generic-run/jobabc/fields/pressure")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["shape"] == [4, 5]
    assert len(body["data"]) == 4
