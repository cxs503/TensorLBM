"""Tests for cylinder-flow parameter-scan solver endpoint."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch
    from fastapi.testclient import TestClient


def test_cylinder_flow_scan_submits_multiple_jobs(
    client: TestClient, monkeypatch: MonkeyPatch
) -> None:
    from backend.routers import solver

    calls: list[dict[str, object]] = []

    def _fake_submit(
        *, name: str, job_type: str, config: dict[str, object], fn: object
    ) -> str:
        calls.append({"name": name, "job_type": job_type, "config": config, "fn": fn})
        return f"job-{len(calls)}"

    monkeypatch.setattr(solver.job_manager, "submit", _fake_submit)

    payload = {
        "nx": 60,
        "ny": 24,
        "u_in": 0.05,
        "re_values": [50.0, 80.0, 120.0],
        "radius": 4.0,
        "n_steps": 10,
        "output_interval": 5,
        "physics": {"turbulence_model": "smagorinsky_les"},
    }
    r = client.post("/api/solve/cylinder-flow/scan", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["parameter"] == "re"
    assert data["values"] == payload["re_values"]
    assert data["job_ids"] == ["job-1", "job-2", "job-3"]
    assert len(calls) == 3
    assert all(c["job_type"] == "cylinder_flow" for c in calls)
    assert all(c["config"]["scan"]["group"] == data["scan_group"] for c in calls)
    assert [c["config"]["scan"]["value"] for c in calls] == payload["re_values"]
    assert all(c["config"]["physics"]["turbulence_model"] == "smagorinsky_les" for c in calls)


def test_cylinder_flow_scan_requires_at_least_two_values(client: TestClient) -> None:
    r = client.post(
        "/api/solve/cylinder-flow/scan",
        json={
            "nx": 60,
            "ny": 24,
            "u_in": 0.05,
            "re_values": [50.0],
            "radius": 4.0,
            "n_steps": 10,
            "output_interval": 5,
        },
    )
    assert r.status_code == 422
