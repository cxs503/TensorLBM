"""Tests for orchestration and AI governance APIs."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch
    from fastapi.testclient import TestClient


def test_orchestration_templates(client: TestClient) -> None:
    r = client.get("/api/orchestration/templates")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["count"] >= 3
    ids = {t["template_id"] for t in data["templates"]}
    assert "cylinder_re_sweep" in ids


def test_orchestration_submit_cylinder_template(
    client: TestClient, monkeypatch: MonkeyPatch,
) -> None:
    from backend.routers import solver

    calls: list[dict] = []

    def _fake_submit(*, name: str, job_type: str, config: dict, fn: object) -> str:
        calls.append({"name": name, "job_type": job_type, "config": config, "fn": fn})
        return f"job-{len(calls)}"

    monkeypatch.setattr(solver.job_manager, "submit", _fake_submit)

    payload = {
        "template_id": "cylinder_re_sweep",
        "base_config": {"nx": 80, "ny": 30, "n_steps": 10, "output_interval": 5},
        "sweep": [{"name": "re", "values": [60.0, 80.0]}],
        "orchestration": {"max_retries": 1, "resource_label": "cuda:0"},
    }
    r = client.post("/api/orchestration/experiments/submit", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["submitted"] == 2
    assert len(data["job_ids"]) == 2
    assert len(calls) == 2
    assert all(c["job_type"] == "cylinder_flow" for c in calls)


def test_orchestration_kpis(client: TestClient) -> None:
    r = client.get("/api/orchestration/kpis")
    assert r.status_code == 200
    data = r.json()
    assert "max_workers" in data
    assert "parallel_efficiency" in data


def test_ai_confidence_gate(client: TestClient) -> None:
    r = client.post(
        "/api/ai/governance/confidence-gate",
        json={
            "prediction": 1.05,
            "baseline": 1.0,
            "uncertainty": 0.05,
            "max_relative_error": 0.1,
            "max_uncertainty": 0.1,
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["accepted"] is True
    assert data["recommended_action"] == "accept_ai"


def test_ai_active_learning_prioritize(client: TestClient) -> None:
    r = client.post(
        "/api/ai/governance/active-learning/prioritize",
        json={
            "top_k": 2,
            "candidates": [
                {"sample_id": "a", "uncertainty": 0.9, "novelty": 0.1, "impact": 0.1},
                {"sample_id": "b", "uncertainty": 0.3, "novelty": 0.9, "impact": 0.2},
                {"sample_id": "c", "uncertainty": 0.7, "novelty": 0.4, "impact": 0.8},
            ],
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["count"] == 2
    assert data["selected"][0]["score"] >= data["selected"][1]["score"]
