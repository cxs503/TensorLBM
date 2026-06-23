"""Tests for orchestration and AI governance APIs."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
    assert "external_aero_e2e_pilot" in ids


def test_orchestration_submit_cylinder_template(
    client: TestClient,
    monkeypatch: MonkeyPatch,
) -> None:
    from backend.routers import solver

    calls: list[dict[str, Any]] = []

    def _fake_submit(*, name: str, job_type: str, config: dict[str, Any], fn: object) -> str:
        calls.append({"name": name, "job_type": job_type, "config": config, "fn": fn})
        return f"job-{len(calls)}"

    monkeypatch.setattr(solver.job_manager, "submit", _fake_submit)

    payload = {
        "template_id": "cylinder_re_sweep",
        "base_config": {"nx": 80, "ny": 30, "n_steps": 10, "output_interval": 5},
        "sweep": [{"name": "re", "values": [60.0, 80.0]}],
    }
    r = client.post("/api/orchestration/experiments/submit", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["submitted"] == 2
    assert len(data["job_ids"]) == 2
    assert len(calls) == 2
    assert all(c["job_type"] == "cylinder_flow" for c in calls)


def test_orchestration_submit_multifactor_template(
    client: TestClient,
    monkeypatch: MonkeyPatch,
) -> None:
    from backend.routers import solver

    calls: list[dict[str, Any]] = []

    def _fake_submit(*, name: str, job_type: str, config: dict[str, Any], fn: object) -> str:
        calls.append({"name": name, "job_type": job_type, "config": config, "fn": fn})
        return f"job-{len(calls)}"

    monkeypatch.setattr(solver.job_manager, "submit", _fake_submit)

    payload = {
        "template_id": "cylinder_multi_factor_doe",
        "base_config": {"nx": 80, "ny": 30, "n_steps": 10, "output_interval": 5},
        "sweep": [
            {"name": "re", "values": [60.0, 80.0]},
            {"name": "u_in", "values": [0.05, 0.08]},
        ],
        "objective": {"metric": "mean_cd_last", "goal": "minimize"},
    }
    r = client.post("/api/orchestration/experiments/submit", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["submitted"] == 4
    assert data["job_count"] == 4
    assert len(data["job_ids"]) == 4
    assert len(calls) == 4
    assert data["objective"]["metric"] == "mean_cd_last"


def test_orchestration_kpis(client: TestClient) -> None:
    r = client.get("/api/orchestration/kpis")
    assert r.status_code == 200
    data = r.json()
    assert "max_workers" in data
    assert "parallel_efficiency" in data


def test_orchestration_study_summary(client: TestClient, job_manager, waiter) -> None:
    def _writer(job: object, cd: float, re_value: float) -> dict[str, bool]:
        job.output_dir.joinpath("forces.csv").write_text(
            f"step,Cd,Cl\n0,{cd},0.0\n1,{cd},0.0\n",
        )
        job.diagnostics.extend([{"step": 0, "Cd": cd}, {"step": 1, "Cd": cd}])
        return {"ok": True, "re": re_value}

    shared = {
        "group": "study-summary-1",
        "variables": [{"name": "re", "values": [80.0, 100.0]}],
        "objective": {"metric": "mean_cd_last", "goal": "minimize"},
        "constraints": [{"metric": "steady_state_score", "operator": ">=", "value": 0.5}],
    }
    job1 = job_manager.submit(
        name="study-a",
        job_type="cylinder_flow",
        config={"study": {**shared, "design_point": {"re": 80.0}}},
        fn=lambda job: _writer(job, 1.2, 80.0),
    )
    job2 = job_manager.submit(
        name="study-b",
        job_type="cylinder_flow",
        config={"study": {**shared, "design_point": {"re": 100.0}}},
        fn=lambda job: _writer(job, 0.8, 100.0),
    )
    waiter(job1, timeout=10.0)
    waiter(job2, timeout=10.0)

    r = client.get("/api/orchestration/studies/study-summary-1/summary")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["job_count"] == 2
    assert data["eligible_jobs"] == 2
    assert data["best_job"]["job_id"] == job2
    assert data["best_job"]["design_point"]["re"] == 100.0


def test_orchestration_submit_external_aero_pilot(
    client: TestClient,
    monkeypatch: MonkeyPatch,
) -> None:
    from backend.routers import solver

    async def _fake_parametric_study(_req: object) -> dict[str, object]:
        return {
            "study_group": "fake-group",
            "job_count": 2,
            "job_ids": ["pilot-job-1", "pilot-job-2"],
            "design_matrix": [{"re": 80.0}, {"re": 120.0}],
        }

    monkeypatch.setattr(solver, "parametric_study", _fake_parametric_study)

    r = client.post(
        "/api/orchestration/experiments/submit",
        json={"template_id": "external_aero_e2e_pilot"},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["template_id"] == "external_aero_e2e_pilot"
    assert data["submitted"] == 2
    assert data["gate_scenario"] == "external_aerodynamics"
    assert data["workflow"] == "external_aero_e2e_pilot"
    assert "next_step" in data


def test_orchestration_gap_assessment(client: TestClient) -> None:
    r = client.get("/api/orchestration/gap-assessment")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["benchmarked_against"] == ["PowerFLOW", "XFlow"]
    assert data["count"] >= 8
    assert len(data["immediate_actions"]) == 3
    assert any(row["priority"] == "P0" for row in data["categories"])


def test_orchestration_regression_dashboard(client: TestClient, job_manager, waiter) -> None:
    def _runner(_job: object) -> dict[str, float]:
        return {
            "cd_error_max": 2.0,
            "cl_error_max": 1.0,
            "yplus_max": 80.0,
            "convergence_max": 5e-5,
        }

    job_id = job_manager.submit(
        name="external-aero-gate-case",
        job_type="cylinder_flow",
        config={"study": {"gate_scenario": "external_aerodynamics"}},
        fn=_runner,
    )
    waiter(job_id, timeout=10.0)

    r = client.get("/api/orchestration/regression-dashboard")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["axis"] == ["version", "accuracy", "cost"]
    assert "external_aerodynamics" in data["accuracy"]["gate_rollup"]
    rollup = data["accuracy"]["gate_rollup"]["external_aerodynamics"]
    assert rollup["evaluated"] >= 1
    assert rollup["pass_rate"] is not None


def test_orchestration_hpc_dashboard(client: TestClient, job_manager, waiter) -> None:
    def _runner(job: object) -> dict[str, bool]:
        job.config.setdefault("hpc_info", {}).update({
            "backend": "slurm",
            "partition": "gpu",
            "cluster_state": "RUNNING",
            "cluster_elapsed_seconds": 120.0,
            "estimated_cluster_cost": 0.6,
            "retry_count": 1,
        })
        return {"ok": True}

    job_id = job_manager.submit(
        name="hpc-dash-case",
        job_type="cylinder_flow",
        config={"orchestration": {"cost_rate_per_second": 0.005}},
        fn=_runner,
    )
    waiter(job_id, timeout=10.0)

    r = client.get("/api/orchestration/hpc-dashboard")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["count"] >= 1
    assert data["backends"].get("slurm", 0) >= 1
    assert data["estimated_cluster_cost_total"] >= 0.6


def test_release_gate_block_and_history(client: TestClient, job_manager, waiter) -> None:
    def _failing_runner(_job: object) -> dict[str, float]:
        return {"cd_error_max": 99.0}

    job_id = job_manager.submit(
        name="release-gate-fail",
        job_type="cylinder_flow",
        config={"study": {"group": "release-study", "gate_scenario": "external_aerodynamics"}},
        fn=_failing_runner,
    )
    waiter(job_id, timeout=10.0)

    r = client.post(
        "/api/orchestration/release-gates/evaluate",
        json={
            "version": "v1.2.3",
            "study_group": "release-study",
            "require_completed_jobs": 1,
            "min_acceptance_pass_rate": 1.0,
            "max_avg_runtime_seconds": 1e-6,
            "promote_as_baseline": True,
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["blocked"] is True
    assert data["decision"] == "blocked"

    hist = client.get("/api/orchestration/release-gates/history")
    assert hist.status_code == 200, hist.text
    assert hist.json()["count"] >= 1

    baselines = client.get("/api/orchestration/release-gates/baselines")
    assert baselines.status_code == 200, baselines.text
    assert "engineering_full" in baselines.json()["profiles"]


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


def test_ai_policy_adaptive_confidence_and_drift(client: TestClient) -> None:
    p = client.post(
        "/api/ai/governance/policies",
        json={
            "scenario": "external_aero",
            "model_id": "m1",
            "max_relative_error": 0.05,
            "max_uncertainty": 0.08,
            "max_ci_half_width": 0.03,
            "drift_threshold": 0.1,
            "require_human_review_error": 0.09,
            "require_human_review_uncertainty": 0.12,
        },
    )
    assert p.status_code == 200, p.text

    gate = client.post(
        "/api/ai/governance/confidence-gate",
        json={
            "scenario": "external_aero",
            "model_id": "m1",
            "prediction": 1.2,
            "baseline": 1.0,
            "uncertainty": 0.13,
            "ci_half_width": 0.01,
            "max_relative_error": 0.2,
            "max_uncertainty": 0.2,
        },
    )
    assert gate.status_code == 200, gate.text
    gate_data = gate.json()
    assert gate_data["human_review_required"] is True
    assert gate_data["recommended_action"] == "manual_review_required"
    assert gate_data["thresholds"]["max_relative_error"] == 0.05

    drift = client.post(
        "/api/ai/governance/drift-monitor",
        json={
            "scenario": "external_aero",
            "model_id": "m1",
            "baseline_mean": 1.0,
            "current_mean": 1.3,
            "baseline_std": 0.05,
            "current_std": 0.08,
            "sample_count": 32,
        },
    )
    assert drift.status_code == 200, drift.text
    drift_data = drift.json()
    assert drift_data["drifted"] is True
