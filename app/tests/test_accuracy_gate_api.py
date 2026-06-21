"""Tests for industrial accuracy baseline and regression report endpoints."""
from __future__ import annotations


def _mock_accuracy_result() -> dict[str, object]:
    return {
        "cavity": [
            {"re": 100, "rmse_u": 0.08, "rmse_v": 0.11},
            {"re": 400, "rmse_u": 0.09, "rmse_v": 0.12},
        ],
        "bfs": [
            {"re": 100, "xr_star": 5.2},
            {"re": 200, "xr_star": 6.0},
        ],
        "rotating_cylinder": [
            {"spin_ratio": 1.0, "cl_mean": 0.08, "cd_mean": 1.3},
            {"spin_ratio": 2.0, "cl_mean": 0.18, "cd_mean": 1.5},
        ],
    }


def test_accuracy_baselines_endpoint(client):
    r = client.get("/api/benchmarks/accuracy/baselines")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "profiles" in body
    assert "ci_fast" in body["profiles"]
    assert "engineering_full" in body["profiles"]


def test_accuracy_report_endpoint_for_completed_accuracy_job(client, job_manager, waiter):
    def _runner(_job):
        return _mock_accuracy_result()

    job_id = job_manager.submit(
        name="mock-accuracy",
        job_type="benchmark_accuracy",
        config={"fast": True},
        fn=_runner,
    )
    final = waiter(job_id, timeout=30.0)
    assert final["status"] == "completed"

    r = client.get(f"/api/benchmarks/accuracy/report/{job_id}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["report_type"] == "accuracy_regression"
    assert body["profile"] == "ci_fast"
    assert body["gate"]["gate_passed"] is True
    assert body["gate"]["checks_total"] > 0


def test_accuracy_report_rejects_non_accuracy_job(client, job_manager, waiter):
    def _runner(_job):
        return {"ok": True}

    job_id = job_manager.submit(
        name="not-accuracy",
        job_type="unit_test",
        config={},
        fn=_runner,
    )
    final = waiter(job_id, timeout=30.0)
    assert final["status"] == "completed"

    r = client.get(f"/api/benchmarks/accuracy/report/{job_id}")
    assert r.status_code == 422
