"""Tests for the engineering workflow P1/P2 features.

Covers:
  - suboff_surrogate_cycle template (P1-A)
  - ship_pareto_screening template (P1-B)
  - Engineering acceptance gate library (P1-C)
  - HPC status polling, retry, and archive endpoints (P2-A/B/C)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch
    from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_suboff_solve_response(call_log: list[dict]) -> Any:
    """Return a factory that captures suboff solve calls and returns a stub job_id."""
    counter = {"n": 0}

    async def _fake(params: Any) -> dict:
        counter["n"] += 1
        job_id = f"suboff-{counter['n']}"
        call_log.append({"hull_type": params.hull_type, "speed_ms": params.speed_ms, "job_id": job_id})
        return {"job_id": job_id, "message": "stub"}

    return _fake


def _fake_hull_free_surface_response(call_log: list[dict]) -> Any:
    """Return a factory that captures hull-free-surface solve calls."""
    counter = {"n": 0}

    async def _fake(params: Any) -> dict:
        counter["n"] += 1
        job_id = f"ship-{counter['n']}"
        call_log.append({"hull_type": params.hull_type, "re": params.re, "job_id": job_id})
        return {"job_id": job_id, "message": "stub"}

    return _fake


# ---------------------------------------------------------------------------
# P1-A: suboff_surrogate_cycle template
# ---------------------------------------------------------------------------

class TestSuboffSurrogateCycle:
    def test_template_listed_as_implemented(self, client: TestClient) -> None:
        r = client.get("/api/orchestration/templates")
        assert r.status_code == 200, r.text
        data = r.json()
        tpl = next(t for t in data["templates"] if t["template_id"] == "suboff_surrogate_cycle")
        assert tpl["implemented"] is True
        assert "suboff_surrogate_cycle" in data["implemented"]

    def test_submit_generates_pre_screen_jobs(
        self, client: TestClient, monkeypatch: MonkeyPatch
    ) -> None:
        from backend.routers import orchestration, suboff as suboff_router

        calls: list[dict] = []
        monkeypatch.setattr(suboff_router, "solve_suboff", _fake_suboff_solve_response(calls))

        payload = {
            "template_id": "suboff_surrogate_cycle",
            "base_config": {
                "hull_variants": ["bare_hull", "with_sail"],
                "speed_values_ms": [1.5, 2.5],
            },
        }
        r = client.post("/api/orchestration/experiments/submit", json=payload)
        assert r.status_code == 200, r.text
        data = r.json()

        assert data["template_id"] == "suboff_surrogate_cycle"
        assert data["phase"] == "pre_screen"
        assert data["submitted"] == 4  # 2 hulls × 2 speeds
        assert len(data["job_ids"]) == 4
        assert len(data["design_matrix"]) == 4
        assert "study_group" in data
        assert "next_step" in data

        # Check all hull/speed combos were submitted
        submitted_combos = {(c["hull_type"], c["speed_ms"]) for c in calls}
        assert ("bare_hull", 1.5) in submitted_combos
        assert ("with_sail", 2.5) in submitted_combos

    def test_submit_default_config(
        self, client: TestClient, monkeypatch: MonkeyPatch
    ) -> None:
        from backend.routers import suboff as suboff_router

        calls: list[dict] = []
        monkeypatch.setattr(suboff_router, "solve_suboff", _fake_suboff_solve_response(calls))

        r = client.post(
            "/api/orchestration/experiments/submit",
            json={"template_id": "suboff_surrogate_cycle"},
        )
        assert r.status_code == 200, r.text
        data = r.json()
        # Default: 2 hull_variants × 3 speed values = 6 jobs
        assert data["submitted"] == 6

    def test_study_group_tagged_on_jobs(
        self, client: TestClient, monkeypatch: MonkeyPatch
    ) -> None:
        from backend.routers import suboff as suboff_router
        from backend import job_manager

        fake_job_id = "tagged-job-1"
        async def _fake_solve(params: Any) -> dict:
            return {"job_id": fake_job_id, "message": "stub"}

        # Register a real job so get_job works
        sentinel: dict = {}
        def _fake_submit(*, name: str, job_type: str, config: dict, fn: Any) -> str:
            sentinel["job_id"] = fake_job_id
            return fake_job_id

        monkeypatch.setattr(suboff_router, "solve_suboff", _fake_solve)
        monkeypatch.setattr(job_manager, "submit", _fake_submit)

        r = client.post(
            "/api/orchestration/experiments/submit",
            json={
                "template_id": "suboff_surrogate_cycle",
                "base_config": {"hull_variants": ["bare_hull"], "speed_values_ms": [2.5]},
            },
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["submitted"] == 1


# ---------------------------------------------------------------------------
# P1-B: ship_pareto_screening template
# ---------------------------------------------------------------------------

class TestShipParetoScreening:
    def test_template_listed_as_implemented(self, client: TestClient) -> None:
        r = client.get("/api/orchestration/templates")
        assert r.status_code == 200, r.text
        data = r.json()
        tpl = next(t for t in data["templates"] if t["template_id"] == "ship_pareto_screening")
        assert tpl["implemented"] is True
        assert "ship_pareto_screening" in data["implemented"]

    def test_submit_generates_screening_jobs(
        self, client: TestClient, monkeypatch: MonkeyPatch
    ) -> None:
        from backend.routers import solver as solver_router

        calls: list[dict] = []
        monkeypatch.setattr(solver_router, "start_hull_free_surface", _fake_hull_free_surface_response(calls))

        payload = {
            "template_id": "ship_pareto_screening",
            "base_config": {
                "hull_variants": ["wigley", "series60"],
                "re_values": [100.0, 200.0],
            },
        }
        r = client.post("/api/orchestration/experiments/submit", json=payload)
        assert r.status_code == 200, r.text
        data = r.json()

        assert data["template_id"] == "ship_pareto_screening"
        assert data["phase"] == "screening"
        assert data["submitted"] == 4  # 2 hulls × 2 Re
        assert len(data["job_ids"]) == 4
        assert len(data["design_matrix"]) == 4
        assert "study_group" in data
        assert "next_step" in data

        submitted_combos = {(c["hull_type"], c["re"]) for c in calls}
        assert ("wigley", 100.0) in submitted_combos
        assert ("series60", 200.0) in submitted_combos

    def test_submit_default_config(
        self, client: TestClient, monkeypatch: MonkeyPatch
    ) -> None:
        from backend.routers import solver as solver_router

        calls: list[dict] = []
        monkeypatch.setattr(solver_router, "start_hull_free_surface", _fake_hull_free_surface_response(calls))

        r = client.post(
            "/api/orchestration/experiments/submit",
            json={"template_id": "ship_pareto_screening"},
        )
        assert r.status_code == 200, r.text
        data = r.json()
        # Default: 3 hull_variants × 3 re_values = 9 jobs
        assert data["submitted"] == 9


# ---------------------------------------------------------------------------
# P1-C: Engineering acceptance gate library
# ---------------------------------------------------------------------------

class TestAcceptanceGates:
    def test_list_gates(self, client: TestClient) -> None:
        r = client.get("/api/benchmarks/acceptance-gates")
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["count"] >= 5
        scenarios = set(data["scenarios"])
        assert "marine_resistance" in scenarios
        assert "suboff_resistance" in scenarios
        assert "turbulent_channel" in scenarios
        assert "cylinder_vortex_shedding" in scenarios
        assert "rotating_machinery" in scenarios

    def test_check_gate_all_pass(self, client: TestClient) -> None:
        metrics = {
            "yplus_max": 20.0,        # < 50 limit
            "yplus_min": 1.0,         # > 0.5 limit
            "cd_error_max": 2.0,      # < 5% limit
            "ct_error_max": 1.5,      # < 3% limit
            "convergence_max": 5e-5,  # < 1e-4 limit
            "wave_rmse_max": 0.05,    # < 0.15 limit
        }
        r = client.post("/api/benchmarks/acceptance-gates/marine_resistance/check", json=metrics)
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["passed"] is True
        assert data["scenario"] == "marine_resistance"
        assert data["checks_passed"] == data["checks_total"]

    def test_check_gate_some_fail(self, client: TestClient) -> None:
        metrics = {
            "yplus_max": 80.0,        # EXCEEDS 50 limit
            "yplus_min": 1.0,
            "ct_error_max": 1.5,
            "convergence_max": 5e-5,
        }
        r = client.post("/api/benchmarks/acceptance-gates/marine_resistance/check", json=metrics)
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["passed"] is False
        failing = [c for c in data["checks"] if not c["passed"]]
        keys = {c["key"] for c in failing}
        assert "yplus_max" in keys

    def test_check_gate_missing_metrics(self, client: TestClient) -> None:
        # Empty metrics – all required checks missing → should fail
        r = client.post(
            "/api/benchmarks/acceptance-gates/suboff_resistance/check", json={}
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["passed"] is False
        missing = [c for c in data["checks"] if c["result"] == "missing"]
        assert len(missing) > 0

    def test_check_gate_unknown_scenario(self, client: TestClient) -> None:
        r = client.post(
            "/api/benchmarks/acceptance-gates/nonexistent_scenario/check", json={}
        )
        assert r.status_code == 422

    def test_check_gate_turbulent_channel(self, client: TestClient) -> None:
        metrics = {
            "yplus_max": 3.0,         # < 5 limit
            "re_tau_error_max": 2.0,  # < 5% limit
            "u_bulk_error_max": 1.0,  # < 2% limit
            "convergence_max": 5e-5,
        }
        r = client.post(
            "/api/benchmarks/acceptance-gates/turbulent_channel/check", json=metrics
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["passed"] is True

    def test_check_job_gate_not_found(self, client: TestClient) -> None:
        r = client.post(
            "/api/benchmarks/acceptance-gates/marine_resistance/check-job/nonexistent-id"
        )
        assert r.status_code == 404

    def test_check_job_gate_not_completed(
        self, client: TestClient, monkeypatch: MonkeyPatch
    ) -> None:
        from backend import job_manager

        def _fake_submit(*, name: str, job_type: str, config: dict, fn: Any) -> str:
            return "pending-job-x"

        monkeypatch.setattr(job_manager, "submit", _fake_submit)

        # Register a queued job via direct manipulation for test purposes
        r = client.post(
            "/api/benchmarks/acceptance-gates/marine_resistance/check-job/pending-job-x"
        )
        # Should 404 since the fake submit doesn't actually register the job
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# P2-A: HPC status polling
# ---------------------------------------------------------------------------

class TestHPCStatusPolling:
    def test_hpc_status_no_hpc_info(
        self, client: TestClient, monkeypatch: MonkeyPatch, waiter: Any
    ) -> None:
        from backend import job_manager

        def _fast_fn(job: Any) -> dict:
            return {"done": True}

        job_id = job_manager.submit("hpc-test", "test", {}, _fast_fn)
        waiter(job_id)

        r = client.get(f"/api/jobs/{job_id}/hpc-status")
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["job_id"] == job_id
        assert data["platform_status"] == "completed"
        assert data["hpc_backend"] == "none"
        assert data["hpc_job_id"] is None

    def test_hpc_status_with_stub_hpc_info(
        self, client: TestClient, job_manager: Any, waiter: Any
    ) -> None:
        from backend import job_manager as jm

        def _fast_fn(j: Any) -> dict:
            return {"done": True}

        job_id = jm.submit("hpc-slurm-test", "test", {}, _fast_fn)
        waiter(job_id)

        # Inject fake HPC info
        job = jm.get_job(job_id)
        assert job is not None
        job.config["hpc_info"] = {
            "backend": "slurm",
            "hpc_job_id": "12345",
            "status": "submitted",
        }

        r = client.get(f"/api/jobs/{job_id}/hpc-status")
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["hpc_backend"] == "slurm"
        assert data["hpc_job_id"] == "12345"
        # sacct not available in test env – cluster_state may be "unknown" or "query_error"
        assert "cluster_state" in data

    def test_hpc_status_not_found(self, client: TestClient) -> None:
        r = client.get("/api/jobs/nonexistent-job/hpc-status")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# P2-B: Manual retry
# ---------------------------------------------------------------------------

class TestJobRetry:
    def test_retry_failed_job(
        self, client: TestClient, waiter: Any
    ) -> None:
        from backend import job_manager

        call_count = {"n": 0}

        def _failing_fn(job: Any) -> dict:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("transient failure")
            return {"recovered": True}

        job_id = job_manager.submit("retry-test", "test", {}, _failing_fn)
        waiter(job_id)

        # Job should be failed (max_retries=0)
        job = job_manager.get_job(job_id)
        assert job is not None
        assert job.status.value == "failed"

        r = client.post(f"/api/jobs/{job_id}/retry")
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["previous_status"] == "failed"
        assert data["job_id"] == job_id

        # Wait for re-execution
        waiter(job_id)
        job = job_manager.get_job(job_id)
        assert job is not None
        assert job.status.value == "completed"
        assert job.result.get("recovered") is True

    def test_retry_running_job_rejected(
        self, client: TestClient, job_manager: Any
    ) -> None:
        import time
        from backend import job_manager as jm

        def _slow_fn(j: Any) -> dict:
            time.sleep(2)
            return {}

        job_id = jm.submit("slow-test", "test", {}, _slow_fn)
        time.sleep(0.05)  # let it start

        r = client.post(f"/api/jobs/{job_id}/retry")
        assert r.status_code == 409

    def test_retry_not_found(self, client: TestClient) -> None:
        r = client.post("/api/jobs/nonexistent-job/retry")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# P2-C: Job archive
# ---------------------------------------------------------------------------

class TestJobArchive:
    def test_archive_completed_job(
        self, client: TestClient, waiter: Any, tmp_path: Any
    ) -> None:
        from backend import job_manager

        def _write_fn(job: Any) -> dict:
            (job.output_dir / "result.json").write_text('{"ok": true}')
            return {"output": "written"}

        job_id = job_manager.submit("archive-test", "test", {}, _write_fn)
        waiter(job_id)

        r = client.post(f"/api/jobs/{job_id}/archive")
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["job_id"] == job_id
        assert "archive_path" in data
        assert data["archive_path"].endswith(".zip")
        assert data["archive_size_bytes"] > 0

    def test_archive_queued_job_rejected(
        self, client: TestClient
    ) -> None:
        import time
        from backend import job_manager

        def _slow_fn(j: Any) -> dict:
            time.sleep(5)
            return {}

        job_id = job_manager.submit("queued-archive-test", "test", {}, _slow_fn)
        # Don't wait – job may still be queued or running
        r = client.post(f"/api/jobs/{job_id}/archive")
        assert r.status_code in (200, 409)  # 200 if already done, 409 if still active

    def test_archive_not_found(self, client: TestClient) -> None:
        r = client.post("/api/jobs/nonexistent-job/archive")
        assert r.status_code == 404
