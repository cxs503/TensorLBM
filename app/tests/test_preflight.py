"""Unit tests for the pre-flight engineering validation wizard.

``POST /api/preprocess/preflight``

Tests verify the structure and correctness of the preflight response without
running any actual solvers.
"""
from __future__ import annotations


class TestPreflightSchema:
    def test_stable_parameters_ok(self, client):
        """All-green checks for a physically stable configuration."""
        r = client.post("/api/preprocess/preflight", json={
            "solver_type": "cylinder_flow",
            "nx": 120, "ny": 60,
            "u_in": 0.05, "re": 100.0, "radius": 8.0,
            "n_steps": 2000, "output_interval": 200,
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert "checks" in body
        assert "recommendations" in body
        assert "memory_mb" in body
        assert "suggested_n_steps" in body
        assert "suggested_output_interval" in body
        assert "yplus_first_cell_m" in body
        # No errors for stable params
        errors = [c for c in body["checks"] if c["status"] == "error"]
        assert errors == [], errors

    def test_tau_instability_detected(self, client):
        """Very high Re + small grid should trigger tau < 0.51 error."""
        r = client.post("/api/preprocess/preflight", json={
            "solver_type": "cylinder_flow",
            "nx": 10, "ny": 10,
            "u_in": 0.3, "re": 10000.0, "radius": 2.0,
        })
        assert r.status_code == 200, r.text
        body = r.json()
        tau_check = next((c for c in body["checks"] if c["name"] == "tau_stability"), None)
        assert tau_check is not None
        assert tau_check["status"] == "error"

    def test_high_mach_warning(self, client):
        """u_in > Ma limit should produce a Mach warning."""
        r = client.post("/api/preprocess/preflight", json={
            "solver_type": "cylinder_flow",
            "nx": 40, "ny": 40,
            "u_in": 0.35, "re": 50.0, "radius": 5.0,
        })
        assert r.status_code == 200, r.text
        body = r.json()
        mach_check = next((c for c in body["checks"] if c["name"] == "mach_number"), None)
        assert mach_check is not None
        assert mach_check["status"] == "warning"

    def test_grid_too_large_3d_error(self, client):
        """3D grid exceeding limit should produce an error."""
        r = client.post("/api/preprocess/preflight", json={
            "solver_type": "sphere_flow_d3q27",
            "nx": 512, "ny": 512, "nz": 512,
        })
        assert r.status_code == 200, r.text
        body = r.json()
        grid_check = next((c for c in body["checks"] if c["name"] == "grid_size"), None)
        assert grid_check is not None
        assert grid_check["status"] == "error"

    def test_n_steps_exceeds_limit(self, client):
        """n_steps > 200000 should produce an error."""
        r = client.post("/api/preprocess/preflight", json={
            "solver_type": "cylinder_flow",
            "nx": 80, "ny": 40, "n_steps": 300_000,
        })
        assert r.status_code == 200, r.text
        body = r.json()
        step_check = next((c for c in body["checks"] if c["name"] == "n_steps"), None)
        assert step_check is not None
        assert step_check["status"] == "error"

    def test_memory_estimate_computed(self, client):
        """Memory estimate should be a positive float."""
        r = client.post("/api/preprocess/preflight", json={
            "solver_type": "cylinder_flow",
            "nx": 100, "ny": 50,
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["memory_mb"] is not None
        assert body["memory_mb"] > 0.0

    def test_yplus_estimate_returned(self, client):
        """y+ first-cell estimate returned when physical params provided."""
        r = client.post("/api/preprocess/preflight", json={
            "solver_type": "cylinder_flow",
            "phys_length_m": 1.0,
            "phys_velocity_ms": 10.0,
            "phys_nu_m2s": 1.5e-5,
            "target_yplus": 1.0,
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["yplus_first_cell_m"] is not None
        assert body["yplus_first_cell_m"] > 0.0

    def test_suggested_n_steps_when_omitted(self, client):
        """When n_steps is not given, a suggestion should be returned."""
        r = client.post("/api/preprocess/preflight", json={
            "solver_type": "cylinder_flow",
            "nx": 120, "ny": 60,
            "u_in": 0.05, "re": 200.0, "radius": 10.0,
            # n_steps intentionally omitted
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["suggested_n_steps"] is not None
        assert body["suggested_n_steps"] >= 1

    def test_minimal_request_no_crash(self, client):
        """An almost-empty request must not crash (return 200 with empty checks)."""
        r = client.post("/api/preprocess/preflight", json={"solver_type": "any"})
        assert r.status_code == 200, r.text

    def test_output_interval_warning(self, client):
        """output_interval > n_steps should warn."""
        r = client.post("/api/preprocess/preflight", json={
            "solver_type": "cylinder_flow",
            "nx": 60, "ny": 30,
            "n_steps": 100, "output_interval": 500,
        })
        assert r.status_code == 200, r.text
        body = r.json()
        oi_check = next((c for c in body["checks"] if c["name"] == "output_interval"), None)
        assert oi_check is not None
        assert oi_check["status"] == "warning"

    def test_3d_d3q27_memory_higher_than_d2q9(self, client):
        """D3Q27 memory estimate should exceed D2Q9 for same nx×ny."""
        r2d = client.post("/api/preprocess/preflight", json={
            "solver_type": "cylinder_flow",
            "nx": 32, "ny": 32,
        })
        r3d = client.post("/api/preprocess/preflight", json={
            "solver_type": "sphere_flow_d3q27",
            "nx": 32, "ny": 32, "nz": 32,
        })
        assert r2d.status_code == 200
        assert r3d.status_code == 200
        mem_2d = r2d.json()["memory_mb"]
        mem_3d = r3d.json()["memory_mb"]
        assert mem_3d > mem_2d
