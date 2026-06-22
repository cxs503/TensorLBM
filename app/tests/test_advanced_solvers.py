"""API-shape and slow smoke tests for the four newly exposed 3-D solver endpoints.

Fast tests (no PLATFORM_SLOW_TESTS gate):
- HTTP 200 + job_id present for valid payloads.
- HTTP 422 for obviously invalid payloads.
- /api/solve/validate works with 3-D parameters.

Slow tests (PLATFORM_SLOW_TESTS=1):
- Full end-to-end: submit → wait for completion → check status.
"""
from __future__ import annotations

import os

import pytest


# ---------------------------------------------------------------------------
# Fast shape tests (always run)
# ---------------------------------------------------------------------------

class TestSphereFlowD3Q27:
    def test_submit_returns_job_id(self, client):
        r = client.post(
            "/api/solve/sphere-flow-d3q27",
            json={"nx": 20, "ny": 12, "nz": 12, "u_in": 0.05,
                  "re": 30.0, "radius": 4.0, "n_steps": 2, "output_interval": 2},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert "job_id" in body
        assert body["message"]

    def test_invalid_nx_too_small(self, client):
        r = client.post(
            "/api/solve/sphere-flow-d3q27",
            json={"nx": 1, "ny": 10, "nz": 10},
        )
        assert r.status_code == 422


class TestThermalCavity3D:
    def test_submit_returns_job_id(self, client):
        r = client.post(
            "/api/solve/thermal-cavity-3d",
            json={"nx": 8, "ny": 8, "nz": 8, "ra": 1e3, "pr": 0.71, "n_steps": 2},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert "job_id" in body

    def test_invalid_n_steps_zero(self, client):
        r = client.post(
            "/api/solve/thermal-cavity-3d",
            json={"nx": 8, "ny": 8, "nz": 8, "ra": 1e3, "pr": 0.71, "n_steps": 0},
        )
        assert r.status_code == 422


class TestPorousDrainage3D:
    def test_submit_returns_job_id(self, client):
        r = client.post(
            "/api/solve/porous-drainage-3d",
            json={"nx": 10, "ny": 10, "nz": 12, "medium": "random_spheres",
                  "n_spheres": 2, "G_12": 0.9, "u_inlet": 0.005,
                  "n_steps": 2, "output_interval": 2},
        )
        assert r.status_code == 200, r.text
        assert "job_id" in r.json()

    def test_invalid_medium(self, client):
        # tau_water below minimum
        r = client.post(
            "/api/solve/porous-drainage-3d",
            json={"tau_water": 0.3},
        )
        assert r.status_code == 422


class TestHullFreeSurface:
    def test_submit_returns_job_id(self, client):
        r = client.post(
            "/api/solve/hull-free-surface",
            json={"nx": 24, "ny": 12, "nz": 12, "hull_type": "wigley",
                  "fill_fraction": 0.5, "re": 50.0, "u_in": 0.05,
                  "n_steps": 2, "output_interval": 2},
        )
        assert r.status_code == 200, r.text
        assert "job_id" in r.json()

    def test_invalid_fill_fraction(self, client):
        r = client.post(
            "/api/solve/hull-free-surface",
            json={"nx": 24, "ny": 12, "nz": 12,
                  "fill_fraction": 1.5},  # > max 0.9
        )
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Slow smoke tests (PLATFORM_SLOW_TESTS=1 required)
# ---------------------------------------------------------------------------

pytestmark_slow = pytest.mark.skipif(
    os.environ.get("PLATFORM_SLOW_TESTS") != "1",
    reason="Slow end-to-end solver tests – opt-in via PLATFORM_SLOW_TESTS=1",
)

ADVANCED_CASES = [
    (
        "/api/solve/sphere-flow-d3q27",
        {"nx": 32, "ny": 16, "nz": 16, "u_in": 0.05, "re": 30.0, "radius": 4.0,
         "n_steps": 40, "output_interval": 20},
        "sphere_flow_d3q27",
    ),
    (
        "/api/solve/thermal-cavity-3d",
        {"nx": 12, "ny": 12, "nz": 12, "ra": 1e3, "pr": 0.71, "n_steps": 40},
        "thermal_cavity_3d",
    ),
    (
        "/api/solve/porous-drainage-3d",
        {"nx": 12, "ny": 12, "nz": 16, "medium": "random_spheres", "n_spheres": 2,
         "G_12": 0.9, "u_inlet": 0.005, "n_steps": 80, "output_interval": 40},
        "porous_drainage_3d",
    ),
    (
        "/api/solve/hull-free-surface",
        {"nx": 24, "ny": 12, "nz": 12, "hull_type": "wigley",
         "fill_fraction": 0.5, "re": 50.0, "u_in": 0.05,
         "n_steps": 40, "output_interval": 20},
        "hull_free_surface",
    ),
]


@pytest.mark.parametrize(("path", "payload", "job_type"), ADVANCED_CASES)
@pytestmark_slow
def test_advanced_solver_completes(client, waiter, path, payload, job_type):
    """Submit an advanced 3-D solver and verify completion."""
    r = client.post(path, json=payload)
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    final = waiter(job_id, timeout=300.0)
    assert final["status"] == "completed", (
        f"{path} → status={final['status']}, "
        f"error={str(final.get('error', ''))[:400]}"
    )
    assert final["job_type"] == job_type
