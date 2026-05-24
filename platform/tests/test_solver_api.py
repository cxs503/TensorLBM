"""Smoke-tests for every simulation endpoint exposed under ``/api/solve``.

These tests submit *very* small simulations and wait for completion in the
same process via the job manager.  They verify that the full pipeline
(``submit → run → produce outputs → terminal status``) works end-to-end
for every solver registered on the platform.

Marked as ``slow`` and opt-in via ``PLATFORM_SLOW_TESTS=1``.  By default
only the fast API-shape tests in ``test_platform_basic.py``,
``test_preprocess_api.py``, ``test_cad_api.py``, ``test_job_manager.py``,
``test_jobs_api.py`` and ``test_postprocess_api.py`` run.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("PLATFORM_SLOW_TESTS") != "1",
    reason="Solver smoke tests opt-in via PLATFORM_SLOW_TESTS=1",
)

import pytest

# Each entry: (endpoint, payload, expected job_type prefix).  Payloads use
# the smallest grids/step counts that still exercise the solver to keep
# the full suite fast on CPU-only CI.
SOLVER_CASES = [
    (
        "/api/solve/cylinder-flow",
        {
            "nx": 60, "ny": 24, "u_in": 0.05, "re": 50.0,
            "radius": 4.0, "n_steps": 40, "output_interval": 20,
        },
        "cylinder_flow",
    ),
    (
        "/api/solve/lid-driven-cavity",
        {
            "nx": 32, "u_lid": 0.1, "re": 50.0,
            "n_steps": 40, "output_interval": 20,
        },
        "lid_driven_cavity",
    ),
    (
        "/api/solve/backward-facing-step",
        {
            "nx": 60, "ny": 20, "step_h": 6, "x_step": 12,
            "u_in": 0.05, "re": 50.0,
            "n_steps": 40, "output_interval": 20,
        },
        "backward_facing_step",
    ),
    (
        "/api/solve/turbulent-channel",
        {
            "nx": 32, "ny": 24, "re_tau": 50.0, "u_tau": 0.005,
            "smagorinsky_cs": 0.1,
            "n_steps": 40, "averaging_start": 20, "output_interval": 20,
        },
        "turbulent_channel",
    ),
    (
        "/api/solve/pipeline-flow",
        {
            "nx": 60, "ny": 40, "diameter": 8.0, "gap_ratio": 0.5,
            "u_in": 0.05, "re": 50.0,
            "n_steps": 40, "output_interval": 20,
        },
        "pipeline_flow",
    ),
    (
        "/api/solve/dam-break",
        {
            "nx": 50, "ny": 30, "dam_width": 15, "model": "cg",
            "rho_heavy": 0.8, "rho_light": 0.4, "G": 0.9,
            "tau": 1.0, "g": 5e-5,
            "n_steps": 40, "output_interval": 20,
        },
        "dam_break",
    ),
    (
        "/api/solve/sloshing-tank",
        {
            "nx": 40, "ny": 32, "water_level": 16,
            "rho_water": 0.8, "rho_air": 0.4, "G": 0.9,
            "tau": 1.0, "g": 2e-5, "forcing_amp": 0.0,
            "forcing_omega": 0.0,
            "n_steps": 40, "output_interval": 20,
        },
        "sloshing_tank",
    ),
    (
        "/api/solve/sphere-flow",
        {
            "nx": 30, "ny": 20, "nz": 20, "u_in": 0.05, "re": 20.0,
            "radius": 3.0,
            "n_steps": 20, "output_interval": 10,
        },
        "sphere_flow",
    ),
    (
        "/api/solve/ship-hull",
        {
            "nx": 40, "ny": 20, "nz": 16, "u_in": 0.05, "re": 50.0,
            "hull_length": 20.0, "hull_beam": 4.0, "hull_draft": 4.0,
            "smagorinsky_cs": 0.1, "wave_amp": 0.0, "wave_period": 100.0,
            "n_steps": 20, "output_interval": 10,
        },
        "ship_hull",
    ),
    (
        "/api/solve/porous-drainage",
        {
            "nx": 40, "ny": 24, "medium": "random_cylinders",
            "model": "cg", "porosity": 0.6,
            "n_steps": 40, "output_interval": 20,
        },
        "porous_drainage",
    ),
]


@pytest.mark.parametrize(("path", "payload", "job_type"), SOLVER_CASES)
def test_solver_endpoint_runs(client, waiter, path, payload, job_type):
    """Submit a solver job and verify it reaches the COMPLETED state."""
    r = client.post(path, json=payload)
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    assert job_id

    final = waiter(job_id, timeout=180.0)
    assert final["status"] == "completed", (
        f"{path} → status={final['status']}, "
        f"error={final.get('error')[:400] if final.get('error') else None}"
    )
    assert final["job_type"] == job_type
    # All solvers write run_metadata.json into their output directory
    files_r = client.get(f"/api/jobs/{job_id}/files")
    assert files_r.status_code == 200
    paths = [f["path"] for f in files_r.json()["files"]]
    assert any("run_metadata.json" in p for p in paths), paths


def test_solver_validation_error(client):
    """Pydantic validation should reject obviously invalid payloads."""
    # nx below minimum (ge=20)
    r = client.post("/api/solve/cylinder-flow", json={"nx": 1, "ny": 1})
    assert r.status_code == 422


def test_solver_failure_marks_job_failed(client, waiter):
    """A solver that cannot run on the requested device should fail cleanly.

    Submitting a CUDA job on a CPU-only host must record an error and mark
    the job as ``failed`` without crashing the server.
    """
    import torch

    if torch.cuda.is_available():
        pytest.skip("CUDA available – cannot reliably force a device failure")

    r = client.post(
        "/api/solve/cylinder-flow",
        json={
            "nx": 60, "ny": 24, "u_in": 0.05, "re": 50.0,
            "radius": 4.0, "n_steps": 10, "output_interval": 5,
            "device": "cuda:0",
        },
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    final = waiter(job_id, timeout=60.0)
    assert final["status"] == "failed"
    assert final["error"]
