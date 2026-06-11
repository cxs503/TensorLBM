"""Physics capability contract tests for solver endpoints."""
from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    ("path", "payload", "needle"),
    [
        (
            "/api/solve/cylinder-flow",
            {
                "nx": 60,
                "ny": 24,
                "u_in": 0.05,
                "re": 50.0,
                "radius": 4.0,
                "n_steps": 5,
                "output_interval": 5,
                "physics": {"flow_type": "multiphase"},
            },
            "Flow type",
        ),
        (
            "/api/solve/sloshing-tank",
            {
                "nx": 40,
                "ny": 30,
                "water_level": 12,
                "n_steps": 5,
                "output_interval": 5,
                "physics": {
                    "flow_type": "multiphase",
                    "turbulence_model": "smagorinsky_les",
                },
            },
            "Turbulence model",
        ),
        (
            "/api/solve/porous-drainage",
            {
                "nx": 40,
                "ny": 24,
                "porosity": 0.5,
                "n_steps": 5,
                "output_interval": 5,
                "physics": {"flow_type": "multiphase", "multiphase_model": "fe"},
            },
            "Multiphase model",
        ),
    ],
)
def test_solver_rejects_invalid_physics_matrix_combinations(client, path, payload, needle):
    r = client.post(path, json=payload)
    assert r.status_code == 422
    assert needle in r.text


def test_solver_accepts_supported_dynamic_les_combo(client):
    r = client.post(
        "/api/solve/turbulent-channel",
        json={
            "nx": 32,
            "ny": 16,
            "re_tau": 50.0,
            "u_tau": 0.005,
            "smagorinsky_cs": 0.1,
            "n_steps": 5,
            "averaging_start": 0,
            "output_interval": 5,
            "physics": {"turbulence_model": "dynamic_smagorinsky_les"},
        },
    )
    assert r.status_code == 200, r.text
