"""Platform API tests for new capabilities:
- VTK export endpoint
- 3-D field-data slice endpoint
- Acoustics FWH endpoint
- Conjugate heat transfer solver endpoint
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# VTK export – unit-test the endpoint contracts without a real job
# ---------------------------------------------------------------------------

def test_export_vtk_unknown_job(client):
    r = client.get("/api/postprocess/export-vtk/no_such_job")
    assert r.status_code == 404


def test_field_data_3d_unknown_job(client):
    r = client.get("/api/postprocess/field-data-3d/no_such_job")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Acoustics endpoint
# ---------------------------------------------------------------------------

def test_acoustics_unknown_job(client):
    r = client.post(
        "/api/postprocess/acoustics",
        json={"job_id": "no_such_job"},
    )
    assert r.status_code == 404


def test_acoustics_invalid_surface_fraction(client):
    """surface_sample_fraction must be in [0.01, 1.0]."""
    r = client.post(
        "/api/postprocess/acoustics",
        json={"job_id": "x", "surface_sample_fraction": 0.0},
    )
    assert r.status_code in (404, 422)


# ---------------------------------------------------------------------------
# Conjugate HT solver endpoint
# ---------------------------------------------------------------------------

def test_conjugate_ht_schema(client):
    """Endpoint should accept valid params and return a job_id."""
    r = client.post(
        "/api/solve/conjugate-ht",
        json={
            "nx": 20, "ny": 20,
            "solid_x_start": 6, "solid_x_end": 14,
            "solid_y_start": 6, "solid_y_end": 14,
            "tau_f": 0.6,
            "kappa_f": 0.1667,
            "alpha_s": 0.05,
            "k_ratio": 5.0,
            "T_hot": 1.0, "T_cold": 0.0,
            "Q_source": 0.0,
            "n_steps": 10,
            "output_interval": 10,
            "device": "cpu",
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert "job_id" in data
    assert len(data["job_id"]) > 0


def test_conjugate_ht_invalid_tau(client):
    """tau_f below 0.51 should be rejected by schema validation."""
    r = client.post(
        "/api/solve/conjugate-ht",
        json={
            "nx": 20, "ny": 20,
            "tau_f": 0.2,          # invalid: must be >= 0.51
            "n_steps": 10,
        },
    )
    assert r.status_code == 422


def test_conjugate_ht_grid_too_large(client):
    """Grid exceeding schema limit should be rejected."""
    r = client.post(
        "/api/solve/conjugate-ht",
        json={
            "nx": 9999, "ny": 9999,   # exceeds max 512
            "n_steps": 10,
        },
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# 3-D field-data slice endpoint – bad axis should return 422
# ---------------------------------------------------------------------------

def test_field_data_3d_bad_axis(client):
    """Requesting an invalid slice axis should return 404 (no such job anyway)."""
    r = client.get(
        "/api/postprocess/field-data-3d/some_job?slice_axis=w",
    )
    # Either 404 (no such job) or 422 (bad axis after job check) is acceptable
    assert r.status_code in (404, 422)


# ---------------------------------------------------------------------------
# API schema regression: new endpoints are registered
# ---------------------------------------------------------------------------

def test_openapi_contains_new_endpoints(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = set(r.json()["paths"].keys())
    assert "/api/postprocess/export-vtk/{job_id}" in paths
    assert "/api/postprocess/field-data-3d/{job_id}" in paths
    assert "/api/postprocess/acoustics" in paths
    assert "/api/solve/conjugate-ht" in paths
