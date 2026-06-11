"""Tests for post-processing endpoints.

Skipped by default (they run a real cylinder simulation per test).  Enable
with ``PLATFORM_SLOW_TESTS=1``.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("PLATFORM_SLOW_TESTS") != "1",
    reason="Post-processing tests opt-in via PLATFORM_SLOW_TESTS=1",
)


def _run_cylinder_job(client, waiter) -> str:
    r = client.post(
        "/api/solve/cylinder-flow",
        json={
            "nx": 60, "ny": 24, "u_in": 0.05, "re": 50.0,
            "radius": 4.0, "n_steps": 40, "output_interval": 20,
        },
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    waiter(job_id, timeout=180.0)
    return job_id


def test_summary_unknown_job(client):
    r = client.get("/api/postprocess/summary/unknown")
    assert r.status_code == 404


def test_summary_for_completed_job(client, waiter):
    job_id = _run_cylinder_job(client, waiter)
    r = client.get(f"/api/postprocess/summary/{job_id}")
    assert r.status_code == 200
    data = r.json()
    assert data["job_id"] == job_id
    assert data["status"] == "completed"
    assert data["png_files"] >= 0
    assert isinstance(data["metadata"], dict)


def test_velocity_profile(client, waiter):
    job_id = _run_cylinder_job(client, waiter)
    r = client.post(
        "/api/postprocess/velocity-profile",
        json={"job_id": job_id, "direction": "y", "position": 0.5},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert len(data["u"]) == len(data["coords"]) > 0
    assert len(data["v"]) == len(data["coords"])


def test_velocity_profile_not_completed(client):
    r = client.post(
        "/api/postprocess/velocity-profile",
        json={"job_id": "unknown", "direction": "y", "position": 0.5},
    )
    assert r.status_code == 404


def test_snapshot_analysis(client, waiter):
    job_id = _run_cylinder_job(client, waiter)
    r = client.get(f"/api/postprocess/snapshot-analysis/{job_id}")
    assert r.status_code == 200
    data = r.json()
    assert data["job_id"] == job_id
    assert data["snapshot_count"] >= 0


def test_csv_endpoint_missing(client, waiter):
    job_id = _run_cylinder_job(client, waiter)
    # Any CSV name not present should produce 404 with a clean message
    r = client.get(f"/api/postprocess/csv/{job_id}/does_not_exist.csv")
    assert r.status_code == 404


def test_csv_endpoint_real_file(client, waiter):
    """If the cylinder solver wrote a 'forces.csv', the parser should return float columns."""
    job_id = _run_cylinder_job(client, waiter)
    files_r = client.get(f"/api/jobs/{job_id}/files")
    csv_files = [f["path"] for f in files_r.json()["files"] if f["path"].endswith(".csv")]
    if not csv_files:
        return  # no CSV produced at this tiny resolution – nothing to assert
    csv_name = csv_files[0].rsplit("/", 1)[-1]
    r = client.get(f"/api/postprocess/csv/{job_id}/{csv_name}")
    assert r.status_code == 200
    data = r.json()
    assert "columns" in data and "data" in data


def test_checkpoints_unknown_job(client):
    r = client.get("/api/postprocess/checkpoints/unknown_job_id")
    assert r.status_code == 404


def test_checkpoints_for_completed_job(client, waiter):
    job_id = _run_cylinder_job(client, waiter)
    r = client.get(f"/api/postprocess/checkpoints/{job_id}")
    assert r.status_code == 200
    data = r.json()
    assert data["job_id"] == job_id
    assert isinstance(data["checkpoints"], list)


def test_field_data_unknown_job(client):
    r = client.get("/api/postprocess/field-data/unknown_job_id")
    assert r.status_code == 404


def test_field_data_velocity_magnitude(client, waiter):
    job_id = _run_cylinder_job(client, waiter)
    r = client.get(f"/api/postprocess/field-data/{job_id}?field=velocity_magnitude")
    # If no checkpoints exist the endpoint returns 422; if checkpoints exist it returns 200.
    assert r.status_code in (200, 422), r.text
    if r.status_code == 200:
        data = r.json()
        assert data["field"] == "velocity_magnitude"
        assert data["nx"] > 0 and data["ny"] > 0
        assert len(data["data"]) == data["nx"] * data["ny"]
        assert len(data["ux"]) == data["nx"] * data["ny"]
        assert len(data["uy"]) == data["nx"] * data["ny"]
        assert data["field_min"] <= data["field_max"]


def test_field_data_all_fields(client, waiter):
    job_id = _run_cylinder_job(client, waiter)
    for field in ("velocity_magnitude", "vorticity", "density", "pressure_coeff", "ux", "uy"):
        r = client.get(f"/api/postprocess/field-data/{job_id}?field={field}")
        assert r.status_code in (200, 422), f"field={field} returned {r.status_code}: {r.text}"


def test_field_data_bad_field(client, waiter):
    job_id = _run_cylinder_job(client, waiter)
    r = client.get(f"/api/postprocess/field-data/{job_id}?field=unknown_field")
    assert r.status_code in (400, 422)
