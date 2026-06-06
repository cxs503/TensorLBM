"""Tests for platform liveness/status, root SPA and frontend fallback."""
from __future__ import annotations


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["service"] == "tensorlbm-platform"


def test_status(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    data = r.json()
    assert data["version"]
    assert "cuda_available" in data
    assert isinstance(data["devices"], list)
    assert "cpu" in data["devices"]
    # The counters must agree
    assert data["total_jobs"] >= (
        data["running_jobs"] + data["completed_jobs"] + data["failed_jobs"]
    )


def test_root_serves_spa(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"<html" in r.content.lower()


def test_spa_fallback(client):
    """Unknown non-API paths should fall back to index.html (client-side routing)."""
    r = client.get("/some/spa/path")
    assert r.status_code == 200
    assert b"<html" in r.content.lower()


def test_openapi(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"]
    # A representative set of expected routes must be present
    expected = [
        "/api/health",
        "/api/status",
        "/api/jobs/",
        "/api/preprocess/units",
        "/api/cad/hull-types",
        "/api/solve/cylinder-flow",
        "/api/solve/cylinder-flow/scan",
        "/api/benchmarks/accuracy",
        "/api/benchmarks/mlups",
    ]
    for p in expected:
        assert p in paths, f"OpenAPI missing route {p}"
