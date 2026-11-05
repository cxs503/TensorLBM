"""Tests for platform liveness/status, root SPA and frontend fallback."""

from __future__ import annotations

from pathlib import Path


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


_FRONTEND_INDEX = Path(__file__).resolve().parent.parent / "frontend-vue" / "dist" / "index.html"
_BUILT_SPA = _FRONTEND_INDEX.is_file()
# frontend-vue/dist is a git-ignored build artifact: CI (and any source
# checkout without `npm run build`) exercises the missing-bundle branch, a
# developer machine with a build exercises the served-bundle branch.  Both
# are real behaviour and both are asserted - no skip, no empty dist shim.
_MISSING_BUNDLE_HINT = "frontend-vue/dist/index.html"


def test_root_serves_spa(client):
    r = client.get("/")
    if _BUILT_SPA:
        assert r.status_code == 200
        assert b"<html" in r.content.lower()
    else:
        # A missing build must be a clear 404 naming the build step, never a
        # 500 RuntimeError from FileResponse.
        assert r.status_code == 404
        assert _MISSING_BUNDLE_HINT in r.json()["detail"]


def test_spa_fallback(client):
    """Unknown non-API paths should fall back to index.html (client-side routing)."""
    r = client.get("/some/spa/path")
    if _BUILT_SPA:
        assert r.status_code == 200
        assert b"<html" in r.content.lower()
    else:
        assert r.status_code == 404
        assert _MISSING_BUNDLE_HINT in r.json()["detail"]


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
        "/api/orchestration/templates",
        "/api/orchestration/studies/{study_group}/summary",
        "/api/ai/governance/confidence-gate",
    ]
    for p in expected:
        assert p in paths, f"OpenAPI missing route {p}"
