"""Tests for the 3-D CAD API endpoints."""
from __future__ import annotations

import base64
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


def _create_model(client: TestClient) -> str:
    req = {
        "source_type": "parametric",
        "hull_type": "series60",
        "length": 80.0,
        "beam": 12.0,
        "draft": 6.0,
        "n_long": 20,
        "n_vert": 12,
    }
    r = client.post("/api/cad/3d/models", json=req)
    assert r.status_code == 200, r.text
    return r.json()["model_id"]


def test_cad3d_create_and_get_stats(client: TestClient) -> None:
    model_id = _create_model(client)
    r = client.get(f"/api/cad/3d/models/{model_id}/stats")
    assert r.status_code == 200, r.text
    mesh = r.json()["mesh"]
    assert mesh["vertex_count"] > 0
    assert mesh["face_count"] > 0


def test_cad3d_mesh_and_update(client: TestClient) -> None:
    model_id = _create_model(client)
    r0 = client.get(f"/api/cad/3d/models/{model_id}/mesh")
    assert r0.status_code == 200
    v0 = r0.json()["stats"]["vertex_count"]

    req = {
        "hull_type": "kcs",
        "length": 90.0,
        "beam": 13.0,
        "draft": 7.0,
        "n_long": 22,
        "n_vert": 14,
    }
    r1 = client.put(f"/api/cad/3d/models/{model_id}", json=req)
    assert r1.status_code == 200, r1.text

    r2 = client.get(f"/api/cad/3d/models/{model_id}/mesh")
    assert r2.status_code == 200
    v1 = r2.json()["stats"]["vertex_count"]
    assert v1 != v0


def test_cad3d_export_and_versions(client: TestClient) -> None:
    model_id = _create_model(client)
    rv = client.get(f"/api/cad/3d/models/{model_id}/versions")
    assert rv.status_code == 200
    versions = rv.json()["versions"]
    assert len(versions) >= 1

    rexp = client.post(f"/api/cad/3d/models/{model_id}/export", json={"fmt": "gltf"})
    assert rexp.status_code == 200, rexp.text
    assert rexp.headers["content-type"].startswith("model/gltf+json")


def test_cad3d_lbm_bridge(client: TestClient) -> None:
    model_id = _create_model(client)
    r = client.post(
        f"/api/cad/3d/models/{model_id}/lbm-mask",
        json={"nx": 40, "ny": 20, "nz": 16, "device": "cpu"},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["solid_cells"] > 0
    assert "stats" in data


def test_cad3d_import_stl_model(client: TestClient) -> None:
    stl = """solid tri
facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 1 0 0
    vertex 0 1 0
  endloop
endfacet
endsolid tri
"""
    req = {
        "source_type": "stl",
        "units": "m",
        "file_b64": base64.b64encode(stl.encode("utf-8")).decode("ascii"),
        "filename": "tri.stl",
    }
    r = client.post("/api/cad/3d/models", json=req)
    assert r.status_code == 200, r.text
    model_id = r.json()["model_id"]

    stats = client.get(f"/api/cad/3d/models/{model_id}/stats")
    assert stats.status_code == 200, stats.text
    mesh = stats.json()["mesh"]
    assert mesh["units"] == "m"
    assert mesh["vertex_count"] == 3
    assert mesh["face_count"] == 1
