"""Tests for the CAD (ship-hull) endpoints."""
from __future__ import annotations


def test_list_hull_types(client):
    r = client.get("/api/cad/hull-types")
    assert r.status_code == 200
    types = r.json()["hull_types"]
    values = {t["value"] for t in types}
    assert values == {"wigley", "series60", "kcs", "kvlcc2", "npl"}


def test_hull_preview(client):
    req = {
        "hull_type": "wigley",
        "length": 60.0,
        "beam": 8.0,
        "draft": 4.0,
        "n_stations": 7,
    }
    r = client.post("/api/cad/preview", json=req)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["image"].startswith("data:image/png;base64,")
    stats = data["stats"]
    # Wigley analytical Cb = 4/9 ≈ 0.444
    assert 0.30 <= stats["Cb"] <= 0.55
    assert stats["hull_type"] == "wigley"
    # All hull form coefficients must be physically valid (in [0, 1])
    for key in ("Cwp", "Cm", "Cp"):
        assert 0.0 < stats[key] <= 1.0, f"{key}={stats[key]} outside (0, 1]"


def test_hull_mask_small(client):
    """Build a very small hull voxel mask and sanity-check the statistics."""
    req = {
        "hull_type": "wigley",
        "nx": 40, "ny": 20, "nz": 16,
        "length": 30.0, "beam": 6.0, "draft": 6.0,
        "device": "cpu",
    }
    r = client.post("/api/cad/hull-mask", json=req)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["image"].startswith("data:image/png;base64,")
    stats = data["stats"]
    assert "Cb_numerical" in stats
    assert 0.0 < stats["Cb_numerical"] < 1.0


def test_lbm_parameters(client):
    req = {
        "length_m": 100.0,
        "speed_ms": 5.0,
        "nu_m2s": 1.139e-6,
        "lbm_length": 100.0,
        "lbm_speed": 0.05,
    }
    r = client.post("/api/cad/lbm-parameters", json=req)
    assert r.status_code == 200, r.text
    data = r.json()
    # Reynolds number Re = U L / nu = 5*100/1.139e-6 ~ 4.4e8
    assert data["re_physical"] > 1e8


def test_resistance_estimate(client):
    req = {
        "hull_type": "series60",
        "length_m": 120.0,
        "beam_m": 20.0,
        "draft_m": 10.0,
        "speed_ms": 8.0,
        "residual_ratio": 0.2,
    }
    r = client.post("/api/cad/resistance-estimate", json=req)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["hull_type"] == "series60"
    assert data["reynolds"] > 1e8
    assert data["cf_ittc57"] > 0.0
    assert data["total_resistance_n"] > data["friction_resistance_n"] > 0.0


def test_export_stl(client):
    req = {
        "hull_type": "wigley",
        "length": 40.0,
        "beam": 8.0,
        "draft": 4.0,
        "n_long": 10,
        "n_vert": 6,
    }
    r = client.post("/api/cad/export-stl", json=req)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("model/stl")
    # ASCII STL files start with 'solid' or are valid binary STL
    body = r.content
    assert len(body) > 100
