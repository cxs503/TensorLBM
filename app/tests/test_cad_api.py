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


# ── SUBOFF mesh3d endpoint ──────────────────────────────────────────────────

class TestSuboffMesh3D:
    """Tests for POST /api/cad/suboff/mesh3d."""

    def _req(self, hull_type="bare_hull", length=80.0, n_axial=20, n_circ=16):
        return {
            "hull_type": hull_type,
            "length": length,
            "radius": 0.0,
            "bow_fraction": 0.233,
            "stern_fraction": 0.252,
            "stern_exponent": 2.0,
            "n_axial": n_axial,
            "n_circ": n_circ,
        }

    def test_bare_hull_returns_positions(self, client):
        r = client.post("/api/cad/suboff/mesh3d", json=self._req("bare_hull"))
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["hull_type"] == "bare_hull"
        n_tri = data["n_triangles"]
        assert n_tri > 0
        assert len(data["positions"]) == n_tri * 9

    def test_with_sail_has_more_triangles(self, client):
        bare = client.post("/api/cad/suboff/mesh3d", json=self._req("bare_hull")).json()
        sail = client.post("/api/cad/suboff/mesh3d", json=self._req("with_sail")).json()
        assert sail["n_triangles"] > bare["n_triangles"], (
            "with_sail should have more triangles than bare_hull (sail box adds 12)"
        )

    def test_full_has_most_triangles(self, client):
        sail = client.post("/api/cad/suboff/mesh3d", json=self._req("with_sail")).json()
        full = client.post("/api/cad/suboff/mesh3d", json=self._req("full")).json()
        assert full["n_triangles"] > sail["n_triangles"], (
            "full should have more triangles than with_sail (4 fin boxes add 48)"
        )

    def test_positions_are_floats(self, client):
        data = client.post("/api/cad/suboff/mesh3d", json=self._req("full")).json()
        pos = data["positions"]
        assert all(isinstance(v, (int, float)) for v in pos[:30]), (
            "All position values should be numeric"
        )

    def test_hull_type_echoed(self, client):
        for ht in ("bare_hull", "with_sail", "full"):
            data = client.post("/api/cad/suboff/mesh3d", json=self._req(ht)).json()
            assert data["hull_type"] == ht

    def test_invalid_hull_type_422(self, client):
        req = self._req()
        req["hull_type"] = "unknown_type"
        r = client.post("/api/cad/suboff/mesh3d", json=req)
        assert r.status_code == 422
