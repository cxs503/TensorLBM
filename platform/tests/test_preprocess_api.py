"""Tests for the pre-processing endpoints."""
from __future__ import annotations

import io
import struct


def test_polygon_mask(client):
    """Square polygon → mask must produce the expected number of obstacle cells."""
    req = {
        "nx": 40,
        "ny": 20,
        "vertices": [[10, 5], [30, 5], [30, 15], [10, 15]],
    }
    r = client.post("/api/preprocess/polygon-mask", json=req)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["nx"] == 40
    assert data["ny"] == 20
    # 20 * 10 = 200 expected interior cells (allow ± a small polygon-fill margin)
    assert 150 <= data["obstacle_cells"] <= 250
    assert data["fluid_cells"] + data["obstacle_cells"] == 40 * 20
    assert data["image"].startswith("data:image/png;base64,")


def test_polygon_mask_invalid_payload(client):
    # FastAPI/Pydantic validation: missing required field
    r = client.post("/api/preprocess/polygon-mask", json={"nx": 10, "ny": 10})
    assert r.status_code == 422


def test_random_porosity_2d(client):
    req = {"nx": 32, "ny": 32, "porosity": 0.4, "sigma": 4.0, "seed": 1234}
    r = client.post("/api/preprocess/random-porosity-2d", json=req)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["nx"] == 32 and data["ny"] == 32
    assert 0.2 <= data["actual_porosity"] <= 0.6
    assert data["image"].startswith("data:image/png;base64,")


def _minimal_binary_stl() -> bytes:
    """Build a one-triangle binary STL describing a degenerate triangle.

    The triangle has zero area, so any reasonable voxeliser will mark no
    solid cells – we only care that the endpoint accepts the upload and
    returns the expected response shape.
    """
    header = b"\x00" * 80
    n_tri = struct.pack("<I", 1)
    normal = struct.pack("<fff", 0.0, 0.0, 1.0)
    v0 = struct.pack("<fff", 0.0, 0.0, 0.0)
    v1 = struct.pack("<fff", 1.0, 0.0, 0.0)
    v2 = struct.pack("<fff", 0.0, 1.0, 0.0)
    attr = struct.pack("<H", 0)
    return header + n_tri + normal + v0 + v1 + v2 + attr


def test_voxelize_stl(client):
    """Upload a tiny STL and verify the basic response contract."""
    stl_bytes = _minimal_binary_stl()
    files = {"file": ("triangle.stl", io.BytesIO(stl_bytes), "model/stl")}
    r = client.post(
        "/api/preprocess/voxelize-stl",
        files=files,
        params={"nx": 16, "ny": 16, "nz": 16},
    )
    # The voxeliser may either succeed or reject the degenerate mesh; in
    # both cases the API should return a well-defined status (no 500s).
    assert r.status_code in (200, 422), r.text
    if r.status_code == 200:
        data = r.json()
        assert data["nx"] == 16
        assert data["solid_cells"] + data["fluid_cells"] == 16 ** 3


def test_voxelize_stl_rejects_non_stl_extension(client):
    files = {"file": ("triangle.txt", io.BytesIO(_minimal_binary_stl()), "text/plain")}
    r = client.post(
        "/api/preprocess/voxelize-stl",
        files=files,
        params={"nx": 16, "ny": 16, "nz": 16},
    )
    assert r.status_code == 422


def test_unit_converter(client):
    req = {
        "phys_length_m": 1.0,
        "phys_velocity_ms": 1.0,
        "phys_nu_m2s": 1e-3,
        "lbm_length": 100.0,
        "lbm_velocity": 0.1,
    }
    r = client.post("/api/preprocess/units", json=req)
    assert r.status_code == 200, r.text
    data = r.json()
    assert abs(data["reynolds_number"] - 1000.0) < 1e-6
    assert data["lbm_tau"] > 0.5
    assert data["stable"] is True
    assert data["mach_number"] > 0
