"""Tests for the ISO8 wall-normal table (XVAL-LBM-ISO8-WALL-NORMAL-TABLE)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from tensorlbm.boundary import wall_normal_iso8 as W  # noqa: E402
import xval_iso8_wall_normal_table as H  # noqa: E402

ART = REPO / "artifacts" / "xval_iso8_wall_normal_table"


# --------------------------------------------------------------------------- #
# provenance: the module tables must still equal the vendored OSS source
# --------------------------------------------------------------------------- #

def test_stencil_matches_vendored_oss_source():
    st = H.extract_stencil()
    assert np.allclose(W.ISO8_WEIGHTS, st["iso8_weights"], rtol=0, atol=0)
    for axis, key in enumerate("xyz"):
        src = sorted((w, tuple(h)) for w, h in
                     ((d["w"] and W.ISO8_WEIGHTS[d["w"]] or 0.0, tuple(d["h"]))
                      for d in st["iso8_pairs"][key]))
        mine = sorted((w, tuple(h)) for w, h in W.ISO8_PAIRS[axis])
        # the extracted json stores the weight *index* under "w"
        idx = {}
        for d in st["iso8_pairs"][key]:
            idx.setdefault(d["w"], W.ISO8_WEIGHTS[d["w"]])
        expect = sorted((W.ISO8_WEIGHTS[d["w"]], tuple(d["h"])) for d in st["iso8_pairs"][key])
        assert mine == expect, f"component {key} drifted from the OSS source"
    k = st["smoothing"]["kernel_d3q27_by_squared_speed"]
    assert np.allclose(W.D3Q27_KERNEL, k, rtol=0, atol=0)
    assert abs(sum(W.D3Q27_KERNEL[o[0] ** 2 + o[1] ** 2 + o[2] ** 2]
                   for o in W.D3Q27_OFFSETS) - 1.0) < 1e-14
    assert len(W.ISO8_PAIRS[0]) == len(W.ISO8_PAIRS[1]) == len(W.ISO8_PAIRS[2]) == 34
    assert st["smoothing"]["n_iterations"] == W.DEFAULT_SMOOTH_ITERATIONS


# --------------------------------------------------------------------------- #
# operator properties
# --------------------------------------------------------------------------- #

def test_gradient_is_exact_on_linear_field_and_isotropic():
    n = 40
    ijk = np.indices((n, n, n)).astype(float)
    gains = []
    for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1),
              (0, 1, 1), (1, 1, 1), (1, -1, 1), (2, -1, 3)):
        v = np.asarray(d, float) / np.linalg.norm(d)
        f = v[0] * ijk[0] + v[1] * ijk[1] + v[2] * ijk[2]
        g = np.stack(W.iso8_gradient(f))
        gains.append(np.linalg.norm(g, axis=0)[5:-5, 5:-5, 5:-5].mean())
    gains = np.array(gains)
    assert gains.max() / gains.min() - 1.0 < 1e-12, "operator is anisotropic"
    assert abs(gains.mean() - W.iso8_stencil_scale()) < 1e-12


def test_lattice_aligned_wall_normal_is_exact():
    n = 24
    walls = np.zeros((n, n, n), np.int8)
    walls[: n // 2] = 1                     # solid wall normal to x
    t = W.wall_normal_table(walls)
    idx = t["indices"]
    phys = W.in_physical_domain(idx, walls.shape)
    # nodes whose whole 5x5x5 support is on the flat part of the wall
    m = phys & (idx[:, 0] >= 4) & (idx[:, 0] <= n - 4)
    assert m.sum() > 100
    interior = m & (idx[:, 1] >= 4) & (idx[:, 1] <= n - 4) & (idx[:, 2] >= 4) & (idx[:, 2] <= n - 4)
    nrm = t["normals"][interior]
    # solid occupies x < n/2, so "into the solid" is -x
    assert np.allclose(nrm, np.array([-1.0, 0.0, 0.0]), atol=1e-12), "flat wall normal must be exact"


def test_normal_points_into_solid_and_is_unit():
    walls = H.sphere_walls(20, 6.0, (10.2, 9.9, 10.4))
    t = W.wall_normal_table(walls)
    assert np.allclose(np.linalg.norm(t["normals"], axis=1), 1.0, atol=1e-13)
    idx = t["indices"]
    r = idx - np.array([10.2, 9.9, 10.4])
    assert np.all((t["normals"] * r).sum(1) < 0), "normals must point into the solid"


# --------------------------------------------------------------------------- #
# regression against the compiled OSS preprocessor (stored fixture)
# --------------------------------------------------------------------------- #

def _load_fixture():
    case = json.loads((ART / "oss_fixture_case.json").read_text())
    rows = np.loadtxt(ART / "oss_fixture_normal_table.csv", delimiter=",", skiprows=1)
    return case, rows


def test_matches_oss_fixture_to_machine_precision():
    if not (ART / "oss_fixture_normal_table.csv").exists():
        pytest.skip("OSS fixture not generated (run scripts/xval_iso8_wall_normal_table.py run)")
    case, rows = _load_fixture()
    walls = (H.sphere_walls(case["n"], case["geom"]["radius"], case["geom"]["center"])
             if case["kind"] == "sphere"
             else H.plane_walls(case["n"], case["geom"]["normal"], case["geom"]["offset"]))
    t = W.wall_normal_table(walls)
    idx = rows[:, :3].astype(np.int64)
    ref = rows[:, 3:]
    assert np.array_equal(idx, t["indices"]), "boundary-node table drifted"
    assert np.abs(ref - t["normals"]).max() < 1e-12, "normals drifted from the OSS table"


def test_smoothing_improves_staircase_normals():
    walls = H.sphere_walls(32, 8.0, (16.2, 15.7, 16.4))
    e = []
    for it in (0, 4):
        t = W.wall_normal_table(walls, n_smooth=it)
        r = t["indices"] - np.array([16.2, 15.7, 16.4])
        r = r / np.linalg.norm(r, axis=1, keepdims=True)
        # normals point into the solid, so the error is 180 deg - the angle
        e.append(180.0 - np.degrees(np.arccos(
            np.clip((t["normals"] * r).sum(1), -1, 1))).mean())
    assert e[1] < e[0], f"smoothing must reduce the normal error ({e})"


# --------------------------------------------------------------------------- #
# L3: platform-side caller (main-solver / platform convention)
# --------------------------------------------------------------------------- #

def test_platform_caller_conventions_and_parity():
    from tensorlbm.boundary import build_wall_normal_table
    walls = H.sphere_walls(24, 8.0, (12.3, 11.7, 12.1))
    t = build_wall_normal_table(walls)
    assert t.n_nodes > 100
    # 0-based indices inside the physical box
    assert t.indices.min() >= 0
    assert (t.indices < np.array(walls.shape)).all()
    # unit normals pointing OUT of the solid
    assert np.allclose(np.linalg.norm(t.normals, axis=1), 1.0, atol=1e-12)
    c = np.array([12.3, 11.7, 12.1])
    r = t.indices.astype(float) + 1.0 - c
    assert np.all((t.normals * r).sum(1) > 0), "platform normals must point out of the solid"
    # exact parity with the raw OSS-convention table (sign flip + reindex + filter)
    raw = W.wall_normal_table(walls)
    keep = W.in_physical_domain(raw["indices"], walls.shape)
    assert np.array_equal(t.indices, raw["indices"][keep] - 1)
    assert np.allclose(t.normals, -raw["normals"][keep], atol=1e-14)
    assert np.array_equal(t.grad_mag, raw["grad_mag"][keep])


def test_platform_caller_rejects_bad_input():
    from tensorlbm.boundary import build_wall_normal_table
    with pytest.raises(ValueError):
        build_wall_normal_table(np.zeros((8, 8), np.int8))
    with pytest.raises(TypeError):
        build_wall_normal_table(np.zeros((8, 8, 8), float))
    with pytest.raises(ValueError):
        build_wall_normal_table(np.zeros((8, 8, 8), np.int8) + 2)


# --------------------------------------------------------------------------- #
# well-posedness certificate: which rows are a property of the geometry?
# --------------------------------------------------------------------------- #

def _unit_err(nrm, idx, kind, geom):
    ref = H.analytic_normals(idx, kind, geom)
    return np.linalg.norm(nrm - ref, axis=1)


def test_well_posed_mask_flat_wall_touches_only_the_faces_the_wall_reaches():
    n = 24
    walls = np.zeros((n, n, n), np.int8)
    walls[: n // 3] = 1                        # solid slab against the x=1 face
    r = W.support_radius()
    assert r == W.GRADIENT_RADIUS + W.DEFAULT_SMOOTH_ITERATIONS
    # probe along x at fixed (j,k): Fortran 1-based indices, as the table reports
    xs = np.arange(1, n + 1)
    idx = np.stack([xs, np.full_like(xs, 12), np.full_like(xs, 15)], axis=1)
    wp = W.well_posed_mask(idx, walls)
    # the slab fills the x=1 face, so every row whose support reaches it is ill posed
    near_lo = idx[:, 0] - r <= 0
    assert near_lo.any() and not wp[near_lo].any()
    # the x=n face is pure fluid there: the replication is exact and harmless
    near_hi = idx[:, 0] + r > n
    assert near_hi.any() and wp[near_hi].all()
    # rows whose whole support is inside the physical box are unaffected
    assert wp[(idx[:, 0] - r > 0) & (idx[:, 0] + r <= n)].all()


def test_well_posed_mask_flags_rows_outside_the_physical_box():
    walls = np.zeros((16, 16, 16), np.int8)
    walls[:8] = 1
    t = W.wall_normal_table(walls)
    idx = t["indices"]
    phys = W.in_physical_domain(idx, walls.shape)
    wp = W.well_posed_mask(idx, walls)
    assert phys.sum() < len(idx), "the overlap band should be listed"
    assert not wp[~phys].any(), "rows outside the box cannot be well posed"
    # a domain face declared as a wall makes its whole band ill posed
    wp2 = W.well_posed_mask(idx, walls, domain_walls=(0, 1, 0, 0, 0, 0))
    assert not wp2[phys & (idx[:, 0] + W.support_radius() > walls.shape[0])].any()


def test_well_posed_certificate_tilted_plane():
    """Every row outside the 15 % band must be ill posed, none inside it."""
    n, geom = 32, {"normal": (1.0, 1.0, 0.0), "offset": 16.0}
    walls = H.plane_walls(n, geom["normal"], geom["offset"])
    t = W.wall_normal_table(walls)
    idx = t["indices"]
    phys = W.in_physical_domain(idx, walls.shape)
    idx, nrm = idx[phys], t["normals"][phys]
    err = _unit_err(nrm, idx, "plane", geom)
    wp = W.well_posed_mask(idx, walls)
    assert wp.sum() > 100 and (~wp).sum() > 100, "both classes must be populated"
    bad = err >= 0.15
    assert bad.sum() > 0, "the tilted plane must still expose junction rows"
    assert not (bad & wp).any(), "a row outside the 15 % band was called well posed"
    assert err[wp].max() < 0.15, f"well-posed max error {err[wp].max():.3f}"


def test_sphere_table_is_fully_well_posed_within_the_band():
    walls = H.sphere_walls(32, 8.0, (16.2, 15.7, 16.4))
    t = W.wall_normal_table(walls)
    idx = t["indices"]
    phys = W.in_physical_domain(idx, walls.shape)
    wp = W.well_posed_mask(idx[phys], walls)
    assert wp.all(), "a sphere inside the box has no wall/face junction"
    err = _unit_err(t["normals"][phys], idx[phys], "sphere",
                    {"radius": 8.0, "center": (16.2, 15.7, 16.4)})
    assert err.max() < 0.15, f"sphere per-row error {err.max():.3f} exceeds 15 %"


def test_platform_caller_reports_and_can_drop_ill_posed_rows():
    from tensorlbm.boundary import build_wall_normal_table
    n, geom = 24, {"normal": (1.0, 1.0, 0.0), "offset": 12.0}
    walls = H.plane_walls(n, geom["normal"], geom["offset"])
    keep = build_wall_normal_table(walls)
    assert keep.well_posed is not None
    assert 0 < keep.well_posed.sum() < keep.n_nodes
    drop = build_wall_normal_table(walls, drop_ill_posed=True)
    assert drop.n_nodes == int(keep.well_posed.sum())
    assert drop.well_posed.all()
    # the surviving rows are exactly the well-posed subset, unchanged
    assert np.array_equal(drop.indices, keep.indices[keep.well_posed])
    assert np.array_equal(drop.normals, keep.normals[keep.well_posed])
