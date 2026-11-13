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
