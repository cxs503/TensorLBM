#!/usr/bin/env python3
"""Regression tests for the DTMB 5415 resistance caliber (m452, fix B).

These cover the three defects the m438 SDAA sweep tripped over:

1. the waterline-clipped STL was **open**, so the voxel body was a pillar with
   Cb ~ 18 (35x the displacement) -> ``_close_waterline.py`` + the geometry
   gate in ``run.py``;
2. ``Cd_total`` was compared against friction-only / Fr-similar references
   (category + similarity error) -> ``resistance_criteria`` frames;
3. a force window that had not settled was reported as a steady Cd ->
   ``_window_stats``.

No GPU, no solver: everything here is a pure function of numbers or of the
committed STL files.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
for _p in (str(_REPO_ROOT / "src"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_spec = importlib.util.spec_from_file_location("dtmb_run", _HERE / "run.py")
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)  # type: ignore[union-attr]

STL_OPEN = _HERE / "geometry" / "dtmb5415_underwater.stl"
STL_CLOSED = _HERE / "geometry" / "dtmb5415_underwater_closed.stl"

# Geometry numbers measured by ``python _close_waterline.py`` / engine setup
# (kept literal so the tests do not need a GPU or a 3 s voxelisation).
L32_OPEN = {"n_solid_cells": 3439, "stair_wall_faces": 2349}   # open STL, L=32
L32_CLOSED = {"n_solid_cells": 79, "stair_wall_faces": 222}    # capped STL, L=32
L48_CLOSED = {"n_solid_cells": 292, "stair_wall_faces": 580}   # capped STL, L=48
S_WET_LB_32 = 153.4   # 3021.7 m2 / (142/32)^2
S_WET_LB_48 = 345.3


def _geom(l_cells: int, measured: dict, s_wet_lb: float) -> dict:
    return run.geometry_resolution_report(
        n_solid_cells=measured["n_solid_cells"],
        dx_m=run.LPP / l_cells,
        stair_wall_faces=measured["stair_wall_faces"],
        smooth_wetted_lb=s_wet_lb,
        near_wall_cells=None,
    )


# --------------------------------------------------------------------------
# 1. wetted area / waterline cap
# --------------------------------------------------------------------------
@pytest.mark.skipif(not STL_OPEN.exists(), reason="STL not committed")
def test_wetted_area_is_geometry_independent():
    """Open and capped STL must give the same wetted area (cap excluded)."""
    open_bd = run.stl_wetted_area_breakdown(STL_OPEN, 48, run.DRAFT)
    assert open_bd["cap_source"] == "none"
    assert open_bd["wetted_area_m2"] == pytest.approx(3021.7, rel=2e-3)
    if STL_CLOSED.exists():
        closed = run.stl_wetted_area_breakdown(STL_CLOSED, 48, run.DRAFT)
        assert closed["cap_source"] == "sidecar"
        # the cap is ~2955 m2: forgetting it would halve every Cd
        assert closed["cap_area_m2"] > 0.5 * closed["wetted_area_m2"]
        assert closed["wetted_area_m2"] == pytest.approx(
            open_bd["wetted_area_m2"], rel=2e-3
        )


def test_cap_exclusion_uses_sidecar_not_heuristics():
    """A file without a sidecar must not be silently de-capped."""
    bd = run.stl_wetted_area_breakdown(STL_OPEN, 48, run.DRAFT)
    assert bd["cap_area_m2"] == 0.0
    assert bd["n_tri_cap"] == 0


def test_open_stl_is_not_watertight():
    """The m438 defect: the waterline clip leaves the body open (942 edges)."""
    import importlib.util as _iu

    spec = _iu.spec_from_file_location("dtmb_close", _HERE / "_close_waterline.py")
    close = _iu.module_from_spec(spec)
    spec.loader.exec_module(close)  # type: ignore[union-attr]
    from tensorlbm.stl_geometry import read_stl

    verts, faces, _ = read_stl(str(STL_OPEN))
    wv, wf = close._welded(np.asarray(verts, float), np.asarray(faces, np.int64).reshape(-1, 3))
    assert close.is_closed(wf) is False
    assert len(close.boundary_edges(wf)) == 942


@pytest.mark.skipif(not STL_CLOSED.exists(), reason="capped STL not committed")
def test_capped_stl_is_watertight():
    """Fix B: every edge of the capped mesh is shared by exactly two triangles."""
    import importlib.util as _iu

    spec = _iu.spec_from_file_location("dtmb_close", _HERE / "_close_waterline.py")
    close = _iu.module_from_spec(spec)
    spec.loader.exec_module(close)  # type: ignore[union-attr]
    from tensorlbm.stl_geometry import read_stl

    verts, faces, _ = read_stl(str(STL_CLOSED))
    wv, wf = close._welded(np.asarray(verts, float), np.asarray(faces, np.int64).reshape(-1, 3))
    assert close.is_closed(wf) is True
    assert len(close.boundary_edges(wf)) == 0
    side = json.loads(Path(str(STL_CLOSED) + ".capinfo.json").read_text())
    assert side["closed"] is True
    assert side["cap_triangle_count"] == 942


# --------------------------------------------------------------------------
# 2. geometry gate
# --------------------------------------------------------------------------
def test_open_stl_body_is_rejected():
    """The unclosed STL voxelises to a pillar: Cb ~ 18 -> UNDER_RESOLVED."""
    g = _geom(32, L32_OPEN, S_WET_LB_32)
    assert g["cb_voxel"] > 3.0
    assert g["cb_error_pct"] > 100.0
    assert g["stair_over_smooth_wetted_area"] > 10.0
    assert g["verdict"] == "UNDER_RESOLVED"


def test_capped_stl_body_is_rejected_below_two_cells_of_draft():
    """T/dx = 1.4 at L=32 -> below the 2-cell gate, even though Cb improves."""
    g = _geom(32, L32_CLOSED, S_WET_LB_32)
    assert g["T_over_dx"] == pytest.approx(1.39, abs=0.01)
    assert g["thickness_resolved"] is False
    assert g["verdict"] == "UNDER_RESOLVED"


def test_capped_stl_body_is_accepted_at_l48():
    g = _geom(48, L48_CLOSED, S_WET_LB_48)
    assert g["T_over_dx"] == pytest.approx(2.08, abs=0.01)
    assert abs(g["cb_error_pct"]) < 15.0
    assert g["stair_over_smooth_wetted_area"] == pytest.approx(1.68, abs=0.1)
    assert g["verdict"] == "VALID"


# --------------------------------------------------------------------------
# 3. acceptance frames
# --------------------------------------------------------------------------
def _criteria(cd_f, cd_tot=None, drift=0.0, geom_verdict="VALID"):
    re = 1000.0
    cf_blasius = 1.328 / math.sqrt(re)
    geom = {"verdict": geom_verdict, "T_over_dx": 2.08, "cb_error_pct": -10.4,
            "stair_over_smooth_wetted_area": 1.68}
    return run.resistance_criteria(
        re=re,
        cd_p=(cd_tot or cd_f) - cd_f,
        cd_f=cd_f,
        cd_tot=cd_tot if cd_tot is not None else cd_f,
        cf_blasius=cf_blasius,
        cf_ittc57=0.075 / (math.log10(re) - 2.0) ** 2,
        ct_exp=4.2e-3,
        exp_fr=0.28,
        fr_run=1e-3 / math.sqrt(9.81 * run.LPP),
        no_wave_floor_pct=-22.4,
        geometry=geom,
        force_window_drift_pct=drift,
    )


def test_exp_frame_is_declared_not_comparable():
    c = _criteria(0.0420)
    exp = c["frames"]["total_vs_exp_CT_fr0.28"]
    assert exp["comparable"] is False
    assert exp["pass"] is None
    assert "no free surface" in exp["reason"]
    assert exp["structural_floor_pct"] == -22.4


def test_total_vs_friction_line_is_a_category_error():
    c = _criteria(0.0420)
    frame = c["frames"]["total_vs_ittc1957"]
    assert frame["comparable"] is False
    assert frame["pass"] is None


def test_friction_vs_blasius_is_the_pass_fail_frame():
    c = _criteria(0.0420)  # exactly Blasius
    f = c["frames"]["friction_vs_blasius_laminar"]
    assert f["comparable"] is True
    assert abs(f["err_pct"]) < 0.05
    assert f["pass"] is True
    assert c["verdict"] == "PASS_IN_FRAME"


def test_band_boundary():
    cf = 1.328 / math.sqrt(1000.0)
    assert _criteria(cf * 1.029)["verdict"] == "PASS_IN_FRAME"
    assert _criteria(cf * 1.031)["verdict"] == "HONEST_FAIL"
    assert _criteria(cf * 0.971)["verdict"] == "PASS_IN_FRAME"
    assert _criteria(cf * 0.969)["verdict"] == "HONEST_FAIL"


def test_geometry_gate_beats_accuracy():
    c = _criteria(0.0420, geom_verdict="UNDER_RESOLVED")
    assert c["verdict"] == "WITHHELD_GEOMETRY"
    assert "T/dx" in c["verdict_reason"]


def test_unsteady_window_is_withheld():
    c = _criteria(0.0420, drift=5.0)
    assert c["verdict"] == "WITHHELD_UNSTEADY"


# --------------------------------------------------------------------------
# 4. force window statistics
# --------------------------------------------------------------------------
def test_window_stats_flags_drift():
    drifting = list(range(100))
    s = run._window_stats(drifting)
    assert s["steady"] is False
    assert s["drift_pct"] == pytest.approx(66.7, abs=0.5)
    flat = [0.042] * 100
    s2 = run._window_stats(flat)
    assert s2["steady"] is True
    assert s2["drift_pct"] == pytest.approx(0.0, abs=1e-9)


def test_window_stats_degenerate_input():
    s = run._window_stats([1.0])
    assert s["steady"] is False