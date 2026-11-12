"""Regression lock for the RB-onset control arm (m16479).

The numbers live in ``artifacts/ra1708_control/control_arm.json``, produced by
``benchmarks/case_pool/rayleigh_benard_onset/control_arm_check.py``.  Two claims
are locked here:

1. the independent continuum control arm (FD + QZ + Richardson) reproduces the
   textbook rigid-rigid value ``27 pi^4 / 4 = 1707.7597`` inside its own 0.05 %
   gate -- this is the precondition that makes any LBM deviation attributable to
   the lattice operator rather than to a wrong reference;
2. the LBM production arms drive ``kH = pi`` (mode m = 1, Gamma = 2), **not** the
   most-unstable ``a_c = 3.11632``, so the admissible reference there is
   ``Ra_c(a = pi)``; the offset is quantified and must stay negligible against
   the case-pool 2 % gate.

The expensive recomputation is opt-in (``RB1708_RECOMPUTE=1``) so the default
test stays a seconds-scale artefact check on a loaded control plane.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "artifacts" / "ra1708_control" / "control_arm.json"
CASE = REPO / "benchmarks" / "case_pool" / "rayleigh_benard_onset"
RA_REF = 1707.7597
A_REF = 3.11632


def _load():
    return json.loads(ART.read_text())


def _driver():
    spec = importlib.util.spec_from_file_location("rb1708_driver_t", CASE / "run_case.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_artifact_present():
    if not ART.exists():
        pytest.skip("artifacts/ra1708_control/control_arm.json absent (run control_arm_check.py)")
    d = _load()
    for key in ("route_c_control", "wavenumber_offset", "inputs"):
        assert key in d, key


def test_control_arm_reproduces_textbook():
    if not ART.exists():
        pytest.skip("control arm artefact absent")
    ctrl = _load()["route_c_control"]
    assert ctrl["control_ok"] is True
    assert abs(ctrl["Ra_c_richardson"] - RA_REF) / RA_REF * 100.0 < ctrl["control_tol_pct"]
    # the same reading must also pass the (much looser) case-pool gate
    assert abs(ctrl["Ra_c_richardson"] - RA_REF) / RA_REF * 100.0 < 2.0


def test_driven_wavenumber_offset_is_quantified_and_small():
    if not ART.exists():
        pytest.skip("control arm artefact absent")
    w = _load()["wavenumber_offset"]
    assert w["a_lbm_mode1_gamma2"] == pytest.approx(3.141592653589793, rel=1e-12)
    # reference at the driven wavenumber is *below* the textbook minimum only by a
    # flat-bottom amount; it must never be used as a >1 % excuse for an LBM miss
    assert w["ra_c_at_a_lbm"] < RA_REF
    assert abs(w["rel_offset_pct"]) < 1.0
    # curvature > 0: the marginal curve has a minimum (flat bottom), so mode m=1
    # cannot be systematically off by the 2 % gate
    assert w["d2Ra_da2_at_a_lbm"] > 0.0


def test_predeclared_unchanged_by_this_round():
    """This round only *diagnoses* the reference; the frozen rule must not drift."""
    drv = _driver()
    assert drv.PREDECLARED["ra_ref"] == pytest.approx(RA_REF)
    assert drv.PREDECLARED["tol_pct"] == pytest.approx(2.0)
    assert drv.PREDECLARED["mode"] == 1
    ny, nx = drv.PREDECLARED["primary_grid"]
    assert nx / ny == 2  # Gamma = 2  =>  kH = pi


@pytest.mark.skipif(os.environ.get("RB1708_RECOMPUTE") != "1",
                    reason="set RB1708_RECOMPUTE=1 to redo the ~75 s continuum control arm")
def test_recompute_matches_artifact():
    drv = _driver()
    ctrl = drv.route_c_control()
    d = _load()
    assert ctrl["Ra_c_richardson"] == pytest.approx(d["route_c_control"]["Ra_c_richardson"], rel=1e-9)