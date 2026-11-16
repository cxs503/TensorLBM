"""Unit tests for the k*xi extrapolation helpers used by
``benchmarks/case_pool/rayleigh_taylor_atwood05/verify/kdelta_similarity_test.py``.

The case report quotes a sharp-interface (k*xi -> 0) residual derived from a
least-squares extrapolation over the similarity ladder.  A quoted extrapolation
is only as trustworthy as the estimator behind it, so the fit forms, the
leave-one-out spread and the estimator definitions are pinned here:

* an exact linear (in k*xi) data set must be recovered to machine precision and
  must extrapolate to its own intercept;
* a curved data set must NOT be reported as linear: the leave-one-out spread has
  to open up, i.e. the diagnostic can actually detect a broken ansatz;
* a purely proportional data set (err = c k*xi) must extrapolate to zero, so the
  "no intercept" hypothesis stays distinguishable from the affine one;
* the artifact reader (``kdelta_convergence.arm_record``) must reproduce the
  numbers that the accepted report quotes, so a later change to the estimator
  definitions cannot silently move the published values.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "case_pool" / "rayleigh_taylor_atwood05"
VERIFY = BENCH / "verify"
if str(VERIFY) not in sys.path:
    sys.path.insert(0, str(VERIFY))

sim = pytest.importorskip("kdelta_similarity_test")
kd = pytest.importorskip("kdelta_convergence")


# ---------------------------------------------------------------------------
# fit forms
# ---------------------------------------------------------------------------
def test_linear_form_recovers_exact_affine_data():
    pts = [(x, -5.5 - 30.4 * x) for x in (0.196, 0.131, 0.098)]
    f = sim.fit_form(pts, "linear")
    e0, m, r2 = kd.ls([p[0] for p in pts], [p[1] for p in pts])
    assert f["e0_pct"] == pytest.approx(e0, abs=1e-12)
    assert f["slope_pct_per_kxi"] == pytest.approx(m, abs=1e-12)
    assert f["r2"] == pytest.approx(r2, abs=1e-12)
    assert f["rms_pct"] == pytest.approx(0.0, abs=1e-12)


def test_quadratic_form_recovers_exact_data_and_differs_from_linear():
    e0, q = -6.0, -100.0
    pts = [(x, e0 + q * x * x) for x in (0.196, 0.131, 0.098)]
    f = sim.fit_form(pts, "quadratic")
    assert f["e0_pct"] == pytest.approx(e0, abs=1e-9)
    assert f["slope_pct_per_kxi2"] == pytest.approx(q, rel=1e-9)
    # the affine form must not reproduce a purely quadratic set exactly
    lin = sim.fit_form(pts, "linear")
    assert lin["rms_pct"] > 1e-3


def test_proportional_form_extrapolates_to_zero_intercept():
    pts = [(x, -66.0 * x) for x in (0.196, 0.131, 0.098)]
    f = sim.fit_form(pts, "proportional")
    assert f["e0_pct"] == 0.0
    assert f["c_pct_per_kxi"] == pytest.approx(-66.0, rel=1e-9)
    assert f["r2"] == pytest.approx(1.0, abs=1e-12)


def test_unknown_form_raises():
    with pytest.raises(ValueError):
        sim.fit_form([(0.1, -1.0), (0.2, -2.0)], "cubic")


# ---------------------------------------------------------------------------
# leave-one-out extrapolation stability
# ---------------------------------------------------------------------------
def test_loo_is_degenerate_on_exact_affine_data():
    """With noiseless affine data every rung removal must give the same e0."""
    pts = [(x, -6.0 - 30.0 * x) for x in (0.196, 0.131, 0.098, 0.065)]
    loo = sim.leave_one_out(pts, "linear")
    assert loo["n"] == 4
    assert loo["e0_spread_pct"] == pytest.approx(0.0, abs=1e-9)
    assert loo["e0_full_pct"] == pytest.approx(-6.0, abs=1e-9)


def test_loo_opens_up_when_the_ansatz_is_wrong():
    """A curved set must produce a non-zero LOO spread (the diagnostic bites)."""
    pts = [(x, -6.0 - 100.0 * x * x) for x in (0.196, 0.131, 0.098, 0.065)]
    loo = sim.leave_one_out(pts, "linear")
    assert loo["n"] == 4
    assert loo["e0_spread_pct"] > 0.2
    assert len(loo["per_dropped_rung"]) == 4


def test_loo_reports_no_result_below_two_points():
    assert sim.leave_one_out([(0.1, -3.0)], "linear") == {"n": 0}


# ---------------------------------------------------------------------------
# artifact reader: the published ladder must stay reproducible
# ---------------------------------------------------------------------------
LADDER_DIR = BENCH / "artifacts" / "runs_kdelta" / "KD_l064"


@pytest.mark.skipif(not (LADDER_DIR / "case.json").is_file(),
                    reason="kdelta ladder artifacts not present in this checkout")
def test_arm_record_reproduces_the_published_coarse_rung():
    rec = kd.arm_record(LADDER_DIR)
    assert rec["lam"] == 64
    assert rec["kxi"] == pytest.approx(2.0 * math.pi * 2.0 / 64.0, rel=1e-12)
    # accepted report: -11.52 % (window estimator, vs sigma_inv) / -10.14 % (vs sigma_visc)
    assert rec["err_W_vs_inv_pct"] == pytest.approx(-11.52, abs=0.02)
    assert rec["err_W_vs_visc_pct"] == pytest.approx(-10.14, abs=0.02)
    # the similarity pins must hold on every rung
    assert rec["zeta"] == pytest.approx(1.538e-2, rel=5e-3)
    assert rec["ma_peak"] < 0.30