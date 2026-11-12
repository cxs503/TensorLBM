"""Tests for the refinement-extrapolation caliber (TensorLBM).

Invariants under test
---------------------
1. A ladder that really follows ``q0 + C h`` is recovered *exactly* (noiseless)
   and its order-1 form is licensed; a wrong-order form is rejected once the
   per-point sigma is small.
2. A *saturated* ladder (no trend) is flagged as such by ``still_converging``
   and its 1/h slope collapses to zero; a form is only *rejected* when the
   residuals are systematic (chi2 cannot reject a form on pure noise -- that
   asymmetry is a property of the test, and the tests pin it explicitly).
3. The verdict must be ``undecided_form`` when the licensed forms disagree by
   more than the tolerance band -- i.e. the caliber refuses to invent a verdict
   the data cannot support.  This is the case-pool RB-onset situation.
4. ``gci_triplet`` needs monotone convergence; oscillatory or saturated
   triplets are refused instead of silently divided by ~0.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.refinement_extrapolation import (  # noqa: E402
    RefinementPoint,
    extrapolation_verdict,
    fit_free_order,
    fit_power_law,
    fit_saturated,
    gci_triplet,
    identify_forms,
    still_converging,
)


def test_first_order_ladder_recovered_exactly():
    q0, C = 100.0, 3.0
    pts = [RefinementPoint(h=1.0 / n, q=q0 + C / n, sigma=1e-9) for n in (8, 16, 32, 64)]
    fit = fit_power_law(pts, power=1.0)
    assert fit["ok"] and fit["licensed"]
    assert abs(fit["q0"] - q0) < 1e-6
    assert abs(fit["C"] - C) < 1e-6
    assert fit["chi2_red"] < 1e-6


def test_wrong_order_form_rejected_with_small_sigma():
    pts = [RefinementPoint(h=1.0 / n, q=100.0 + 3.0 / n, sigma=1e-6) for n in (8, 16, 32)]
    first = fit_power_law(pts, power=1.0)
    second = fit_power_law(pts, power=2.0)
    assert first["licensed"]
    assert not second["licensed"]
    assert "chi2_red" in second["why"]


def test_free_order_recovers_declared_order():
    pts = [RefinementPoint(h=1.0 / n, q=50.0 + 2.0 / n ** 2, sigma=1e-9) for n in (8, 16, 32, 64)]
    fit = fit_free_order(pts)
    assert fit["ok"]
    assert abs(fit["power"] - 2.0) < 0.02
    assert abs(fit["q0"] - 50.0) < 1e-4


def test_saturated_ladder_prefers_the_flat_form():
    """A flat ladder within its own noise: the flat form is licensed, the 1/h
    slope collapses to ~0, and the ladder is flagged as saturated.

    Note the *honesty* property being pinned here: with only random noise the
    chi2 test cannot reject a wrong form (SSR is sigma^2 * dof by construction).
    Rejection power needs systematic residuals -- that is
    ``test_systematic_outlier_rejects_form`` below.
    """
    rng = np.random.default_rng(0)
    truth = 200.0
    pts = [RefinementPoint(h=1.0 / n, q=truth + float(rng.normal(0, 0.05)), sigma=0.05)
           for n in (16, 32, 64, 128)]
    first = fit_power_law(pts, power=1.0)
    sat = fit_saturated(pts)
    assert sat["licensed"] and first["licensed"]
    assert abs(first["C"]) < 3.0 * 0.05 * 128.0     # slope indistinguishable from 0
    sc = still_converging(pts)
    assert sc["saturated_within_noise"]


def test_systematic_outlier_rejects_form():
    """Rejection power: one 4-sigma systematic residual kills the 1/h form."""
    pts = [RefinementPoint(h=1.0 / n, q=100.0 + 3.0 / n + (1.0 if n == 64 else 0.0),
                           sigma=0.05)
           for n in (16, 32, 64)]
    first = fit_power_law(pts, power=1.0)
    assert not first["licensed"], "a 20-sigma residual must not be licensed"
    res = identify_forms(pts)
    assert "h^1" in res["rejected"]


def test_converging_ladder_is_detected():
    pts = [RefinementPoint(h=1.0 / n, q=100.0 + 3.0 / n, sigma=0.001) for n in (16, 32, 64)]
    sc = still_converging(pts)
    assert sc["still_converging"] and not sc["saturated_within_noise"]


def test_verdict_pass_when_band_inside_tolerance():
    pts = [RefinementPoint(h=1.0 / n, q=100.0 + 1.0 / n, sigma=1e-3) for n in (16, 32, 64)]
    res = identify_forms(pts)
    v = extrapolation_verdict(res, ref=100.0, tol_rel=0.02)
    assert v["verdict"] == "pass"


def test_verdict_fail_when_band_outside_tolerance():
    pts = [RefinementPoint(h=1.0 / n, q=110.0 + 1.0 / n, sigma=1e-3) for n in (16, 32, 64)]
    res = identify_forms(pts)
    v = extrapolation_verdict(res, ref=100.0, tol_rel=0.02)
    assert v["verdict"] == "fail"


def test_verdict_form_falsified_when_no_form_licensed():
    pts = [RefinementPoint(h=1.0 / n, q=100.0 + 5.0 * math.sin(7.0 * n), sigma=1e-6)
           for n in (16, 32, 64)]
    res = identify_forms(pts)
    v = extrapolation_verdict(res, ref=100.0, tol_rel=0.02)
    assert not res["licensed"]
    assert v["verdict"] == "form_falsified"


def test_rb_onset_ladder_is_undecided_form():
    """The real CASE-LBM-RB-ONSET-RA1708 ladder (archived artifacts).

    Three grids, bisection resolution floor +-0.4 %.  The single h^1 form says
    Ra_inf = 1755.5 (+2.79 % -> FAIL); the saturated form says 1724.4 (+0.98 %);
    the licensed band spans the whole +-2 % tolerance interval, and the finest
    pair differs by 0.15 sigma.  The caliber must therefore report
    ``undecided_form`` instead of a pass/fail that is really a form choice.
    """
    raw = [(32, 1711.73858423433, 6.891738108795039),
           (48, 1730.353035565442, 7.040612007299549),
           (64, 1731.8585169336582, 7.13070620483161)]
    pts = [RefinementPoint(h=1.0 / n, q=q, sigma=s) for n, q, s in raw]
    res = identify_forms(pts)
    assert res["best_by_aic"] == "h^p"
    first = next(f for f in res["forms"] if f["name"] == "h^1")
    assert abs(first["q0"] - 1755.47) < 0.5
    assert 1.7 < 100.0 * res["band_rel"] < 1.9          # band ~1.785 %
    v = extrapolation_verdict(res, ref=1707.7597, tol_rel=0.02)
    assert v["verdict"] == "undecided_form"
    assert v["band_vs_tol"] > 0.9                        # band ~ the whole gate
    sc = still_converging(pts)
    assert sc["saturated_within_noise"]
    # the three-finest-triplet GCI is applicable (monotone) but saturated
    g = gci_triplet(raw[0][1], raw[1][1], raw[2][1], r=64 / 48)
    assert g["ok"] and g["p_obs"] > 4.0
    assert abs(g["extrapolated"] - 1732.0) < 1.0


def test_gci_triplet_refuses_oscillatory_and_saturated():
    osc = gci_triplet(100.0, 101.0, 100.5, r=2.0)
    assert not osc["ok"] and "oscillat" in osc["reason"]
    sat = gci_triplet(100.0, 100.0, 100.0, r=2.0)
    assert not sat["ok"] and "saturat" in sat["reason"]
    good = gci_triplet(100.0 + 8.0, 100.0 + 2.0, 100.0 + 0.5, r=2.0)
    assert good["ok"] and abs(good["p_obs"] - 2.0) < 1e-6
    assert abs(good["extrapolated"] - 100.0) < 1e-6


def test_audit_script_on_archived_summary(tmp_path):
    """End-to-end: the case-pool audit script runs on the archived summary."""
    import importlib.util

    path = ROOT / "benchmarks/case_pool/rayleigh_benard_onset/grid_ladder_audit.py"
    spec = importlib.util.spec_from_file_location("rb_grid_ladder_audit", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    summary = ROOT / "benchmarks/case_pool/rayleigh_benard_onset/artifacts/summary.json"
    if not summary.exists():
        pytest.skip("archived case summary not present")
    pts, _meta = mod.load_points([summary])
    assert [p["ny"] for p in pts] == [32, 48, 64]
    res = mod.audit(pts)
    assert res["verdict"]["verdict"] in ("undecided_form", "pass", "fail")
    assert res["licensed"], "at least one form must be licensed on the real ladder"
    md = mod.markdown(res)
    assert "verdict = " in md and "licensed forms:" in md