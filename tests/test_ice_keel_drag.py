"""Tests for the IN101-N01 ice-keel drag decomposition (``ice_keel_drag``).

The suite is deliberately cheap: every lattice-Boltzmann run here is a
smoke-scale case on a 8-9 k cell grid, so the whole file runs in seconds.  The
production campaign (``artifacts/IN101_N01``) is not re-run here; what is tested
is the machinery, the invariants it relies on, and the documented findings.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.ice_keel_drag import criteria
from tensorlbm.ice_keel_drag.cases import keel_sweep, smooth_reference
from tensorlbm.underice_drag import (KeelArraySpec, UnderIceChannelConfig,
                                     analytic_frontal_area_ratio, cf_log_law,
                                     dean_cf, keel_geometry_stats,
                                     mirrored_solid, native_available,
                                     run_underice_channel,
                                     run_underice_channel_native,
                                     triangular_keel_mask)

TINY = dict(nx=16, ny=16, nz=34)


def _smooth_cfg(**kw) -> UnderIceChannelConfig:
    base = dict(re_tau=20.0, u_tau=0.002, steps=800, warmup=200,
                sample_interval=20, threads=4, init_profile="parabolic",
                noise_amplitude=0.0)
    base.update(TINY)
    base.update(kw)
    return UnderIceChannelConfig(**base)


# ---------------------------------------------------------------------------
# solver verification: the exact plane-Poiseuille anchor
# ---------------------------------------------------------------------------


def test_smooth_channel_matches_exact_poiseuille():
    """Cf of the forced flat channel equals 18/Re_tau^2 (exact laminar value)."""
    rec = run_underice_channel_native(_smooth_cfg())
    cf_exact = 18.0 / rec["re_tau"] ** 2
    assert abs(rec["cf"] / cf_exact - 1.0) < 0.02, (rec["cf"], cf_exact)
    # the state is stationary by construction (exact steady seed)
    assert rec["stationarity_drift"] < 1e-3


def test_negative_control_form_drag_vanishes():
    """Keels off: the wall-pressure integral is exactly zero and C_d = Cf/2."""
    rec = run_underice_channel_native(_smooth_cfg())
    assert rec["tau_form"] == 0.0
    assert abs(rec["cd_total"] / (0.5 * rec["cf"]) - 1.0) < 1e-12
    assert rec["cd_form"] == 0.0


def test_backends_report_the_same_keys():
    """torch and native records must agree on the shared diagnostic keys."""
    pytest.importorskip("torch")
    if not native_available():
        pytest.skip("no C compiler")
    nat = run_underice_channel_native(_smooth_cfg(steps=400, warmup=100))
    tor = run_underice_channel(_smooth_cfg(steps=400, warmup=100, threads=2))
    shared = ("cf", "cd_total", "cd_skin", "cd_form", "tau_total", "tau_form",
              "tau_me", "tau_skin", "closure_residual",
              "momentum_balance_residual", "cf_log_law", "dean_cf")
    missing = [k for k in shared if k not in nat or k not in tor]
    assert not missing, f"back-ends disagree on key set: missing {missing}"
    # the flat-wall statistics must agree to a few per mille at the same grid
    assert abs(nat["cf"] / tor["cf"] - 1.0) < 5e-3


# ---------------------------------------------------------------------------
# geometry: what the sweep actually voxelises
# ---------------------------------------------------------------------------


def test_frontal_area_ratio_is_linear_in_depth():
    """Measured S_fr/S_plan scales linearly with h_keel at fixed lambda."""
    hs = (2, 3, 4)
    areas = []
    for h in hs:
        spec = KeelArraySpec(nx=32, ny=32, nz=66, h_keel=float(h), coverage=0.6,
                             n_ridges=2, height_modulation=0.0)
        mask = triangular_keel_mask(spec)
        stats = keel_geometry_stats(spec, mask, mirrored_solid(mask))
        areas.append(stats["S_fr_x_over_plan"])
    assert areas[0] > 0
    # exactly linear through the origin: S_fr = 0.125 h for this voxelisation
    ratios = [areas[i + 1] / areas[i] for i in range(len(hs) - 1)]
    expected = [hs[i + 1] / hs[i] for i in range(len(hs) - 1)]
    assert all(abs(r / e - 1.0) < 1e-9 for r, e in zip(ratios, expected)), areas


def test_dense_ridge_set_degenerates():
    """Documented finding: n_ridges=4 collapses the base to 1-2 cells.

    With a pitch of 8 cells a triangular base of lambda = 0.15 is one cell wide,
    so h_keel = 3, 5, 7 voxelise to the *same* mask.  The production table
    therefore uses n_ridges = 2 (pitch 16 cells); this test keeps the finding
    from being silently lost.
    """
    masks = []
    for h in (3, 5, 7):
        spec = KeelArraySpec(nx=32, ny=32, nz=66, h_keel=float(h), coverage=0.15,
                             n_ridges=4, height_modulation=0.0)
        masks.append(triangular_keel_mask(spec))
    assert np.array_equal(masks[0], masks[1]) and np.array_equal(masks[1], masks[2])


def test_analytic_frontal_area_formula():
    # 0.5 lambda h / h_eff
    assert analytic_frontal_area_ratio(0.5, 4.0, 32.0) == pytest.approx(0.5 * 0.5 * 4.0 / 32.0)
    with pytest.raises(ValueError):
        analytic_frontal_area_ratio(0.5, 4.0, 0.0)


# ---------------------------------------------------------------------------
# criteria machinery
# ---------------------------------------------------------------------------


def test_gci_recovers_third_order():
    """A cubic-convergent triplet (r=4/3) gives p ~ 3 and a small GCI."""
    val = lambda h: 1.0 + 0.1 * h**3  # noqa: E731
    r = 4.0 / 3.0
    fine, med, coarse = val(1.0), val(r), val(r * r)
    out = criteria.gci(fine, med, coarse, ratio=r)
    assert out["ok"] and out["p"] == pytest.approx(3.0, abs=0.35)
    # Richardson extrapolation must recover the exact value of the triplet
    assert out["f_exact"] == pytest.approx(1.0, abs=1e-6)
    assert out["gci_fine"] < 0.15


def test_gci_flags_non_monotone():
    out = criteria.gci(1.0, 1.0, 2.0)
    assert not out["ok"]


def test_power_fit_recovers_exponent():
    x = np.array([1.0, 2.0, 4.0, 8.0])
    y = 3.0 * x**1.67
    fit = criteria.fit_power(x, y)
    assert fit["ok"] and fit["exponent"] == pytest.approx(1.67, abs=1e-6)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-9)


def _fake_record(case_id, *, cf, cd_form, cd_total, s_fr, family="keel", h=3.0,
                 lam=0.6, grid=32, re_tau=20.0, tau_total=4e-6):
    return {"case_id": case_id, "family": family, "grid": {"nz": grid},
            "tags": {"h_keel": h, "lam": lam}, "cf": cf, "cd_total": cd_total,
            "cd_form": cd_form, "cd_skin": cd_total - cd_form,
            "tau_total": tau_total, "tau_form": cd_form * 1e-4,
            "tau_skin": (cd_total - cd_form) * 1e-4, "tau_me": tau_total,
            "tau_flux": float("nan"), "re_tau": re_tau, "re_h": 130.0,
            "S_fr_over_plan": s_fr, "analytic_S_fr_over_plan": 0.5 * lam * h / 32.0,
            "geometry": {"S_fr_x_over_plan": s_fr}, "closure_residual": 0.5,
            "momentum_balance_residual": 0.01, "stationarity_drift": 1e-3,
            "u_plus_bulk": 6.7, "log_fit": None}


def test_evaluate_reports_all_five_criteria():
    recs = {}
    for nz, cf in ((34, 5.6e-4), (50, 5.58e-4), (66, 5.57e-4)):
        rid = f"smooth_{nz}"
        rec = _fake_record(rid, cf=cf, cd_form=0.0, cd_total=0.5 * cf, s_fr=0.0,
                           family="smooth", grid=nz)
        rec["re_tau"] = 180.0
        rec["re_h"] = 1.0e4
        recs[rid] = rec
    ref = _fake_record("smooth_ref", cf=2.0e-2, cd_form=0.0, cd_total=1.0e-2,
                       s_fr=0.0, family="smooth_ref")
    recs["smooth_ref"] = ref
    for i, (s_fr, cdf) in enumerate(((0.25, 5e-3), (0.5, 1e-2), (0.75, 1.5e-2))):
        recs[f"keel{i}"] = _fake_record(f"keel{i}", cf=2.1e-2, cd_form=cdf,
                                        cd_total=1.05e-2, s_fr=s_fr)
    out = criteria.evaluate(recs)
    assert set(out["verdicts"]) == {"a", "b", "c", "d", "e"}
    assert out["n_criteria"] == 5
    # the synthetic form drag is exactly proportional to S_fr -> exponent 1
    assert out["criteria"]["b_form_scaling"]["fit"]["exponent"] == pytest.approx(1.0, abs=1e-6)
    assert out["criteria"]["b_form_scaling"]["passed"] is True
    # C_d = 1.05e-2 is inside the observed band
    assert out["criteria"]["d_observed_band"]["n_in_band"] == 3


def test_case_tables_are_self_consistent():
    ref = smooth_reference()
    keels = keel_sweep()
    assert len(keels) == len({c.case_id for c in keels}) > 0
    for c in keels:
        assert c.keel is not None
        # the keel cases must run at the same operating point as the control
        assert c.cfg.re_tau == ref.cfg.re_tau
        assert (c.cfg.nx, c.cfg.ny, c.cfg.nz) == (ref.cfg.nx, ref.cfg.ny, ref.cfg.nz)
        assert c.keel.h_keel <= 0.45 * (c.cfg.nz - 2)


def test_cf_reference_correlations():
    """The two empirical references are monotone and positive where documented."""
    from tensorlbm.underice_drag import B_SMOOTH, KAPPA
    assert cf_log_law(180.0) == pytest.approx(
        2.0 / (math.log(180.0) / KAPPA + B_SMOOTH) ** 2)
    assert 0.0 < dean_cf(1.0e4) < dean_cf(1.0e3)