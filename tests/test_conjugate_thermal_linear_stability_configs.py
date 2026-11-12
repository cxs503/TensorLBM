"""Configuration-resolved linear stability of the Boussinesq cavity / Bénard layer.

Regression target (m16921): the module used to advertise ``Ra_c = 1707.762`` for a
**side-heated** square cavity, citing an "independent discretisation" that in
fact never bracketed a root.  These tests pin the two configurations apart:

* the Bénard layer (gravity parallel to the temperature gradient) *does*
  reproduce Chandrasekhar's ``1707.7597`` — including a grid-convergence
  extrapolation and an independent cross-check against
  :mod:`tensorlbm.rb_continuum_stability`;
* the side-heated cavity does **not** — its neutral point is a factor ~8.5
  higher, and the unbracketed API reports that honestly instead of raising a
  ``KeyError`` downstream.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlbm.conjugate_thermal import stability as S

RA_REF = 1707.7597
A_REF = 3.11632


# ---------------------------------------------------------------------------
# Bénard layer: the configuration that the classical anchor belongs to
# ---------------------------------------------------------------------------
def test_benard_operator_structure():
    ops = S.benard_operator(n=16, a=A_REF)
    M = 15
    assert ops["size"] == 2 * M
    assert ops["configuration"] == "rayleigh_benard_layer"
    assert ops["S0"].shape == (2 * M, 2 * M)
    assert np.iscomplexobj(ops["S0"]) and np.iscomplexobj(ops["S1"])
    # only the buoyancy coupling is Ra-dependent: theta rows stay untouched
    assert np.allclose(ops["S1"][M:, :], 0.0)
    assert np.allclose(ops["S1"][:M, :M], 0.0)
    with pytest.raises(ValueError):
        S.benard_operator(n=3)


def test_benard_marginal_sign_change():
    """Below onset stable, well above unstable: sigma(Ra) must change sign."""
    ops = S.benard_operator(n=16, a=np.pi)
    assert S.benard_growth_rate(1000.0, ops) < 0.0
    assert S.benard_growth_rate(3000.0, ops) > 0.0


def test_benard_critical_rayleigh_converges_to_chandrasekhar():
    """Second-order grid convergence onto 1707.7597 (measured errors)."""
    coarse = S.benard_critical_rayleigh(n=16, a=A_REF)
    fine = S.benard_critical_rayleigh(n=32, a=A_REF)
    assert coarse["status"] == "ok" and fine["status"] == "ok"
    assert abs(coarse["deviation_rel"]) < 0.04      # measured -2.9e-2
    assert abs(fine["deviation_rel"]) < 0.01        # measured -7.6e-3
    # refinement must reduce the error, not wander
    assert abs(fine["deviation_rel"]) < abs(coarse["deviation_rel"])
    # second-order Richardson on the two grids lands on the literature value
    extrap = S.richardson([coarse["ra_c"], fine["ra_c"]])
    assert abs(extrap / RA_REF - 1.0) < 5e-3        # measured -3e-4
    # the marginal growth rate at the reported Ra_c is ~0 by construction
    assert abs(fine["sigma_marginal"]) < 0.05


def test_benard_wavenumber_minimum_is_near_3_116():
    """The most dangerous wavenumber is the classical a_c = 3.11632."""
    scan = S.benard_wavenumber_scan(np.array([2.8, 3.0, 3.11632, 3.3, 3.6]), n=16)
    ras = [g["Ra_c"] for g in scan]
    assert all(r is not None for r in ras)
    best = min(scan, key=lambda g: g["Ra_c"])
    assert abs(best["a"] - A_REF) < 0.2
    # a minimum, not a monotone trend: both neighbours are larger
    centre = [g for g in scan if g["a"] == pytest.approx(A_REF)][0]
    for g in scan:
        if g["a"] in (2.8, 3.6):
            assert g["Ra_c"] > centre["Ra_c"]


def test_benard_continuous_extrapolation_is_within_3_percent():
    """End-to-end path: wavenumber scan + grid extrapolation, judged vs ref."""
    out = S.benard_critical_rayleigh_continuous(pr=0.71, n_list=(16, 32), n_a=5,
                                                refine_iter=4)
    assert abs(out["rel_err_by_grid"][-1]) < 0.02
    assert abs(out["rel_err_extrapolated"]) < 0.03
    assert abs(out["a_by_grid"][-1] - A_REF) < 0.25


def test_benard_growth_rate_agrees_with_rb_continuum_module():
    """Independent formulation cross-check (streamfunction pencil vs Thom vorticity).

    ``rb_continuum_stability`` measures sigma in viscous units (``nu/H^2``) while
    this module uses thermal units (``alpha/H^2``); the conversion is a factor
    ``Pr``.  Without it the two modules disagree by 41 %, which is the trap this
    test documents.
    """
    rb = pytest.importorskip("tensorlbm.rb_continuum_stability")
    pr = 0.71
    for ra in (1500.0, 2500.0):
        here = S.benard_growth_rate(ra, n=64, a=A_REF, pr=pr)
        there = rb.growth(a=A_REF, Ra=ra, Pr=pr, M=100)["sigma"]
        assert here < 0.0 if ra < RA_REF else here > 0.0
        assert pr * there == pytest.approx(here, rel=0.03)   # measured <= 0.9 %
        # naive (unconverted) comparison would be off by ~1/Pr
        assert abs(there / here - 1.0) > 0.3


def test_benard_unbracketed_is_reported_not_invented():
    res = S.benard_critical_rayleigh(n=16, a=A_REF, ra_lo=200.0, ra_hi=400.0)
    assert res["ra_c"] is None and res["status"] == "not_bracketed"
    assert res["deviation_rel"] is None


# ---------------------------------------------------------------------------
# Side-heated cavity: a different configuration, and now honestly labelled
# ---------------------------------------------------------------------------
def test_side_heated_cavity_is_stable_at_the_benard_anchor():
    """sigma(Ra=1707.76) << 0: the classical anchor is the wrong configuration."""
    sigma = S.growth_rate(RA_REF, n=12)
    assert sigma < -5.0            # measured -9.75 at n = 12, -9.1 at n = 16
    # and unstable far above it, so the neutral point is a factor ~8 above
    assert S.growth_rate(3.0e4, n=12) > 5.0


def test_side_heated_unbracketed_reports_all_keys():
    res = S.critical_rayleigh(n=8, ra_lo=500.0, ra_hi=4000.0)
    assert res["ra_c"] is None
    assert res["status"] == "not_bracketed"
    # the fragile key used to be missing -> downstream KeyError
    assert res["deviation_rel"] is None
    assert res["ra_c_reference"] is None
    assert res["configuration"] == "side_heated_square_cavity"
    assert res["sigma_bracket"][1] < 0.0


def test_side_heated_recorded_neutral_point():
    """The recorded measured neutral point is bracketed by the module itself."""
    ra = S.SIDE_HEATED_RA_C_MEASURED_N12
    assert 1.0e4 < ra < 2.0e4
    assert ra / RA_REF > 5.0
    res = S.critical_rayleigh(n=8, ra_lo=0.5 * ra, ra_hi=2.0 * ra, tol=1e-3)
    assert res["status"] == "ok"
    assert res["ra_c"] is not None and abs(res["ra_c"] / ra - 1.0) < 0.25