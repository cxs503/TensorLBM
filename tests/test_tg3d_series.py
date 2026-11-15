"""CI guards for the exact time-Taylor-series reference of the 3-D Taylor–Green vortex.

These tests are cheap (a few seconds) but they assert *invariants*, not merely
"the code ran":

* the initial kinetic energy of the classic field equals the analytic ``A²/8``;
* the recursion collapses to the closed form for the z-invariant initial data
  (where the convective term is a gradient), which pins the viscous/normalisation
  convention;
* the series and an independent pseudo-spectral DNS agree;
* the reference really carries 3-D nonlinearity (enstrophy production above the
  pure-viscous decay, non-zero curl of the convective acceleration);
* the D3Q27 path reproduces the reference inside the declared 8 % band.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.tg3d_lbm_check import (
    compare_to_series,
    run_tg3d_lbm,
    series_reference,
    tg3d_parameters,
)
from tensorlbm.tg3d_series import (
    TG3DSeries,
    convective_acceleration,
    default_nfft,
    spectral_dns_reference,
    taylor_green_classic,
    taylor_green_extruded,
)

AMP = 0.05
BOX = 24.0
NU = 0.02


def _series(ic=None, *, order: int = 8, box: float = BOX, nu: float = NU) -> TG3DSeries:
    return TG3DSeries(ic or taylor_green_classic(AMP), box=box, nu=nu, order=order)


def test_initial_energy_matches_analytic():
    ser = _series()
    # u = A sin x cos y cos z -> <u²> = A²/8 ; v likewise ; E = ½·2·A²/8
    assert ser.energy(0.0) == pytest.approx(AMP**2 / 8.0, rel=1e-13)
    # enstrophy of the classic field: <|ω|²>/2 with |k|=2π/box per factor
    assert ser.enstrophy(0.0) > 0.0


def test_dealiasing_band_is_inactive():
    ser = _series(order=10)
    margin = ser.dealias_margin()
    assert margin["support_peak_index"] < margin["cutoff_index"]
    assert margin["nfft"] == default_nfft(10)


def test_extruded_limit_collapses_to_closed_form():
    """z-invariant IC: (u·∇)u is a gradient ⇒ u(t) = u(0) exp(−ν|k|²t) exactly.

    Tolerance 1e-8: the projector annihilates the (numerically) gradient
    convective term only to ~1e-15 relative, and that residue accumulates over
    separated orders.  A wrong viscous/normalisation convention would instead
    show an O(1) discrepancy, which is what this test is for.
    """
    # high order so the truncation of exp(-nu|k|^2 t) itself is below 1e-12
    ser = _series(taylor_green_extruded(AMP), order=16)
    k = 2.0 * math.pi / BOX
    for t in (20.0, 100.0, 240.0):
        exact = math.exp(-2.0 * NU * 2.0 * k**2 * t)
        assert ser.energy(t) / ser.energy(0.0) == pytest.approx(exact, rel=1e-8)
        assert ser.enstrophy(t) / ser.enstrophy(0.0) == pytest.approx(exact, rel=1e-8)


def test_series_matches_independent_spectral_dns():
    steps = 40
    ser = _series(order=10)
    dns = spectral_dns_reference(
        taylor_green_classic(AMP), box=BOX, nu=NU, steps=steps, dt=1.0, n=24, sample_every=20
    )
    for h in dns["history"]:
        t = h["time"]
        # DNS itself carries an RK4-in-time error at dt = 1 (≈1e-6 here); the
        # point is that two independent code paths agree far below the 8 % band
        assert ser.energy(t) == pytest.approx(h["energy"], rel=1e-5)
        assert ser.enstrophy(t) == pytest.approx(h["enstrophy"], rel=1e-4)


def test_reference_carries_three_dimensional_nonlinearity():
    ser = _series(order=12)
    k = 2.0 * math.pi / BOX
    t = 1.0 / (AMP * k)  # one eddy turnover
    pure = math.exp(-2.0 * NU * 3.0 * k**2 * t)  # single-mode viscous decay
    ratio = ser.enstrophy(t) / ser.enstrophy(0.0)
    assert ratio > 1.02 * pure, "enstrophy production missing: reference is linear"
    # the classic IC has a rotational convective acceleration, the extruded one not
    w_classic = convective_acceleration(taylor_green_classic(AMP), box=BOX, n=24)
    w_extruded = convective_acceleration(taylor_green_extruded(AMP), box=BOX, n=24)
    assert w_classic["curl_over_adv"] > 0.1
    assert w_extruded["curl_over_adv"] < 1e-8


def test_truncation_is_reported_and_small_inside_window():
    ser = _series(order=16, box=32.0, nu=0.01)
    t = 100.0  # T = A k t ≈ 1.0
    tail = ser.relative_tail(t)
    assert tail["enstrophy_step_rel"] < 1e-3
    assert ser.dealias_margin()["margin"] > 0.0


def test_lbm_d3q27_matches_series_inside_case_band():
    n = 24
    amp, re = 0.05, 160.0
    p = tg3d_parameters(n, amplitude=amp, reynolds=re)
    ser = series_reference(n, amplitude=amp, reynolds=re, order=12)
    run = run_tg3d_lbm(n, amplitude=amp, reynolds=re, eddy_times=0.6)
    rows = compare_to_series(run, ser, order=n // 2 - 2)
    last = rows[-1]
    assert abs(last["energy_rel_err_pct"]) < 8.0
    assert abs(last["enstrophy_rel_err_pct"]) < 8.0
    assert abs(last["u_l2_rel_err_pct"]) < 8.0
    assert p["tau"] > 0.5
    # the run must be a genuine 3-D field: w is not identically zero
    assert np.abs(run["final"]["w"]).max() > 0.0