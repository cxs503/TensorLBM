"""Unit + regression tests for :mod:`tensorlbm.weak_compressibility`.

All tests run on deliberately tiny grids so that the file stays inside the
fast CI budget; the *campaign* numbers are produced by
``scripts/run_weak_compressibility_campaign.py`` on the cluster.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlbm import weak_compressibility as wc


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------
def test_equilibrium_moments_are_exact():
    rng = np.random.default_rng(0)
    rho = 1.0 + 1e-3 * rng.standard_normal((5, 6))
    ux = 0.02 * rng.standard_normal((5, 6))
    uy = 0.02 * rng.standard_normal((5, 6))
    f = wc.equilibrium(rho, ux, uy)
    r, x, y = wc.macroscopic(f)
    assert np.allclose(r, rho, atol=1e-14)
    assert np.allclose(x, ux, atol=1e-13)
    assert np.allclose(y, uy, atol=1e-13)


def test_streaming_and_bounce_back_conserve_mass():
    rng = np.random.default_rng(1)
    f = wc.equilibrium(np.ones((6, 8)), 0.05 * np.ones((6, 8)), np.zeros((6, 8)))
    f = f + 1e-6 * rng.standard_normal(f.shape)
    solid = np.zeros((6, 8), dtype=bool)
    solid[0, :] = solid[-1, :] = True
    m0 = f.sum()
    g = wc.stream_periodic(f)
    g, _ = wc.bounce_back(g, solid)
    assert np.isclose(g.sum(), m0, rtol=1e-13)


def test_zou_he_west_reproduces_imposed_velocity():
    """Applying the inlet to a uniform state must reproduce ``u_in`` exactly.

    Regression for the ``f8 <- f7`` typo that silently broke the anti-symmetric
    closure of the Zou--He inlet (it injected a spurious transverse flux).
    """
    ny, nx = 7, 5
    u_in = 0.04
    f = wc.equilibrium(np.ones((ny, nx)), np.full((ny, nx), u_in), np.zeros((ny, nx)))
    inner = f[:, 1:-1, :].copy()
    wc._zou_he_west(inner, u_in, 0.0)
    f[:, 1:-1, :] = inner
    rho, ux, uy = wc.macroscopic(f)
    assert np.allclose(ux[:, 0], u_in, atol=1e-12)
    assert np.allclose(uy[:, 0], 0.0, atol=1e-12)
    assert np.allclose(rho[:, 0], 1.0, atol=1e-12)
    # the transverse flux must stay symmetric (no spurious uy in the bulk)
    assert np.max(np.abs(uy)) < 1e-12


def test_pressure_outlet_reproduces_rho():
    ny, nx = 7, 5
    f = wc.equilibrium(np.ones((ny, nx)), np.full((ny, nx), 0.03), np.zeros((ny, nx)))
    inner = f[:, 1:-1, :].copy()
    wc._zou_he_east(inner, 1.0)
    f[:, 1:-1, :] = inner
    rho, _, _ = wc.macroscopic(f)
    assert np.allclose(rho[:, -1], 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# scaling-law helper
# ---------------------------------------------------------------------------
def test_fit_power_law_recovers_exponent_with_offset():
    machs = np.array([0.01, 0.05, 0.1, 0.2, 0.3])
    offset, pref = 3.0e-2, 0.7
    err = offset + pref * machs**2
    fit = wc.fit_power_law(machs, err, subtract_offset=True)
    assert abs(fit["exponent_excess"] - 2.0) < 1e-3
    assert fit["offset"] == pytest.approx(offset, rel=1e-6)


def test_fit_power_law_plain_exponent():
    machs = np.array([0.01, 0.05, 0.1, 0.2, 0.3])
    fit = wc.fit_power_law(machs, 4.0 * machs**2)
    assert fit["exponent_plain"] == pytest.approx(2.0, abs=1e-10)
    assert fit["raw_loglog_slope"] == pytest.approx(2.0, abs=1e-10)


# ---------------------------------------------------------------------------
# solvers (tiny grids)
# ---------------------------------------------------------------------------
def test_body_force_poiseuille_has_uniform_density():
    """Periodic-streamwise body-force channel: rho is uniform to round-off.

    This is the reason the Ma^2 law is *not* observable in this configuration
    (documented in the campaign report); the test pins that fact.
    """
    rec = wc.run_poiseuille(
        wc.PoiseuilleSetup(mach=0.1, re=5.0, ny=17, nx=6, max_steps=4000, tol=1e-11)
    )
    assert rec["converged"] or rec["steps"] == 4000
    assert rec["drho_over_rho0"] < 1e-10
    num = np.asarray(rec["u_num"])
    exact = np.asarray(rec["u_exact"])
    num = num / num.max()
    exact = exact / exact.max()
    # H=16 cells: the lattice wall-slip / discretisation offset of the
    # bounce-back channel is ~10% here (it scales like H^-2 and is *not* the
    # compressibility error -- the campaign measures it explicitly as the
    # Ma-independent offset v0 of the three-parameter fit).
    assert np.max(np.abs(num - exact)) < 0.15


def test_pressure_channel_is_consistent_and_stable():
    mach = 0.05
    nu = 0.05
    setup = wc.PressureChannelSetup(
        mach=mach, re=mach * wc.CS * 16 / nu, ny=17, nx=32, steps=3000
    )
    rec = wc.run_channel_pressure_driven(setup)
    assert rec["converged"]
    assert np.isfinite(rec["err_inf"])
    # the closed-form hydrostatic estimate must be within a factor of a few
    assert 0.2 < rec["drho_over_rho0"] / rec["drho_analytic"] < 5.0
    # the density field is genuinely non-uniform (unlike the body-force channel)
    assert rec["drho_over_rho0"] > 1e-8
    # centre-line velocity is not grossly wrong
    assert 0.5 < rec["u_max_probe"] / rec["u_max_exact"] < 1.01


def test_acoustic_filter_reduces_density_band_and_keeps_profile():
    common = dict(mach=0.05, nu=0.05, ny=17, nx=32, steps=3000, pulse=1e-3, pulse_sigma=2.0)
    raw = wc.run_channel_acoustic_filter(wc.AcousticFilterSetup(alpha=0.0, **common))
    flt = wc.run_channel_acoustic_filter(wc.AcousticFilterSetup(alpha=0.05, **common))
    assert raw["converged"] and flt["converged"]
    assert flt["drho_acoustic_amplitude"] < raw["drho_acoustic_amplitude"]
    assert abs(flt["err_inf"] - raw["err_inf"]) / raw["err_inf"] < 0.10


def test_acoustic_band_helper_separates_high_frequency():
    t = np.arange(2000, dtype=float)  # one sample per lattice step
    sig = 1e-3 * np.sin(2 * np.pi * t / 3.5) + 1e-6
    band = wc._acoustic_band(sig, 32)
    assert np.max(np.abs(band)) > 0.8e-3


# ---------------------------------------------------------------------------
# verification record
# ---------------------------------------------------------------------------
def _synthetic_sweep(case: str, exponent: float, pref: float, offset: float = 0.0):
    machs = [0.01, 0.05, 0.1, 0.2, 0.3]
    err = [offset + pref * m**exponent for m in machs]
    drho = [0.5 * m**exponent for m in machs]
    return {
        "case": case,
        "machs": machs,
        "err_inf": err,
        "err_l2": err,
        "drho_over_rho0": drho,
        "fit_error_raw": wc.fit_power_law(machs, err),
        "fit_error_excess": wc.fit_power_law(machs, err, subtract_offset=True),
        "fit_drho": wc.fit_power_law(machs, drho),
    }


def test_verification_record_flags_a_correct_ma2_law():
    pois = _synthetic_sweep("poiseuille_mach_sweep", 0.0, 3e-2)
    pc = _synthetic_sweep("pressure_channel_mach_sweep", 2.0, 1.0, offset=1e-3)
    cyl = {
        "machs": [0.01, 0.05, 0.1, 0.2, 0.3],
        "st": [0.165, 0.1653, 0.1656, 0.168, 0.175],
        "cd_mean": [1.30, 1.3005, 1.301, 1.32, 1.35],
        "cl_amplitude": [0.30, 0.3002, 0.3005, 0.31, 0.33],
        "st_ref": 0.165,
        "cd_ref": 1.30,
        "cl_ref": 0.30,
        "rel_st": [0.0, 1.8e-3, 3.6e-3, 1.8e-2, 6.1e-2],
        "rel_cd": [0.0, 3.8e-4, 7.7e-4, 1.5e-2, 3.8e-2],
        "rel_cl": [0.0, 6.7e-4, 1.7e-3, 3.3e-2, 1.0e-1],
        "max_rel_st_low_ma": 3.6e-3,
        "max_rel_cd_low_ma": 7.7e-4,
        "max_rel_cl_low_ma": 1.7e-3,
        "drho_over_rho0": [0.5 * m**2 for m in (0.01, 0.05, 0.1, 0.2, 0.3)],
    }
    rec = wc.build_verification_record(poiseuille=pois, pressure_channel=pc, cylinder=cyl)
    crit = rec["criteria"]
    assert crit["c1b_ma2_scaling_pressure_driven_channel"]["pass"]
    assert crit["c2_strouhal_mach_independence"]["pass"]
    assert crit["c3_density_fluctuation_order"]["pass"]
    assert crit["c5_hydrodynamic_coefficient_independence"]["pass"]
    assert "c4_acoustic_filter" in crit


def test_verification_record_flags_a_bad_exponent():
    bad = _synthetic_sweep("pressure_channel_mach_sweep", 1.0, 1.0)
    cyl = {
        "machs": [0.01, 0.05, 0.1],
        "st": [0.165, 0.2, 0.3],
        "cd_mean": [1.3, 1.6, 2.0],
        "cl_amplitude": [0.3, 0.4, 0.6],
        "st_ref": 0.165,
        "cd_ref": 1.3,
        "cl_ref": 0.3,
        "rel_st": [0.0, 0.21, 0.82],
        "rel_cd": [0.0, 0.23, 0.54],
        "rel_cl": [0.0, 0.33, 1.0],
        "max_rel_st_low_ma": 0.21,
        "max_rel_cd_low_ma": 0.23,
        "max_rel_cl_low_ma": 0.33,
        "drho_over_rho0": [0.5 * m for m in (0.01, 0.05, 0.1)],
    }
    rec = wc.build_verification_record(
        poiseuille=_synthetic_sweep("poiseuille_mach_sweep", 0.0, 3e-2),
        pressure_channel=bad,
        cylinder=cyl,
    )
    crit = rec["criteria"]
    assert not crit["c1b_ma2_scaling_pressure_driven_channel"]["pass"]
    assert not crit["c2_strouhal_mach_independence"]["pass"]
    assert not crit["c3_density_fluctuation_order"]["pass"]
    assert not crit["c5_hydrodynamic_coefficient_independence"]["pass"]