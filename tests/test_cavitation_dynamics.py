"""Tests for the unsteady cavitation-dynamics module (mission P22W2-N09).

The cases are deliberately small (60-100 chordwise cells, short time windows, no
shedding where not needed) so the whole file runs in well under two minutes.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlbm.cavitation_dynamics import (
    CLOUD_REFERENCE,
    ventilation_number,
    SHEET_REFERENCE,
    KunzClosure,
    cavitation_number,
    dominant_frequency,
    naca0015_cp,
    run_dynamics,
    sigma_critical_dynamics,
    sigma_critical_reference,
    steady_cavity_length,
    strouhal,
    thrust_derating,
    thrust_derating_vs_sigma,
    vapour_pressure,
)
from tensorlbm.cavitation_dynamics.kunz_fv import ALPHA_CAVITY, RHO_VAPOUR, RHO_WATER
from tensorlbm.cavitation_dynamics.thrust import (
    BladeGeometry,
    PropellerDesign,
    blade_section_lift_loss,
    propeller_design,
)

C_KUNZ = 800.0   # calibrated value is ~1.6e3; 800 keeps the tests fast and the
                 # qualitative behaviour identical (see validation.json sensitivity table)


# --------------------------------------------------------------------------- #
# 1-3  numbers and definitions
# --------------------------------------------------------------------------- #
def test_vapour_pressure_matches_saturation_table():
    """Antoine fit at 20 C must give the saturated vapour pressure 2.339 kPa."""
    p_v = vapour_pressure(20.0)
    assert 2200.0 < p_v < 2450.0
    assert vapour_pressure(30.0) > p_v            # monotone in temperature


def test_cavitation_number_definition():
    """sigma = (p_inf - p_v) / (0.5 rho U^2) for a hand-computable case."""
    p_v = vapour_pressure(20.0)
    p_inf = p_v + 0.5 * RHO_WATER * 10.0 ** 2 * 1.5
    assert cavitation_number(p_inf, p_v, 10.0, RHO_WATER) == pytest.approx(1.5, rel=1e-12)
    # increasing the ambient pressure at fixed speed increases sigma
    assert cavitation_number(p_inf * 1.2, p_v, 10.0) > 1.5


def test_ventilation_number_scaling_law():
    """Ventilating the cavity (p_cav > p_v) lowers sigma at unchanged depth and speed."""
    pv = vapour_pressure(20.0)
    p_inf = pv + 0.5 * 998.2 * 100.0 * 1.5
    vaporous = ventilation_number(p_inf, pv, 10.0)
    ventilated = ventilation_number(p_inf, pv + 20_000.0, 10.0)
    assert vaporous == pytest.approx(1.5, rel=1e-9)
    assert ventilated < vaporous
    assert ventilation_number(pv, pv, 10.0) == pytest.approx(0.0)


def test_cp_distribution_shape():
    """NACA0015 suction-side Cp: peak near the leading edge, recovery to the TE."""
    x = np.linspace(0.0, 1.0, 201)
    cp = naca0015_cp(x)
    i_pk = int(np.argmin(cp))
    assert 0.0 < x[i_pk] < 0.10                # suction peak close to the LE
    assert cp[i_pk] < -1.5                     # sigma_i = -Cp_min > 1.5
    assert cp[-1] > cp[i_pk] + 1.0             # pressure recovery towards the TE
    assert np.all(cp > -4.0)


# --------------------------------------------------------------------------- #
# 4-5  Kunz closure
# --------------------------------------------------------------------------- #
def test_kunz_source_branches():
    """Evaporation where p < p_v, condensation where p > p_v, nothing at p = p_v."""
    clo = KunzClosure(c_dest=10.0, c_prod=10.0)
    s = np.array([-1.0, 0.0, 1.0])
    alpha = np.array([0.1, 0.1, 0.1])
    src = clo.source(s, alpha)
    assert src[0] > 0.0                        # s < 0 -> evaporation
    assert src[1] == pytest.approx(0.0, abs=1e-15)
    assert src[2] < 0.0                        # s > 0 -> condensation


def test_kunz_source_availability_factors():
    """The closure is bounded: no vapour source at alpha=1, no sink at alpha=0."""
    clo = KunzClosure(c_dest=5.0, c_prod=5.0)
    assert clo.source(np.array([-2.0]), np.array([1.0]))[0] == pytest.approx(0.0)
    assert clo.source(np.array([2.0]), np.array([0.0]))[0] == pytest.approx(0.0)


def test_kunz_mass_transfer_is_rho_v_times_source():
    clo = KunzClosure()
    s, a = np.array([-0.5]), np.array([0.2])
    assert clo.mass_transfer(s, a)[0] == pytest.approx(clo.source(s, a)[0] * RHO_VAPOUR)


# --------------------------------------------------------------------------- #
# 6-9  cavity length, limits and negative controls
# --------------------------------------------------------------------------- #
def test_cavity_length_decreases_with_sigma():
    """Monotone: lower cavitation number -> longer cavity."""
    lengths = [steady_cavity_length(s, n_cells=80, t_end=8.0, c_dest=C_KUNZ, c_prod=C_KUNZ)
               for s in (0.7, 0.9, 1.1)]
    assert lengths[0] > lengths[1] > lengths[2] > 0.0


def test_no_cavity_above_inception():
    """sigma above inception (-Cp_min = 1.97): no vapour at all."""
    res = run_dynamics(2.4, n_cells=80, t_end=6.0, c_dest=C_KUNZ, c_prod=C_KUNZ, shedding=False)
    assert float(np.max(res.alpha_final)) < 1.0e-3
    assert res.l_extent_mean == pytest.approx(0.0, abs=1e-9)


def test_sigma_infinity_alpha_vanishes():
    """E4 negative control: alpha_v -> 0 as sigma -> infinity (monotone decay)."""
    maxima = []
    for sigma in (2.0, 4.0, 8.0):
        res = run_dynamics(sigma, n_cells=80, t_end=6.0, c_dest=C_KUNZ, c_prod=C_KUNZ,
                           shedding=False)
        maxima.append(float(np.max(res.alpha_final)))
    assert maxima[0] >= maxima[1] >= maxima[2]
    assert maxima[-1] < 1.0e-3


def test_single_phase_limit():
    """E5: p_inf >> p_v leaves the flow single phase (no evaporation, tiny transfer)."""
    res = run_dynamics(20.0, n_cells=80, t_end=6.0, c_dest=C_KUNZ, c_prod=C_KUNZ, shedding=False)
    assert float(np.max(res.alpha_final)) < 1.0e-4
    cav = run_dynamics(0.6, n_cells=80, t_end=6.0, c_dest=C_KUNZ, c_prod=C_KUNZ, shedding=False)
    assert res.mass_evap < 1.0e-2 * cav.mass_evap


# --------------------------------------------------------------------------- #
# 10  mass conservation
# --------------------------------------------------------------------------- #
def test_mass_budget_closes():
    """E6: steady-state evaporation balances condensation + outflow (residual < 1 %)."""
    res = run_dynamics(0.8, n_cells=100, t_end=14.0, c_dest=C_KUNZ, c_prod=C_KUNZ,
                       shedding=False, budget_from=7.0)
    assert res.mass_evap > 0.0
    assert abs(res.mass_evap - res.mass_cond - res.mass_out) / res.mass_evap < 0.01
    assert res.mass_balance < 1.0e-6


# --------------------------------------------------------------------------- #
# 11-13  shedding: transition, Strouhal number, spectral cross-check
# --------------------------------------------------------------------------- #
def test_shedding_only_below_critical_number():
    """E7: below the transition the cavity sheds, above it the sheet is stable."""
    shedding = run_dynamics(0.50, n_cells=100, t_end=18.0, c_dest=C_KUNZ, c_prod=C_KUNZ)
    stable = run_dynamics(0.90, n_cells=100, t_end=12.0, c_dest=C_KUNZ, c_prod=C_KUNZ)
    assert shedding.shedding
    assert shedding.shed_times.size >= 5
    assert not stable.shedding


def test_shedding_strouhal_in_band():
    """E7: the measured break-off Strouhal number sits in the 0.2-0.3 band."""
    res = run_dynamics(0.50, n_cells=100, t_end=18.0, c_dest=C_KUNZ, c_prod=C_KUNZ)
    assert 0.20 <= res.st_l <= 0.30
    assert res.st_c > res.st_l          # chord-based number is larger for L < c


def test_measured_frequency_matches_spectrum():
    """The event period and the spectral peak of the vapour mass agree within 10 %."""
    res = run_dynamics(0.50, n_cells=100, t_end=18.0, c_dest=C_KUNZ, c_prod=C_KUNZ)
    f_sig = res.meta["f_from_signal"]
    assert abs(f_sig - res.f_shed) <= 0.10 * res.f_shed


def test_dominant_frequency_recovers_sine():
    """Sanity of the spectral estimator on a synthetic signal."""
    t = np.linspace(0.0, 40.0, 4001)
    y = np.sin(2.0 * np.pi * 0.25 * t) + 0.3 * np.sin(2.0 * np.pi * 0.75 * t)
    assert dominant_frequency(t, y, f_min=0.05, f_max=1.0) == pytest.approx(0.25, rel=0.02)


def test_strouhal_definition():
    assert strouhal(2.0, 0.5, 4.0) == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# 14  critical cavitation number
# --------------------------------------------------------------------------- #
def test_sigma_critical_reference_from_measured_curve():
    """The reference transition number is the inversion of the measured L/c curve."""
    sigma_c = sigma_critical_reference(0.5)
    assert 0.6 < sigma_c < 0.8
    # monotone: a stricter (longer) break-off threshold gives a lower number
    assert sigma_critical_reference(0.4) > sigma_c > sigma_critical_reference(0.6)


def test_sigma_critical_dynamics_is_bracketed():
    out = sigma_critical_dynamics(n_cells=80, t_end=7.0, sigma_lo=0.35, sigma_hi=1.0,
                                  c_dest=C_KUNZ, c_prod=C_KUNZ)
    assert out["converged"]
    assert 0.3 < out["sigma_critical"] < 1.0
    lo, hi = out["bracket"]
    assert lo <= out["sigma_critical"] <= hi


# --------------------------------------------------------------------------- #
# 15-17  thrust derating
# --------------------------------------------------------------------------- #
def test_blade_section_lift_loss_bounds():
    """No cavity above inception, strong but bounded loss at low sigma."""
    assert blade_section_lift_loss(2.5)["lift_loss"] == pytest.approx(0.0)
    loss = blade_section_lift_loss(0.4)["lift_loss"]
    assert 0.2 < loss < 0.75
    # the area-fraction convention is always the smaller (diluted) number
    assert blade_section_lift_loss(0.4, loss_model="area_fraction")["lift_loss"] < loss


def test_thrust_derating_monotone_in_sigma():
    sweep = thrust_derating_vs_sigma([0.4, 0.6, 1.0, 2.5], loss_model="area_fraction")
    der = np.abs(sweep["derating"])
    assert np.all(np.diff(der) <= 1e-6)        # derating shrinks as sigma grows
    assert der[-1] < 0.02                      # no cavitation -> no derating


def test_thrust_derating_at_sigma_06_in_empirical_band():
    """E3: the derating at sigma_0.7R = 0.6 is inside the empirical 10-15 % band."""
    design = propeller_design(0.6)
    res = thrust_derating(design, loss_model="area_fraction")
    assert 0.10 <= abs(res["derating"]) <= 0.15
    assert res["kT_wet"] > 0.05                # a realistic loading, not a flat plate


def test_thrust_derating_resolves_blade_position():
    """The derating is azimuth dependent (immersion depth changes round the disc)."""
    design = propeller_design(0.6)
    res = thrust_derating(design, loss_model="area_fraction", n_theta=12)
    az = res["derating_azimuth"]
    assert az.size == 12
    assert np.ptp(az) > 0.0
    assert np.all(az >= 0.0)


def test_propeller_design_solves_depth_for_sigma():
    design = propeller_design(0.6)
    assert isinstance(design, PropellerDesign)
    assert design.shaft_depth > 0.0
    assert design.sigma_ref() == pytest.approx(0.6, rel=0.02)


# --------------------------------------------------------------------------- #
# 18  references are the in-repo digests (single source of truth)
# --------------------------------------------------------------------------- #
def test_reference_data_is_shared_with_the_repo():
    assert np.allclose(SHEET_REFERENCE.sigma, [1.0, 0.8, 0.6])
    assert np.allclose(SHEET_REFERENCE.l_cav_exp, [0.25, 0.4, 0.65])
    assert CLOUD_REFERENCE.st_exp.size == CLOUD_REFERENCE.sigma.size
    # the power law through the measured points is decreasing in sigma
    assert SHEET_REFERENCE.l_cav_curve(0.8) > SHEET_REFERENCE.l_cav_curve(1.0)


def test_geometry_guard_blade_chord_and_area():
    """Geometry sanity of the propulsor used by the derating calculation."""
    geo = BladeGeometry()
    r = np.array([geo.hub_ratio, 0.7, 0.995])
    c = geo.chord(r)
    assert np.all(c > 0.0)
    assert c[1] == pytest.approx(geo.chord_over_diameter * geo.diameter, rel=1e-9)
    beta = geo.pitch_angle(r)
    assert np.all(beta > 0.0) and beta[0] > beta[1] > beta[2]   # pitch angle falls outboard
    assert ALPHA_CAVITY == 0.5