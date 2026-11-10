"""Tests for the P151W1-N04 cavitation + ventilation validation chain.

Coverage: cavitation/ventilation numbers and regimes, the Kunz closure (branch
signs, single-phase limits, closed-form verification, two-zone mass balance), the
panel suction pressure and the isobaric cavity closure, the sheet/cloud
transition and shedding Strouhal number, and the blade-element thrust derating
with its 5 %-loss critical number.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.cavitation_chain import (
    Kunz0D,
    KunzCoefficients,
    PropellerDesign,
    blade_element_solution,
    cavitation_number,
    classify,
    critical_sigma,
    fit_power_law,
    inception_sigma,
    isobaric_cavity,
    kunz_mass_transfer,
    kunz_timescale,
    lift_loss,
    regime,
    shedding,
    sigma_critical_thrust,
    sigma_from_depth,
    steady_state_vapour_fraction,
    strouhal_band,
    suction_pressure,
    thrust_derating,
    ventilation_number,
    vapour_pressure,
)
from tensorlbm.cavitation_chain.thrust import design_for_targets

RHO_L, RHO_V, U_REF = 998.2, 0.01729, 10.0
SIGMA = 1.0


def _q() -> float:
    return 0.5 * RHO_L * U_REF**2


# ---------------------------------------------------------------------------
# cavitation / ventilation numbers
# ---------------------------------------------------------------------------


def test_vapour_pressure_water_20c():
    pv = vapour_pressure(20.0)
    assert 2.2e3 < pv < 2.5e3  # steam tables: 2.339 kPa


def test_cavitation_number_definition_and_regimes():
    p_v = vapour_pressure(20.0)
    q = _q()
    assert cavitation_number(p_v + 2 * q, p_v, RHO_L, U_REF) == pytest.approx(2.0)
    assert regime(3.0, 1.8, 0.9) == "single_phase"
    assert regime(1.8, 1.8, 0.9) == "inception"
    assert regime(1.2, 1.8, 0.9) == "sheet"
    assert regime(0.7, 1.8, 0.9) == "cloud"
    assert inception_sigma(-1.8) == pytest.approx(1.8)
    with pytest.raises(ValueError):
        inception_sigma(0.5)


def test_ventilation_number_promotes_cavity():
    p_v = vapour_pressure(20.0)
    p_gas = p_v + 0.5 * _q()  # air at a higher cavity pressure
    sigma_v = ventilation_number(101325.0, p_gas, RHO_L, U_REF)
    sigma_c = cavitation_number(101325.0, p_v, RHO_L, U_REF)
    assert sigma_v < sigma_c
    assert sigma_c - sigma_v == pytest.approx(0.5)


def test_sigma_from_depth_monotone():
    s0, s5 = sigma_from_depth(10.0, 0.0), sigma_from_depth(10.0, 5.0)
    assert s5 > s0 > 0.0
    assert (s5 - s0) == pytest.approx(RHO_L * 9.80665 * 5.0 / _q(), rel=1e-9)


def test_power_law_fit_recovers_exponent():
    x = np.logspace(-1, 1, 7)
    y = 0.25 * x ** (-1.9)
    fit = fit_power_law(x, y)
    assert fit.exponent == pytest.approx(-1.9, rel=1e-10)
    assert fit.prefactor == pytest.approx(0.25, rel=1e-10)


# ---------------------------------------------------------------------------
# Kunz closure
# ---------------------------------------------------------------------------


def test_kunz_branch_signs_and_exclusivity():
    p_v = vapour_pressure(20.0)
    q = _q()
    ev, cd = kunz_mass_transfer(p_v - SIGMA * q, p_v, 0.7, 0.3, RHO_L, RHO_V, U_REF)
    assert ev > 0.0 and cd == 0.0
    ev2, cd2 = kunz_mass_transfer(p_v + SIGMA * q, p_v, 0.7, 0.3, RHO_L, RHO_V, U_REF)
    assert ev2 == 0.0 and cd2 > 0.0
    ev3, cd3 = kunz_mass_transfer(p_v, p_v, 0.7, 0.3, RHO_L, RHO_V, U_REF)
    assert ev3 == 0.0 and cd3 == 0.0


def test_kunz_rate_matches_closed_form():
    """m_dot_evap / rho_v must equal C_dest alpha_l |p_v-p| / (t_inf q)."""
    p_v, t_inf, c = vapour_pressure(20.0), 0.01, 0.1
    q = _q()
    a_v = 0.3
    ev, _ = kunz_mass_transfer(
        p_v - 0.7 * q, p_v, 1.0 - a_v, a_v, RHO_L, RHO_V, U_REF, t_inf=t_inf
    )
    dalpha_dt = float(ev) / RHO_V
    expected = 1.0 * (1.0 - a_v) * 0.7 / t_inf
    assert dalpha_dt == pytest.approx(expected, rel=1e-12)


def test_kunz_zero_vapour_means_no_condensation():
    p_v = vapour_pressure(20.0)
    ev, cd = kunz_mass_transfer(
        p_v + 1e3 * _q(), p_v, 1.0, 0.0, RHO_L, RHO_V, U_REF,
        coeffs=KunzCoefficients(weighting="physical"),
    )
    assert float(ev) == 0.0 and float(cd) == 0.0


def test_kunz_zero_liquid_means_no_evaporation():
    p_v = vapour_pressure(20.0)
    ev, _ = kunz_mass_transfer(p_v - 1e3 * _q(), p_v, 0.0, 1.0, RHO_L, RHO_V, U_REF)
    assert float(ev) == 0.0


def test_kunz_timescale_is_one_flow_time_at_inception():
    assert kunz_timescale(1.0, 0.01) == pytest.approx(0.01)
    assert kunz_timescale(2.0, 0.01) == pytest.approx(0.005)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_kunz_rk4_matches_analytic(sign):
    p_v = vapour_pressure(20.0)
    ko = Kunz0D(rho_l=RHO_L, rho_v=RHO_V, U=U_REF, t_inf=0.01, alpha_v0=0.05)
    p = p_v + sign * SIGMA * _q()
    t = np.linspace(0.0, 0.02, 201)
    tr = ko.march(np.full(t.size, p), p_v, t[1] - t[0], alpha_v0=0.05)
    ana = ko.analytic(t, p, p_v, alpha_v0=0.05)
    assert np.max(np.abs(tr["alpha_v"] - ana)) < 1e-6


def test_kunz_two_zone_mass_balance():
    bal = steady_state_vapour_fraction(
        dp_cavity=0.5 * _q(), dp_condensation=2.0 * 0.5 * _q(), alpha_v_cavity=1.0e-4
    )
    assert bal["relative_imbalance"] < 1e-12
    assert 0.0 < bal["alpha_v_condensation_zone"] <= 1.0


def test_kunz_high_pressure_single_phase_limit():
    p_v = vapour_pressure(20.0)
    ko = Kunz0D(rho_l=RHO_L, rho_v=RHO_V, U=U_REF, t_inf=0.01)
    # no vapour at all and p >> p_v: no mass transfer (single-phase limit)
    assert abs(ko.rate(0.0, p_v + 1e3 * _q(), p_v)) == 0.0
    # vapour present and p >> p_v: condensation, i.e. vapour disappears
    assert ko.rate(0.5, p_v + 1e3 * _q(), p_v) < 0.0


# ---------------------------------------------------------------------------
# section pressure, isobaric cavity, lift loss
# ---------------------------------------------------------------------------


def test_panel_lift_matches_thin_aerofoil():
    """The Hess-Smith engine must reproduce the thin-aerofoil lift slope."""
    sp = suction_pressure("naca0015", 5.0, n_foil=160)
    assert sp.cp_min < -1.0
    assert sp.x_cp_min < 0.10  # suction peak at the leading edge
    assert abs(sp.cl_panel - sp.cl_thin_airfoil) / sp.cl_thin_airfoil < 0.15


def test_isobaric_cavity_shrinks_with_sigma():
    sp = suction_pressure("naca0015", 5.0, n_foil=160)
    l = [isobaric_cavity(s, pressure=sp)["l_cav"] for s in (1.4, 1.2, 1.0, 0.8, 0.6)]
    assert all(x > 0.0 for x in l)
    # L/c grows as sigma falls
    assert all(a < b for a, b in zip(l[:-1], l[1:]))
    assert isobaric_cavity(sp.sigma_inception * 1.05, pressure=sp)["active"] is False
    assert isobaric_cavity(sp.sigma_inception * 1.05, pressure=sp)["l_cav"] == 0.0


def test_lift_loss_sign_and_monotonicity():
    sp = suction_pressure("naca0015", 5.0, n_foil=160)
    losses = [-lift_loss(s, pressure=sp) for s in (1.2, 1.0, 0.8, 0.6)]
    assert all(x > 0.0 for x in losses)
    assert all(a < b for a, b in zip(losses[:-1], losses[1:]))


# ---------------------------------------------------------------------------
# sheet / cloud transition and shedding
# ---------------------------------------------------------------------------


def test_critical_sigma_ordering_and_criterion():
    sp = suction_pressure("naca0015", 5.0, n_foil=160)
    out = critical_sigma(pressure=sp, n_foil=160)
    assert out["bracketed"] is True
    assert 0.0 < out["sigma_critical"] < sp.sigma_inception
    assert out["l_cav_at_critical"] == pytest.approx(0.5, abs=0.02)
    assert classify(out["sigma_critical"] * 0.9, pressure=sp,
                    sigma_critical=out["sigma_critical"]) == "cloud"
    assert classify(out["sigma_critical"] * 1.1, pressure=sp,
                    sigma_critical=out["sigma_critical"]) == "sheet"


def test_shedding_strouhal_band():
    band = strouhal_band()
    assert band["inside_band"] is True
    assert 0.2 <= band["st_cavity"] <= 0.3
    sp = suction_pressure("naca0015", 5.0, n_foil=160)
    est = shedding(0.6, u_inf=10.0, chord=0.1, pressure=sp, sigma_critical=0.8)
    assert est.f_shed > 0.0
    assert est.st_cavity == pytest.approx(0.25)
    # St_c = f c/U = St_L * (c/L_cav)
    assert est.st_chord == pytest.approx(est.st_cavity / est.l_over_c)
    assert est.regime == "cloud"


# ---------------------------------------------------------------------------
# thrust derating
# ---------------------------------------------------------------------------


def test_design_for_targets_hits_area_and_loading():
    d = design_for_targets(target_area_ratio=0.65, target_kt=0.22, advance_ratio=0.7)
    assert d.blade_area_ratio() == pytest.approx(0.65, rel=2e-3)
    kt0 = blade_element_solution(1e6, d, wetted_only=True)["kt_wetted"]
    assert kt0 == pytest.approx(0.22, rel=2e-3)


def test_thrust_derating_monotone_and_vanishes():
    d = PropellerDesign(n_strips=12)
    dk = [thrust_derating(s, d) for s in (0.6, 0.8, 1.0, 1.5, 3.0)]
    assert all(a > b for a, b in zip(dk[:-1], dk[1:]))
    assert dk[0] > 0.05
    assert dk[-1] == pytest.approx(0.0, abs=1e-9)


def test_sigma_critical_thrust_is_the_five_percent_point():
    d = PropellerDesign(n_strips=16)
    out = sigma_critical_thrust(0.05, d)
    assert out["bracketed"] is True
    assert thrust_derating(out["sigma_critical"], d) == pytest.approx(0.05, abs=2e-3)


def test_propeller_kt_is_wetted_at_high_sigma():
    d = PropellerDesign(n_strips=12)
    sol = blade_element_solution(1e6, d)
    assert sol["delta_kt_ratio"] == pytest.approx(0.0, abs=1e-9)
    assert sol["cavitating_area_fraction"] == 0.0
    low = blade_element_solution(0.5, d)
    assert low["delta_kt_ratio"] > 0.0
    assert low["cavitating_area_fraction"] > 0.0