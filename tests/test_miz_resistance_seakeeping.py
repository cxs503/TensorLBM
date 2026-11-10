"""Tests for the MIZ resistance / seakeeping coupling modules (P46W3-N04).

Covers the physics of ``tensorlbm.miz_resistance`` and
``tensorlbm.miz_seakeeping``: analytic limits, monotonicity, negative controls,
the five machine acceptance criteria and the energy ledger closure.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.miz_resistance import (
    G_ACCEL, RHO_W, SIGMA_F_DEFAULT, SQUIRE_WINDOW, MIZShip,
    attenuation_correlation, bending_length, bending_stress,
    characteristic_floe_size, energy_ledger, flexural_rigidity,
    flexural_wavelength, flexural_wavenumber, floe_drag_power_density,
    floe_size_distribution, fractured_energy_per_floe, group_velocity,
    ice_resistance, ittc57_friction_coefficient, local_wave_height,
    open_water_resistance, pierson_moskowitz, resistance_coupled,
    resistance_superposed, scattering_attenuation, superposition_error,
    wave_added_resistance, wave_attenuation_coefficient, wave_energy_profile)
from tensorlbm.miz_seakeeping import (
    K_B_ICE, hull_coefficients, ice_added_mass_factor, ice_damping_factor,
    ice_restoring_factor, miz_hydro_coefficients, rao_curves,
    roll_damping_power, roll_peak_reduction, significant_roll)

SHIP = MIZShip()


# ---------------------------------------------------------------------------
# flexural-gravity dispersion
# ---------------------------------------------------------------------------
def test_dispersion_reduces_to_open_water_for_thin_ice():
    """h_ice -> 0 must recover omega^2 = g k (deep water gravity waves)."""
    for Tp in (6.0, 8.0, 10.0, 14.0):
        omega = 2.0 * math.pi / Tp
        k = flexural_wavenumber(omega, 1e-8)
        assert k == pytest.approx(omega ** 2 / G_ACCEL, rel=1e-6)


def test_dispersion_satisfies_plate_relation_and_lengthens_waves():
    """The Newton solution must satisfy the dispersion relation, and flexural
    waves must be longer than the open-water gravity wave of the same period."""
    h = 0.5
    D = flexural_rigidity(h)
    for Tp in (6.0, 8.0, 10.0):
        omega = 2.0 * math.pi / Tp
        k = flexural_wavenumber(omega, h)
        residual = G_ACCEL * k + D * k ** 5 / RHO_W - omega ** 2
        assert abs(residual) < 1e-8 * omega ** 2
        k_ow = omega ** 2 / G_ACCEL
        assert k < k_ow                       # flexural waves are longer
        assert flexural_wavelength(omega, h) > 2.0 * math.pi / k_ow


def test_finite_depth_dispersion():
    """Shallow water: omega^2 = g k tanh(kd), so k grows as depth decreases."""
    omega = 2.0 * math.pi / 10.0
    k_deep = flexural_wavenumber(omega, 0.5)
    k_shallow = flexural_wavenumber(omega, 0.5, depth=40.0)
    assert k_shallow > k_deep
    resid = (G_ACCEL * k_shallow +
             flexural_rigidity(0.5) * k_shallow ** 5 / RHO_W) * \
        math.tanh(k_shallow * 40.0) - omega ** 2
    assert abs(resid) < 1e-6 * omega ** 2


def test_group_velocity_matches_numerical_derivative():
    """cg = domega/dk must agree with the analytical group velocity."""
    D = flexural_rigidity(0.5)

    def omega_of_k(k, depth=None):
        th = math.tanh(k * depth) if depth else 1.0
        return math.sqrt((G_ACCEL * k + D * k ** 5 / RHO_W) * th)

    for omega_t in (2.0 * math.pi / 6.0, 2.0 * math.pi / 10.0):
        k = flexural_wavenumber(omega_t, 0.5)
        dk = 1e-6 * k
        num = (omega_of_k(k + dk) - omega_of_k(k - dk)) / (2.0 * dk)
        assert group_velocity(omega_t, k, 0.5) == pytest.approx(num, rel=1e-6)
        # finite depth
        kd = flexural_wavenumber(omega_t, 0.5, 40.0)
        num_d = (omega_of_k(kd + dk, 40.0) - omega_of_k(kd - dk, 40.0)) / (2 * dk)
        assert group_velocity(omega_t, kd, 0.5, depth=40.0) == pytest.approx(
            num_d, rel=1e-5)
        assert group_velocity(omega_t, kd, 0.5, depth=40.0) > 0.0


def test_bending_length_scales_with_thickness():
    lb1 = bending_length(0.5)
    lb2 = bending_length(1.0)
    # l_b ~ h^{3/4}
    assert lb2 / lb1 == pytest.approx(2.0 ** 0.75, rel=1e-6)


# ---------------------------------------------------------------------------
# breakup / floe size
# ---------------------------------------------------------------------------
def test_bending_stress_scaling():
    base = bending_stress(1.0, 0.06, 0.5)
    assert bending_stress(2.0, 0.06, 0.5) == pytest.approx(2.0 * base)
    assert bending_stress(1.0, 0.12, 0.5) == pytest.approx(4.0 * base)
    assert bending_stress(1.0, 0.06, 1.0) == pytest.approx(2.0 * base)


def test_floe_size_inside_squire_window_over_sweep():
    lo, hi = SQUIRE_WINDOW
    for Hs in (1.0, 1.5, 2.0, 2.5):
        for Tp in (6.0, 8.0, 10.0, 12.0):
            for h in (0.3, 0.5, 1.0, 1.5):
                a = characteristic_floe_size(Hs, Tp, h)
                lam = flexural_wavelength(2.0 * math.pi / Tp, h)
                assert lo <= a / lam <= hi, (Hs, Tp, h, a / lam)


def test_floe_size_physical_trends():
    """Stronger waves -> smaller floes; stronger/thicker ice -> larger floes."""
    # long-wave state where the bending-length floor does not bind
    base = characteristic_floe_size(1.5, 12.0, 0.5)
    assert base > bending_length(0.5)
    # bigger waves -> smaller floes
    assert characteristic_floe_size(2.5, 12.0, 0.5) < base
    # stronger ice -> larger floes
    assert characteristic_floe_size(1.5, 12.0, 0.5, sigma_f=1.0e6) > base
    # longer waves -> larger floes
    assert base > characteristic_floe_size(1.5, 8.0, 0.5)
    # the bending length is a hard lower bound for every state
    for Hs in (1.0, 2.5):
        for Tp in (6.0, 12.0):
            for h in (0.3, 0.5, 1.0, 1.5):
                assert characteristic_floe_size(Hs, Tp, h) >= \
                    0.999 * bending_length(h)


def test_floe_size_distribution_normalised_and_inside_window():
    a, p, stats = floe_size_distribution(1.5, 8.0, 0.5)
    assert len(a) == len(p) > 0
    assert np.all(a > 0.0)
    assert np.all(p > 0.0)
    assert np.all(np.diff(a) >= 0.0)
    assert stats['n_breaking_bands'] > 0
    lo, hi = SQUIRE_WINDOW
    assert lo <= stats['ratio_mean'] <= hi
    assert stats['a_p10'] <= stats['a_mean'] <= stats['a_p90']
    # discrete weights sum to one
    assert pytest.approx(1.0, rel=1e-12) == float(np.sum(p))


def test_floe_size_distribution_no_breakup_case():
    """A calm sea cannot break the ice: no floes are produced."""
    a, p, stats = floe_size_distribution(0.02, 12.0, 1.5)
    assert stats['n_breaking_bands'] == 0
    assert len(a) == 0 and len(p) == 0


# ---------------------------------------------------------------------------
# attenuation
# ---------------------------------------------------------------------------
def test_attenuation_scattering_cubic_in_floe_size():
    """alpha_scat = C (ka)^4/(3 gamma a) -> scales as a^3 at fixed k."""
    a1 = scattering_attenuation(0.5, 5.0, 0.05)
    a2 = scattering_attenuation(0.5, 10.0, 0.05)
    assert a2 / a1 == pytest.approx(8.0, rel=1e-9)
    assert scattering_attenuation(0.0, 10.0, 0.05) == 0.0


def test_attenuation_monotone_in_concentration_and_correlation():
    a = characteristic_floe_size(1.5, 8.0, 0.5)
    k = flexural_wavenumber(2.0 * math.pi / 8.0, 0.5)
    Cs = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    alphas = [wave_attenuation_coefficient(c, a, 0.5, k) for c in Cs]
    assert alphas[0] > 0.0
    assert all(b > a_ for a_, b in zip(alphas, alphas[1:]))
    corr = attenuation_correlation(Cs, a, 0.5, k)
    assert corr['r'] >= 0.9
    # magnitude check against MIZ field measurements (1e-3 .. 1e-2 1/m)
    assert 1e-4 < alphas[-1] < 2e-2


def test_energy_profile_matches_exponential_law():
    alpha = 0.002
    x = np.linspace(0.0, 500.0, 51)
    E = wave_energy_profile(1000.0, alpha, x)
    assert E[0] == pytest.approx(1000.0)
    slope = np.polyfit(x, np.log(E), 1)[0]
    assert slope == pytest.approx(-2.0 * alpha, rel=1e-9)


def test_pierson_moskowitz_variance():
    """Integral of the PM spectrum equals Hs^2/16."""
    omega = np.linspace(1e-3, 20.0, 200000)
    for Hs, Tp in ((1.5, 8.0), (2.5, 10.0)):
        S = pierson_moskowitz(omega, Hs, Tp)
        m0 = np.trapezoid(S, omega)
        assert m0 == pytest.approx(Hs ** 2 / 16.0, rel=5e-3)


# ---------------------------------------------------------------------------
# resistance
# ---------------------------------------------------------------------------
def test_ittc57_friction_line():
    assert ittc57_friction_coefficient(1e9) == pytest.approx(1.536e-3, rel=5e-3)
    assert ittc57_friction_coefficient(0.0) == 0.0
    assert ittc57_friction_coefficient(1e7) > ittc57_friction_coefficient(1e9)


def test_open_water_resistance_trends():
    slow = MIZShip(speed_kt=8.0)
    fast = MIZShip(speed_kt=12.0)
    assert open_water_resistance(fast) > open_water_resistance(slow) > 0.0
    # order of magnitude for a 100 m / 7800 m^3 hull at 10 kt
    assert 20e3 < open_water_resistance(SHIP) < 200e3


def test_wave_added_resistance_scales_with_Hs_squared():
    r1 = wave_added_resistance(SHIP, 1.0, 8.0)
    r2 = wave_added_resistance(SHIP, 2.0, 8.0)
    assert r2 == pytest.approx(4.0 * r1, rel=1e-9)
    assert wave_added_resistance(SHIP, 0.0, 8.0) == 0.0


def test_ice_resistance_negative_control_and_trends():
    zero = ice_resistance(SHIP, 0.0, 10.0, 0.5)
    assert zero['R_ice'] == 0.0 and zero['R_break'] == 0.0
    r_lo = ice_resistance(SHIP, 0.2, 10.0, 0.5)['R_ice']
    r_mid = ice_resistance(SHIP, 0.5, 10.0, 0.5)['R_ice']
    r_hi = ice_resistance(SHIP, 0.8, 10.0, 0.5)['R_ice']
    assert r_lo < r_mid < r_hi
    # thicker ice -> more resistance
    assert ice_resistance(SHIP, 0.5, 10.0, 1.0)['R_ice'] > r_mid
    # smaller floes -> more fracture events per metre -> more resistance
    assert ice_resistance(SHIP, 0.5, 5.0, 0.5)['R_break'] > \
        ice_resistance(SHIP, 0.5, 20.0, 0.5)['R_break']
    # all three components present
    comp = ice_resistance(SHIP, 0.5, 10.0, 0.5)
    assert comp['R_break'] > 0 and comp['R_subm'] > 0 and comp['R_fric'] > 0
    assert comp['R_ice'] == pytest.approx(
        comp['R_break'] + comp['R_subm'] + comp['R_fric'])


def test_fracture_energy_positive_and_scaling():
    e1 = fractured_energy_per_floe(10.0, 0.5)
    e2 = fractured_energy_per_floe(20.0, 0.5)
    assert e2 == pytest.approx(2.0 * e1, rel=1e-9)
    # G_c = sigma_f^2 h / (2E) -> 12.5 J/m^2 for the default ice props
    G_c = SIGMA_F_DEFAULT ** 2 * 0.5 / (2.0 * 5.0e9)
    assert G_c == pytest.approx(12.5, rel=1e-9)


def test_superposition_vs_coupled_within_ten_percent():
    for C in (0.2, 0.3, 0.5, 0.7, 0.9):
        for Hs in (1.0, 1.5, 2.5):
            se = superposition_error(SHIP, Hs, 8.0, C, 0.5, x_ship=250.0)
            assert se['abs_rel_error'] <= 0.10, (C, Hs, se['rel_error'])
            assert se['R_coupled'] > 0.0


def test_local_wave_height_decays_and_is_consistent():
    Hs_loc, alpha = local_wave_height(1.5, 8.0, 0.5, 0.5, 250.0)
    assert 0.0 < Hs_loc < 1.5
    assert Hs_loc == pytest.approx(1.5 * math.exp(-alpha * 250.0))
    Hs0, alpha0 = local_wave_height(1.5, 8.0, 0.0, 0.5, 250.0)
    assert alpha0 == 0.0 and Hs0 == pytest.approx(1.5)


def test_coupled_model_degenerates_to_superposition_without_ice():
    """No ice: coupled and superposed must agree to machine precision."""
    sup = resistance_superposed(SHIP, 1.5, 8.0, 0.0, 0.5, x_ship=250.0)
    cpl = resistance_coupled(SHIP, 1.5, 8.0, 0.0, 0.5, 250.0)
    assert cpl['R_ice'] == 0.0
    assert cpl['R_total'] == pytest.approx(sup['R_total'], rel=1e-12)


# ---------------------------------------------------------------------------
# seakeeping
# ---------------------------------------------------------------------------
def test_ice_hydrodynamic_factors_are_unity_without_ice():
    assert ice_added_mass_factor(0.0) == 1.0
    assert ice_damping_factor(0.0, 10.0, 0.5) == 1.0
    assert ice_restoring_factor(0.0, 0.5, 6.0) == 1.0
    assert ice_added_mass_factor(0.5) > 1.0
    assert ice_damping_factor(0.5, 10.0, 0.5) > 1.0
    # monotone in concentration
    assert ice_damping_factor(0.8, 10.0, 0.5) > ice_damping_factor(0.2, 10.0, 0.5)
    assert K_B_ICE == pytest.approx(1.0)


def test_miz_coefficients_reduce_to_open_water():
    ow = hull_coefficients(SHIP)
    miz = miz_hydro_coefficients(SHIP, 0.0, 10.0, 0.5)
    for key in ('A33', 'B33', 'C33', 'A44', 'B44', 'C44', 'A55', 'B55', 'C55'):
        assert miz[key] == pytest.approx(ow[key], rel=1e-12)
    miz2 = miz_hydro_coefficients(SHIP, 0.5, 10.0, 0.5)
    assert miz2['B44'] > ow['B44'] and miz2['zeta_roll'] > 0.0


def test_natural_frequencies_plausible():
    ow = hull_coefficients(SHIP)
    # roll period 8..20 s for a 100 m hull with GM_T = 1.2 m
    T_roll = 2.0 * math.pi / ow['omega_n_roll']
    assert 8.0 < T_roll < 20.0
    T_heave = 2.0 * math.pi / ow['omega_n_heave']
    assert 3.0 < T_heave < 10.0


def test_roll_rao_reduction_in_miz():
    for C in (0.2, 0.3, 0.5, 0.7, 0.9):
        for h in (0.3, 0.5, 1.0):
            a = characteristic_floe_size(1.5, 8.0, h)
            rp = roll_peak_reduction(SHIP, C, a, h)
            assert rp['reduction'] >= 0.20, (C, h, rp['reduction'])
            assert rp['peak_roll_miz'] < rp['peak_roll_ow']
            assert rp['zeta_roll_miz'] > rp['zeta_roll_ow']


def test_roll_extremum_negative_control():
    """C_floe = 0 must reproduce the open-water RAO curve exactly."""
    a = characteristic_floe_size(1.5, 8.0, 0.5)
    r = rao_curves(SHIP, 0.0, a, 0.5)
    assert np.allclose(r['roll_ow'], r['roll_miz'], rtol=1e-12, atol=1e-15)
    assert np.allclose(r['heave_ow'], r['heave_miz'], rtol=1e-12, atol=1e-15)
    assert np.allclose(r['pitch_ow'], r['pitch_miz'], rtol=1e-12, atol=1e-15)


def test_rao_curves_physical_bounds():
    r = rao_curves(SHIP, 0.5, characteristic_floe_size(1.5, 8.0, 0.5), 0.5)
    assert np.all(r['heave_ow'] > 0.0)
    assert np.all(np.isfinite(r['roll_ow']))
    # heave RAO peak close to unity (long-wave limit) for a hull of this size
    assert 0.5 < r['heave_ow'].max() < 2.0
    # pitch RAO peak in rad/m: a few degrees per metre
    assert math.radians(1.0) < r['pitch_ow'].max() < math.radians(15.0)
    # roll RAO peak: rad per metre of wave amplitude
    assert 0.01 < r['roll_ow'].max() < 0.5


def test_significant_roll_reduced_in_ice():
    a = characteristic_floe_size(1.5, 8.0, 0.5)
    ow = significant_roll(1.5, 8.0, SHIP, 0.0, a, 0.5)
    miz = significant_roll(1.5, 8.0, SHIP, 0.5, a, 0.5)
    assert 0.0 < miz < ow
    assert math.degrees(ow) < 30.0            # linear theory sanity bound


def test_roll_damping_power_positive_and_grows_with_ice():
    a = characteristic_floe_size(1.5, 8.0, 0.5)
    p_ow = roll_damping_power(SHIP, 1.5, 8.0, 0.0, a, 0.5)
    p_miz = roll_damping_power(SHIP, 1.5, 8.0, 0.5, a, 0.5)
    assert p_ow > 0.0
    assert p_miz > p_ow
    # energy consistency: the damping power must stay below the wave power
    # crossing the ship beam (23.7 kW/m for Hs = 1.5 m, Tp = 8 s)
    P_in = RHO_W * G_ACCEL * 1.5 ** 2 / 8.0 * group_velocity(
        2 * math.pi / 8.0, flexural_wavenumber(2 * math.pi / 8.0, 0.5), 0.5)
    assert p_miz < P_in * SHIP.B


# ---------------------------------------------------------------------------
# energy ledger
# ---------------------------------------------------------------------------
def test_floe_drag_power_density_positive_and_scales_with_C():
    p1 = floe_drag_power_density(0.2, 10.0, 0.5, 0.785, 0.75)
    p2 = floe_drag_power_density(0.4, 10.0, 0.5, 0.785, 0.75)
    assert p2 == pytest.approx(2.0 * p1, rel=1e-9)
    assert floe_drag_power_density(0.0, 10.0, 0.5, 0.785, 0.75) == 0.0
    # cubic in the orbital velocity (amplitude^3)
    p3 = floe_drag_power_density(0.2, 10.0, 0.5, 0.785, 1.5)
    assert p3 == pytest.approx(8.0 * p1, rel=1e-9)


def test_energy_ledger_closure():
    for C in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        for Hs, Tp, h in ((1.0, 6.0, 0.3), (1.5, 8.0, 0.5), (2.5, 10.0, 1.0)):
            led = energy_ledger(C, characteristic_floe_size(Hs, Tp, h), h,
                                Hs, Tp, 300.0, ship=SHIP)
            assert led['residual'] <= 0.05, (C, Hs, Tp, h, led['residual'])
            assert led['P_in'] > 0.0
            assert 0.0 < led['P_trans'] < led['P_in']
            assert led['P_fract'] > 0.0
            assert led['P_fract'] / led['P_att'] < 0.05   # fracture is tiny
            assert led['P_scat'] <= led['P_att'] + 1e-9


def test_energy_ledger_no_ice_limit():
    """No ice: no attenuation, no breakup, only the ship's damping channel."""
    led = energy_ledger(0.0, 10.0, 0.5, 1.5, 8.0, 300.0, ship=None)
    assert led['alpha'] == 0.0
    assert led['P_att'] == pytest.approx(0.0, abs=1e-9)
    assert led['P_trans'] == pytest.approx(led['P_in'], rel=1e-12)
    assert led['P_fract'] == 0.0
    assert led['residual'] < 1e-12
