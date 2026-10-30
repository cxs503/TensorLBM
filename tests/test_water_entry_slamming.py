"""Analytical Wagner water-entry reference checks; no LBM execution."""

import math

import numpy as np
import pytest

from tensorlbm.wagner_theory import (
    SphereEntryParams,
    WedgeEntryParams,
    dimensionless_force,
    wagner_sphere_force,
    wagner_sphere_force_coefficient,
    wagner_sphere_wetted_radius,
    wagner_wedge_added_mass_force,
    wagner_wedge_force_slope,
    wagner_wedge_plate_added_mass_force,
    wagner_wedge_pressure,
    wagner_wedge_slamming_coefficient,
    wagner_wedge_total_force,
    wagner_wetted_halfwidth,
)


def test_wedge_wetted_halfwidth_and_coefficient_follow_wagner_formula() -> None:
    params = WedgeEntryParams(beta=math.radians(30.0), v_entry=0.05)
    c1 = wagner_wetted_halfwidth(10.0, params)
    assert wagner_wetted_halfwidth(20.0, params) == pytest.approx(2.0 * c1)
    # C_F = F / (rho V^2 c) = pi^2 / (4 tan beta): fixed 2026-10-30 (was pi/tan,
    # inconsistent with the force and with the function's own docstring).
    assert wagner_wedge_slamming_coefficient(params) == pytest.approx(
        math.pi**2 / (4.0 * math.tan(params.beta))
    )
    assert wagner_wedge_slamming_coefficient(params) == pytest.approx(
        wagner_wedge_added_mass_force(10.0, params) / (params.rho * params.v_entry**2 * c1)
    )


def test_wedge_added_mass_pressure_is_symmetric_and_zero_outside_wetted_region() -> None:
    params = WedgeEntryParams(beta=math.radians(30.0), v_entry=0.05)
    c = wagner_wetted_halfwidth(10.0, params)
    x = np.linspace(0.1 * c, 0.9 * c, 20)
    np.testing.assert_allclose(
        wagner_wedge_pressure(x, 10.0, params, include_convective=False),
        wagner_wedge_pressure(-x, 10.0, params, include_convective=False),
    )
    assert wagner_wedge_pressure(2.0 * c, 10.0, params) == 0.0


def test_sphere_early_entry_scaling_is_dimensionally_consistent() -> None:
    params = SphereEntryParams(radius=6.0, v_entry=0.05)
    h1, h2 = 0.1, 0.4
    assert wagner_sphere_wetted_radius(h2, params.radius) == pytest.approx(
        2.0 * wagner_sphere_wetted_radius(h1, params.radius)
    )
    assert wagner_sphere_force(h2, params) == pytest.approx(2.0 * wagner_sphere_force(h1, params))
    h_over_r = 0.1
    computed = dimensionless_force(
        wagner_sphere_force(h_over_r * params.radius, params),
        params.rho,
        params.v_entry,
        params.radius,
    )
    assert computed == pytest.approx(wagner_sphere_force_coefficient(h_over_r), rel=0.25)


# ---------------------------------------------------------------------------
# Anchor regression (added 2026-10-30 with the water-entry benchmark prereg):
# the wedge force must equal the integral of the model's own pressure field,
# the closed form quoted by De Rosis & Tafuni (2022) Eq.(45), and the
# documented pi^2/4 convention.  The pre-2026-10-30 code returned 4/pi times
# this value.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("deadrise_deg", [2.0, 3.0, 4.0, 25.0])
def test_wedge_force_equals_pressure_integral(deadrise_deg: float) -> None:
    params = WedgeEntryParams(beta=math.radians(deadrise_deg), v_entry=1.0, rho=1000.0)
    t = 0.002
    c = wagner_wetted_halfwidth(t, params)
    x = np.linspace(-c, c, 200001)
    leading = np.trapezoid(wagner_wedge_pressure(x, t, params, include_convective=False), x)
    assert wagner_wedge_total_force(t, params) == pytest.approx(leading, rel=2e-6)
    assert wagner_wedge_added_mass_force(t, params) == pytest.approx(leading, rel=2e-6)


@pytest.mark.parametrize("deadrise_deg", [2.0, 3.0, 4.0])
def test_wedge_force_matches_de_rosis_eq45_closed_form(deadrise_deg: float) -> None:
    """Eq.(45): F_W = rho pi^2 v^2 a_W / (4 tan zeta), a_W = pi v t / (2 tan zeta)."""
    beta = math.radians(deadrise_deg)
    params = WedgeEntryParams(beta=beta, v_entry=1.0, rho=1000.0)
    t = 0.002
    a_w = math.pi * params.v_entry * t / (2.0 * math.tan(beta))
    eq45 = params.rho * math.pi**2 * params.v_entry**2 * a_w / (4.0 * math.tan(beta))
    assert wagner_wedge_added_mass_force(t, params) == pytest.approx(eq45, rel=1e-12)
    # documented closed form rho V^3 pi^2 t / (8 tan^2 zeta)
    closed = params.rho * params.v_entry**3 * math.pi**3 * t / (8.0 * math.tan(beta) ** 2)
    assert wagner_wedge_added_mass_force(t, params) == pytest.approx(closed, rel=1e-12)


@pytest.mark.parametrize("deadrise_deg", [2.0, 3.0, 4.0])
def test_wedge_force_slope_is_the_anchor(deadrise_deg: float) -> None:
    """dF/dt / (rho V^3) = pi^3 / (8 tan^2 zeta); F is exactly linear in t."""
    beta = math.radians(deadrise_deg)
    params = WedgeEntryParams(beta=beta, v_entry=0.05, rho=1.0)
    slope = wagner_wedge_force_slope(params)
    dimless = slope / (params.rho * params.v_entry**3)
    assert dimless == pytest.approx(math.pi**3 / (8.0 * math.tan(beta) ** 2), rel=1e-12)
    # and it must be the derivative of the force, not an independent formula
    f = wagner_wedge_added_mass_force(20.0, params)
    assert f / 20.0 == pytest.approx(slope, rel=1e-12)
    t1, t2 = 100.0, 300.0
    numeric = (wagner_wedge_added_mass_force(t2, params) - wagner_wedge_added_mass_force(t1, params)) / (
        t2 - t1
    )
    assert numeric == pytest.approx(slope, rel=1e-10)


def test_wedge_plate_variant_is_exactly_twice_and_both_are_documented() -> None:
    params = WedgeEntryParams(beta=math.radians(2.0), v_entry=1.0, rho=1000.0)
    t = 0.002
    assert wagner_wedge_plate_added_mass_force(t, params) == pytest.approx(
        2.0 * wagner_wedge_added_mass_force(t, params)
    )
    # The pre-fix value: 4/pi * the pressure-integral force (kept as a guard so
    # nobody silently reintroduces it).
    legacy = 4.0 / math.pi * wagner_wedge_added_mass_force(t, params)
    assert wagner_wedge_total_force(t, params) != pytest.approx(legacy)
    assert wagner_wedge_total_force(t, params) == pytest.approx(
        wagner_wedge_added_mass_force(t, params)
    )
