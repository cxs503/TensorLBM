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
    wagner_wedge_pressure,
    wagner_wedge_slamming_coefficient,
    wagner_wetted_halfwidth,
)


def test_wedge_wetted_halfwidth_and_coefficient_follow_wagner_formula() -> None:
    params = WedgeEntryParams(beta=math.radians(30.0), v_entry=0.05)
    c1 = wagner_wetted_halfwidth(10.0, params)
    assert wagner_wetted_halfwidth(20.0, params) == pytest.approx(2.0 * c1)
    assert wagner_wedge_slamming_coefficient(params) == pytest.approx(math.pi / math.tan(params.beta))


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
