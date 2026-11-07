"""Tests for the wall-law references, Blasius solution and friction correlations."""

from __future__ import annotations

import math

import pytest

from tensorlbm.validation import (
    B_LOG,
    KAPPA,
    blasius_plate_cf,
    blasius_solution,
    channel_cf_from_u_tau,
    comparison_report,
    dean_channel_cf,
    fit_power_law,
    friction_velocity_from_cf,
    ittc57_cf,
    log_law_u_plus,
    plate_mesh_budget,
    reichardt_u_plus,
    schoenherr_cf,
    schultz_grunow_cf,
    solve_friction_velocity,
    spalding_u_plus,
    spalding_y_plus,
    y_plus,
)

LITERATURE_WALL_SHEAR = 0.3320573362151963


def test_blasius_wall_shear_matches_literature() -> None:
    solution = blasius_solution()
    assert solution.wall_derivative == pytest.approx(LITERATURE_WALL_SHEAR, rel=1.0e-8)
    assert solution.cf_coefficient == pytest.approx(0.6641146724, rel=1.0e-8)
    # f'(eta) -> 1 far from the wall
    assert solution.u_over_u_edge(9.0) == pytest.approx(1.0, abs=5.0e-3)


def test_blasius_local_cf_follows_the_inverse_square_root_law() -> None:
    ratio = blasius_plate_cf(1.0e5) / blasius_plate_cf(4.0e5)
    assert ratio == pytest.approx(2.0, rel=1.0e-12)


def test_spalding_and_reichardt_laws_agree_in_the_log_layer() -> None:
    for y_plus_value in (50.0, 100.0, 300.0):
        assert spalding_u_plus(y_plus_value) == pytest.approx(
            log_law_u_plus(y_plus_value), rel=0.03
        )
        assert reichardt_u_plus(y_plus_value) == pytest.approx(
            log_law_u_plus(y_plus_value), rel=0.03
        )


def test_spalding_inversion_round_trip() -> None:
    for y_plus_value in (1.0, 5.0, 12.0, 30.0, 100.0, 500.0, 2000.0):
        u_plus_value = spalding_u_plus(y_plus_value)
        assert spalding_y_plus(u_plus_value) == pytest.approx(y_plus_value, rel=1.0e-9)


def test_wall_laws_are_monotone_and_continuous_at_the_wall() -> None:
    previous = 0.0
    for y_plus_value in (0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0, 1000.0):
        value = spalding_u_plus(y_plus_value)
        assert value > previous
        previous = value
    # small y+ limits
    assert spalding_u_plus(1.0e-6) == pytest.approx(1.0e-6, rel=1.0e-6)
    assert reichardt_u_plus(1.0e-6) == pytest.approx(0.0, abs=1.0e-4)
    assert math.isfinite(log_law_u_plus(1.0))


def test_friction_velocity_inversion_recovers_the_input() -> None:
    nu = 1.0e-5
    for law in ("spalding", "reichardt", "log"):
        for y_plus_value in (30.0, 80.0, 200.0):
            u_tau_true = 0.03
            wall_distance = y_plus_value * nu / u_tau_true
            law_function = (
                spalding_u_plus if law == "spalding" else reichardt_u_plus
                if law == "reichardt" else log_law_u_plus
            )
            velocity = u_tau_true * law_function(y_plus_value)
            recovered, recovered_y_plus, _ = solve_friction_velocity(
                velocity, wall_distance, nu, law=law
            )
            assert recovered == pytest.approx(u_tau_true, rel=1.0e-8)
            assert recovered_y_plus == pytest.approx(y_plus_value, rel=1.0e-8)


def test_friction_velocity_inversion_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        solve_friction_velocity(0.0, 1.0e-4, 1.0e-5)
    with pytest.raises(ValueError):
        solve_friction_velocity(0.1, 0.0, 1.0e-5)
    with pytest.raises(ValueError):
        solve_friction_velocity(0.1, 1.0e-4, 1.0e-5, law="not-a-law")


def test_y_plus_definition() -> None:
    assert y_plus(0.01, 0.02, 1.0e-5) == pytest.approx(20.0)


def test_reference_friction_lines_are_physical() -> None:
    reynolds_values = [1.0e5, 1.0e6, 1.0e7, 1.0e8, 1.0e9]
    for line in (ittc57_cf, dean_channel_cf, schultz_grunow_cf, schoenherr_cf):
        values = [line(re_value) for re_value in reynolds_values]
        assert all(value > 0.0 for value in values)
        assert all(later < earlier for earlier, later in zip(values, values[1:], strict=False))
    # ITTC-1957 value at Re = 1e7 is a widely quoted reference number
    assert ittc57_cf(1.0e7) == pytest.approx(0.003, rel=1.0e-6)
    # Schoenherr is bracketed by the explicit lines at model scale (+/- 8%)
    assert schoenherr_cf(1.0e7) == pytest.approx(0.00293, abs=5.0e-5)


def test_channel_cf_and_friction_velocity_are_consistent() -> None:
    u_tau = friction_velocity_from_cf(0.004, 1.0)
    assert channel_cf_from_u_tau(u_tau, 1.0) == pytest.approx(0.004, rel=1.0e-12)


def test_mesh_budget_negative_control_is_monotone_and_huge() -> None:
    light = plate_mesh_budget(1.0e6, 4.356, treatment="wall_resolved")
    heavy = plate_mesh_budget(1.0e8, 4.356, treatment="wall_resolved")
    assert heavy.total_cells > light.total_cells
    assert heavy.cells_wall_normal > light.cells_wall_normal
    report = comparison_report([1.0e6, 3.15e7], 4.356)
    ratios = [row["cell_ratio"] for row in report["rows"]]
    assert ratios == sorted(ratios)
    assert ratios[-1] > 1.0e3  # wall resolution costs thousands of times more cells
    # geometric stretching reduces the wall-normal count below the uniform mesh
    uniform = plate_mesh_budget(3.15e7, 4.356, treatment="wall_resolved", dy_plus_first_cell=50.0)
    stretched = plate_mesh_budget(
        3.15e7,
        4.356,
        treatment="wall_modeled",
        dy_plus_first_cell=50.0,
        growth_ratio=1.2,
    )
    assert stretched.cells_wall_normal < uniform.cells_wall_normal


def test_power_law_fit_recovers_a_known_exponent() -> None:
    reynolds = [1.0e6, 3.0e6, 1.0e7, 3.0e7]
    cf_values = [0.075 * value**-0.15 for value in reynolds]
    fit = fit_power_law(reynolds, cf_values)
    assert fit.exponent == pytest.approx(0.15, rel=1.0e-9)
    assert fit.amplitude == pytest.approx(0.075, rel=1.0e-9)
    assert fit.r_squared == pytest.approx(1.0, abs=1.0e-12)
    with pytest.raises(ValueError):
        fit_power_law([1.0, 1.0], [0.1, 0.2])


def test_b_log_constant_is_documented_value() -> None:
    assert KAPPA == pytest.approx(0.41)
    assert B_LOG == pytest.approx(5.2)