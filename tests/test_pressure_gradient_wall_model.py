from __future__ import annotations

import pytest
import torch

from tensorlbm.pressure_gradient_wall_model import (
    pressure_gradient_eddy_viscosity,
    pressure_gradient_equilibrium_velocity,
    solve_pressure_gradient_equilibrium_wall_shear,
)


def test_duprat_eddy_viscosity_matches_open_source_reference() -> None:
    # libWallModelledLES DupratEddyViscosity.Value reference, using its
    # published defaults kappa=0.4, APlus=18 and beta=0.78.
    actual = pressure_gradient_eddy_viscosity(
        torch.tensor((0.01, 0.1), dtype=torch.float64),
        torch.tensor(0.04, dtype=torch.float64),
        torch.tensor(0.1, dtype=torch.float64),
        8.0e-6,
        model="duprat",
    )
    expected = torch.tensor(
        (0.00021063538749341106, 0.007392720890351022),
        dtype=torch.float64,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-14, atol=2e-16)


def test_laminar_pressure_gradient_solution_matches_analytic_shear() -> None:
    speed = torch.tensor((0.08, 0.08, 0.08), dtype=torch.float64)
    distance = torch.tensor((1.5, 2.0, 3.0), dtype=torch.float64)
    acceleration = torch.tensor((-2e-5, 0.0, 2e-5), dtype=torch.float64)
    nu = 0.01
    result = solve_pressure_gradient_equilibrium_wall_shear(
        speed,
        distance,
        acceleration,
        nu,
        mixing_length=False,
    )
    expected_tau = nu * speed / distance - 0.5 * acceleration * distance
    assert torch.all(result.attached)
    assert result.shear_stress_over_density == pytest.approx(
        expected_tau, rel=2e-12, abs=2e-14,
    )
    assert result.residual == pytest.approx(torch.zeros(3), abs=2e-13)


def test_zero_gradient_manufactured_turbulent_state_is_recovered() -> None:
    expected = torch.tensor((0.002, 0.003, 0.004), dtype=torch.float64)
    distance = torch.tensor((1.0, 2.0, 4.0), dtype=torch.float64)
    speed = pressure_gradient_equilibrium_velocity(
        expected,
        distance,
        0.0,
        3.0e-5,
    )
    result = solve_pressure_gradient_equilibrium_wall_shear(
        speed,
        distance,
        0.0,
        3.0e-5,
    )
    assert torch.all(result.attached)
    assert result.friction_velocity == pytest.approx(expected, rel=2e-11)
    assert result.residual == pytest.approx(torch.zeros(3), abs=2e-13)


def test_adverse_gradient_lowers_attached_shear_at_fixed_exchange_speed() -> None:
    speed = torch.tensor((0.055,), dtype=torch.float64)
    gradients = (-1.0e-7, 0.0, 1.0e-7)
    friction = []
    for acceleration in gradients:
        result = solve_pressure_gradient_equilibrium_wall_shear(
            speed,
            1.0,
            acceleration,
            3.0e-6,
        )
        assert bool(result.attached.item())
        friction.append(float(result.friction_velocity.item()))
    assert friction[0] > friction[1] > friction[2]


def test_attached_solver_flags_required_reverse_shear_as_separation() -> None:
    # In the laminar limit U(tau_w=0)=a_p*y^2/(2*nu)=0.1.  An observed
    # exchange speed of 0.05 therefore requires negative wall shear.
    result = solve_pressure_gradient_equilibrium_wall_shear(
        torch.tensor((0.05,), dtype=torch.float64),
        2.0,
        5.0e-4,
        0.01,
        mixing_length=False,
    )
    assert bool(result.separated.item())
    assert not bool(result.attached.item())
    assert result.friction_velocity.item() == 0.0
    assert result.shear_stress_over_density.item() == 0.0


def test_duprat_pressure_scale_can_retain_attached_adverse_solution() -> None:
    speed = torch.tensor((0.055,), dtype=torch.float64)
    distance = 1.0775
    acceleration = 6.0e-6
    nu = 3.2727272727272725e-6
    van_driest = solve_pressure_gradient_equilibrium_wall_shear(
        speed,
        distance,
        acceleration,
        nu,
    )
    duprat = solve_pressure_gradient_equilibrium_wall_shear(
        speed,
        distance,
        acceleration,
        nu,
        eddy_viscosity_model="duprat",
    )
    assert bool(van_driest.separated.item())
    assert bool(duprat.attached.item())
    assert duprat.friction_velocity.item() > 0.0
    assert abs(duprat.residual.item()) < 1.0e-12


def test_duprat_uses_gradient_magnitude_separately_from_signed_ode_source() -> None:
    speed = torch.tensor((0.05,), dtype=torch.float64)
    signed_streamwise = torch.tensor((0.0,), dtype=torch.float64)
    without_crossflow = solve_pressure_gradient_equilibrium_wall_shear(
        speed,
        1.0,
        signed_streamwise,
        1.0e-5,
        eddy_viscosity_model="duprat",
    )
    with_crossflow = solve_pressure_gradient_equilibrium_wall_shear(
        speed,
        1.0,
        signed_streamwise,
        1.0e-5,
        pressure_gradient_magnitude_acceleration=1.0e-5,
        eddy_viscosity_model="duprat",
    )
    assert bool(without_crossflow.attached.item())
    assert bool(with_crossflow.attached.item())
    assert with_crossflow.friction_velocity.item() != pytest.approx(
        without_crossflow.friction_velocity.item(), rel=1.0e-4,
    )


def test_nonfinite_sample_fails_closed_without_poisoning_finite_sample() -> None:
    result = solve_pressure_gradient_equilibrium_wall_shear(
        torch.tensor((0.05, float("nan"))),
        torch.tensor((1.0, 1.0)),
        0.0,
        1.0e-4,
    )
    assert result.finite.tolist() == [True, False]
    assert result.attached.tolist() == [True, False]
    assert torch.isfinite(result.friction_velocity[0])
    assert result.friction_velocity[1].item() == 0.0
    assert torch.isnan(result.residual[1])


def test_invalid_quadrature_and_iteration_contracts_are_rejected() -> None:
    value = torch.tensor((0.05,))
    with pytest.raises(ValueError, match="quadrature_points"):
        pressure_gradient_equilibrium_velocity(
            value, 1.0, 0.0, 1.0e-4, quadrature_points=8.5,
        )
    with pytest.raises(ValueError, match="iterations"):
        solve_pressure_gradient_equilibrium_wall_shear(
            value, 1.0, 0.0, 1.0e-4, iterations=True,
        )
    with pytest.raises(ValueError, match="eddy_viscosity_model"):
        pressure_gradient_equilibrium_velocity(
            value,
            1.0,
            0.0,
            1.0e-4,
            eddy_viscosity_model="invented",
        )
