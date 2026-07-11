"""Unit tests for D3Q27 link-wise moving-wall momentum exchange."""

import torch

from tensorlbm.d3q27 import moving_wall_linkwise_me_force_torque


def test_zero_wall_velocity_reduces_to_stationary_momentum_exchange():
    outgoing = torch.tensor([0.125, 0.375], dtype=torch.float64)
    directions = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
    weights = torch.tensor([2.0 / 27.0, 2.0 / 27.0], dtype=torch.float64)
    wall_velocity = torch.zeros((2, 3), dtype=torch.float64)
    positions = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)

    reflected, link_force, force, torque = moving_wall_linkwise_me_force_torque(
        outgoing, directions, weights, wall_velocity, positions, origin=(0.0, 0.0, 0.0)
    )

    expected_link_force = -2.0 * outgoing[:, None] * directions
    assert torch.allclose(reflected, outgoing)
    assert torch.allclose(link_force, expected_link_force)
    assert torch.allclose(force, expected_link_force.sum(dim=0))
    assert torch.allclose(torque, torch.cross(positions, expected_link_force, dim=1).sum(dim=0))


def test_rotational_wall_velocity_reverses_torque_with_omega_sign():
    # Two mirrored z-directed links cancel stationary torque.  The moving-wall
    # contribution remains and must be odd in angular velocity about x.
    outgoing = torch.tensor([0.2, 0.2], dtype=torch.float64)
    directions = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    weights = torch.tensor([2.0 / 27.0, 2.0 / 27.0], dtype=torch.float64)
    positions = torch.tensor([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)

    def torque_for(omega: float) -> torch.Tensor:
        wall_velocity = torch.stack((
            torch.zeros(2, dtype=torch.float64),
            torch.zeros(2, dtype=torch.float64),
            omega * positions[:, 1],
        ), dim=1)
        return moving_wall_linkwise_me_force_torque(
            outgoing, directions, weights, wall_velocity, positions, origin=(0.0, 0.0, 0.0)
        )[3]

    torque_plus = torque_for(0.01)
    torque_minus = torque_for(-0.01)
    assert torch.allclose(torque_plus, -torque_minus, atol=1e-14)
    assert torque_plus[0].abs() > 0.0


def test_linkwise_force_is_finite_for_finite_inputs():
    reflected, link_force, force, torque = moving_wall_linkwise_me_force_torque(
        outgoing=torch.tensor([0.1]),
        directions=torch.tensor([[1.0, -1.0, 1.0]]),
        weights=torch.tensor([1.0 / 216.0]),
        wall_velocity=torch.tensor([[0.0, 0.02, -0.01]]),
        positions=torch.tensor([[2.0, -3.0, 1.0]]),
        origin=(1.0, 1.0, 1.0),
    )

    for value in (reflected, link_force, force, torque):
        assert torch.isfinite(value).all()
