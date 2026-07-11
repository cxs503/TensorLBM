"""Closed-periodic control-volume checks for D3Q27 moving-wall exchange."""

import math

import pytest
import torch

from tensorlbm.d3q27 import (
    C,
    OPPOSITE,
    W,
    control_volume_momentum_balance27,
    moving_wall_linkwise_me_force_torque,
)


def _closed_periodic_bounceback_packets(omega: float):
    """Return one periodic packet per wall link before and after reflection.

    A packet has no external face: its incident population is removed and its
    reflected population is placed in the opposite D3Q27 direction. It is a
    minimal closed control volume for which linkwise ME must be exact.
    """
    directions = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
         [0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    outgoing = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.1], dtype=torch.float64)
    positions = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0],
         [0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    wall_velocity = torch.zeros((len(outgoing), 3), dtype=torch.float64)
    wall_velocity[:, 2] = omega * positions[:, 1]
    q = torch.tensor(
        [next(i for i, c in enumerate(C.tolist()) if c == direction.tolist()) for direction in directions],
        dtype=torch.long,
    )
    reflected, _, force_on_fluid, torque_on_fluid = moving_wall_linkwise_me_force_torque(
        outgoing, directions, W[q].double(), wall_velocity, positions,
    )
    before = torch.zeros((27, len(outgoing)), dtype=torch.float64)
    after = torch.zeros_like(before)
    packet_index = torch.arange(len(outgoing))
    before[q, packet_index] = outgoing
    after[OPPOSITE[q], packet_index] = reflected
    return before, after, force_on_fluid, torque_on_fluid


def test_closed_periodic_moving_wall_me_equals_distribution_momentum_change():
    before, after, force_on_fluid, torque_on_fluid = _closed_periodic_bounceback_packets(0.01)

    report = control_volume_momentum_balance27(
        before, after, force_on_fluid, max_lattice_speed=0.01,
    )

    assert report.max_mach == pytest.approx(math.sqrt(3.0) * 0.01)
    assert report.within_tolerance(atol=1e-14, rtol=0.0)
    assert torch.allclose(report.distribution_momentum_change, force_on_fluid, atol=1e-14)
    assert torch.allclose(report.force_on_wall, -force_on_fluid)
    torque_on_wall = -torque_on_fluid
    assert torch.allclose(torque_on_wall, -torque_on_fluid)
    assert report.force_on_wall[0] > 0.0
    assert torque_on_wall[0] < 0.0


def test_closed_periodic_control_volume_reverses_rotational_reaction_with_omega():
    plus = _closed_periodic_bounceback_packets(0.01)
    minus = _closed_periodic_bounceback_packets(-0.01)
    plus_report = control_volume_momentum_balance27(plus[0], plus[1], plus[2], max_lattice_speed=0.01)
    minus_report = control_volume_momentum_balance27(minus[0], minus[1], minus[2], max_lattice_speed=0.01)

    # The axial stationary control load is even; rotational torque and wall
    # reaction are odd in omega.
    assert torch.allclose(plus_report.force_on_wall, minus_report.force_on_wall, atol=1e-14)
    assert torch.allclose(plus[3], -minus[3], atol=1e-14)
    assert plus_report.within_tolerance(atol=1e-14, rtol=0.0)
    assert minus_report.within_tolerance(atol=1e-14, rtol=0.0)


def test_control_volume_residual_bounds_unreported_open_face_or_source_term():
    before, after, force, _ = _closed_periodic_bounceback_packets(0.01)
    # An additional +x population represents a face flux or source that was
    # deliberately not included in the linkwise wall-ME accumulator.
    after[1, 0] += 0.003
    report = control_volume_momentum_balance27(before, after, force, max_lattice_speed=0.01)

    assert report.residual_norm == pytest.approx(0.003)
    assert report.within_tolerance(atol=0.003 + 1e-14, rtol=0.0)
    assert not report.within_tolerance(atol=0.002, rtol=0.0)


def test_control_volume_diagnostic_rejects_non_low_mach_speed():
    before, after, force, _ = _closed_periodic_bounceback_packets(0.01)
    with pytest.raises(ValueError, match="invalid low-Mach control-volume diagnostic"):
        control_volume_momentum_balance27(
            before, after, force, max_lattice_speed=0.1 / math.sqrt(3.0),
        )
