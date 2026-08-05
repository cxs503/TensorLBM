from __future__ import annotations

import pytest
import torch

from tensorlbm.two_point_wall_diagnostics import (
    estimate_two_point_log_slope_friction_velocity,
    summarize_two_point_log_slope,
)


def test_two_point_log_slope_recovers_exact_friction_velocity() -> None:
    kappa = 0.41
    expected = torch.tensor((0.02, 0.03), dtype=torch.float64)
    inner_distance = torch.tensor((1.0, 2.0), dtype=torch.float64)
    outer_distance = torch.tensor((2.0, 8.0), dtype=torch.float64)
    inner_speed = torch.tensor((0.4, 0.5), dtype=torch.float64)
    outer_speed = inner_speed + expected / kappa * torch.log(
        outer_distance / inner_distance,
    )

    actual, valid = estimate_two_point_log_slope_friction_velocity(
        inner_speed,
        outer_speed,
        inner_distance,
        outer_distance,
    )

    torch.testing.assert_close(actual, expected)
    assert valid.tolist() == [True, True]
    summary = summarize_two_point_log_slope(actual, valid)
    assert summary.valid_samples == 2
    assert summary.mean_friction_velocity == pytest.approx(0.025)


def test_two_point_log_slope_rejects_non_increasing_profile() -> None:
    friction_velocity, valid = estimate_two_point_log_slope_friction_velocity(
        torch.tensor((0.5,)),
        torch.tensor((0.4,)),
        torch.tensor((1.0,)),
        torch.tensor((2.0,)),
    )

    assert valid.item() is False
    assert friction_velocity.item() == 0.0
    summary = summarize_two_point_log_slope(friction_velocity, valid)
    assert summary.valid_samples == 0
    assert summary.rejected_fraction == 1.0
