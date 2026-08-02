from __future__ import annotations

import torch

from tensorlbm.wall_pressure_gradient import (
    aggregate_wall_pressure_gradient_summaries,
    sample_wall_tangential_pressure_gradient,
)


def test_fluid_only_fit_recovers_linear_tangential_gradient() -> None:
    shape = (7, 7, 7)
    z, y, x = torch.meshgrid(
        *(torch.arange(size, dtype=torch.float64) for size in shape),
        indexing="ij",
    )
    pressure = 2.0 * x + 3.0 * y + 5.0 * z
    solid = torch.zeros(shape, dtype=torch.bool)
    solid[:, :3, :] = True
    pressure[solid] = 1.0e9
    active = torch.zeros_like(solid)
    active[1:-1, 3, 1:-1] = True
    nx = torch.zeros(shape, dtype=torch.float64)
    ny = torch.zeros(shape, dtype=torch.float64)
    nz = torch.zeros(shape, dtype=torch.float64)
    ny[active] = 1.0

    result = sample_wall_tangential_pressure_gradient(
        pressure,
        solid,
        active,
        (nx, ny, nz),
    )

    assert result.valid_nodes == result.requested_nodes
    expected = torch.full_like(result.magnitude, (2.0**2 + 5.0**2) ** 0.5)
    torch.testing.assert_close(result.magnitude, expected)


def test_rank_deficient_neighbourhood_fails_closed() -> None:
    shape = (5, 5, 5)
    pressure = torch.ones(shape)
    active = torch.zeros(shape, dtype=torch.bool)
    active[2, 2, 2] = True
    solid = torch.ones(shape, dtype=torch.bool)
    solid[2, 2, 2] = False
    solid[2, 2, 3] = False
    normal = torch.zeros(shape)
    normal[active] = 1.0

    result = sample_wall_tangential_pressure_gradient(
        pressure,
        solid,
        active,
        (normal, torch.zeros_like(normal), torch.zeros_like(normal)),
    )

    assert result.requested_nodes == 1
    assert result.valid_nodes == 0
    assert result.rejected_fraction == 1.0


def test_pressure_gradient_aggregate_uses_exact_counts() -> None:
    result = aggregate_wall_pressure_gradient_summaries(
        [
            {
                "requested_samples": 10,
                "valid_samples": 8,
                "minimum": 0.1,
                "mean": 2.0,
                "maximum": 20.0,
                "le_one_samples": 3,
                "gt_ten_samples": 2,
                "gradient_scheme": "fluid_only_weighted_least_squares_26",
            },
            {
                "requested_samples": 5,
                "valid_samples": 4,
                "minimum": 0.2,
                "mean": 5.0,
                "maximum": 30.0,
                "le_one_samples": 1,
                "gt_ten_samples": 1,
                "gradient_scheme": "fluid_only_weighted_least_squares_26",
            },
        ],
    )

    assert result.requested_sample_exposures == 15
    assert result.valid_sample_exposures == 12
    assert result.rejected_fraction == 0.2
    assert result.mean == 3.0
    assert result.fraction_le_one == 4 / 12
    assert result.fraction_gt_ten == 3 / 12
