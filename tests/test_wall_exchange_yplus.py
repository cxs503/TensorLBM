from __future__ import annotations

import pytest
import torch

from tensorlbm import (
    aggregate_wall_exchange_yplus_summaries,
    summarize_wall_exchange_yplus,
)


def test_wall_exchange_yplus_admits_finite_in_range_distribution() -> None:
    summary = summarize_wall_exchange_yplus(
        torch.tensor([20.0, 40.0, 100.0, 300.0, 900.0]),
        lower_bound=30.0,
        upper_bound=1000.0,
        minimum_in_range_fraction=0.8,
    )

    assert summary.requested_samples == 5
    assert summary.finite_samples == 5
    assert summary.below_range_samples == 1
    assert summary.in_range_samples == 4
    assert summary.above_range_samples == 0
    assert summary.in_range_fraction == pytest.approx(0.8)
    assert summary.median_y_plus == pytest.approx(100.0)
    assert summary.admitted is True


def test_wall_exchange_yplus_rejects_nonfinite_and_out_of_range_samples() -> None:
    summary = summarize_wall_exchange_yplus(
        torch.tensor([40.0, 80.0, 1200.0, float("nan")]),
        minimum_in_range_fraction=0.5,
    )

    assert summary.finite_fraction == pytest.approx(0.75)
    assert summary.in_range_fraction == pytest.approx(2.0 / 3.0)
    assert summary.above_range_samples == 1
    assert summary.admitted is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lower_bound": -1.0},
        {"lower_bound": 30.0, "upper_bound": 30.0},
        {"minimum_in_range_fraction": 1.1},
    ],
)
def test_wall_exchange_yplus_rejects_invalid_policy(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        summarize_wall_exchange_yplus(torch.ones(4), **kwargs)


def test_wall_exchange_yplus_empty_input_fails_closed() -> None:
    summary = summarize_wall_exchange_yplus(torch.empty(0))

    assert summary.finite_samples == 0
    assert summary.in_range_fraction == 0.0
    assert summary.admitted is False


def test_wall_exchange_yplus_aggregate_uses_exact_exposure_counts() -> None:
    first = summarize_wall_exchange_yplus(
        torch.tensor([20.0, 40.0, 80.0]),
        minimum_in_range_fraction=0.6,
    )
    second = summarize_wall_exchange_yplus(
        torch.tensor([100.0, 1200.0]),
        minimum_in_range_fraction=0.6,
    )

    aggregate = aggregate_wall_exchange_yplus_summaries(
        [
            first.to_dict(),
            second.to_dict(),
        ]
    )

    assert aggregate.requested_sample_exposures == 5
    assert aggregate.in_range_sample_exposures == 3
    assert aggregate.in_range_fraction == pytest.approx(0.6)
    assert aggregate.mean_y_plus == pytest.approx(288.0)
    assert aggregate.admitted is True


def test_wall_exchange_yplus_aggregate_rejects_mixed_policies() -> None:
    first = summarize_wall_exchange_yplus(torch.tensor([100.0]))
    second = summarize_wall_exchange_yplus(
        torch.tensor([100.0]),
        upper_bound=500.0,
    )

    with pytest.raises(ValueError, match="different policies"):
        aggregate_wall_exchange_yplus_summaries(
            [
                first.to_dict(),
                second.to_dict(),
            ]
        )
