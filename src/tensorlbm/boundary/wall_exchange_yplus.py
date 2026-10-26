"""Common applicability evidence for wall-model exchange locations.

Wall-modelled LES needs more than a nominal exchange distance.  The sampled
velocity must also place a sufficiently large fraction of the active wall
nodes inside the declared wall-law ``y+`` interval.  This module keeps that
policy and its diagnostics independent of any particular geometry or solver
runner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass(frozen=True)
class WallExchangeYPlusSummary:
    """Distribution and applicability decision for active exchange samples."""

    requested_samples: int
    finite_samples: int
    minimum_y_plus: float | None
    percentile05_y_plus: float | None
    median_y_plus: float | None
    mean_y_plus: float | None
    percentile95_y_plus: float | None
    maximum_y_plus: float | None
    lower_bound_y_plus: float
    upper_bound_y_plus: float
    below_range_samples: int
    in_range_samples: int
    above_range_samples: int
    finite_fraction: float
    in_range_fraction: float
    minimum_in_range_fraction: float
    admitted: bool

    def to_dict(self) -> dict[str, float | int | bool | None]:
        return {
            "requested_samples": self.requested_samples,
            "finite_samples": self.finite_samples,
            "minimum_y_plus": self.minimum_y_plus,
            "percentile05_y_plus": self.percentile05_y_plus,
            "median_y_plus": self.median_y_plus,
            "mean_y_plus": self.mean_y_plus,
            "percentile95_y_plus": self.percentile95_y_plus,
            "maximum_y_plus": self.maximum_y_plus,
            "lower_bound_y_plus": self.lower_bound_y_plus,
            "upper_bound_y_plus": self.upper_bound_y_plus,
            "below_range_samples": self.below_range_samples,
            "in_range_samples": self.in_range_samples,
            "above_range_samples": self.above_range_samples,
            "finite_fraction": self.finite_fraction,
            "in_range_fraction": self.in_range_fraction,
            "minimum_in_range_fraction": self.minimum_in_range_fraction,
            "admitted": self.admitted,
        }


@dataclass(frozen=True)
class WallExchangeYPlusAggregate:
    """Exact count-weighted aggregate over repeated wall diagnostics."""

    observations: int
    requested_sample_exposures: int
    finite_sample_exposures: int
    minimum_y_plus: float | None
    mean_y_plus: float | None
    maximum_y_plus: float | None
    lower_bound_y_plus: float
    upper_bound_y_plus: float
    below_range_sample_exposures: int
    in_range_sample_exposures: int
    above_range_sample_exposures: int
    finite_fraction: float
    in_range_fraction: float
    minimum_in_range_fraction: float
    admitted: bool

    def to_dict(self) -> dict[str, float | int | bool | None]:
        return dict(vars(self))


def summarize_wall_exchange_yplus(
    y_plus: torch.Tensor,
    *,
    lower_bound: float = 30.0,
    upper_bound: float = 1000.0,
    minimum_in_range_fraction: float = 0.9,
) -> WallExchangeYPlusSummary:
    """Summarize measured exchange ``y+`` and fail closed on invalid samples.

    Bounds are an explicit run policy, not a universal turbulence constant.
    The defaults cover the broad equilibrium wall-model interval already used
    by TensorLBM's high-Re planning tools.  Validation campaigns may narrow
    the interval after independent flat-plate/channel evidence.
    """
    if not math.isfinite(lower_bound) or lower_bound < 0.0:
        raise ValueError("lower y+ bound must be finite and non-negative")
    if not math.isfinite(upper_bound) or upper_bound <= lower_bound:
        raise ValueError("upper y+ bound must be finite and exceed the lower bound")
    if not 0.0 <= minimum_in_range_fraction <= 1.0:
        raise ValueError("minimum in-range fraction must lie in [0,1]")
    if not isinstance(y_plus, torch.Tensor):
        raise TypeError("y_plus must be a torch.Tensor")

    values = y_plus.detach().reshape(-1)
    requested = int(values.numel())
    finite_mask = torch.isfinite(values)
    finite = values[finite_mask]
    finite_count = int(finite.numel())
    finite_fraction = finite_count / requested if requested else 0.0
    if not finite_count:
        return WallExchangeYPlusSummary(
            requested_samples=requested,
            finite_samples=0,
            minimum_y_plus=None,
            percentile05_y_plus=None,
            median_y_plus=None,
            mean_y_plus=None,
            percentile95_y_plus=None,
            maximum_y_plus=None,
            lower_bound_y_plus=lower_bound,
            upper_bound_y_plus=upper_bound,
            below_range_samples=0,
            in_range_samples=0,
            above_range_samples=0,
            finite_fraction=finite_fraction,
            in_range_fraction=0.0,
            minimum_in_range_fraction=minimum_in_range_fraction,
            admitted=False,
        )

    below = int((finite < lower_bound).sum().item())
    above = int((finite > upper_bound).sum().item())
    in_range = finite_count - below - above
    in_range_fraction = in_range / finite_count
    quantiles = torch.quantile(
        finite.to(dtype=torch.float64),
        torch.tensor((0.05, 0.5, 0.95), device=finite.device, dtype=torch.float64),
    )
    return WallExchangeYPlusSummary(
        requested_samples=requested,
        finite_samples=finite_count,
        minimum_y_plus=float(finite.min().item()),
        percentile05_y_plus=float(quantiles[0].item()),
        median_y_plus=float(quantiles[1].item()),
        mean_y_plus=float(finite.mean().item()),
        percentile95_y_plus=float(quantiles[2].item()),
        maximum_y_plus=float(finite.max().item()),
        lower_bound_y_plus=lower_bound,
        upper_bound_y_plus=upper_bound,
        below_range_samples=below,
        in_range_samples=in_range,
        above_range_samples=above,
        finite_fraction=finite_fraction,
        in_range_fraction=in_range_fraction,
        minimum_in_range_fraction=minimum_in_range_fraction,
        admitted=(
            finite_count == requested
            and requested > 0
            and in_range_fraction >= minimum_in_range_fraction
        ),
    )


def aggregate_wall_exchange_yplus_summaries(
    summaries: Sequence[Mapping[str, float | int | bool | None]],
) -> WallExchangeYPlusAggregate:
    """Combine repeated summaries without pretending quantiles are additive.

    Counts, extrema and the count-weighted mean are exact.  Percentiles are
    intentionally absent because reconstructing them from per-observation
    percentiles would be mathematically invalid.
    """
    if not summaries:
        raise ValueError("at least one y+ summary is required")
    lower = float(summaries[0]["lower_bound_y_plus"])
    upper = float(summaries[0]["upper_bound_y_plus"])
    required = float(summaries[0]["minimum_in_range_fraction"])
    for item in summaries[1:]:
        policy = (
            float(item["lower_bound_y_plus"]),
            float(item["upper_bound_y_plus"]),
            float(item["minimum_in_range_fraction"]),
        )
        if policy != (lower, upper, required):
            raise ValueError("cannot aggregate y+ summaries with different policies")

    def count(item: Mapping, summary_key: str, aggregate_key: str) -> int:
        key = summary_key if summary_key in item else aggregate_key
        return int(item[key])

    requested = sum(
        count(
            item,
            "requested_samples",
            "requested_sample_exposures",
        )
        for item in summaries
    )
    finite = sum(
        count(
            item,
            "finite_samples",
            "finite_sample_exposures",
        )
        for item in summaries
    )
    below = sum(
        count(
            item,
            "below_range_samples",
            "below_range_sample_exposures",
        )
        for item in summaries
    )
    in_range = sum(
        count(
            item,
            "in_range_samples",
            "in_range_sample_exposures",
        )
        for item in summaries
    )
    above = sum(
        count(
            item,
            "above_range_samples",
            "above_range_sample_exposures",
        )
        for item in summaries
    )
    finite_means = [
        (
            count(item, "finite_samples", "finite_sample_exposures"),
            float(item["mean_y_plus"]),
        )
        for item in summaries
        if count(item, "finite_samples", "finite_sample_exposures")
        and item["mean_y_plus"] is not None
    ]
    minima = [
        float(item["minimum_y_plus"]) for item in summaries if item["minimum_y_plus"] is not None
    ]
    maxima = [
        float(item["maximum_y_plus"]) for item in summaries if item["maximum_y_plus"] is not None
    ]
    finite_fraction = finite / requested if requested else 0.0
    in_range_fraction = in_range / finite if finite else 0.0
    return WallExchangeYPlusAggregate(
        observations=sum(int(item.get("observations", 1)) for item in summaries),
        requested_sample_exposures=requested,
        finite_sample_exposures=finite,
        minimum_y_plus=min(minima) if minima else None,
        mean_y_plus=(
            sum(count * mean for count, mean in finite_means) / finite if finite else None
        ),
        maximum_y_plus=max(maxima) if maxima else None,
        lower_bound_y_plus=lower,
        upper_bound_y_plus=upper,
        below_range_sample_exposures=below,
        in_range_sample_exposures=in_range,
        above_range_sample_exposures=above,
        finite_fraction=finite_fraction,
        in_range_fraction=in_range_fraction,
        minimum_in_range_fraction=required,
        admitted=(requested > 0 and finite == requested and in_range_fraction >= required),
    )


__all__ = [
    "WallExchangeYPlusAggregate",
    "WallExchangeYPlusSummary",
    "aggregate_wall_exchange_yplus_summaries",
    "summarize_wall_exchange_yplus",
]
