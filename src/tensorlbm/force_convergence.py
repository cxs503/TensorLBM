"""Time-stationarity diagnostics for CFD force histories.

Instantaneous agreement with a reference value is not convergence.  This
module reduces a force history to equal-duration batch means and checks three
independent failure modes: early/late mean drift, a persistent linear trend,
and insufficient Student-t confidence in the time mean.  Batch range remains
reported as an unsteady-load diagnostic but is not itself a mean-convergence
criterion.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


_T975 = (
    math.inf, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365,
    2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
    2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069,
    2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)


def _student_t_975(degrees_of_freedom: int) -> float:
    if degrees_of_freedom < 1:
        return math.inf
    if degrees_of_freedom < len(_T975):
        return _T975[degrees_of_freedom]
    return 1.96


@dataclass(frozen=True)
class ForceStationarityReport:
    sample_count: int
    block_size: int
    block_count: int
    mean: float
    block_means: tuple[float, ...]
    relative_range_pct: float
    half_mean_drift_pct: float
    linear_trend_pct: float
    standard_error_pct: float
    confidence95_half_width_pct: float
    finite: bool
    sufficiently_sampled: bool

    def meets(self, relative_tolerance_pct: float) -> bool:
        if relative_tolerance_pct < 0.0:
            raise ValueError("relative_tolerance_pct must be non-negative")
        return (
            self.finite
            and self.sufficiently_sampled
            and self.half_mean_drift_pct <= relative_tolerance_pct
            and self.linear_trend_pct <= relative_tolerance_pct
            and self.confidence95_half_width_pct <= relative_tolerance_pct
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def assess_force_stationarity(
    samples: Sequence[float],
    *,
    block_size: int,
    minimum_blocks: int = 4,
) -> ForceStationarityReport:
    """Assess stationarity from complete, non-overlapping block means."""
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if minimum_blocks < 2:
        raise ValueError("minimum_blocks must be at least two")
    values = tuple(float(value) for value in samples)
    finite = bool(values) and all(math.isfinite(value) for value in values)
    block_count = len(values) // block_size
    blocks = tuple(
        sum(values[index * block_size:(index + 1) * block_size]) / block_size
        for index in range(block_count)
    ) if finite else ()
    if not blocks:
        return ForceStationarityReport(
            sample_count=len(values), block_size=block_size,
            block_count=0, mean=math.nan, block_means=(),
            relative_range_pct=math.inf, half_mean_drift_pct=math.inf,
            linear_trend_pct=math.inf, standard_error_pct=math.inf,
            confidence95_half_width_pct=math.inf,
            finite=finite, sufficiently_sampled=False,
        )

    mean = sum(blocks) / len(blocks)
    scale = abs(mean)
    if scale <= 1e-30:
        scale = 1e-30
    relative_range = (max(blocks) - min(blocks)) / scale * 100.0
    if len(blocks) == 1:
        return ForceStationarityReport(
            sample_count=len(values), block_size=block_size,
            block_count=1, mean=mean, block_means=blocks,
            relative_range_pct=relative_range,
            half_mean_drift_pct=math.inf,
            linear_trend_pct=math.inf,
            standard_error_pct=math.inf,
            confidence95_half_width_pct=math.inf,
            finite=finite, sufficiently_sampled=False,
        )
    split = len(blocks) // 2
    early = sum(blocks[:split]) / split
    late_count = len(blocks) - split
    late = sum(blocks[split:]) / late_count
    half_drift = abs(late - early) / scale * 100.0

    x_mean = (len(blocks) - 1) / 2.0
    denominator = sum((index - x_mean) ** 2 for index in range(len(blocks)))
    slope = (
        sum((index - x_mean) * (value - mean) for index, value in enumerate(blocks))
        / denominator
        if denominator > 0.0 else math.inf
    )
    trend = abs(slope) * max(len(blocks) - 1, 1) / scale * 100.0
    if len(blocks) > 1:
        variance = sum((value - mean) ** 2 for value in blocks) / (len(blocks) - 1)
        standard_error = math.sqrt(variance / len(blocks)) / scale * 100.0
    else:
        standard_error = math.inf
    return ForceStationarityReport(
        sample_count=len(values), block_size=block_size,
        block_count=len(blocks), mean=mean, block_means=blocks,
        relative_range_pct=relative_range,
        half_mean_drift_pct=half_drift,
        linear_trend_pct=trend, standard_error_pct=standard_error,
        confidence95_half_width_pct=(
            _student_t_975(len(blocks) - 1) * standard_error
        ),
        finite=finite, sufficiently_sampled=len(blocks) >= minimum_blocks,
    )


__all__ = ["ForceStationarityReport", "assess_force_stationarity"]
