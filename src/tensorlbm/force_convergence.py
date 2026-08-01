"""Time-stationarity diagnostics for CFD force histories.

Instantaneous agreement with a reference value is not convergence.  This
module reduces a force history to equal-duration block means and checks three
independent failure modes: block-to-block oscillation, early/late mean drift,
and a persistent linear trend.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from collections.abc import Sequence


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
    finite: bool
    sufficiently_sampled: bool

    def meets(self, relative_tolerance_pct: float) -> bool:
        if relative_tolerance_pct < 0.0:
            raise ValueError("relative_tolerance_pct must be non-negative")
        return (
            self.finite
            and self.sufficiently_sampled
            and self.relative_range_pct <= relative_tolerance_pct
            and self.half_mean_drift_pct <= relative_tolerance_pct
            and self.linear_trend_pct <= relative_tolerance_pct
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
            finite=finite, sufficiently_sampled=False,
        )

    mean = sum(blocks) / len(blocks)
    scale = abs(mean)
    if scale <= 1e-30:
        scale = 1e-30
    relative_range = (max(blocks) - min(blocks)) / scale * 100.0
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
        finite=finite, sufficiently_sampled=len(blocks) >= minimum_blocks,
    )


__all__ = ["ForceStationarityReport", "assess_force_stationarity"]
