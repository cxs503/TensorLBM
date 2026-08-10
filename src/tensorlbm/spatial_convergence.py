"""Resolution-sequence convergence and discretisation uncertainty evidence."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class SpatialConvergenceAssessment:
    resolutions: tuple[float, ...]
    values: tuple[float, ...]
    monotonic: bool
    observed_order: float
    extrapolated_value: float
    finest_relative_error_pct: float
    relative_fit_rms_pct: float
    finite: bool

    def meets(
        self,
        *,
        maximum_finest_error_pct: float,
        maximum_fit_rms_pct: float,
        minimum_order: float = 0.5,
    ) -> bool:
        if min(maximum_finest_error_pct, maximum_fit_rms_pct, minimum_order) < 0.0:
            raise ValueError("convergence targets must be non-negative")
        return (
            self.finite
            and self.monotonic
            and self.observed_order >= minimum_order
            and self.finest_relative_error_pct <= maximum_finest_error_pct
            and self.relative_fit_rms_pct <= maximum_fit_rms_pct
        )


def _linear_fit(x: tuple[float, ...], y: tuple[float, ...]) -> tuple[float, float, float]:
    count = len(x)
    mean_x = sum(x) / count
    mean_y = sum(y) / count
    denominator = sum((value - mean_x) ** 2 for value in x)
    if denominator <= 1e-30:
        return mean_y, 0.0, math.inf
    slope = sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(x, y, strict=True)
    ) / denominator
    intercept = mean_y - slope * mean_x
    residual = sum(
        (value - (intercept + slope * coordinate)) ** 2
        for coordinate, value in zip(x, y, strict=True)
    )
    return intercept, slope, residual


def assess_spatial_convergence(
    resolutions: Sequence[float],
    values: Sequence[float],
    *,
    minimum_order_search: float = 0.1,
    maximum_order_search: float = 8.0,
) -> SpatialConvergenceAssessment:
    """Fit ``phi(N)=phi_inf+a*N**(-p)`` to at least three resolutions."""
    n = tuple(float(value) for value in resolutions)
    phi = tuple(float(value) for value in values)
    if len(n) != len(phi) or len(n) < 3:
        raise ValueError("at least three paired resolutions and values are required")
    if not all(math.isfinite(value) and value > 0.0 for value in n):
        raise ValueError("resolutions must be finite and positive")
    if any(right <= left for left, right in zip(n, n[1:], strict=False)):
        raise ValueError("resolutions must be strictly increasing")
    if not 0.0 < minimum_order_search < maximum_order_search:
        raise ValueError("invalid observed-order search interval")
    finite = all(math.isfinite(value) for value in phi)
    differences = tuple(
        right - left for left, right in zip(phi, phi[1:], strict=False)
    )
    monotonic = finite and (
        all(value >= 0.0 for value in differences)
        or all(value <= 0.0 for value in differences)
    ) and any(abs(value) > 0.0 for value in differences)
    if not finite:
        return SpatialConvergenceAssessment(
            n, phi, False, math.nan, math.nan, math.inf, math.inf, False,
        )

    def fit(order: float) -> tuple[float, float, float]:
        return _linear_fit(tuple(value ** (-order) for value in n), phi)

    grid_points = 400
    orders = [
        minimum_order_search
        + index * (maximum_order_search - minimum_order_search) / grid_points
        for index in range(grid_points + 1)
    ]
    best_index = min(range(len(orders)), key=lambda index: fit(orders[index])[2])
    left = orders[max(0, best_index - 1)]
    right = orders[min(grid_points, best_index + 1)]
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = right - golden * (right - left)
    x2 = left + golden * (right - left)
    for _ in range(60):
        if fit(x1)[2] <= fit(x2)[2]:
            right, x2 = x2, x1
            x1 = right - golden * (right - left)
        else:
            left, x1 = x1, x2
            x2 = left + golden * (right - left)
    observed_order = 0.5 * (left + right)
    extrapolated, _, residual = fit(observed_order)
    finest_error = (
        abs(phi[-1] - extrapolated) / max(abs(extrapolated), 1e-30) * 100.0
    )
    fit_rms = math.sqrt(residual / len(phi))
    fit_rms_pct = fit_rms / max(abs(extrapolated), 1e-30) * 100.0
    return SpatialConvergenceAssessment(
        resolutions=n,
        values=phi,
        monotonic=monotonic,
        observed_order=observed_order,
        extrapolated_value=extrapolated,
        finest_relative_error_pct=finest_error,
        relative_fit_rms_pct=fit_rms_pct,
        finite=True,
    )


__all__ = [
    "SpatialConvergenceAssessment",
    "assess_spatial_convergence",
]
