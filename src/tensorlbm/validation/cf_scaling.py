"""Scale-effect calibration of the skin-friction coefficient, ``Cf(Re)``.

Wall functions enter the resistance prediction as a *friction* model, so the
scale effect of ``C_f`` with Reynolds number is the quantity that has to be
calibrated rather than a single-point value.  This module fits

    ``C_f = A Re^{-n}``

to a set of measured/reported friction coefficients by log-log least squares,
and compares the fitted line against the ITTC-1957 / Dean / Schultz-Grunow
reference lines over the calibrated band and at a held-out point.

Typical use: feed the ``Ct``/``Cf`` values produced by the wall-modelled LES
ladder for the SUBOFF speed anchors and check that (i) the fitted exponent is
within ``exponent_tolerance`` of the reference line and (ii) the pointwise
``C_f`` error stays below the accuracy gate.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

from .wall_law import dean_channel_cf, ittc57_cf, schultz_grunow_cf

__all__ = [
    "CfFit",
    "CfScalingReport",
    "REFERENCE_LINES",
    "calibrate_cf_scaling",
    "fit_power_law",
]

REFERENCE_LINES: dict[str, Callable[[float], float]] = {
    "ittc57": ittc57_cf,
    "dean_channel": dean_channel_cf,
    "schultz_grunow": schultz_grunow_cf,
}


@dataclass(frozen=True)
class CfFit:
    """Power-law fit ``C_f = amplitude * Re^{-exponent}``."""

    amplitude: float
    exponent: float
    exponent_standard_error: float
    r_squared: float
    sample_count: int

    def evaluate(self, reynolds: float) -> float:
        return self.amplitude * reynolds ** (-self.exponent)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_power_law(
    reynolds: Sequence[float], cf_values: Sequence[float]
) -> CfFit:
    """Log-log least squares fit of ``C_f = A Re^{-n}``."""
    if len(reynolds) != len(cf_values):
        raise ValueError("reynolds and cf_values must have the same length")
    if len(reynolds) < 2:
        raise ValueError("a power-law fit needs at least two samples")
    xs = [math.log(float(value)) for value in reynolds]
    ys = [math.log(float(value)) for value in cf_values]
    for value, re_value, cf_value in zip(ys, reynolds, cf_values, strict=True):
        if not math.isfinite(value) or re_value <= 0.0 or cf_value <= 0.0:
            raise ValueError("log-log fit requires positive finite inputs")
    count = len(xs)
    x_mean = sum(xs) / count
    y_mean = sum(ys) / count
    sxx = sum((x - x_mean) ** 2 for x in xs)
    sxy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    if sxx == 0.0:
        raise ValueError("all Reynolds numbers are identical")
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys, strict=True)]
    sse = sum(value**2 for value in residuals)
    sst = sum((y - y_mean) ** 2 for y in ys)
    r_squared = 1.0 - sse / sst if sst > 0.0 else 1.0
    # standard error of the slope (count > 2 required for a meaningful estimate)
    if count > 2:
        standard_error = math.sqrt(sse / (count - 2) / sxx)
    else:
        standard_error = float("nan")
    return CfFit(
        amplitude=math.exp(intercept),
        exponent=-slope,
        exponent_standard_error=standard_error,
        r_squared=r_squared,
        sample_count=count,
    )


@dataclass(frozen=True)
class CfScalingReport:
    """Outcome of a ``Cf(Re)`` scale-effect calibration."""

    reference_line: str
    fit: CfFit
    reference_exponent: float
    exponent_deviation: float
    exponent_deviation_pct: float
    errors_pct: tuple[float, ...]
    maximum_error_pct: float
    mean_error_pct: float
    holdout_reynolds: float | None
    holdout_reference: float | None
    holdout_predicted: float | None
    holdout_error_pct: float | None
    monotone_decreasing: bool
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fit"] = self.fit.to_dict()
        return payload


def _reference_exponent(
    reference: Callable[[float], float], reynolds: Sequence[float]
) -> float:
    re_low, re_high = min(reynolds), max(reynolds)
    return (
        math.log(reference(re_low) / reference(re_high))
        / math.log(re_high / re_low)
    )


def calibrate_cf_scaling(
    reynolds: Sequence[float],
    cf_values: Sequence[float],
    *,
    reference_line: str = "ittc57",
    exponent_tolerance: float = 0.02,
    maximum_error_pct: float = 3.0,
    holdout_reynolds: float | None = None,
) -> CfScalingReport:
    """Fit the scale effect and judge it against a reference friction line."""
    if reference_line not in REFERENCE_LINES:
        raise ValueError(
            f"unknown reference line {reference_line!r}; "
            f"expected one of {sorted(REFERENCE_LINES)}"
        )
    reference = REFERENCE_LINES[reference_line]
    ordered = sorted(zip(reynolds, cf_values, strict=True), key=lambda item: item[0])
    re_sorted = [float(item[0]) for item in ordered]
    cf_sorted = [float(item[1]) for item in ordered]
    fit = fit_power_law(re_sorted, cf_sorted)
    reference_exponent = _reference_exponent(reference, re_sorted)
    deviation = fit.exponent - reference_exponent
    errors = [
        abs(measured - reference(re_value)) / reference(re_value) * 100.0
        for re_value, measured in zip(re_sorted, cf_sorted, strict=True)
    ]
    maximum = max(errors)
    mean = sum(errors) / len(errors)
    monotone = all(
        later < earlier for earlier, later in zip(cf_sorted, cf_sorted[1:], strict=False)
    )

    holdout_reference = holdout_predicted = holdout_error = None
    if holdout_reynolds is not None:
        holdout_reference = reference(float(holdout_reynolds))
        holdout_predicted = fit.evaluate(float(holdout_reynolds))
        holdout_error = (
            abs(holdout_predicted - holdout_reference) / holdout_reference * 100.0
        )

    reasons = []
    if abs(deviation) > exponent_tolerance:
        reasons.append(
            "exponent deviation %.4f exceeds tolerance %.4f" % (deviation, exponent_tolerance)
        )
    if maximum > maximum_error_pct:
        reasons.append(
            "maximum C_f error %.2f%% exceeds gate %.2f%%" % (maximum, maximum_error_pct)
        )
    if not monotone:
        reasons.append("simulated C_f(Re) is not monotone decreasing")

    return CfScalingReport(
        reference_line=reference_line,
        fit=fit,
        reference_exponent=reference_exponent,
        exponent_deviation=deviation,
        exponent_deviation_pct=abs(deviation) / abs(reference_exponent) * 100.0,
        errors_pct=tuple(errors),
        maximum_error_pct=maximum,
        mean_error_pct=mean,
        holdout_reynolds=holdout_reynolds,
        holdout_reference=holdout_reference,
        holdout_predicted=holdout_predicted,
        holdout_error_pct=holdout_error,
        monotone_decreasing=monotone,
        verdict="PASS" if not reasons else "FAIL: " + "; ".join(reasons),
    )