"""Fail-closed far-field and outlet-sensitivity planning for SUBOFF D3Q27.

This module is deliberately solver-agnostic: it reports geometric and
convective observability requirements, but never applies or changes a boundary
condition.  Lengths are lattice cells and convection times are lattice steps.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite
from typing import Mapping

_REQUIRED_METRICS = ("Ct", "Cp", "wake", "flx")


@dataclass(frozen=True)
class OutletSensitivityTolerances:
    """Absolute paired-run acceptance limits for required outlet metrics."""

    ct_absolute: float
    cp_absolute: float
    wake_absolute: float
    flx_absolute: float

    def as_mapping(self) -> dict[str, float]:
        return {
            "Ct": self.ct_absolute,
            "Cp": self.cp_absolute,
            "wake": self.wake_absolute,
            "flx": self.flx_absolute,
        }


@dataclass(frozen=True)
class OutletMetricResult:
    baseline: float | None
    candidate: float | None
    tolerance: float
    difference: float | None
    passed: bool


@dataclass(frozen=True)
class OutletSensitivityResult:
    """Pairwise result; ``accepted`` is true only when every metric passes."""

    accepted: bool
    metric_results: dict[str, OutletMetricResult]
    reasons: tuple[str, ...]


def _finite_positive(value: object, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return numeric


def build_suboff_far_field_metadata(
    *, nx: int, ny: int, nz: int, hull_length: float, u_in: float,
    hull_center_x: float, transient_steps: int | None = None,
) -> dict[str, object]:
    """Calculate and validate physical-domain distances and convection times.

    The outlet lies at the final physical lattice plane, ``x = nx - 1``.
    Therefore the reported requirements cover bow-to-stern hull convection,
    stern-to-outlet wake clearance, and inlet-to-outlet domain traversal.
    """
    nx_i, ny_i, nz_i = int(nx), int(ny), int(nz)
    if nx_i < 2 or ny_i < 1 or nz_i < 1:
        raise ValueError("physical domain dimensions require nx >= 2 and ny/nz >= 1")
    length = _finite_positive(hull_length, "hull_length")
    speed = _finite_positive(u_in, "u_in")
    center = float(hull_center_x)
    if not isfinite(center):
        raise ValueError("hull_center_x must be finite")
    if transient_steps is not None and int(transient_steps) < 0:
        raise ValueError("transient_steps must be non-negative")

    inlet_x, outlet_x = 0.0, float(nx_i - 1)
    bow_x, stern_x = center - 0.5 * length, center + 0.5 * length
    distances = {
        "inlet_to_bow": bow_x - inlet_x,
        "hull_length": stern_x - bow_x,
        "stern_to_outlet": outlet_x - stern_x,
        "inlet_to_outlet": outlet_x - inlet_x,
    }
    metadata: dict[str, object] = {
        "schema": "suboff-d3q27-far-field-v1",
        "units": {"length": "lattice_cells", "time": "lattice_steps"},
        "domain_shape": {"nx": nx_i, "ny": ny_i, "nz": nz_i},
        "streamwise_domain": {"inlet_x": inlet_x, "outlet_x": outlet_x},
        "hull_x_bounds": (bow_x, stern_x),
        "u_in": speed,
        "distances": distances,
        "convection_steps": {key: value / speed for key, value in distances.items()},
        "required_transient_steps": ceil(distances["inlet_to_outlet"] / speed),
    }
    if transient_steps is not None:
        metadata["transient_steps"] = int(transient_steps)
        metadata["transient_steps_satisfy_outlet_convection"] = (
            int(transient_steps) >= metadata["required_transient_steps"]
        )
    validate_suboff_far_field_metadata(metadata)
    return metadata


def validate_suboff_far_field_metadata(metadata: Mapping[str, object]) -> None:
    """Reject incomplete, inconsistent, or nonphysical planning metadata."""
    try:
        shape = metadata["domain_shape"]
        streamwise = metadata["streamwise_domain"]
        bounds = metadata["hull_x_bounds"]
        distances = metadata["distances"]
        convection = metadata["convection_steps"]
        speed = _finite_positive(metadata["u_in"], "u_in")
        nx = int(shape["nx"])  # type: ignore[index]
        inlet, outlet = float(streamwise["inlet_x"]), float(streamwise["outlet_x"])  # type: ignore[index]
        bow, stern = float(bounds[0]), float(bounds[1])  # type: ignore[index]
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError("incomplete far-field physical-domain metadata") from exc
    if nx < 2 or not all(isfinite(value) for value in (inlet, outlet, bow, stern)):
        raise ValueError("invalid far-field physical-domain metadata")
    if inlet != 0.0 or outlet != float(nx - 1):
        raise ValueError("invalid inlet/outlet physical-domain metadata")
    if bow <= inlet:
        raise ValueError("inlet-to-bow distance must be positive")
    if stern <= bow:
        raise ValueError("hull-length distance must be positive")
    if stern >= outlet:
        raise ValueError("stern-to-outlet distance must be positive")
    expected = {
        "inlet_to_bow": bow - inlet,
        "hull_length": stern - bow,
        "stern_to_outlet": outlet - stern,
        "inlet_to_outlet": outlet - inlet,
    }
    for key, expected_distance in expected.items():
        try:
            actual = float(distances[key])  # type: ignore[index]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"missing {key.replace('_', '-') } distance metadata") from exc
        if not isfinite(actual) or actual <= 0.0:
            raise ValueError(f"{key.replace('_', '-')} distance must be positive")
        if abs(actual - expected_distance) > 1e-9:
            raise ValueError(f"inconsistent {key.replace('_', '-')} distance metadata")
        try:
            time = float(convection[key])  # type: ignore[index]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"missing {key.replace('_', '-')} convection metadata") from exc
        if not isfinite(time) or time <= 0.0 or abs(time - actual / speed) > 1e-9:
            raise ValueError(f"inconsistent {key.replace('_', '-')} convection metadata")


def assess_outlet_distance_sensitivity(
    *, baseline: Mapping[str, object], candidate: Mapping[str, object],
    tolerances: OutletSensitivityTolerances,
) -> OutletSensitivityResult:
    """Apply an all-metrics, non-finite-is-failure outlet-distance gate."""
    tolerance_by_metric = tolerances.as_mapping()
    metric_results: dict[str, OutletMetricResult] = {}
    reasons: list[str] = []
    for metric in _REQUIRED_METRICS:
        tolerance = float(tolerance_by_metric[metric])
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(f"{metric} tolerance must be finite and non-negative")
        if metric not in baseline or metric not in candidate:
            metric_results[metric] = OutletMetricResult(None, None, tolerance, None, False)
            reasons.append(f"missing required metric: {metric}")
            continue
        try:
            reference, observed = float(baseline[metric]), float(candidate[metric])
        except (TypeError, ValueError):
            reference, observed = float("nan"), float("nan")
        if not isfinite(reference) or not isfinite(observed):
            metric_results[metric] = OutletMetricResult(reference, observed, tolerance, None, False)
            reasons.append(f"non-finite required metric: {metric}")
            continue
        difference = abs(observed - reference)
        passed = difference <= tolerance
        metric_results[metric] = OutletMetricResult(reference, observed, tolerance, difference, passed)
        if not passed:
            reasons.append(f"{metric} difference {difference:g} exceeds tolerance {tolerance:g}")
    return OutletSensitivityResult(not reasons, metric_results, tuple(reasons))
