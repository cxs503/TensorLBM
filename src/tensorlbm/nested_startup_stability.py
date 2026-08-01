"""Fail-closed startup-stability assessment for nested LBM trajectories.

The assessment is deliberately independent of geometry and force accuracy.
It answers only whether a nested-grid startup remained numerically admissible
while reaching the requested collision Reynolds number.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class NestedStartupAssessment:
    status: str
    startup_stability_pass: bool
    target_reynolds_reached: bool
    health_snapshots: int
    target_reynolds_steps: int
    all_levels_finite: bool
    minimum_population: float | None
    minimum_density: float | None
    maximum_density: float | None
    maximum_speed: float | None
    maximum_reflux_residual: float | None
    maximum_transfer_limited_fraction: float | None
    maximum_collision_limited_fraction: float | None
    failure_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_values(values: list[Any]) -> list[float]:
    return [
        float(value) for value in values
        if value is not None and math.isfinite(float(value))
    ]


def assess_nested_startup(
    result: dict[str, Any],
    *,
    maximum_speed: float = 0.3,
    maximum_limited_fraction: float = 1.0e-3,
    maximum_reflux_residual: float = 1.0e-6,
    require_target_reynolds: bool = True,
) -> NestedStartupAssessment:
    """Assess numerical startup health without making a force claim."""
    if not 0.0 < maximum_speed < 1.0:
        raise ValueError("maximum speed must lie in (0,1)")
    if maximum_limited_fraction < 0.0:
        raise ValueError("maximum limited fraction must be non-negative")
    if maximum_reflux_residual < 0.0:
        raise ValueError("maximum reflux residual must be non-negative")

    payload = result.get("result", {})
    health = payload.get("population_health", [])
    levels = [level for record in health for level in record.get("levels", [])]
    interfaces = [
        interface
        for record in health
        for interface in record.get("interfaces", [])
    ]
    populations = _finite_values([
        level.get("minimum_population") for level in levels
    ])
    densities_min = _finite_values([
        level.get("minimum_density") for level in levels
    ])
    densities_max = _finite_values([
        level.get("maximum_density") for level in levels
    ])
    speeds = _finite_values([level.get("maximum_speed") for level in levels])
    reflux = _finite_values([
        interface.get("maximum_reflux_residual") for interface in interfaces
    ])
    transfer_limiter = _finite_values([
        interface.get("restriction_limited_fraction")
        for interface in interfaces
    ])

    target = float(result.get("configuration", {}).get("resolved_reynolds", 0.0))
    step_records = payload.get("steps", [])
    target_steps = sum(
        math.isclose(
            float(record.get("collision_resolved_reynolds", math.nan)),
            target,
            rel_tol=1.0e-12,
            abs_tol=0.0,
        )
        for record in step_records
    ) if target > 0.0 else 0
    target_reached = target_steps > 0 or any(
        bool(record.get("target_reynolds_reached")) for record in health
    )

    all_finite = bool(levels) and all(bool(level.get("finite")) for level in levels)
    minimum_population = min(populations) if populations else None
    minimum_density = min(densities_min) if densities_min else None
    maximum_density_seen = max(densities_max) if densities_max else None
    maximum_speed_seen = max(speeds) if speeds else None
    maximum_reflux_seen = max(reflux) if reflux else None
    maximum_transfer_limiter = max(transfer_limiter) if transfer_limiter else None
    collision_limiter = payload.get("maximum_positivity_limited_fraction")
    maximum_collision_limiter = (
        float(collision_limiter) if collision_limiter is not None else None
    )

    reasons: list[str] = []
    if not health or not levels:
        reasons.append("missing_population_health")
    if not all_finite:
        reasons.append("nonfinite_level_state")
    if minimum_population is None or minimum_population <= 0.0:
        reasons.append("nonpositive_or_missing_population")
    if minimum_density is None or minimum_density <= 0.0:
        reasons.append("nonpositive_or_missing_density")
    if maximum_density_seen is None:
        reasons.append("missing_density_maximum")
    if maximum_speed_seen is None or maximum_speed_seen > maximum_speed:
        reasons.append("weakly_compressible_speed_gate_failed")
    if maximum_reflux_seen is None or maximum_reflux_seen > maximum_reflux_residual:
        reasons.append("reflux_residual_gate_failed")
    if (
        maximum_transfer_limiter is None
        or maximum_transfer_limiter > maximum_limited_fraction
    ):
        reasons.append("transfer_limiter_gate_failed")
    if (
        maximum_collision_limiter is None
        or maximum_collision_limiter > maximum_limited_fraction
    ):
        reasons.append("collision_limiter_gate_failed")
    if require_target_reynolds and not target_reached:
        reasons.append("target_reynolds_not_reached")

    passed = not reasons
    return NestedStartupAssessment(
        status="startup_stability_pass" if passed else "startup_stability_fail",
        startup_stability_pass=passed,
        target_reynolds_reached=target_reached,
        health_snapshots=len(health),
        target_reynolds_steps=target_steps,
        all_levels_finite=all_finite,
        minimum_population=minimum_population,
        minimum_density=minimum_density,
        maximum_density=maximum_density_seen,
        maximum_speed=maximum_speed_seen,
        maximum_reflux_residual=maximum_reflux_seen,
        maximum_transfer_limited_fraction=maximum_transfer_limiter,
        maximum_collision_limited_fraction=maximum_collision_limiter,
        failure_reasons=tuple(reasons),
    )
