"""Reproducible A/B comparison for nested-LBM health log records."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

_PREFIX = "nested health "


def read_nested_health_log(path: str | Path) -> list[dict[str, Any]]:
    """Read complete JSON health lines, ignoring unrelated or partial lines."""
    records: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(_PREFIX):
            continue
        try:
            record = json.loads(line[len(_PREFIX):])
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and isinstance(record.get("step"), int):
            records.append(record)
    return records


def _metrics(record: dict[str, Any]) -> dict[str, float | int | bool | None]:
    levels = record.get("levels", [])
    interfaces = record.get("interfaces", [])
    speeds = [float(level["maximum_speed"]) for level in levels]
    populations = [float(level["minimum_population"]) for level in levels]
    reflux = [
        abs(float(interface["maximum_reflux_residual"]))
        for interface in interfaces
    ]
    return {
        "step": int(record["step"]),
        "target_reynolds_reached": bool(record.get("target_reynolds_reached")),
        "maximum_speed": max(speeds) if speeds else None,
        "minimum_population": min(populations) if populations else None,
        "maximum_reflux_residual": max(reflux) if reflux else None,
        "maximum_collision_limited_fraction": record.get(
            "maximum_collision_limited_fraction",
        ),
    }


def compare_nested_health(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare two trajectories only at exactly matching health steps."""
    baseline_by_step = {int(record["step"]): _metrics(record) for record in baseline}
    candidate_by_step = {int(record["step"]): _metrics(record) for record in candidate}
    common_steps = sorted(baseline_by_step.keys() & candidate_by_step.keys())
    if not common_steps:
        raise ValueError("health trajectories have no common steps")

    aligned: list[dict[str, Any]] = []
    for step in common_steps:
        base = baseline_by_step[step]
        trial = candidate_by_step[step]
        base_speed = base["maximum_speed"]
        trial_speed = trial["maximum_speed"]
        speed_ratio = None
        if isinstance(base_speed, float) and isinstance(trial_speed, float):
            speed_ratio = trial_speed / base_speed if base_speed > 0.0 else None
        aligned.append({
            "step": step,
            "baseline": base,
            "candidate": trial,
            "candidate_to_baseline_speed_ratio": speed_ratio,
        })

    def values(side: str, metric: str) -> list[float]:
        return [
            float(item[side][metric])
            for item in aligned
            if item[side][metric] is not None
            and math.isfinite(float(item[side][metric]))
        ]

    baseline_speeds = values("baseline", "maximum_speed")
    candidate_speeds = values("candidate", "maximum_speed")
    baseline_populations = values("baseline", "minimum_population")
    candidate_populations = values("candidate", "minimum_population")
    latest = aligned[-1]

    def first_step(
        side: str,
        metric: str,
        predicate: Callable[[float], bool],
    ) -> int | None:
        for item in aligned:
            value = item[side][metric]
            if value is not None and predicate(float(value)):
                return int(item["step"])
        return None

    speed_threshold_steps = {
        str(threshold): {
            side: first_step(
                side, "maximum_speed", lambda value, limit=threshold: value >= limit,
            )
            for side in ("baseline", "candidate")
        }
        for threshold in (0.1, 0.15, 0.3)
    }
    return {
        "schema": "tensorlbm-nested-health-comparison-v1",
        "common_step_count": len(common_steps),
        "first_common_step": common_steps[0],
        "latest_common_step": common_steps[-1],
        "baseline_maximum_speed": max(baseline_speeds),
        "candidate_maximum_speed": max(candidate_speeds),
        "baseline_minimum_population": min(baseline_populations),
        "candidate_minimum_population": min(candidate_populations),
        "latest_candidate_to_baseline_speed_ratio": latest[
            "candidate_to_baseline_speed_ratio"
        ],
        "instability_onset": {
            "speed_threshold_steps": speed_threshold_steps,
            "population_at_or_below_1e-8_step": {
                side: first_step(
                    side, "minimum_population", lambda value: value <= 1.0e-8,
                )
                for side in ("baseline", "candidate")
            },
            "collision_limiter_step": {
                side: first_step(
                    side,
                    "maximum_collision_limited_fraction",
                    lambda value: value > 0.0,
                )
                for side in ("baseline", "candidate")
            },
        },
        "aligned_steps": aligned,
    }


__all__ = ["compare_nested_health", "read_nested_health_log"]
