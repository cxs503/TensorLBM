"""Fail-closed planning manifests for segmented D3Q27 SUBOFF campaigns.

This module plans a campaign only: it neither starts ``torchrun`` nor writes a
checkpoint.  The resulting manifest has the schema consumed by
:mod:`tensorlbm.suboff_segmented_run`; its Ct block sums are deliberately
``None`` until a completed campaign records real measurements.
"""
from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from pathlib import Path
from typing import Any


_MANIFEST_SCHEMA = "suboff-d3q27-segmented-run-v1"
_CHECKPOINT_FORMAT = "suboff-d3q27-cumulant-xslab-v1"
_REQUIRED_METADATA = frozenset((
    "format", "nx", "ny", "nz", "hull_length", "re", "u_in", "y_val",
    "world_size", "rank", "nx_local", "q",
))


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer >= {minimum}")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer >= {minimum}") from exc
    if result != value or result < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return result


def _positive_finite(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite and positive") from exc
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _checkpoint_template(value: object) -> str:
    if not isinstance(value, str) or not value or "{end_step}" not in value:
        raise ValueError("checkpoint_path_template must be a safe relative path containing {end_step}")
    try:
        rendered_zero = value.format(end_step=0)
        rendered_one = value.format(end_step=1)
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError("checkpoint_path_template must contain only a valid {end_step} placeholder") from exc
    for rendered in (rendered_zero, rendered_one):
        path = Path(rendered)
        if path.is_absolute() or ".." in path.parts or path == Path("."):
            raise ValueError("checkpoint_path_template must be a safe relative path")
    if rendered_zero == rendered_one:
        raise ValueError("checkpoint_path_template must produce a distinct path per end_step")
    return value


def _validated_metadata(value: object) -> tuple[dict[str, object], int]:
    metadata = dict(_mapping(value, "checkpoint_metadata"))
    missing = _REQUIRED_METADATA - metadata.keys()
    if missing:
        raise ValueError(f"checkpoint_metadata missing required fields: {sorted(missing)}")
    if metadata["format"] != _CHECKPOINT_FORMAT:
        raise ValueError("unsupported checkpoint metadata format")
    world_size = _integer(metadata["world_size"], "checkpoint_metadata.world_size", minimum=1)
    rank = _integer(metadata["rank"], "checkpoint_metadata.rank")
    if rank != 0:
        # The current evaluator's campaign metadata is the rank-zero canonical
        # record and checks all rank artifacts against it.
        raise ValueError("checkpoint_metadata.rank must be 0 for the evaluator-compatible campaign manifest")
    if rank >= world_size:
        raise ValueError("checkpoint_metadata.rank must be less than world_size")
    nx = _integer(metadata["nx"], "checkpoint_metadata.nx", minimum=1)
    nx_local = _integer(metadata["nx_local"], "checkpoint_metadata.nx_local", minimum=1)
    if nx_local * world_size != nx:
        raise ValueError("checkpoint_metadata nx_local/world_size must exactly partition nx")
    for field in ("ny", "nz", "q"):
        _integer(metadata[field], f"checkpoint_metadata.{field}", minimum=1)
    for field in ("hull_length", "re", "u_in", "y_val"):
        _positive_finite(metadata[field], f"checkpoint_metadata.{field}")
    return metadata, world_size


def build_suboff_segmented_campaign_manifest(
    static_config: Mapping[str, object], *, segment_steps: int, segment_count: int,
) -> dict[str, object]:
    """Build an evaluator-compatible, non-executing segmented campaign plan.

    A Ct collection start explicitly requested by a caller must occur *after*
    both the configured warmup and the outlet-convection transient.  The
    planner never silently advances an unsafe requested Ct start; instead it
    rejects it.  With no explicit request, sampling begins exactly at the first
    safe step and segments are split into wholly eligible blocks.
    """
    config = _mapping(static_config, "static_config")
    metadata, world_size = _validated_metadata(config.get("checkpoint_metadata"))
    far_field = _mapping(config.get("far_field"), "far_field")
    required_transient = _integer(
        far_field.get("required_transient_steps"), "far_field.required_transient_steps"
    )
    if far_field.get("transient_steps_satisfy_outlet_convection") is not True:
        raise ValueError("outlet convection requirement is not satisfied")
    transient = _integer(config.get("transient_steps"), "transient_steps")
    if transient < required_transient:
        raise ValueError("outlet convection transient is incomplete")
    warmup = _integer(config.get("warmup_steps"), "warmup_steps")
    denominator = _positive_finite(
        config.get("dynamic_pressure_wetted_area"), "dynamic_pressure_wetted_area"
    )
    template = _checkpoint_template(config.get("checkpoint_path_template"))
    length = _integer(segment_steps, "segment_steps", minimum=1)
    count = _integer(segment_count, "segment_count", minimum=1)

    sample_gate = max(transient, warmup)
    requested_ct_start = config.get("ct_start_step")
    if requested_ct_start is None:
        ct_start = sample_gate + 1
    else:
        ct_start = _integer(requested_ct_start, "ct_start_step", minimum=1)
        if ct_start <= transient:
            raise ValueError("Ct cannot start before outlet convection is complete")
        if ct_start <= warmup:
            raise ValueError("Ct cannot start before warmup is complete")

    segments: list[dict[str, object]] = []
    blocks: list[dict[str, object]] = []
    for index in range(count):
        start = index * length
        end = start + length
        checkpoint = template.format(end_step=end)
        segments.append({"start_step": start, "end_step": end, "checkpoint": checkpoint})
        first = max(start + 1, ct_start)
        if first <= end:
            blocks.append({
                "first_sample_step": first,
                "last_sample_step": end,
                "friction_sum": None,
                "pressure_sum": None,
                "drag_samples": end - first + 1,
            })
    if not blocks:
        raise ValueError("campaign has no wholly post-gate Ct samples")

    return {
        "schema": _MANIFEST_SCHEMA,
        "planning_only": True,
        "planned_completed_step": segments[-1]["end_step"],
        "checkpoint_metadata": metadata,
        "expected_ranks": list(range(world_size)),
        "far_field": dict(far_field),
        "transient_steps": transient,
        "warmup_steps": warmup,
        "dynamic_pressure_wetted_area": denominator,
        "segments": segments,
        "blocks": blocks,
    }
