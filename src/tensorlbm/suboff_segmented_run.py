"""Fail-closed manifest evaluation for segmented D3Q27 SUBOFF campaigns.

The evaluator is reporting-only.  It does not launch a solver, modify a
checkpoint, or alter a boundary condition.  A Ct is produced only when a
complete checkpoint chain and wholly post-transient/post-warmup sample blocks
are present.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

import torch

_MANIFEST_SCHEMA = "suboff-d3q27-segmented-run-v1"
_CHECKPOINT_FORMAT = "suboff-d3q27-cumulant-xslab-v1"
_REQUIRED_CHECKPOINT_FIELDS = frozenset((
    "metadata", "step", "owned_populations", "target_mass", "mass_cadence",
    "friction_sum", "pressure_sum", "drag_samples",
))


@dataclass(frozen=True)
class SegmentedRunResult:
    """Accepted, traceable Ct reduction from independent post-warmup blocks."""

    accepted: bool
    completed_step: int
    included_blocks: int
    drag_samples: int
    ct_friction: float
    ct_pressure: float
    ct_total: float


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    # bool is technically an int but never a legitimate campaign coordinate.
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if result != value or result < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return result


def _finite_positive(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite and positive") from exc
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _finite(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _checkpoint_directory(root: Path, reference: object) -> Path:
    if not isinstance(reference, str) or not reference:
        raise ValueError("segment checkpoint must be a non-empty relative path")
    candidate = Path(reference)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("segment checkpoint must be a safe relative path")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    if resolved_root not in resolved.parents and resolved != resolved_root:
        raise ValueError("segment checkpoint escapes manifest root")
    if not resolved.is_dir():
        raise ValueError(f"missing checkpoint directory: {reference}")
    return resolved


def _validate_checkpoint(directory: Path, *, expected_step: int,
                         campaign_metadata: Mapping[str, Any]) -> None:
    world_size = _integer(campaign_metadata.get("world_size"), "checkpoint_metadata.world_size", minimum=1)
    if campaign_metadata.get("format") != _CHECKPOINT_FORMAT:
        raise ValueError("unsupported checkpoint metadata format")
    for rank in range(world_size):
        checkpoint_file = directory / f"rank{rank:04d}.pt"
        if not checkpoint_file.is_file():
            raise ValueError(f"missing checkpoint rank file: {checkpoint_file.name}")
        try:
            payload = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
        except Exception as exc:  # a corrupt artifact is not evidence for Ct
            raise ValueError(f"unreadable checkpoint: {checkpoint_file.name}") from exc
        payload_map = _mapping(payload, f"checkpoint {checkpoint_file.name}")
        missing = _REQUIRED_CHECKPOINT_FIELDS - payload_map.keys()
        if missing:
            raise ValueError(f"checkpoint continuation metadata missing: {sorted(missing)}")
        if payload_map["metadata"] != dict(campaign_metadata):
            raise ValueError(f"checkpoint metadata mismatch: {checkpoint_file.name}")
        if _integer(payload_map["step"], f"checkpoint {checkpoint_file.name} step") != expected_step:
            raise ValueError(f"checkpoint step mismatch: {checkpoint_file.name}")
        if not isinstance(payload_map["owned_populations"], torch.Tensor):
            raise ValueError(f"checkpoint populations missing: {checkpoint_file.name}")
        populations = payload_map["owned_populations"]
        nx_local = _integer(campaign_metadata.get("nx_local"), "checkpoint_metadata.nx_local", minimum=1)
        ny = _integer(campaign_metadata.get("ny"), "checkpoint_metadata.ny", minimum=1)
        nz = _integer(campaign_metadata.get("nz"), "checkpoint_metadata.nz", minimum=1)
        q = _integer(campaign_metadata.get("q"), "checkpoint_metadata.q", minimum=1)
        if tuple(populations.shape) != (q, nz, ny, nx_local):
            raise ValueError(f"checkpoint population shape mismatch: {checkpoint_file.name}")
        for field in ("target_mass", "friction_sum", "pressure_sum"):
            _finite(payload_map[field], f"checkpoint {checkpoint_file.name} {field}")
        _integer(payload_map["mass_cadence"], f"checkpoint {checkpoint_file.name} mass_cadence", minimum=1)
        _integer(payload_map["drag_samples"], f"checkpoint {checkpoint_file.name} drag_samples")


def evaluate_suboff_segmented_run(manifest: Mapping[str, object], *, root: str | Path) -> SegmentedRunResult:
    """Validate an interrupted-run campaign and reduce its eligible Ct blocks.

    ``segments`` must make one gap-free chain beginning at step zero. Every
    boundary has a complete rank checkpoint with exact static metadata.  Ct
    blocks must be entirely after both the required outlet-convection transient
    and configured warmup; intersecting a gate is rejected rather than clipped.
    """
    data = _mapping(manifest, "manifest")
    if data.get("schema") != _MANIFEST_SCHEMA:
        raise ValueError("unsupported segmented-run manifest schema")
    campaign_metadata = _mapping(data.get("checkpoint_metadata"), "checkpoint_metadata")
    far_field = _mapping(data.get("far_field"), "far_field")
    required_transient = _integer(far_field.get("required_transient_steps"), "far_field.required_transient_steps")
    if far_field.get("transient_steps_satisfy_outlet_convection") is not True:
        raise ValueError("outlet convection requirement is not satisfied")
    transient = _integer(data.get("transient_steps"), "transient_steps")
    if transient < required_transient:
        raise ValueError("required transient/outlet convection is incomplete")
    warmup = _integer(data.get("warmup_steps"), "warmup_steps")
    sample_gate = max(transient, warmup)
    denominator = _finite_positive(data.get("dynamic_pressure_wetted_area"), "dynamic_pressure_wetted_area")

    segments = data.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("manifest requires at least one segment")
    previous_end = 0
    root_path = Path(root)
    for index, raw_segment in enumerate(segments):
        segment = _mapping(raw_segment, f"segment {index}")
        start = _integer(segment.get("start_step"), f"segment {index} start_step")
        end = _integer(segment.get("end_step"), f"segment {index} end_step", minimum=1)
        if start != previous_end or end <= start:
            raise ValueError("segment step ranges must be continuous, ordered, and non-empty")
        _validate_checkpoint(_checkpoint_directory(root_path, segment.get("checkpoint")),
                             expected_step=end, campaign_metadata=campaign_metadata)
        previous_end = end

    blocks = data.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("Ct requires at least one post-warmup block")
    friction_sum = pressure_sum = 0.0
    drag_samples = 0
    previous_block_last = sample_gate
    for index, raw_block in enumerate(blocks):
        block = _mapping(raw_block, f"block {index}")
        first = _integer(block.get("first_sample_step"), f"block {index} first_sample_step", minimum=1)
        last = _integer(block.get("last_sample_step"), f"block {index} last_sample_step", minimum=1)
        if last < first or last > previous_end:
            raise ValueError("Ct block sample range is outside the completed campaign")
        if first <= sample_gate:
            raise ValueError("Ct block starts before warmup/transient gate; only wholly post-warmup blocks are allowed")
        if first <= previous_block_last:
            raise ValueError("Ct blocks must be ordered and non-overlapping")
        count = _integer(block.get("drag_samples"), f"block {index} drag_samples", minimum=1)
        if count != last - first + 1:
            raise ValueError("Ct block drag_samples must equal its inclusive sample range")
        friction_sum += _finite(block.get("friction_sum"), f"block {index} friction_sum")
        pressure_sum += _finite(block.get("pressure_sum"), f"block {index} pressure_sum")
        drag_samples += count
        previous_block_last = last

    if drag_samples == 0:  # retained as a defensive fail-closed invariant
        raise ValueError("Ct requires post-warmup drag samples")
    ct_friction = friction_sum / drag_samples / denominator
    ct_pressure = pressure_sum / drag_samples / denominator
    return SegmentedRunResult(True, previous_end, len(blocks), drag_samples,
                              ct_friction, ct_pressure, ct_friction + ct_pressure)
