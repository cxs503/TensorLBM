"""Fail-closed, hash-bound audit artifacts for segmented SUBOFF campaigns.

The artifact is derived from checkpoint evidence.  In particular, a manifest's
Ct block totals are accepted only when they equal accumulator differences across
adjacent checkpoint boundaries; they are never treated as primary evidence.
"""
from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any

import torch

_ARTIFACT_SCHEMA = "suboff-d3q27-campaign-audit-v1"
_MANIFEST_SCHEMA = "suboff-d3q27-segmented-run-v1"
_FORMAT = "suboff-d3q27-cumulant-xslab-v1"
_FLOAT_REL_TOL = 1.0e-12
_FLOAT_ABS_TOL = 1.0e-12
_REQUIRED = frozenset((
    "metadata", "step", "owned_populations", "target_mass", "mass_cadence",
    "friction_sum", "pressure_sum", "drag_samples",
))


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if result != value or result < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return result


def _finite(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _close(left: float | int, right: float | int) -> bool:
    """Versioned tolerance for scalar all-reduce/checkpoint roundoff."""
    return abs(float(left) - float(right)) <= _FLOAT_ABS_TOL + _FLOAT_REL_TOL * max(
        abs(float(left)), abs(float(right))
    )


def _root_file(root: Path, reference: object, rank: int) -> Path:
    if not isinstance(reference, str) or not reference:
        raise ValueError("checkpoint must be a non-empty relative path")
    relative = Path(reference)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("checkpoint must be a safe relative path")
    base = root.resolve()
    result = (base / relative / f"rank{rank:04d}.pt").resolve()
    if base not in result.parents:
        raise ValueError("checkpoint escapes artifact root")
    if not result.is_file():
        raise ValueError(f"missing checkpoint rank file: {result.name}")
    return result


def _load_checkpoint(path: Path, metadata: Mapping[str, Any], rank: int,
                     step: int) -> tuple[dict[str, object], dict[str, object]]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"unreadable checkpoint: {path.name}") from exc
    data = _mapping(payload, f"checkpoint {path.name}")
    missing = _REQUIRED - data.keys()
    if missing:
        raise ValueError(f"checkpoint missing fields: {sorted(missing)}")
    expected = dict(metadata)
    expected["rank"] = rank
    if data["metadata"] != expected:
        raise ValueError(f"checkpoint metadata mismatch: {path.name}")
    if _integer(data["step"], f"checkpoint {path.name} step") != step:
        raise ValueError(f"checkpoint step mismatch: {path.name}")
    populations = data["owned_populations"]
    if not isinstance(populations, torch.Tensor):
        raise ValueError(f"checkpoint populations missing: {path.name}")
    if not populations.is_floating_point() or not bool(torch.isfinite(populations).all().item()):
        raise ValueError(f"checkpoint populations must be floating and finite: {path.name}")
    shape = (_integer(metadata.get("q"), "metadata.q", 1),
             _integer(metadata.get("nz"), "metadata.nz", 1),
             _integer(metadata.get("ny"), "metadata.ny", 1),
             _integer(metadata.get("nx_local"), "metadata.nx_local", 1))
    if tuple(populations.shape) != shape:
        raise ValueError(f"checkpoint population shape mismatch: {path.name}")
    accumulator = {field: _finite(data[field], f"checkpoint {path.name} {field}")
                   for field in ("target_mass", "friction_sum", "pressure_sum")}
    accumulator["mass_cadence"] = _integer(data["mass_cadence"],
                                             f"checkpoint {path.name} mass_cadence", 1)
    accumulator["drag_samples"] = _integer(data["drag_samples"],
                                             f"checkpoint {path.name} drag_samples")
    return accumulator, {
        "path": str(path), "sha256": sha256(path.read_bytes()).hexdigest(),
        "step": step, "rank": rank, "metadata": expected,
        "population_shape": list(shape), "accumulator": accumulator,
    }


def _build(manifest: Mapping[str, object], root: str | Path) -> dict[str, object]:
    data = _mapping(manifest, "manifest")
    if data.get("schema") != _MANIFEST_SCHEMA:
        raise ValueError("unsupported segmented-run manifest schema")
    metadata = dict(_mapping(data.get("checkpoint_metadata"), "checkpoint_metadata"))
    if metadata.get("format") != _FORMAT or _integer(metadata.get("rank"), "metadata.rank") != 0:
        raise ValueError("unsupported canonical checkpoint metadata")
    world_size = _integer(metadata.get("world_size"), "metadata.world_size", 1)
    if _integer(metadata.get("nx"), "metadata.nx", 1) != world_size * _integer(metadata.get("nx_local"), "metadata.nx_local", 1):
        raise ValueError("metadata nx partition mismatch")
    ranks = data.get("expected_ranks")
    if ranks != list(range(world_size)):
        raise ValueError("expected_ranks must exactly cover world_size")
    segments = data.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("artifact requires at least one checkpoint segment")

    root_path = Path(root)
    boundaries: list[tuple[int, dict[int, dict[str, object]]]] = []
    records: list[dict[str, object]] = []
    previous_end = 0
    for index, raw in enumerate(segments):
        segment = _mapping(raw, f"segment {index}")
        start = _integer(segment.get("start_step"), f"segment {index} start_step")
        end = _integer(segment.get("end_step"), f"segment {index} end_step", 1)
        if start != previous_end or end <= start:
            raise ValueError("segments must be continuous and non-empty")
        accumulators: dict[int, dict[str, object]] = {}
        for rank in range(world_size):
            accumulator, record = _load_checkpoint(_root_file(root_path, segment.get("checkpoint"), rank), metadata, rank, end)
            accumulators[rank] = accumulator
            record["checkpoint"] = segment["checkpoint"]
            records.append(record)
        # These accumulators are the result of an all_reduce at every sample,
        # so each rank must retain the same global cumulative values.  The
        # checkpoint continuation controls are also global, rank-invariant
        # state: differing values make a resume chain non-auditable.
        for field in ("target_mass", "mass_cadence", "friction_sum", "pressure_sum", "drag_samples"):
            canonical = accumulators[0][field]
            if field in ("mass_cadence", "drag_samples"):
                mismatch = any(accumulators[rank][field] != canonical
                               for rank in range(1, world_size))
            else:
                mismatch = any(not _close(accumulators[rank][field], canonical)
                               for rank in range(1, world_size))
            if mismatch:
                if field in ("target_mass", "mass_cadence"):
                    raise ValueError(f"rank {field} continuation mismatch")
                raise ValueError(f"rank {field} accumulator mismatch")
        if boundaries:
            prior = boundaries[-1][1][0]
            for field in ("target_mass", "mass_cadence"):
                matches = (
                    accumulators[0][field] == prior[field]
                    if field == "mass_cadence"
                    else _close(accumulators[0][field], prior[field])
                )
                if not matches:
                    raise ValueError(f"{field} checkpoint-boundary continuation mismatch")
        boundaries.append((end, accumulators))
        previous_end = end

    blocks = data.get("blocks")
    if not isinstance(blocks, list):
        raise ValueError("manifest blocks must be a list")
    if not blocks:
        return {"schema": _ARTIFACT_SCHEMA, "manifest_schema": _MANIFEST_SCHEMA,
                "checkpoint_metadata": metadata,
                "expected_ranks": list(range(world_size)), "checkpoints": records,
                "blocks": [],
                "ct": {"status": "withheld", "reason": "no_post_gate_blocks"}}
    denominator = _finite(data.get("dynamic_pressure_wetted_area"),
                          "dynamic_pressure_wetted_area")
    if denominator <= 0.0:
        raise ValueError("dynamic_pressure_wetted_area must be finite and > 0")
    audited_blocks: list[dict[str, object]] = []
    previous_last = 0
    used_boundaries: set[tuple[int, int]] = set()
    for index, raw in enumerate(blocks):
        block = _mapping(raw, f"block {index}")
        first = _integer(block.get("first_sample_step"), f"block {index} first_sample_step", 1)
        last = _integer(block.get("last_sample_step"), f"block {index} last_sample_step", 1)
        if first <= previous_last:
            raise ValueError("blocks must be strictly ordered and non-overlapping")
        match = next(((boundaries[i - 1], boundaries[i]) for i in range(1, len(boundaries))
                      if first == boundaries[i - 1][0] + 1 and last == boundaries[i][0]), None)
        if match is None:
            raise ValueError("block must be exactly between adjacent checkpoint boundaries")
        before, after = match
        boundary_key = (before[0], after[0])
        if boundary_key in used_boundaries:
            raise ValueError("blocks must be strictly ordered and non-overlapping")
        used_boundaries.add(boundary_key)
        # Rank 0 is canonical only after the boundary-wide equality check
        # above; summing here would multiply an already-global all_reduce.
        expected = {
            field: (_finite(after[1][0][field], f"checkpoint {field}")
                    - _finite(before[1][0][field], f"checkpoint {field}"))
            for field in ("friction_sum", "pressure_sum")
        }
        expected["drag_samples"] = (
            _integer(after[1][0]["drag_samples"], "checkpoint drag_samples")
            - _integer(before[1][0]["drag_samples"], "checkpoint drag_samples")
        )
        sample_count = _integer(block.get("drag_samples"), f"block {index} drag_samples", 1)
        if sample_count != last - first + 1 or sample_count != expected["drag_samples"]:
            raise ValueError("drag_samples delta mismatch")
        for field in ("friction_sum", "pressure_sum"):
            if not _close(_finite(block.get(field), f"block {index} {field}"), expected[field]):
                raise ValueError(f"{field} delta mismatch")
        audited_blocks.append({"first_sample_step": first, "last_sample_step": last,
                               "friction_sum": expected["friction_sum"], "pressure_sum": expected["pressure_sum"],
                               "drag_samples": int(expected["drag_samples"])})
        previous_last = last
    total_samples = sum(block["drag_samples"] for block in audited_blocks)
    friction = sum(block["friction_sum"] for block in audited_blocks) / total_samples / denominator
    pressure = sum(block["pressure_sum"] for block in audited_blocks) / total_samples / denominator
    total = friction + pressure
    if not all(isfinite(value) for value in (friction, pressure, total)):
        raise ValueError("Ct must be finite")
    return {"schema": _ARTIFACT_SCHEMA, "manifest_schema": _MANIFEST_SCHEMA,
            "checkpoint_metadata": metadata, "expected_ranks": list(range(world_size)),
            "checkpoints": records, "blocks": audited_blocks,
            "ct": {"status": "computed", "physical_validation": "not_verified",
                   "dynamic_pressure_wetted_area": denominator,
                   "ct_friction": friction, "ct_pressure": pressure,
                   "ct_total": total}}


def build_suboff_campaign_audit_artifact(manifest: Mapping[str, object], *, root: str | Path) -> dict[str, object]:
    """Construct a fail-closed audit artifact from existing checkpoint evidence."""
    return _build(manifest, root)


def validate_suboff_campaign_audit_artifact(artifact: Mapping[str, object], manifest: Mapping[str, object], *, root: str | Path) -> bool:
    """Rebuild evidence and require byte-hash-bound artifact equality."""
    supplied = _mapping(artifact, "artifact")
    if supplied.get("schema") != _ARTIFACT_SCHEMA:
        raise ValueError("unsupported campaign audit artifact schema")
    supplied_records = supplied.get("checkpoints")
    if not isinstance(supplied_records, list):
        raise ValueError("artifact checkpoints must be a list")
    root_path = Path(root)
    for record in supplied_records:
        item = _mapping(record, "artifact checkpoint")
        path = item.get("path")
        if not isinstance(path, str):
            raise ValueError("artifact checkpoint path missing")
        resolved = Path(path).resolve()
        if root_path.resolve() not in resolved.parents or not resolved.is_file():
            raise ValueError("artifact checkpoint path invalid")
        if item.get("sha256") != sha256(resolved.read_bytes()).hexdigest():
            raise ValueError("checkpoint sha256 mismatch")
    expected = _build(manifest, root)
    if dict(supplied) != expected:
        raise ValueError("campaign audit artifact content mismatch")
    return True
