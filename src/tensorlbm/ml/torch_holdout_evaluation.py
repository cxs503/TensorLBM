"""Data-only, evidence-gated evaluation for field-dataset holdout snapshots.

This boundary materializes the complete dataset to validate every supplied payload,
then summarizes only the selected validation or test snapshot references.  It has
no executable-model, parameter, or persistence interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite
from types import MappingProxyType
from typing import Any, Mapping

from tensorlbm.data import FieldDatasetR2
from tensorlbm.ml.contracts import TaskKind, TrainingBackend, TrainingSpec
from tensorlbm.ml.torch_dataset_materialize import FieldSnapshotReference, materialize_torch_field_dataset


_ALLOWED_SPLITS = ("val", "test")
_COMPONENT_NAMES = ("u_x", "u_y")


@dataclass(frozen=True, slots=True)
class HoldoutSampleEvidence:
    """Immutable selected-snapshot identity and validated field reference."""

    sample_id: str
    group_id: str
    source_case_id: str
    source_trajectory_id: str
    field_blob_hash: str
    grid_shape: tuple[int, int]


@dataclass(frozen=True, slots=True)
class HoldoutEvaluationRecord:
    """Data-quality evidence only; the fields are explicitly not model metrics."""

    status: str
    data_only: bool
    not_model_evaluation: bool
    split: str
    dataset_fingerprint: str
    sample_count: int
    sample_ids: tuple[str, ...]
    group_ids: Mapping[str, str]
    source_case_ids: Mapping[str, str]
    source_trajectory_ids: Mapping[str, str]
    field_blob_hashes: Mapping[str, str]
    samples: tuple[HoldoutSampleEvidence, ...]
    grid_shapes: tuple[tuple[int, int], ...]
    component_finite_counts: Mapping[str, int]
    component_nonfinite_counts: Mapping[str, int]
    component_min: Mapping[str, float | None]
    component_max: Mapping[str, float | None]
    component_mean: Mapping[str, float | None]
    component_abs_mean: Mapping[str, float | None]


def _require_compatible_spec(spec: TrainingSpec) -> None:
    if not isinstance(spec, TrainingSpec):
        raise TypeError("spec must be a TrainingSpec")
    if spec.backend is not TrainingBackend.TORCH:
        raise ValueError("holdout data evaluation requires the TORCH backend")
    if spec.task is not TaskKind.FIELD_RECONSTRUCTION:
        raise ValueError("holdout data evaluation requires FIELD_RECONSTRUCTION")


def _finite_summary(values: Any) -> tuple[int, int, float | None, float | None, float | None, float | None]:
    """Summarize a CPU adapter value through its basic Python representation only."""
    flattened: list[float] = []

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        flattened.append(float(value))

    visit(values.tolist())
    finite = [value for value in flattened if isfinite(value)]
    count = len(finite)
    nonfinite = len(flattened) - count
    if not finite:
        return count, nonfinite, None, None, None, None
    return count, nonfinite, min(finite), max(finite), fsum(finite) / count, fsum(abs(value) for value in finite) / count


def _selected_evidence(reference: FieldSnapshotReference) -> HoldoutSampleEvidence:
    field = reference.field_provenance
    if field.component_labels != _COMPONENT_NAMES:
        raise ValueError("selected snapshot component labels are invalid")
    if len(field.shape) != 3 or field.shape[2] != len(_COMPONENT_NAMES):
        raise ValueError("selected snapshot field shape is invalid")
    return HoldoutSampleEvidence(
        sample_id=reference.sample_id,
        group_id=reference.group_id,
        source_case_id=reference.source_case_id,
        source_trajectory_id=reference.source_trajectory_id,
        field_blob_hash=field.blob_sha256,
        grid_shape=(field.shape[0], field.shape[1]),
    )


def evaluate_evidence_gated_holdout(
    spec: TrainingSpec,
    dataset: FieldDatasetR2,
    payloads: Mapping[str, Mapping[str, bytes]],
    split: str = "test",
) -> HoldoutEvaluationRecord:
    """Return selected val/test field-quality summaries after full evidence materialization."""
    if split not in _ALLOWED_SPLITS:
        raise ValueError("split must be val or test; train is never eligible for holdout data evaluation")
    _require_compatible_spec(spec)
    if not isinstance(dataset, FieldDatasetR2):
        raise TypeError("dataset must be a FieldDatasetR2")

    materialized = materialize_torch_field_dataset(spec, dataset, payloads)
    if dataset.training_input_fingerprint() != materialized.training_input_fingerprint:
        raise ValueError("dataset changed while processing")
    selected = getattr(materialized, split)
    if not selected:
        raise ValueError(f"holdout split {split!r} must be non-empty")

    summaries = {name: [] for name in _COMPONENT_NAMES}
    evidence: list[HoldoutSampleEvidence] = []
    for reference in selected:
        if len(reference.snapshot) != len(_COMPONENT_NAMES):
            raise ValueError("selected snapshot must contain exactly two velocity components")
        evidence.append(_selected_evidence(reference))
        for name, component in zip(_COMPONENT_NAMES, reference.snapshot, strict=True):
            summaries[name].append(_finite_summary(component))

    finite_counts: dict[str, int] = {}
    nonfinite_counts: dict[str, int] = {}
    minimums: dict[str, float | None] = {}
    maximums: dict[str, float | None] = {}
    means: dict[str, float | None] = {}
    abs_means: dict[str, float | None] = {}
    for name, parts in summaries.items():
        finite_count = sum(part[0] for part in parts)
        nonfinite_count = sum(part[1] for part in parts)
        finite_mins = [part[2] for part in parts if part[2] is not None]
        finite_maxs = [part[3] for part in parts if part[3] is not None]
        weighted_means = [part[4] * part[0] for part in parts if part[4] is not None]
        weighted_abs_means = [part[5] * part[0] for part in parts if part[5] is not None]
        finite_counts[name] = finite_count
        nonfinite_counts[name] = nonfinite_count
        minimums[name] = min(finite_mins) if finite_mins else None
        maximums[name] = max(finite_maxs) if finite_maxs else None
        means[name] = fsum(weighted_means) / finite_count if finite_count else None
        abs_means[name] = fsum(weighted_abs_means) / finite_count if finite_count else None

    return HoldoutEvaluationRecord(
        status="holdout data evaluation completed",
        data_only=True,
        not_model_evaluation=True,
        split=split,
        dataset_fingerprint=materialized.training_input_fingerprint,
        sample_count=len(selected),
        sample_ids=tuple(reference.sample_id for reference in selected),
        group_ids=MappingProxyType({item.sample_id: item.group_id for item in evidence}),
        source_case_ids=MappingProxyType({item.sample_id: item.source_case_id for item in evidence}),
        source_trajectory_ids=MappingProxyType({item.sample_id: item.source_trajectory_id for item in evidence}),
        field_blob_hashes=MappingProxyType({item.sample_id: item.field_blob_hash for item in evidence}),
        samples=tuple(evidence),
        grid_shapes=tuple(item.grid_shape for item in evidence),
        component_finite_counts=MappingProxyType(finite_counts),
        component_nonfinite_counts=MappingProxyType(nonfinite_counts),
        component_min=MappingProxyType(minimums),
        component_max=MappingProxyType(maximums),
        component_mean=MappingProxyType(means),
        component_abs_mean=MappingProxyType(abs_means),
    )


__all__ = ["HoldoutEvaluationRecord", "HoldoutSampleEvidence", "evaluate_evidence_gated_holdout"]
