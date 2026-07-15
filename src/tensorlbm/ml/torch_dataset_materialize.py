"""Cold multi-snapshot adapter that delegates validated fields to the Torch materializer.

This module assembles immutable dataset references only.  It never trains a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from tensorlbm.data import FieldDatasetR2
from tensorlbm.ml.contracts import TaskKind, TrainingBackend, TrainingSpec, validate_training_spec
from tensorlbm.ml.torch_materialize import MaterializationProvenance, materialize_torch_velocity_snapshots


_SPLITS = ("train", "val", "test")


@dataclass(frozen=True, slots=True)
class FieldSnapshotReference:
    """One materialized velocity pair bound to its immutable source metadata."""

    sample_id: str
    group_id: str
    source_case_id: str
    source_trajectory_id: str
    snapshot: tuple[Any, Any]
    field_provenance: MaterializationProvenance


@dataclass(frozen=True, slots=True)
class DatasetMaterializationRecord:
    """Immutable cold-path result; it is not a training result or model artifact."""

    training_input_fingerprint: str
    sample_records: tuple[FieldSnapshotReference, ...]
    split_ids: Mapping[str, tuple[str, ...]]
    split_counts: Mapping[str, int]
    train: tuple[FieldSnapshotReference, ...]
    val: tuple[FieldSnapshotReference, ...]
    test: tuple[FieldSnapshotReference, ...]


def _freeze_payloads(
    dataset: FieldDatasetR2, payloads: Mapping[str, Mapping[str, bytes]]
) -> Mapping[str, Mapping[str, bytes]]:
    if not isinstance(payloads, Mapping):
        raise TypeError("payloads must be a mapping from sample_id to payload mappings")
    frozen: dict[str, Mapping[str, bytes]] = {}
    for sample_id, inner in payloads.items():
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("payload sample_id must be a non-empty string")
        if not isinstance(inner, Mapping):
            raise TypeError("each sample payload must be a mapping")
        copied: dict[str, bytes] = {}
        for array_id, payload in inner.items():
            if not isinstance(array_id, str) or not array_id.strip():
                raise ValueError("payload array_id must be a non-empty string")
            if not isinstance(payload, bytes):
                raise TypeError("payload values must be bytes")
            copied[array_id] = bytes(payload)
        frozen[sample_id] = MappingProxyType(copied)

    expected_samples = {sample.sample_id: sample for sample in dataset.samples}
    supplied = set(frozen)
    unknown = supplied - set(expected_samples)
    missing = set(expected_samples) - supplied
    if unknown:
        raise ValueError(f"payloads contains unknown sample_id values: {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"payloads is missing sample_id values: {', '.join(sorted(missing))}")
    return MappingProxyType(frozen)


def _validate_payload_array_keys(dataset: FieldDatasetR2, payloads: Mapping[str, Mapping[str, bytes]]) -> None:
    """Require every frozen inner payload map to match the post-freeze validated product exactly."""
    for sample in dataset.samples:
        expected_arrays = {array.array_id for array in sample.product.arrays}
        supplied_arrays = set(payloads[sample.sample_id])
        unknown_arrays = supplied_arrays - expected_arrays
        missing_arrays = expected_arrays - supplied_arrays
        if unknown_arrays:
            raise ValueError(
                f"payloads[{sample.sample_id!r}] contains unknown array_id values: {', '.join(sorted(unknown_arrays))}"
            )
        if missing_arrays:
            raise ValueError(
                f"payloads[{sample.sample_id!r}] is missing array_id values: {', '.join(sorted(missing_arrays))}"
            )


def _validate_compatibility(spec: TrainingSpec, dataset: FieldDatasetR2) -> None:
    if not isinstance(spec, TrainingSpec):
        raise TypeError("spec must be a TrainingSpec")
    if spec.backend is not TrainingBackend.TORCH:
        raise ValueError("dataset materialization requires the TORCH backend")
    if spec.task is not TaskKind.FIELD_RECONSTRUCTION:
        raise ValueError("dataset materialization requires FIELD_RECONSTRUCTION")
    validate_training_spec(spec)
    if spec.dataset.dataset_id != dataset.dataset_id:
        raise ValueError("TrainingSpec dataset id must match FieldDatasetR2")
    if spec.dataset.version != dataset.version:
        raise ValueError("TrainingSpec dataset version must match FieldDatasetR2")
    if spec.dataset.task_name != dataset.task_name:
        raise ValueError("TrainingSpec dataset task must match FieldDatasetR2")
    if dataset.task_name != "field reconstruction":
        raise ValueError("FieldDatasetR2 task must be field reconstruction")


def materialize_torch_field_dataset(
    spec: TrainingSpec,
    dataset: FieldDatasetR2,
    payloads: Mapping[str, Mapping[str, bytes]],
) -> DatasetMaterializationRecord:
    """Materialize every R2 sample through the existing CPU Torch field adapter.

    The complete immutable record is constructed only after every sample succeeds.
    """
    if not isinstance(dataset, FieldDatasetR2):
        raise TypeError("dataset must be a FieldDatasetR2")
    frozen_payloads = _freeze_payloads(dataset, payloads)
    dataset.validate_for_use()
    fingerprint = dataset.training_input_fingerprint()
    _validate_compatibility(spec, dataset)
    _validate_payload_array_keys(dataset, frozen_payloads)
    samples = tuple(dataset.samples)
    split_ids = MappingProxyType({split: tuple(dataset.splits[split]) for split in _SPLITS})

    records: list[FieldSnapshotReference] = []
    by_id: dict[str, FieldSnapshotReference] = {}
    for sample in samples:
        if dataset.training_input_fingerprint() != fingerprint:
            raise ValueError("dataset changed while processing")
        snapshots, provenance = materialize_torch_velocity_snapshots(
            spec, sample.product, frozen_payloads[sample.sample_id]
        )
        if len(snapshots) != 1:
            raise ValueError("single-product materializer must return exactly one snapshot")
        snapshot = snapshots[0]
        if len(snapshot) != 2:
            raise ValueError("single-product materializer must return a velocity pair")
        record = FieldSnapshotReference(
            sample.sample_id,
            sample.group_id,
            sample.source_case_id,
            sample.source_trajectory_id,
            snapshot,
            provenance,
        )
        records.append(record)
        by_id[sample.sample_id] = record

    if dataset.training_input_fingerprint() != fingerprint:
        raise ValueError("dataset changed while processing")
    split_counts = MappingProxyType({split: len(split_ids[split]) for split in _SPLITS})
    split_records = {split: tuple(by_id[sample_id] for sample_id in split_ids[split]) for split in _SPLITS}
    return DatasetMaterializationRecord(
        fingerprint,
        tuple(records),
        split_ids,
        split_counts,
        split_records["train"],
        split_records["val"],
        split_records["test"],
    )


__all__ = [
    "DatasetMaterializationRecord",
    "FieldSnapshotReference",
    "materialize_torch_field_dataset",
]
