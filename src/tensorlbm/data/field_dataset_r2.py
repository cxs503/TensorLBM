"""Cold metadata catalogue for leakage-safe multi-field input selection."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Any, Mapping

from tensorlbm.runtime import RunManifest
from tensorlbm.data.field_r2 import (
    ArrayEncoding,
    ArrayManifestR2,
    AxisSpec,
    BlobRef,
    FieldDataProductR2,
)


def _text(value: object, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _freeze(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        raise TypeError("lineage must not contain payload bytes")
    if isinstance(value, Mapping):
        return MappingProxyType({_text(key, "lineage key"): _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    raise TypeError(f"unsupported lineage value: {type(value).__name__}")


def _frozen_lineage(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("lineage must be a mapping")
    frozen = _freeze(value)
    assert isinstance(frozen, Mapping)
    return frozen

def _validate_field_product_current(product: FieldDataProductR2) -> None:
    """Recursively reconstruct nested field metadata after hostile mutation."""
    arrays: list[ArrayManifestR2] = []
    for array in product.arrays:
        if not isinstance(array, ArrayManifestR2):
            raise TypeError("product arrays must be ArrayManifestR2")
        axes = tuple(AxisSpec(axis.name, axis.semantic, axis.length) for axis in array.axes)
        encoding = ArrayEncoding(
            array.encoding.format,
            array.encoding.dtype,
            array.encoding.order,
            array.encoding.byte_order,
        )
        blob = BlobRef(
            array.blob_ref.blob_id,
            array.blob_ref.uri,
            array.blob_ref.byte_size,
            array.blob_ref.sha256,
            array.blob_ref.media_type,
        )
        arrays.append(
            ArrayManifestR2(
                array.array_id,
                array.role,
                array.shape,
                axes,
                array.units,
                encoding,
                blob,
                array.component_labels,
            )
        )
    FieldDataProductR2(
        product.product_id,
        product.run_manifest,
        product.source_artifact_id,
        tuple(arrays),
        product.lineage,
    )

@dataclass(frozen=True, slots=True)
class FieldSampleRefR2:
    """A group-labelled reference to a PASS-gated field product, never field bytes."""

    sample_id: str
    product: FieldDataProductR2
    group_id: str
    source_case_id: str
    source_trajectory_id: str

    def __post_init__(self) -> None:
        _text(self.sample_id, "sample_id")
        _text(self.group_id, "group_id")
        _text(self.source_case_id, "source_case_id")
        _text(self.source_trajectory_id, "source_trajectory_id")
        if not isinstance(self.product, FieldDataProductR2):
            raise TypeError("product must be a FieldDataProductR2")
        self._validate_product()

    def _validate_product(self) -> None:
        """Rebuild product structure and runtime PASS gate without materializing blobs."""
        _validate_field_product_current(self.product)

    def validate_for_use(self) -> None:
        """Fail closed after hostile mutation of a ref, product, run, or blob metadata."""
        FieldSampleRefR2(
            self.sample_id,
            self.product,
            self.group_id,
            self.source_case_id,
            self.source_trajectory_id,
        )


def _validate_splits(samples: tuple[FieldSampleRefR2, ...], splits: object) -> Mapping[str, tuple[str, ...]]:
    if not isinstance(splits, Mapping):
        raise TypeError("splits must be a mapping")
    expected = {"train", "val", "test"}
    if set(splits) != expected:
        raise ValueError("splits must contain exactly train, val, and test")
    sample_ids = tuple(sample.sample_id for sample in samples)
    known = set(sample_ids)
    assigned: set[str] = set()
    membership: dict[str, str] = {}
    frozen: dict[str, tuple[str, ...]] = {}
    for split in ("train", "val", "test"):
        members = splits[split]
        if not isinstance(members, tuple):
            raise TypeError(f"splits[{split!r}] must be a tuple")
        for sample_id in members:
            _text(sample_id, f"splits[{split!r}] member")
            if sample_id not in known:
                raise ValueError(f"splits contains unknown sample_id: {sample_id}")
            if sample_id in assigned:
                raise ValueError("splits must not overlap")
            assigned.add(sample_id)
            membership[sample_id] = split
        frozen[split] = tuple(members)
    if not frozen["train"]:
        raise ValueError("splits train must be non-empty")
    if assigned != known:
        raise ValueError("splits must assign every sample exactly once")
    for field in ("group_id", "source_case_id", "source_trajectory_id"):
        seen: dict[str, str] = {}
        for sample in samples:
            value = getattr(sample, field)
            split = membership[sample.sample_id]
            previous = seen.setdefault(value, split)
            if previous != split:
                raise ValueError(f"{field} must not cross splits")
    return MappingProxyType(frozen)

def _canonical_value(value: object) -> object:
    """Encode every immutable runtime/data metadata leaf deterministically without embedding bytes."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"$bytes_sha256": sha256(value).hexdigest(), "$bytes_size": len(value)}
    if isinstance(value, Mapping):
        canonical: dict[str, object] = {}
        for key, item in value.items():
            canonical[_text(key, "canonical mapping key")] = _canonical_value(item)
        return canonical
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        canonical_items = [_canonical_value(item) for item in value]
        return sorted(
            canonical_items,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
        )
    raise TypeError(f"not canonical metadata: {type(value).__name__}")

def _canonical_run(run: RunManifest) -> dict[str, object]:
    return {
        "run_id": run.run_id,
        "model_identity": run.model_identity,
        "config": run.config,
        "code_sha": run.code_sha,
        "environment": run.environment,
        "artifacts": [
            {
                "artifact_id": artifact.artifact_id,
                "media_type": artifact.media_type,
                "sha256": artifact.sha256,
                "size_bytes": artifact.size_bytes,
                "metadata": artifact.metadata,
            }
            for artifact in run.artifacts
        ],
        "metrics": [
            {
                "metric_id": metric.metric_id,
                "value": metric.value,
                "unit": metric.unit,
                "artifact_id": metric.artifact_id,
                "evidence_pointer": metric.evidence_pointer,
            }
            for metric in run.metrics
        ],
        "validation_status": run.validation_status.value,
        "validation_reason": run.validation_reason,
    }


def _canonical_encoding(array: ArrayManifestR2) -> dict[str, str]:
    return {
        "format": array.encoding.format,
        "dtype": array.encoding.dtype,
        "order": array.encoding.order.value,
        "byte_order": array.encoding.byte_order.value,
    }

@dataclass(frozen=True, slots=True)
class FieldDatasetR2:
    """Pure multi-snapshot catalogue contract; it does not define training or outputs."""

    dataset_id: str
    version: str
    task_name: str
    samples: tuple[FieldSampleRefR2, ...]
    splits: Mapping[str, tuple[str, ...]]
    lineage: Mapping[str, Any]

    def __post_init__(self) -> None:
        _text(self.dataset_id, "dataset_id")
        _text(self.version, "version")
        _text(self.task_name, "task_name")
        samples = tuple(self.samples)
        if not samples or not all(isinstance(sample, FieldSampleRefR2) for sample in samples):
            raise ValueError("samples must be non-empty FieldSampleRefR2 values")
        sample_ids = tuple(sample.sample_id for sample in samples)
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("samples must have unique sample_id values")
        for sample in samples:
            sample.validate_for_use()
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "splits", _validate_splits(samples, self.splits))
        object.__setattr__(self, "lineage", _frozen_lineage(self.lineage))

    def validate_for_use(self) -> None:
        """Rebuild every reference, product/run gate, split, and lineage invariant."""
        FieldDatasetR2(
            self.dataset_id,
            self.version,
            self.task_name,
            self.samples,
            self.splits,
            self.lineage,
        )

    def training_input_fingerprint(self) -> str:
        """Hash canonical input references; this does not materialize any field payload."""
        self.validate_for_use()
        document = {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "task_name": self.task_name,
            "lineage": self.lineage,
            "samples": [
                {
                    "sample_id": sample.sample_id,
                    "group_id": sample.group_id,
                    "source_case_id": sample.source_case_id,
                    "source_trajectory_id": sample.source_trajectory_id,
                    "product": {
                        "product_id": sample.product.product_id,
                        "source_artifact_id": sample.product.source_artifact_id,
                        "lineage": sample.product.lineage,
                        "run": _canonical_run(sample.product.run_manifest),
                        "arrays": [
                            {
                                "array_id": array.array_id,
                                "role": array.role.value,
                                "shape": array.shape,
                                "units": array.units,
                                "axes": [
                                    {"name": axis.name, "semantic": axis.semantic.value, "length": axis.length}
                                    for axis in array.axes
                                ],
                                "component_labels": array.component_labels,
                                "encoding": _canonical_encoding(array),
                                "blob": {
                                    "blob_id": array.blob_ref.blob_id,
                                    "sha256": array.blob_ref.sha256,
                                    "byte_size": array.blob_ref.byte_size,
                                    "media_type": array.blob_ref.media_type,
                                    "uri": array.blob_ref.uri,
                                },
                            }
                            for array in sample.product.arrays
                        ],
                    },
                }
                for sample in sorted(self.samples, key=lambda item: item.sample_id)
            ],
            "splits": {split: self.splits[split] for split in ("train", "val", "test")},
        }
        encoded = json.dumps(_canonical_value(document), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        return sha256(encoded).hexdigest()


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"not canonical metadata: {type(value).__name__}")


__all__ = ["FieldDatasetR2", "FieldSampleRefR2"]
