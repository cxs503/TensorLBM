"""Immutable cold-path data contracts; this module only validates metadata."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from tensorlbm.runtime import RunManifest, ValidationStatus, validate_run_manifest


def _require_text(value: object, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _freeze(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool, bytes)):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType({_require_text(key, "mapping key"): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    raise TypeError(f"unsupported lineage value: {type(value).__name__}")


def _immutable_mapping(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    frozen = _freeze(value)
    assert isinstance(frozen, Mapping)
    return frozen


def _positive_shape(value: object) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value:
        raise ValueError("shape must be a non-empty tuple")
    if any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in value):
        raise ValueError("shape must contain positive non-boolean integers")
    return value


@dataclass(frozen=True, slots=True)
class FieldProduct:
    """One field artifact, gated only by a revalidated completed runtime manifest."""

    product_id: str
    run_manifest: RunManifest
    artifact_id: str
    field_name: str
    shape: tuple[int, ...]
    dtype: str
    units: str
    quality_status: ValidationStatus
    lineage: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_text(self.product_id, "product_id")
        if not isinstance(self.run_manifest, RunManifest):
            raise TypeError("run_manifest must be a RunManifest")
        try:
            validate_run_manifest(self.run_manifest)
        except (TypeError, ValueError) as error:
            raise ValueError("run manifest must pass runtime evidence validation") from error
        _require_text(self.artifact_id, "artifact_id")
        _require_text(self.field_name, "field_name")
        _positive_shape(self.shape)
        _require_text(self.dtype, "dtype")
        _require_text(self.units, "units")
        if not isinstance(self.quality_status, ValidationStatus):
            raise TypeError("quality_status must be a ValidationStatus")
        if self.quality_status is not self.run_manifest.validation_status:
            raise ValueError("quality_status must equal run_manifest.validation_status")
        if self.artifact_id not in {artifact.artifact_id for artifact in self.run_manifest.artifacts}:
            raise ValueError("artifact_id must reference an artifact in run_manifest")
        object.__setattr__(self, "lineage", _immutable_mapping(self.lineage, "lineage"))

    @property
    def is_training_eligible(self) -> bool:
        """PASS is only the declared runtime-validation gate, not a physical-truth claim."""
        return self.quality_status is ValidationStatus.PASS

    def require_current_training_eligibility(self) -> None:
        """Revalidate evidence at dataset-use time; never trust construction alone."""
        try:
            validate_run_manifest(self.run_manifest)
        except (TypeError, ValueError) as error:
            raise ValueError(f"product {self.product_id} has invalid runtime evidence") from error
        if self.quality_status is not self.run_manifest.validation_status:
            raise ValueError(f"product {self.product_id} quality status no longer matches runtime validation")
        if self.quality_status is not ValidationStatus.PASS:
            raise ValueError(f"product {self.product_id} is not PASS-gated")
        if self.artifact_id not in {artifact.artifact_id for artifact in self.run_manifest.artifacts}:
            raise ValueError(f"product {self.product_id} artifact binding is no longer valid")


@dataclass(frozen=True, slots=True)
class DatasetSampleRef:
    """Traceable locator for a product field; it contains no field values."""

    product_id: str
    run_id: str
    artifact_id: str
    field_name: str
    shape: tuple[int, ...]
    dtype: str
    units: str

    def __post_init__(self) -> None:
        _require_text(self.product_id, "product_id")
        _require_text(self.run_id, "run_id")
        _require_text(self.artifact_id, "artifact_id")
        _require_text(self.field_name, "field_name")
        _positive_shape(self.shape)
        _require_text(self.dtype, "dtype")
        _require_text(self.units, "units")

    @classmethod
    def from_product(cls, product: FieldProduct) -> DatasetSampleRef:
        if not isinstance(product, FieldProduct):
            raise TypeError("product must be a FieldProduct")
        return cls(
            product_id=product.product_id,
            run_id=product.run_manifest.run_id,
            artifact_id=product.artifact_id,
            field_name=product.field_name,
            shape=product.shape,
            dtype=product.dtype,
            units=product.units,
        )


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Immutable split manifest; readiness rejects every non-eligible product."""

    dataset_id: str
    version: str
    products: tuple[FieldProduct, ...]
    task_name: str
    group_splits: Mapping[str, tuple[str, ...]]
    lineage: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_text(self.dataset_id, "dataset_id")
        _require_text(self.version, "version")
        _require_text(self.task_name, "task_name")
        products = tuple(self.products)
        if not products:
            raise ValueError("products must be non-empty")
        if not all(isinstance(product, FieldProduct) for product in products):
            raise TypeError("products must contain FieldProduct values")
        product_ids = tuple(product.product_id for product in products)
        if len(set(product_ids)) != len(product_ids):
            raise ValueError("products must have unique product_id values")
        if not isinstance(self.group_splits, Mapping):
            raise TypeError("group_splits must be a mapping")
        expected_groups = {"train", "val", "test"}
        if set(self.group_splits) != expected_groups:
            raise ValueError("group_splits must contain exactly train, val, and test")
        frozen_splits: dict[str, tuple[str, ...]] = {}
        assigned: set[str] = set()
        known_products = set(product_ids)
        for group in ("train", "val", "test"):
            members = self.group_splits[group]
            if not isinstance(members, tuple):
                raise TypeError(f"group_splits[{group!r}] must be a tuple")
            for product_id in members:
                _require_text(product_id, f"group_splits[{group!r}] member")
                if product_id not in known_products:
                    raise ValueError(f"group_splits contains unknown product_id: {product_id}")
                if product_id in assigned:
                    raise ValueError("group_splits must not overlap")
                assigned.add(product_id)
            frozen_splits[group] = tuple(members)
        if not frozen_splits["train"]:
            raise ValueError("group_splits train must be non-empty")
        if assigned != known_products:
            raise ValueError("group_splits must assign every product exactly once")
        object.__setattr__(self, "products", products)
        object.__setattr__(self, "group_splits", MappingProxyType(frozen_splits))
        object.__setattr__(self, "lineage", _immutable_mapping(self.lineage, "lineage"))

    def require_training_ready(self) -> None:
        """Revalidate every product and its evidence at the training-use decision point."""
        rejected: list[str] = []
        for product in self.products:
            try:
                product.require_current_training_eligibility()
            except (TypeError, ValueError):
                rejected.append(product.product_id)
        if rejected:
            raise ValueError(f"dataset is not training ready; ineligible product_ids: {', '.join(rejected)}")


__all__ = ["DatasetManifest", "DatasetSampleRef", "FieldProduct"]
