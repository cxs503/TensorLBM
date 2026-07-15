"""R2 cold-path field-byte contracts; metadata binds an external single NPY blob.

This module never reads a URI, starts training, or materializes an array.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import re
from types import MappingProxyType
from typing import Any, Mapping

from tensorlbm.runtime import RunManifest, ValidationStatus, validate_run_manifest


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FILE_URI = re.compile(r"file:///[^\n]+\Z")


def _text(value: object, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive non-boolean integer")
    return value


def _freeze(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        raise TypeError("metadata must not contain payload bytes")
    if isinstance(value, Mapping):
        return MappingProxyType({_text(key, "lineage key"): _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    raise TypeError(f"unsupported metadata value: {type(value).__name__}")


class AxisSemantic(str, Enum):
    SAMPLE = "SAMPLE"
    SPATIAL = "SPATIAL"
    COMPONENT = "COMPONENT"


class ArrayRole(str, Enum):
    FEATURE = "FEATURE"
    TARGET = "TARGET"
    MASK = "MASK"
    AUXILIARY = "AUXILIARY"


class MemoryOrder(str, Enum):
    C = "C"
    F = "F"


class ByteOrder(str, Enum):
    LITTLE = "LITTLE"
    BIG = "BIG"


@dataclass(frozen=True, slots=True)
class AxisSpec:
    name: str
    semantic: AxisSemantic
    length: int

    def __post_init__(self) -> None:
        _text(self.name, "axis name")
        if not isinstance(self.semantic, AxisSemantic):
            raise TypeError("axis semantic must be an AxisSemantic")
        _positive_int(self.length, "axis length")


@dataclass(frozen=True, slots=True)
class ArrayEncoding:
    format: str
    dtype: str
    order: MemoryOrder
    byte_order: ByteOrder

    def __post_init__(self) -> None:
        if self.format != "NPY":
            raise ValueError("format must be NPY")
        if self.dtype not in {"float32", "float64", "int32", "uint8", "bool"}:
            raise ValueError("dtype must be a supported concrete scalar dtype")
        if not isinstance(self.order, MemoryOrder) or not isinstance(self.byte_order, ByteOrder):
            raise TypeError("order and byte_order must be enum values")


ArrayEncoding.NPY_FLOAT32_C_LITTLE = ArrayEncoding("NPY", "float32", MemoryOrder.C, ByteOrder.LITTLE)


@dataclass(frozen=True, slots=True)
class BlobRef:
    blob_id: str
    uri: str
    byte_size: int
    sha256: str
    media_type: str

    def __post_init__(self) -> None:
        _text(self.blob_id, "blob_id")
        if (
            not isinstance(self.uri, str)
            or not _FILE_URI.fullmatch(self.uri)
            or "latest" in self.uri.split("/")
        ):
            raise ValueError("uri must be an absolute file:// path without /latest")
        _positive_int(self.byte_size, "byte_size")
        if not isinstance(self.sha256, str) or not _SHA256.fullmatch(self.sha256):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
        _text(self.media_type, "media_type")

    def validate_current(self) -> None:
        """Reconstruct all immutable fields after hostile object.__setattr__."""
        BlobRef(self.blob_id, self.uri, self.byte_size, self.sha256, self.media_type)

    def validate_blob_bytes(self, payload: object) -> None:
        self.validate_current()
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        if len(payload) != self.byte_size:
            raise ValueError("payload size does not match blob byte_size")
        if sha256(payload).hexdigest() != self.sha256:
            raise ValueError("payload sha256 does not match blob sha256")


def _shape_size(shape: tuple[int, ...]) -> int:
    size = 1
    for dimension in shape:
        size *= dimension
    return size


def _npy_header(payload: object) -> tuple[str, tuple[int, ...], MemoryOrder, ByteOrder, int]:
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    magic = bytes((147, 78, 85, 77, 80, 89))
    if len(payload) < 10 or payload[:6] != magic:
        raise ValueError("payload is not a valid NPY stream")
    major, minor = payload[6], payload[7]
    if (major, minor) == (1, 0):
        header_size, start = int.from_bytes(payload[8:10], "little"), 10
    elif major in (2, 3):
        if len(payload) < 12:
            raise ValueError("truncated NPY header")
        header_size, start = int.from_bytes(payload[8:12], "little"), 12
    else:
        raise ValueError("unsupported NPY version")
    end = start + header_size
    if end > len(payload):
        raise ValueError("truncated NPY header")
    try:
        header = ast.literal_eval(payload[start:end].decode("latin1").strip())
    except (SyntaxError, ValueError, UnicodeDecodeError) as error:
        raise ValueError("invalid NPY header") from error
    if not isinstance(header, dict) or set(header) != {"descr", "fortran_order", "shape"}:
        raise ValueError("NPY header must contain only descr, fortran_order, and shape")
    descr, fortran, shape = header["descr"], header["fortran_order"], header["shape"]
    if not isinstance(descr, str) or not isinstance(fortran, bool) or not isinstance(shape, tuple):
        raise ValueError("invalid NPY metadata types")
    if not shape or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape):
        raise ValueError("NPY shape must be a non-empty positive integer tuple")
    dtype_map = {
        "f4": ("float32", 4),
        "f8": ("float64", 8),
        "i4": ("int32", 4),
        "u1": ("uint8", 1),
        "b1": ("bool", 1),
        "?": ("bool", 1),
    }
    if len(descr) < 2 or descr[0] not in "<>=|" or descr[1:] not in dtype_map:
        if "O" in descr:
            raise ValueError("object NPY dtype is forbidden")
        raise ValueError("unsupported NPY dtype")
    prefix = descr[0]
    if prefix == "=" or prefix == "|":
        raise ValueError("ambiguous or byte-order-not-applicable NPY dtype is forbidden")
    byte_order = ByteOrder.BIG if prefix == ">" else ByteOrder.LITTLE
    dtype, item_size = dtype_map[descr[1:]]
    return dtype, shape, MemoryOrder.F if fortran else MemoryOrder.C, byte_order, end + item_size * _shape_size(shape)


@dataclass(frozen=True, slots=True)
class ArrayManifestR2:
    array_id: str
    role: ArrayRole
    shape: tuple[int, ...]
    axes: tuple[AxisSpec, ...]
    units: str
    encoding: ArrayEncoding
    blob_ref: BlobRef
    component_labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        _text(self.array_id, "array_id")
        if not isinstance(self.role, ArrayRole):
            raise TypeError("role must be an ArrayRole")
        shape = tuple(self.shape)
        if not shape or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in shape):
            raise ValueError("shape must be a non-empty tuple of positive non-boolean integers")
        axes = tuple(self.axes)
        if len(axes) != len(shape) or not all(isinstance(axis, AxisSpec) for axis in axes):
            raise ValueError("axes must contain one AxisSpec per shape dimension")
        if len({axis.name for axis in axes}) != len(axes):
            raise ValueError("axis names must be unique")
        if tuple(axis.length for axis in axes) != shape:
            raise ValueError("axis lengths must equal shape")
        components = [axis for axis in axes if axis.semantic is AxisSemantic.COMPONENT]
        if len(components) > 1:
            raise ValueError("only one component axis is allowed")
        labels = self.component_labels
        if labels is not None:
            if not isinstance(labels, tuple) or not components:
                raise ValueError("component_labels require exactly one component axis")
            if len(labels) != components[0].length or len(set(labels)) != len(labels):
                raise ValueError("component_labels must be unique and match component axis length")
            for label in labels:
                _text(label, "component label")
        _text(self.units, "units")
        if not isinstance(self.encoding, ArrayEncoding) or not isinstance(self.blob_ref, BlobRef):
            raise TypeError("encoding and blob_ref must be contract values")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "axes", axes)

    def verify_payload(self, payload: object) -> None:
        self.validate_current()
        self.blob_ref.validate_blob_bytes(payload)
        assert isinstance(payload, bytes)
        dtype, shape, order, byte_order, expected_size = _npy_header(payload)
        if len(payload) != expected_size:
            raise ValueError("NPY payload body is truncated or has trailing bytes")
        if (dtype, shape, order, byte_order) != (self.encoding.dtype, self.shape, self.encoding.order, self.encoding.byte_order):
            raise ValueError("NPY dtype, shape, order, or byte_order does not match manifest")

    def validate_current(self) -> None:
        """Reconstruct validation to fail closed after hostile object.__setattr__."""
        ArrayManifestR2(
            self.array_id,
            self.role,
            self.shape,
            self.axes,
            self.units,
            self.encoding,
            self.blob_ref,
            self.component_labels,
        )


@dataclass(frozen=True, slots=True)
class FieldDataProductR2:
    """Cold metadata product; PASS gating does not make it training-ready."""

    product_id: str
    run_manifest: RunManifest
    source_artifact_id: str
    arrays: tuple[ArrayManifestR2, ...]
    lineage: Mapping[str, Any]

    def __post_init__(self) -> None:
        _text(self.product_id, "product_id")
        self._validate_run()
        _text(self.source_artifact_id, "source_artifact_id")
        if self.source_artifact_id not in {artifact.artifact_id for artifact in self.run_manifest.artifacts}:
            raise ValueError("source_artifact_id must exist in run_manifest")
        arrays = tuple(self.arrays)
        if not arrays or not all(isinstance(array, ArrayManifestR2) for array in arrays):
            raise ValueError("arrays must be non-empty ArrayManifestR2 values")
        if len({array.array_id for array in arrays}) != len(arrays):
            raise ValueError("arrays must have unique array_id values")
        if not isinstance(self.lineage, Mapping):
            raise TypeError("lineage must be a mapping")
        object.__setattr__(self, "arrays", arrays)
        object.__setattr__(self, "lineage", _freeze(self.lineage))

    def _validate_run(self) -> None:
        if not isinstance(self.run_manifest, RunManifest):
            raise TypeError("run_manifest must be a RunManifest")
        try:
            validate_run_manifest(self.run_manifest)
        except (TypeError, ValueError) as error:
            raise ValueError("runtime evidence must validate") from error
        if self.run_manifest.validation_status is not ValidationStatus.PASS:
            raise ValueError("runtime evidence must have PASS status")

    def validate_for_use(self, payloads: Mapping[str, bytes]) -> None:
        """Reconstruct product invariants before validating any external field bytes."""
        FieldDataProductR2(
            self.product_id,
            self.run_manifest,
            self.source_artifact_id,
            self.arrays,
            self.lineage,
        )
        self._validate_run()
        arrays = self.arrays
        if not isinstance(arrays, tuple) or not arrays or not all(isinstance(array, ArrayManifestR2) for array in arrays):
            raise ValueError("arrays are no longer valid contract values")
        if len({array.array_id for array in arrays}) != len(arrays):
            raise ValueError("arrays no longer have unique array_id values")
        if not isinstance(payloads, Mapping) or set(payloads) != {array.array_id for array in arrays}:
            raise ValueError("payloads must contain exactly the declared array IDs")
        if self.source_artifact_id not in {artifact.artifact_id for artifact in self.run_manifest.artifacts}:
            raise ValueError("source_artifact_id no longer exists in run_manifest")
        for array in arrays:
            array.verify_payload(payloads[array.array_id])


__all__ = [
    "ArrayEncoding", "ArrayManifestR2", "ArrayRole", "AxisSemantic", "AxisSpec", "BlobRef", "ByteOrder",
    "FieldDataProductR2", "MemoryOrder",
]
