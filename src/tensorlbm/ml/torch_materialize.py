"""Narrow CPU-only FieldDataProductR2 velocity materialization for future Torch consumers.

This module accepts caller-supplied bytes only.  It validates evidence and layout,
then returns one detached velocity snapshot; it does not perform training.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Mapping

import numpy as np
import torch

from tensorlbm.data.field_r2 import (
    ArrayManifestR2,
    ArrayRole,
    AxisSemantic,
    ByteOrder,
    FieldDataProductR2,
    MemoryOrder,
)
from tensorlbm.ml.contracts import TaskKind, TrainingBackend, TrainingSpec, validate_training_spec


@dataclass(frozen=True, slots=True)
class MaterializationProvenance:
    """Immutable metadata binding a returned snapshot to its validated field blob."""

    product_id: str
    run_id: str
    array_id: str
    blob_sha256: str
    shape: tuple[int, ...]
    dtype: str
    units: str
    order: str
    component_labels: tuple[str, ...]


def _velocity_array(product: FieldDataProductR2) -> ArrayManifestR2:
    arrays = tuple(array for array in product.arrays if array.role is ArrayRole.FEATURE)
    if len(arrays) != 1 or arrays[0].array_id != "velocity":
        raise ValueError("product must contain exactly one FEATURE array named velocity")
    array = arrays[0]
    if len(array.shape) != 3 or array.shape[2] != 2:
        raise ValueError("velocity array shape must be (ny, nx, 2)")
    if (
        tuple(axis.name for axis in array.axes) != ("y", "x", "component")
        or tuple(axis.semantic for axis in array.axes)
        != (AxisSemantic.SPATIAL, AxisSemantic.SPATIAL, AxisSemantic.COMPONENT)
    ):
        raise ValueError("velocity axes must be Y, X, COMPONENT without a sample axis")
    if array.component_labels != ("u_x", "u_y"):
        raise ValueError("velocity component_labels must be ('u_x', 'u_y')")
    if (
        array.encoding.dtype != "float32"
        or array.encoding.order is not MemoryOrder.C
        or array.encoding.byte_order is not ByteOrder.LITTLE
    ):
        raise ValueError("velocity encoding must be little-endian C-order float32")
    if not array.units.strip():
        raise ValueError("velocity units must be non-empty")
    return array


def _validate_spec(spec: TrainingSpec, units: str) -> None:
    validate_training_spec(spec)
    if spec.backend is not TrainingBackend.TORCH:
        raise ValueError("materialization requires the TORCH backend")
    if spec.task is not TaskKind.FIELD_RECONSTRUCTION:
        raise ValueError("materialization requires FIELD_RECONSTRUCTION")
    signature = spec.signature
    if (
        signature.inputs != ("velocity",)
        or signature.outputs != ("velocity",)
        or dict(signature.units) != {"velocity": units}
    ):
        raise ValueError("signature must self-reconstruct velocity with the array units")


def materialize_torch_velocity_snapshots(
    spec: TrainingSpec,
    product: FieldDataProductR2,
    payloads: Mapping[str, bytes],
) -> tuple[tuple[tuple[torch.Tensor, torch.Tensor], ...], MaterializationProvenance]:
    """Return one detached CPU ``(u_x, u_y)`` snapshot and its field provenance."""
    if not isinstance(product, FieldDataProductR2):
        raise TypeError("product must be a FieldDataProductR2")
    array = _velocity_array(product)
    _validate_spec(spec, array.units)
    if not isinstance(payloads, Mapping):
        raise TypeError("payloads must be a mapping")
    frozen_payloads: dict[str, bytes] = {}
    for array_id in {array.array_id for array in product.arrays}:
        payload = payloads[array_id]
        if not isinstance(payload, bytes):
            raise TypeError("payloads must contain bytes values")
        # bytes(payload) makes the use-time value explicit even for a bytes subclass.
        frozen_payloads[array_id] = bytes(payload)
    product.validate_for_use(frozen_payloads)

    payload = frozen_payloads[array.array_id]
    decoded = np.load(BytesIO(payload), allow_pickle=False)
    if (
        not isinstance(decoded, np.ndarray)
        or decoded.dtype != np.dtype("<f4")
        or decoded.shape != array.shape
        or not decoded.flags.c_contiguous
    ):
        raise ValueError("decoded velocity payload must be C-order little-endian float32 with manifest shape")

    velocity = torch.from_numpy(decoded)
    snapshot = (velocity[..., 0].clone(), velocity[..., 1].clone())
    provenance = MaterializationProvenance(
        product_id=product.product_id,
        run_id=product.run_manifest.run_id,
        array_id=array.array_id,
        blob_sha256=array.blob_ref.sha256,
        shape=array.shape,
        dtype=array.encoding.dtype,
        units=array.units,
        order=array.encoding.order.value,
        component_labels=("u_x", "u_y"),
    )
    return (snapshot,), provenance


__all__ = ["MaterializationProvenance", "materialize_torch_velocity_snapshots"]
