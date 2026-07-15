"""TDD coverage for the narrow R1 FieldDataProductR2-to-Torch adapter."""

from __future__ import annotations

import ast
from hashlib import sha256
import io
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
import torch

from tensorlbm.data import DatasetManifest, FieldProduct
from tensorlbm.data.field_r2 import (
    ArrayEncoding,
    ArrayManifestR2,
    ArrayRole,
    AxisSemantic,
    AxisSpec,
    BlobRef,
    FieldDataProductR2,
)
from tensorlbm.ml import ModelSignature, TaskKind, TrainingBackend, TrainingSpec
from tensorlbm.runtime import ArtifactManifest, MetricEvidence, RunManifest, ValidationStatus

_CODE_SHA = "a" * 40


def _npy_bytes(values: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, values, allow_pickle=False)
    return stream.getvalue()


def _run(status: ValidationStatus = ValidationStatus.PASS) -> RunManifest:
    evidence = json.dumps({"drag": 1.25}).encode("utf-8")
    artifact = ArtifactManifest.from_bytes("metrics", "application/json", evidence)
    return RunManifest(
        run_id="field-run-r2",
        model_identity={"case": "fixture"},
        config={"grid": 8},
        code_sha=_CODE_SHA,
        environment={"backend": "recorded"},
        artifacts=(artifact,),
        metrics=(MetricEvidence("drag", 1.25, "1", artifact.artifact_id, "/drag"),),
        validation_status=status,
        validation_reason="reviewed runtime evidence",
    )


def _product(
    values: np.ndarray | None = None,
    *,
    status: ValidationStatus = ValidationStatus.PASS,
    shape: tuple[int, ...] = (2, 3, 2),
    axes: tuple[AxisSpec, ...] | None = None,
    labels: tuple[str, ...] = ("u_x", "u_y"),
    encoding: ArrayEncoding = ArrayEncoding.NPY_FLOAT32_C_LITTLE,
) -> tuple[FieldDataProductR2, dict[str, bytes]]:
    values = values if values is not None else np.arange(12, dtype=np.float32).reshape(2, 3, 2)
    payload = _npy_bytes(values)
    blob = BlobRef("velocity-npy", "file:///fixtures/velocity.npy", len(payload), sha256(payload).hexdigest(), "application/x-npy")
    array = ArrayManifestR2(
        array_id="velocity",
        role=ArrayRole.FEATURE,
        shape=shape,
        axes=axes
        or tuple(
            (
                AxisSpec("y", AxisSemantic.SPATIAL, shape[0]),
                AxisSpec("x", AxisSemantic.SPATIAL, shape[1]),
                AxisSpec("component", AxisSemantic.COMPONENT, shape[2]),
            )
            if len(shape) == 3
            else (
                AxisSpec("sample", AxisSemantic.SAMPLE, shape[0]),
                AxisSpec("y", AxisSemantic.SPATIAL, shape[1]),
                AxisSpec("x", AxisSemantic.SPATIAL, shape[2]),
                AxisSpec("component", AxisSemantic.COMPONENT, shape[3]),
            )
        ),
        units="m/s",
        encoding=encoding,
        blob_ref=blob,
        component_labels=labels,
    )
    run = _run(status)
    return (
        FieldDataProductR2("velocity-r2", run, "metrics", (array,), {"source": "fixture"}),
        {"velocity": payload},
    )


def _spec(
    *,
    backend: TrainingBackend = TrainingBackend.TORCH,
    task: TaskKind = TaskKind.FIELD_RECONSTRUCTION,
    signature_ok: bool = True,
) -> TrainingSpec:
    run = _run()
    field = FieldProduct("dataset-velocity", run, "metrics", "velocity", (2, 3, 2), "float32", "m/s", ValidationStatus.PASS, {})
    dataset = DatasetManifest("dataset", "r1", (field,), "field-reconstruction", {"train": ("dataset-velocity",), "val": (), "test": ()}, {})
    signature = ModelSignature(inputs=("velocity",), outputs=("target",), units={"velocity": "m/s", "target": "m/s"})
    spec = TrainingSpec("torch-materialize-r1", dataset, task, signature, backend, {})
    if signature_ok:
        # The existing framework-neutral constructor reserves distinct input/output names.
        # R1's self-reconstruction adapter validates the explicit use-time representation.
        object.__setattr__(signature, "outputs", ("velocity",))
        object.__setattr__(signature, "units", MappingProxyType({"velocity": "m/s"}))
    return spec



class _SwitchingPayloads(dict[str, bytes]):
    """Returns trusted bytes once, then a distinct untrusted payload if reread."""

    def __init__(self, trusted: bytes, untrusted: bytes) -> None:
        super().__init__({"velocity": trusted})
        self._trusted = trusted
        self._untrusted = untrusted
        self._reads = 0

    def __getitem__(self, key: str) -> bytes:
        self._reads += 1
        return self._trusted if self._reads == 1 else self._untrusted


def test_materializes_verified_single_velocity_snapshot_and_provenance() -> None:
    from tensorlbm.ml.torch_materialize import materialize_torch_velocity_snapshots

    values = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
    product, payloads = _product(values)
    snapshots, provenance = materialize_torch_velocity_snapshots(_spec(), product, payloads)

    assert len(snapshots) == 1
    ux, uy = snapshots[0]
    assert ux.device.type == "cpu" and uy.device.type == "cpu"
    assert ux.dtype is torch.float32 and uy.dtype is torch.float32
    assert tuple(ux.shape) == (2, 3) and tuple(uy.shape) == (2, 3)
    assert torch.equal(ux, torch.tensor(values[..., 0]))
    assert torch.equal(uy, torch.tensor(values[..., 1]))
    assert provenance.product_id == "velocity-r2"
    assert provenance.run_id == "field-run-r2"
    assert provenance.array_id == "velocity"
    assert provenance.blob_sha256 == sha256(payloads["velocity"]).hexdigest()
    assert provenance.shape == (2, 3, 2)
    assert provenance.dtype == "float32"
    assert provenance.units == "m/s"
    assert provenance.order == "C"
    assert provenance.component_labels == ("u_x", "u_y")


def test_materialization_uses_the_same_frozen_bytes_for_validation_and_decode() -> None:
    from tensorlbm.ml.torch_materialize import materialize_torch_velocity_snapshots

    trusted = np.zeros((2, 3, 2), dtype=np.float32)
    product, trusted_payloads = _product(trusted)
    untrusted = _npy_bytes(np.full((2, 3, 2), 99.0, dtype=np.float32))
    snapshots, provenance = materialize_torch_velocity_snapshots(
        _spec(), product, _SwitchingPayloads(trusted_payloads["velocity"], untrusted)
    )
    assert torch.equal(snapshots[0][0], torch.zeros((2, 3), dtype=torch.float32))
    assert provenance.blob_sha256 == sha256(trusted_payloads["velocity"]).hexdigest()


def test_use_time_gates_tampering_and_rejects_wrong_backend_task_or_signature() -> None:
    from tensorlbm.ml.torch_materialize import materialize_torch_velocity_snapshots

    product, payloads = _product()
    with pytest.raises(ValueError, match="sha256"):
        materialize_torch_velocity_snapshots(_spec(), product, {"velocity": payloads["velocity"][:-1] + b"x"})
    with pytest.raises(ValueError, match="not supported"):
        materialize_torch_velocity_snapshots(_spec(backend=TrainingBackend.PADDLE), product, payloads)
    with pytest.raises(ValueError, match="FIELD_RECONSTRUCTION"):
        materialize_torch_velocity_snapshots(_spec(task=TaskKind.SURROGATE), product, payloads)
    with pytest.raises(ValueError, match="signature"):
        materialize_torch_velocity_snapshots(_spec(signature_ok=False), product, payloads)
    withheld, withheld_payloads = _product()
    object.__setattr__(withheld.run_manifest, "validation_status", ValidationStatus.WITHHELD)
    with pytest.raises(ValueError, match="PASS"):
        materialize_torch_velocity_snapshots(_spec(), withheld, withheld_payloads)


@pytest.mark.parametrize(
    ("shape", "axes", "labels", "encoding"),
    [
        ((1, 2, 3, 2), None, ("u_x", "u_y"), ArrayEncoding.NPY_FLOAT32_C_LITTLE),
        ((2, 3, 2), (AxisSpec("sample", AxisSemantic.SAMPLE, 2), AxisSpec("x", AxisSemantic.SPATIAL, 3), AxisSpec("component", AxisSemantic.COMPONENT, 2)), ("u_x", "u_y"), ArrayEncoding.NPY_FLOAT32_C_LITTLE),
        ((2, 3, 2), None, ("u", "v"), ArrayEncoding.NPY_FLOAT32_C_LITTLE),
        ((2, 3, 2), None, ("u_x", "u_y"), ArrayEncoding("NPY", "float64", ArrayEncoding.NPY_FLOAT32_C_LITTLE.order, ArrayEncoding.NPY_FLOAT32_C_LITTLE.byte_order)),
    ],
)
def test_rejects_non_velocity_layout_components_or_dtype(shape, axes, labels, encoding) -> None:
    from tensorlbm.ml.torch_materialize import materialize_torch_velocity_snapshots

    values = np.zeros(shape, dtype=np.float64 if encoding.dtype == "float64" else np.float32)
    product, payloads = _product(values, shape=shape, axes=axes, labels=labels, encoding=encoding)
    with pytest.raises(ValueError):
        materialize_torch_velocity_snapshots(_spec(), product, payloads)


def test_adapter_loads_without_pickle_and_has_no_training_or_gpu_execution_api(monkeypatch) -> None:
    from tensorlbm.ml.torch_materialize import materialize_torch_velocity_snapshots

    product, payloads = _product()
    original_load = np.load
    calls: list[bool] = []

    def checked_load(*args, **kwargs):
        calls.append(kwargs.get("allow_pickle"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", checked_load)
    materialize_torch_velocity_snapshots(_spec(), product, payloads)
    assert calls == [False]

    source = Path("src/tensorlbm/ml/torch_materialize.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.name.lower() for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    call_names = {
        (node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id).lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    imported_modules = {
        alias.name.lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module.lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert not {"train", "fit", "train_flow_transformer"} & names
    assert not {"train", "fit", "train_flow_transformer"} & call_names
    assert not any("trainer" in module or "train_flow_transformer" in module for module in imported_modules)
    assert "cuda" not in source.lower()
    assert "sdaa" not in source.lower()
    assert "uri" not in source.lower()
