"""TDD coverage for the cold multi-snapshot Torch materialization adapter."""

from __future__ import annotations

import ast
from hashlib import sha256
import io
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pytest
import torch

from tensorlbm.data import DatasetManifest, FieldDatasetR2, FieldProduct, FieldSampleRefR2
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


def _run(index: int) -> RunManifest:
    evidence = json.dumps({"drag": 1.25, "run": index}).encode("utf-8")
    artifact = ArtifactManifest.from_bytes("metrics", "application/json", evidence)
    return RunManifest(
        run_id=f"run-{index}",
        model_identity={"case": index},
        config={"grid": 8},
        code_sha=_CODE_SHA,
        environment={"backend": "recorded"},
        artifacts=(artifact,),
        metrics=(MetricEvidence("drag", 1.25, "1", artifact.artifact_id, "/drag"),),
        validation_status=ValidationStatus.PASS,
        validation_reason="reviewed runtime evidence",
    )


def _field_product(index: int, values: np.ndarray) -> tuple[FieldDataProductR2, bytes]:
    payload = _npy_bytes(values)
    array = ArrayManifestR2(
        array_id="velocity",
        role=ArrayRole.FEATURE,
        shape=tuple(values.shape),
        axes=(
            AxisSpec("y", AxisSemantic.SPATIAL, values.shape[0]),
            AxisSpec("x", AxisSemantic.SPATIAL, values.shape[1]),
            AxisSpec("component", AxisSemantic.COMPONENT, values.shape[2]),
        ),
        units="m/s",
        encoding=ArrayEncoding.NPY_FLOAT32_C_LITTLE,
        blob_ref=BlobRef(
            f"velocity-{index}",
            f"file:///fixtures/velocity-{index}.npy",
            len(payload),
            sha256(payload).hexdigest(),
            "application/x-npy",
        ),
        component_labels=("u_x", "u_y"),
    )
    return FieldDataProductR2(f"product-{index}", _run(index), "metrics", (array,), {"fixture": index}), payload


def _dataset() -> tuple[FieldDatasetR2, dict[str, Mapping[str, bytes]]]:
    samples: list[FieldSampleRefR2] = []
    payloads: dict[str, Mapping[str, bytes]] = {}
    for index, split in enumerate(("train", "val", "test"), start=1):
        values = np.full((2, 3, 2), index, dtype=np.float32)
        product, payload = _field_product(index, values)
        sample_id = f"sample-{split}"
        samples.append(FieldSampleRefR2(sample_id, product, f"group-{split}", f"case-{split}", f"trajectory-{split}"))
        payloads[sample_id] = {"velocity": payload}
    return (
        FieldDatasetR2(
            "field-dataset-r2",
            "r2",
            "field reconstruction",
            tuple(samples),
            {"train": ("sample-train",), "val": ("sample-val",), "test": ("sample-test",)},
            {"curator": "test"},
        ),
        payloads,
    )


def _spec(dataset: FieldDatasetR2, *, dataset_id: str | None = None, version: str | None = None, task: TaskKind = TaskKind.FIELD_RECONSTRUCTION, backend: TrainingBackend = TrainingBackend.TORCH) -> TrainingSpec:
    products = tuple(
        FieldProduct(
            sample.product.product_id,
            sample.product.run_manifest,
            sample.product.source_artifact_id,
            "velocity",
            sample.product.arrays[0].shape,
            "float32",
            "m/s",
            ValidationStatus.PASS,
            {},
        )
        for sample in dataset.samples
    )
    manifest = DatasetManifest(
        dataset_id or dataset.dataset_id,
        version or dataset.version,
        products,
        dataset.task_name,
        {"train": tuple(product.product_id for product in products), "val": (), "test": ()},
        {},
    )
    signature = ModelSignature(("velocity",), ("target",), {"velocity": "m/s", "target": "m/s"})
    object.__setattr__(signature, "outputs", ("velocity",))
    object.__setattr__(signature, "units", MappingProxyType({"velocity": "m/s"}))
    return TrainingSpec("dataset-materialize-r1", manifest, task, signature, backend, {})


def test_materializes_all_splits_as_cpu_float32_snapshot_references() -> None:
    from tensorlbm.ml.torch_dataset_materialize import materialize_torch_field_dataset

    dataset, payloads = _dataset()
    record = materialize_torch_field_dataset(_spec(dataset), dataset, payloads)

    assert record.training_input_fingerprint == dataset.training_input_fingerprint()
    assert record.split_ids == MappingProxyType({"train": ("sample-train",), "val": ("sample-val",), "test": ("sample-test",)})
    assert record.split_counts == MappingProxyType({"train": 1, "val": 1, "test": 1})
    assert tuple(sample.sample_id for sample in record.train) == ("sample-train",)
    assert tuple(sample.sample_id for sample in record.val) == ("sample-val",)
    assert tuple(sample.sample_id for sample in record.test) == ("sample-test",)
    for sample in record.sample_records:
        ux, uy = sample.snapshot
        assert ux.device.type == "cpu" and uy.device.type == "cpu"
        assert ux.dtype is torch.float32 and uy.dtype is torch.float32
        assert sample.group_id.startswith("group-")
        assert sample.source_case_id.startswith("case-")
        assert sample.source_trajectory_id.startswith("trajectory-")
        assert sample.field_provenance.product_id.startswith("product-")
    with pytest.raises(TypeError):
        record.split_ids["train"] = ()  # type: ignore[index]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"dataset_id": "wrong"}, "dataset id"),
        ({"version": "wrong"}, "dataset version"),
        ({"task": TaskKind.SURROGATE}, "FIELD_RECONSTRUCTION"),
        ({"backend": TrainingBackend.PADDLE}, "TORCH"),
    ],
)
def test_rejects_training_spec_incompatible_with_field_dataset(kwargs, match: str) -> None:
    from tensorlbm.ml.torch_dataset_materialize import materialize_torch_field_dataset

    dataset, payloads = _dataset()
    with pytest.raises(ValueError, match=match):
        materialize_torch_field_dataset(_spec(dataset, **kwargs), dataset, payloads)


def test_freezes_outer_and_inner_payload_mappings_before_materialization(monkeypatch) -> None:
    import tensorlbm.ml.torch_dataset_materialize as adapter

    dataset, payloads = _dataset()
    trusted = dict(payloads)
    switched = _npy_bytes(np.full((2, 3, 2), 99, dtype=np.float32))

    class SwitchingInner(dict[str, bytes]):
        def __init__(self, first: bytes) -> None:
            super().__init__({"velocity": first})
            self.reads = 0

        def items(self):
            self.reads += 1
            return {"velocity": self["velocity"] if self.reads == 1 else switched}.items()

    payloads["sample-train"] = SwitchingInner(trusted["sample-train"]["velocity"])
    original = adapter.materialize_torch_velocity_snapshots
    observed: list[Mapping[str, bytes]] = []

    def checked(spec, product, inner):
        observed.append(inner)
        return original(spec, product, inner)

    monkeypatch.setattr(adapter, "materialize_torch_velocity_snapshots", checked)
    record = adapter.materialize_torch_field_dataset(_spec(dataset), dataset, payloads)
    assert len(record.sample_records) == 3
    assert len(observed) == 3
    assert observed[0]["velocity"] == trusted["sample-train"]["velocity"]
    assert isinstance(observed[0], MappingProxyType)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda payloads: payloads.pop("sample-test"), "missing sample_id"),
        (lambda payloads: payloads.update({"unknown": {"velocity": b"x"}}), "unknown sample_id"),
        (lambda payloads: payloads["sample-train"].update({"unexpected": b"x"}), "unknown array_id"),
        (lambda payloads: payloads.__setitem__("sample-train", {}), "missing array_id"),
    ],
)
def test_rejects_missing_or_unknown_sample_or_array_payloads_before_any_materializer_call(monkeypatch, mutate, match: str) -> None:
    import tensorlbm.ml.torch_dataset_materialize as adapter

    dataset, payloads = _dataset()
    mutate(payloads)
    called: list[str] = []
    monkeypatch.setattr(adapter, "materialize_torch_velocity_snapshots", lambda *args: called.append("called"))
    with pytest.raises(ValueError, match=match):
        adapter.materialize_torch_field_dataset(_spec(dataset), dataset, payloads)
    assert called == []


def test_dataset_mutation_during_payload_freeze_is_revalidated_before_snapshot_materialization() -> None:
    from tensorlbm.ml.torch_dataset_materialize import materialize_torch_field_dataset

    dataset, payloads = _dataset()

    class SplitMutatingInner(dict[str, bytes]):
        def items(self):
            object.__setattr__(
                dataset,
                "splits",
                {"train": ("sample-val",), "val": ("sample-train",), "test": ("sample-test",)},
            )
            return super().items()

    payloads["sample-train"] = SplitMutatingInner(payloads["sample-train"])
    record = materialize_torch_field_dataset(_spec(dataset), dataset, payloads)
    assert record.training_input_fingerprint == dataset.training_input_fingerprint()
    assert record.split_ids["train"] == ("sample-val",)
    assert tuple(ref.sample_id for ref in record.train) == ("sample-val",)



def test_dataset_mutation_during_materialization_is_rejected_before_later_samples(monkeypatch) -> None:
    import tensorlbm.ml.torch_dataset_materialize as adapter

    dataset, payloads = _dataset()
    original = adapter.materialize_torch_velocity_snapshots
    called: list[str] = []

    def mutating(spec, product, inner):
        called.append(product.product_id)
        if len(called) == 1:
            object.__setattr__(dataset.samples[1], "group_id", "attacked-group")
        return original(spec, product, inner)

    monkeypatch.setattr(adapter, "materialize_torch_velocity_snapshots", mutating)
    with pytest.raises(ValueError, match="dataset changed while processing"):
        adapter.materialize_torch_field_dataset(_spec(dataset), dataset, payloads)
    assert called == ["product-1"]


def test_bad_payload_does_not_materialize_later_samples_or_return_partial_record(monkeypatch) -> None:
    import tensorlbm.ml.torch_dataset_materialize as adapter

    dataset, payloads = _dataset()
    payloads["sample-train"] = {"velocity": b"not-a-valid-npy"}
    original = adapter.materialize_torch_velocity_snapshots
    called: list[str] = []

    def tracked(spec, product, inner):
        called.append(product.product_id)
        return original(spec, product, inner)

    monkeypatch.setattr(adapter, "materialize_torch_velocity_snapshots", tracked)
    with pytest.raises(ValueError):
        adapter.materialize_torch_field_dataset(_spec(dataset), dataset, payloads)
    assert called == ["product-1"]


def test_adapter_delegates_to_existing_materializer_and_has_no_direct_torch_or_training_api() -> None:
    source = Path("src/tensorlbm/ml/torch_dataset_materialize.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    calls = {
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "torch" not in imports
    assert "materialize_torch_velocity_snapshots" in calls
    lowered = source.lower()
    for forbidden in ("trainer", "optimizer", "cuda", "sdaa", "train_flow_transformer", ".to(", "uri"):
        assert forbidden not in lowered
