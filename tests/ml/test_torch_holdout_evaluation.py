"""TDD coverage for the data-only, evidence-gated holdout evaluator."""

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


def _npy_bytes(values: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, values, allow_pickle=False)
    return stream.getvalue()


def _product(index: int, values: np.ndarray) -> tuple[FieldDataProductR2, bytes]:
    payload = _npy_bytes(values)
    artifact = ArtifactManifest.from_bytes("metrics", "application/json", b'{"fixture": 1.0}')
    run = RunManifest(
        run_id=f"holdout-run-{index}",
        model_identity={"case": index},
        config={"grid": [2, 2]},
        code_sha="d" * 40,
        environment={"backend": "recorded"},
        artifacts=(artifact,),
        metrics=(MetricEvidence("fixture", 1.0, "1", artifact.artifact_id, "/fixture"),),
        validation_status=ValidationStatus.PASS,
        validation_reason="fixture evidence reviewed",
    )
    array = ArrayManifestR2(
        array_id="velocity",
        role=ArrayRole.FEATURE,
        shape=(2, 2, 2),
        axes=(
            AxisSpec("y", AxisSemantic.SPATIAL, 2),
            AxisSpec("x", AxisSemantic.SPATIAL, 2),
            AxisSpec("component", AxisSemantic.COMPONENT, 2),
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
    return FieldDataProductR2(f"product-{index}", run, "metrics", (array,), {"fixture": index}), payload


def _inputs() -> tuple[TrainingSpec, FieldDatasetR2, dict[str, dict[str, bytes]]]:
    samples: list[FieldSampleRefR2] = []
    payloads: dict[str, dict[str, bytes]] = {}
    values_by_split = {
        "train-a": np.full((2, 2, 2), 101.0, dtype=np.float32),
        "train-b": np.full((2, 2, 2), 102.0, dtype=np.float32),
        "val": np.array([[[3.0, -4.0], [5.0, -6.0]], [[7.0, -8.0], [9.0, -10.0]]], dtype=np.float32),
        "test": np.array([[[11.0, -12.0], [13.0, -14.0]], [[15.0, -16.0], [17.0, -18.0]]], dtype=np.float32),
    }
    for index, (split, values) in enumerate(values_by_split.items(), start=1):
        product, payload = _product(index, values)
        sample_id = f"sample-{split}"
        samples.append(FieldSampleRefR2(sample_id, product, f"group-{split}", f"case-{split}", f"trajectory-{split}"))
        payloads[sample_id] = {"velocity": payload}
    dataset = FieldDatasetR2(
        "holdout-dataset-r1",
        "r1",
        "field reconstruction",
        tuple(samples),
        {
            "train": ("sample-train-a", "sample-train-b"),
            "val": ("sample-val",),
            "test": ("sample-test",),
        },
        {"curator": "test"},
    )
    fields = tuple(
        FieldProduct(sample.product.product_id, sample.product.run_manifest, "metrics", "velocity", (2, 2, 2), "float32", "m/s", ValidationStatus.PASS, {})
        for sample in samples
    )
    manifest = DatasetManifest(dataset.dataset_id, dataset.version, fields, dataset.task_name, {"train": tuple(field.product_id for field in fields), "val": (), "test": ()}, {})
    signature = ModelSignature(("velocity",), ("target",), {"velocity": "m/s", "target": "m/s"})
    object.__setattr__(signature, "outputs", ("velocity",))
    object.__setattr__(signature, "units", MappingProxyType({"velocity": "m/s"}))
    return TrainingSpec("holdout-evaluation-r1", manifest, TaskKind.FIELD_RECONSTRUCTION, signature, TrainingBackend.TORCH, {}), dataset, payloads


def test_evaluates_only_test_snapshot_data_and_reports_immutable_quality_evidence() -> None:
    from tensorlbm.ml.torch_holdout_evaluation import evaluate_evidence_gated_holdout

    spec, dataset, payloads = _inputs()
    record = evaluate_evidence_gated_holdout(spec, dataset, payloads)

    assert record.status == "holdout data evaluation completed"
    assert record.data_only is True and record.not_model_evaluation is True
    assert record.split == "test"
    assert record.dataset_fingerprint == dataset.training_input_fingerprint()
    assert record.sample_ids == ("sample-test",)
    assert record.group_ids == MappingProxyType({"sample-test": "group-test"})
    assert record.source_case_ids == MappingProxyType({"sample-test": "case-test"})
    assert record.source_trajectory_ids == MappingProxyType({"sample-test": "trajectory-test"})
    assert record.field_blob_hashes == MappingProxyType({"sample-test": sha256(payloads["sample-test"]["velocity"]).hexdigest()})
    assert record.samples[0].group_id == "group-test"
    assert record.samples[0].source_case_id == "case-test"
    assert record.samples[0].source_trajectory_id == "trajectory-test"
    assert record.samples[0].field_blob_hash == sha256(payloads["sample-test"]["velocity"]).hexdigest()
    assert record.grid_shapes == ((2, 2),)
    assert record.component_finite_counts == MappingProxyType({"u_x": 4, "u_y": 4})
    assert record.component_nonfinite_counts == MappingProxyType({"u_x": 0, "u_y": 0})
    assert record.component_min == MappingProxyType({"u_x": 11.0, "u_y": -18.0})
    assert record.component_max == MappingProxyType({"u_x": 17.0, "u_y": -12.0})
    assert record.component_mean == MappingProxyType({"u_x": 14.0, "u_y": -15.0})
    assert record.component_abs_mean == MappingProxyType({"u_x": 14.0, "u_y": 15.0})
    with pytest.raises(TypeError):
        record.component_mean["u_x"] = 0.0  # type: ignore[index]


def test_val_is_allowed_and_uses_only_val_sentinel_values() -> None:
    from tensorlbm.ml import evaluate_evidence_gated_holdout

    spec, dataset, payloads = _inputs()
    record = evaluate_evidence_gated_holdout(spec, dataset, payloads, split="val")

    assert record.split == "val"
    assert record.sample_ids == ("sample-val",)
    assert record.component_mean == MappingProxyType({"u_x": 6.0, "u_y": -7.0})
    assert record.component_abs_mean == MappingProxyType({"u_x": 6.0, "u_y": 7.0})


@pytest.mark.parametrize("split", ("val", "test"))
def test_rejects_empty_selected_holdout_split(split: str) -> None:
    from tensorlbm.ml.torch_holdout_evaluation import evaluate_evidence_gated_holdout

    spec, dataset, payloads = _inputs()
    replacement = {name: ids for name, ids in dataset.splits.items()}
    replacement[split] = ()
    if split == "val":
        replacement["test"] = ("sample-val", "sample-test")
    else:
        replacement["val"] = ("sample-val", "sample-test")
    object.__setattr__(dataset, "splits", replacement)
    with pytest.raises(ValueError, match="must be non-empty"):
        evaluate_evidence_gated_holdout(spec, dataset, payloads, split=split)


@pytest.mark.parametrize("split", ("train", "", "validation", "TEST"))
def test_rejects_non_holdout_splits_before_any_adapter_call(monkeypatch, split: str) -> None:
    import tensorlbm.ml.torch_holdout_evaluation as evaluation

    spec, dataset, payloads = _inputs()
    called: list[str] = []
    monkeypatch.setattr(evaluation, "materialize_torch_field_dataset", lambda *args: called.append("adapter"))
    with pytest.raises(ValueError, match="val or test"):
        evaluation.evaluate_evidence_gated_holdout(spec, dataset, payloads, split=split)
    assert called == []


def test_invalid_selected_payload_fails_before_any_holdout_record_is_returned() -> None:
    from tensorlbm.ml.torch_holdout_evaluation import evaluate_evidence_gated_holdout

    spec, dataset, payloads = _inputs()
    payloads["sample-test"]["velocity"] = b"invalid selected bytes"
    with pytest.raises(ValueError):
        evaluate_evidence_gated_holdout(spec, dataset, payloads)


def test_dataset_spec_mismatch_is_rejected() -> None:
    from tensorlbm.ml.torch_holdout_evaluation import evaluate_evidence_gated_holdout

    spec, dataset, payloads = _inputs()
    wrong = TrainingSpec("wrong", DatasetManifest("wrong-id", "r1", spec.dataset.products, spec.dataset.task_name, spec.dataset.group_splits, {}), spec.task, spec.signature, spec.backend, {})
    with pytest.raises(ValueError, match="dataset id"):
        evaluate_evidence_gated_holdout(wrong, dataset, payloads)


def test_unselected_payloads_are_still_evidence_validated_and_dataset_toctou_propagates(monkeypatch) -> None:
    import tensorlbm.ml.torch_holdout_evaluation as evaluation

    spec, dataset, payloads = _inputs()
    payloads["sample-train-a"]["velocity"] = b"invalid unselected bytes"
    with pytest.raises(ValueError):
        evaluation.evaluate_evidence_gated_holdout(spec, dataset, payloads)

    spec, dataset, payloads = _inputs()
    original = evaluation.materialize_torch_field_dataset

    def mutate(*args):
        record = original(*args)
        object.__setattr__(dataset.samples[0], "group_id", "attacked")
        return record

    monkeypatch.setattr(evaluation, "materialize_torch_field_dataset", mutate)
    with pytest.raises(ValueError, match="dataset changed"):
        evaluation.evaluate_evidence_gated_holdout(spec, dataset, payloads)


def test_cpu_snapshots_and_boundary_never_imports_or_calls_training_or_artifact_apis() -> None:
    from tensorlbm.ml.torch_dataset_materialize import materialize_torch_field_dataset
    from tensorlbm.ml.torch_holdout_evaluation import evaluate_evidence_gated_holdout

    spec, dataset, payloads = _inputs()
    materialized = materialize_torch_field_dataset(spec, dataset, payloads)
    assert all(pair[0].device.type == "cpu" and pair[0].dtype is torch.float32 for ref in materialized.test for pair in (ref.snapshot,))
    record = evaluate_evidence_gated_holdout(spec, dataset, payloads)
    assert record.sample_count == 1

    source = Path("src/tensorlbm/ml/torch_holdout_evaluation.py").read_text(encoding="utf-8")
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
    assert "materialize_torch_field_dataset" in calls
    assert not any("trainer" in name.lower() or name.startswith("train_") for name in imports | calls)
    assert not any("load" in name.lower() or "weight" in name.lower() for name in calls)
    lowered = source.lower()
    for forbidden in ("cuda", "sdaa", "gpu"):
        assert forbidden not in lowered
