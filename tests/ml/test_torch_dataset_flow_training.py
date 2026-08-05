"""TDD coverage for evidence-gated multi-snapshot CPU field-dataset smoke training."""

from __future__ import annotations

import ast
from dataclasses import replace
from hashlib import sha256
import io
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

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

_CODE_SHA = "c" * 40


def _npy_bytes(values: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, values, allow_pickle=False)
    return stream.getvalue()


def _product(index: int, values: np.ndarray) -> tuple[FieldDataProductR2, bytes]:
    payload = _npy_bytes(values)
    artifact = ArtifactManifest.from_bytes("metrics", "application/json", b'{"fixture": 1.0}')
    run = RunManifest(
        run_id=f"dataset-flow-run-{index}",
        model_identity={"case": index},
        config={"grid": [2, 2]},
        code_sha=_CODE_SHA,
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
    for index, split in enumerate(("train-a", "train-b", "val", "test"), start=1):
        product, payload = _product(index, np.full((2, 2, 2), index / 10, dtype=np.float32))
        sample_id = f"sample-{split}"
        samples.append(FieldSampleRefR2(sample_id, product, f"group-{split}", f"case-{split}", f"trajectory-{split}"))
        payloads[sample_id] = {"velocity": payload}
    dataset = FieldDatasetR2(
        "dataset-flow-r1",
        "r1",
        "field reconstruction",
        tuple(samples),
        {"train": ("sample-train-a", "sample-train-b"), "val": ("sample-val",), "test": ("sample-test",)},
        {"curator": "test"},
    )
    fields = tuple(
        FieldProduct(sample.product.product_id, sample.product.run_manifest, "metrics", "velocity", (2, 2, 2), "float32", "m/s", ValidationStatus.PASS, {})
        for sample in samples
    )
    manifest = DatasetManifest(
        dataset.dataset_id, dataset.version, fields, dataset.task_name,
        {"train": tuple(field.product_id for field in fields), "val": (), "test": ()}, {},
    )
    signature = ModelSignature(("velocity",), ("target",), {"velocity": "m/s", "target": "m/s"})
    object.__setattr__(signature, "outputs", ("velocity",))
    object.__setattr__(signature, "units", MappingProxyType({"velocity": "m/s"}))
    return TrainingSpec("dataset-flow-smoke-r1", manifest, TaskKind.FIELD_RECONSTRUCTION, signature, TrainingBackend.TORCH, {}), dataset, payloads


def _mini_arch_and_config():
    from tensorlbm.ai.transformer import FlowTransformerArch, FlowTransformerTrainConfig

    return (
        FlowTransformerArch(d_model=4, n_heads=1, n_layers=1, ffn_dim=8, dropout=0.0, max_tokens=4),
        FlowTransformerTrainConfig(epochs=1, batch_size=1, learning_rate=1e-3, device="cpu"),
    )


def test_runs_real_cpu_training_on_train_split_only_and_writes_complete_provenance(tmp_path: Path, monkeypatch) -> None:
    import tensorlbm.ml.torch_dataset_flow_training as execution

    spec, dataset, payloads = _inputs()
    arch, config = _mini_arch_and_config()
    out = tmp_path / "artifacts" / "dataset-flow.npz"
    received: list[tuple[object, object]] = []
    real_trainer = execution.train_flow_transformer_self_supervised

    def tracked(snapshots, *args, **kwargs):
        received.extend(snapshots)
        return real_trainer(snapshots, *args, **kwargs)

    monkeypatch.setattr(execution, "train_flow_transformer_self_supervised", tracked)
    record = execution.run_evidence_gated_field_dataset_flow_reconstruction(spec, dataset, payloads, out, arch, config)

    metadata_path = Path(f"{out}.json")
    provenance_path = Path(f"{out}.provenance.json")
    assert record.status == "training execution completed"
    assert record.smoke_only is True and record.n_snapshots == 2 and record.grid == (2, 2)
    assert len(received) == 2
    assert [float(pair[0][0, 0]) for pair in received] == pytest.approx([0.1, 0.2])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert metadata["family"] == "flow_transformer_ssl" and metadata["backend"] == "torch"
    assert provenance["dataset_fingerprint"] == dataset.training_input_fingerprint()
    assert provenance["splits"]["train"]["sample_ids"] == ["sample-train-a", "sample-train-b"]
    assert provenance["splits"]["val"]["sample_ids"] == ["sample-val"]
    assert provenance["splits"]["test"]["sample_ids"] == ["sample-test"]
    for split in ("train", "val", "test"):
        assert provenance["splits"][split]["count"] == len(provenance["splits"][split]["sample_ids"])
        sample = provenance["splits"][split]["samples"][0]
        assert {"group_id", "source_case_id", "source_trajectory_id", "field_provenance"} <= set(sample)
        assert len(sample["field_provenance"]["blob_sha256"]) == 64
    assert provenance["files"]["weights"]["sha256"] == sha256(out.read_bytes()).hexdigest()
    assert provenance["files"]["metadata"]["sha256"] == sha256(metadata_path.read_bytes()).hexdigest()
    claimed = provenance.pop("provenance_sha256")
    assert claimed == sha256(json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def test_val_or_test_bytes_are_evidence_validated_but_never_trainer_inputs(tmp_path: Path, monkeypatch) -> None:
    import tensorlbm.ml.torch_dataset_flow_training as execution

    spec, dataset, payloads = _inputs()
    arch, config = _mini_arch_and_config()
    received: list[tuple[object, object]] = []

    def fake_trainer(snapshots, out, *args, **kwargs):
        received.extend(snapshots)
        Path(out).write_bytes(b"weights")
        Path(f"{out}.json").write_text('{"family":"flow_transformer_ssl","backend":"torch"}', encoding="utf-8")
        return {"family": "flow_transformer_ssl", "backend": "torch"}

    monkeypatch.setattr(execution, "train_flow_transformer_self_supervised", fake_trainer)
    execution.run_evidence_gated_field_dataset_flow_reconstruction(spec, dataset, payloads, tmp_path / "one.npz", arch, config)
    baseline = [float(pair[0][0, 0]) for pair in received]
    received.clear()
    changed = {sample: dict(inner) for sample, inner in payloads.items()}
    changed["sample-val"]["velocity"] = _npy_bytes(np.full((2, 2, 2), 9.0, dtype=np.float32))
    with pytest.raises(ValueError, match="payload"):
        execution.run_evidence_gated_field_dataset_flow_reconstruction(spec, dataset, changed, tmp_path / "changed.npz", arch, config)
    assert received == []
    assert baseline == pytest.approx([0.1, 0.2])


@pytest.mark.parametrize("device", ("cuda", "sdaa", " cpu ", "cpu:0"))
def test_non_exact_cpu_rejects_before_adapter_or_trainer_and_creates_no_files(tmp_path: Path, monkeypatch, device: str) -> None:
    import tensorlbm.ml.torch_dataset_flow_training as execution

    spec, dataset, payloads = _inputs()
    arch, config = _mini_arch_and_config()
    called: list[str] = []
    monkeypatch.setattr(execution, "materialize_torch_field_dataset", lambda *args: called.append("adapter"))
    monkeypatch.setattr(execution, "train_flow_transformer_self_supervised", lambda *args, **kwargs: called.append("trainer"))
    out = tmp_path / f"{device}.npz"
    with pytest.raises(ValueError, match="device='cpu'"):
        execution.run_evidence_gated_field_dataset_flow_reconstruction(spec, dataset, payloads, out, arch, replace(config, device=device))
    assert called == []
    assert not any(Path(f"{out}{suffix}").exists() for suffix in ("", ".json", ".provenance.json"))


def test_requires_at_least_two_train_snapshots_and_cleans_every_partial_on_failure(tmp_path: Path, monkeypatch) -> None:
    import tensorlbm.ml.torch_dataset_flow_training as execution

    spec, dataset, payloads = _inputs()
    arch, config = _mini_arch_and_config()
    object.__setattr__(dataset, "splits", {"train": ("sample-train-a",), "val": ("sample-train-b", "sample-val"), "test": ("sample-test",)})
    with pytest.raises(ValueError, match="at least 2 train"):
        execution.run_evidence_gated_field_dataset_flow_reconstruction(spec, dataset, payloads, tmp_path / "single.npz", arch, config)
    spec, dataset, payloads = _inputs()
    out = tmp_path / "failed.npz"

    def explode(_snapshots, path, *args, **kwargs):
        Path(path).write_bytes(b"partial")
        Path(f"{path}.json").write_text("{}", encoding="utf-8")
        raise RuntimeError("injected trainer failure")

    monkeypatch.setattr(execution, "train_flow_transformer_self_supervised", explode)
    with pytest.raises(RuntimeError, match="injected trainer failure"):
        execution.run_evidence_gated_field_dataset_flow_reconstruction(spec, dataset, payloads, out, arch, config)
    assert not any(Path(f"{out}{suffix}").exists() for suffix in ("", ".json", ".provenance.json"))


def test_detects_dataset_toctou_before_training_and_writer_is_a_delegating_boundary(tmp_path: Path, monkeypatch) -> None:
    import tensorlbm.ml.torch_dataset_flow_training as execution

    spec, dataset, payloads = _inputs()
    arch, config = _mini_arch_and_config()
    real_adapter = execution.materialize_torch_field_dataset

    def mutate(*args):
        record = real_adapter(*args)
        object.__setattr__(dataset.samples[0], "group_id", "attacked")
        return record

    monkeypatch.setattr(execution, "materialize_torch_field_dataset", mutate)
    with pytest.raises(ValueError, match="dataset changed"):
        execution.run_evidence_gated_field_dataset_flow_reconstruction(spec, dataset, payloads, tmp_path / "toctou.npz", arch, config)
    source = Path("src/tensorlbm/ml/torch_dataset_flow_training.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = {node.func.id if isinstance(node.func, ast.Name) else node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))}
    assert {"materialize_torch_field_dataset", "train_flow_transformer_self_supervised"} <= calls
    assert not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree))
    assert "torch" not in {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    lowered = source.lower()
    for forbidden in ("optimizer", "cuda", "sdaa", "model("):
        assert forbidden not in lowered
    assert not any(isinstance(node, ast.ClassDef) and node.name != "DatasetTrainingExecutionRecord" for node in ast.walk(tree))
