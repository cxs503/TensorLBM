"""TDD coverage for evidence-gated, CPU-only Torch flow smoke training."""

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

_CODE_SHA = "b" * 40


def _npy_bytes(values: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, values, allow_pickle=False)
    return stream.getvalue()


def _run() -> RunManifest:
    evidence = json.dumps({"fixture": 1.0}).encode("utf-8")
    artifact = ArtifactManifest.from_bytes("metrics", "application/json", evidence)
    return RunManifest(
        run_id="torch-flow-field-run",
        model_identity={"case": "smoke-fixture"},
        config={"grid": [2, 2]},
        code_sha=_CODE_SHA,
        environment={"backend": "recorded"},
        artifacts=(artifact,),
        metrics=(MetricEvidence("fixture", 1.0, "1", artifact.artifact_id, "/fixture"),),
        validation_status=ValidationStatus.PASS,
        validation_reason="fixture evidence reviewed",
    )


def _inputs() -> tuple[TrainingSpec, FieldDataProductR2, dict[str, bytes]]:
    values = np.arange(8, dtype=np.float32).reshape(2, 2, 2) / 10.0
    payload = _npy_bytes(values)
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
            "velocity-npy",
            "file:///fixtures/velocity.npy",
            len(payload),
            sha256(payload).hexdigest(),
            "application/x-npy",
        ),
        component_labels=("u_x", "u_y"),
    )
    run = _run()
    product = FieldDataProductR2("velocity-product", run, "metrics", (array,), {"source": "fixture"})
    field = FieldProduct(
        "velocity-field", run, "metrics", "velocity", (2, 2, 2), "float32", "m/s", ValidationStatus.PASS, {}
    )
    dataset = DatasetManifest(
        "torch-flow-dataset", "r1", (field,), "field-reconstruction", {"train": ("velocity-field",), "val": (), "test": ()}, {}
    )
    signature = ModelSignature(inputs=("velocity",), outputs=("target",), units={"velocity": "m/s", "target": "m/s"})
    object.__setattr__(signature, "outputs", ("velocity",))
    object.__setattr__(signature, "units", MappingProxyType({"velocity": "m/s"}))
    spec = TrainingSpec(
        "torch-flow-smoke-r1", dataset, TaskKind.FIELD_RECONSTRUCTION, signature, TrainingBackend.TORCH, {"purpose": "smoke"}
    )
    return spec, product, {"velocity": payload}


def _mini_arch_and_config():
    from tensorlbm.ai.transformer import FlowTransformerArch, FlowTransformerTrainConfig

    return (
        FlowTransformerArch(d_model=4, n_heads=1, n_layers=1, ffn_dim=8, dropout=0.0, max_tokens=4),
        FlowTransformerTrainConfig(epochs=1, batch_size=1, learning_rate=1e-3, device="cpu"),
    )


def test_runs_real_cpu_smoke_training_and_writes_verified_evidence(tmp_path: Path) -> None:
    from tensorlbm.ml.torch_flow_training import run_evidence_gated_flow_reconstruction

    spec, product, payloads = _inputs()
    arch, config = _mini_arch_and_config()
    out = tmp_path / "artifacts" / "smoke-flow.npz"

    record = run_evidence_gated_flow_reconstruction(spec, product, payloads, out, arch, config)

    metadata_path = Path(f"{out}.json")
    provenance_path = Path(f"{out}.provenance.json")
    assert record.status == "training execution completed"
    assert record.smoke_only is True
    assert record.n_snapshots == 1
    assert record.grid == (2, 2)
    assert Path(record.weights_path) == out and out.stat().st_size > 0
    assert Path(record.metadata_path) == metadata_path and metadata_path.stat().st_size > 0
    assert Path(record.provenance_path) == provenance_path and provenance_path.stat().st_size > 0
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert metadata["family"] == "flow_transformer_ssl"
    assert metadata["backend"] == "torch"
    assert metadata["n_snapshots"] == 1 and metadata["grid"] == [2, 2]
    assert provenance["schema"] == "tensorlbm.training-provenance.r1"
    assert provenance["training_spec"]["run_id"] == spec.run_id
    assert provenance["dataset"] == {"id": "torch-flow-dataset", "version": "r1"}
    assert provenance["field_data"]["product_id"] == product.product_id
    assert provenance["field_data"]["blob_sha256"] == sha256(payloads["velocity"]).hexdigest()
    assert provenance["files"]["weights"]["sha256"] == sha256(out.read_bytes()).hexdigest()
    assert provenance["files"]["metadata"]["sha256"] == sha256(metadata_path.read_bytes()).hexdigest()
    claimed = provenance.pop("provenance_sha256")
    canonical = json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert claimed == sha256(canonical).hexdigest()


def test_trainer_failure_never_returns_completed_record_or_provenance(tmp_path: Path, monkeypatch) -> None:
    import tensorlbm.ml.torch_flow_training as training

    spec, product, payloads = _inputs()
    arch, config = _mini_arch_and_config()
    out = tmp_path / "failed.npz"

    def explode(*args, **kwargs):
        partial = Path(args[1])
        partial.write_bytes(b"partial weights")
        Path(f"{partial}.json").write_text("{}", encoding="utf-8")
        raise RuntimeError("injected trainer failure")

    monkeypatch.setattr(training, "train_flow_transformer_self_supervised", explode)
    with pytest.raises(RuntimeError, match="injected trainer failure"):
        training.run_evidence_gated_flow_reconstruction(spec, product, payloads, out, arch, config)
    assert not Path(f"{out}.provenance.json").exists()


def test_rejects_bad_payload_before_trainer_and_refuses_existing_output(tmp_path: Path, monkeypatch) -> None:
    import tensorlbm.ml.torch_flow_training as training

    spec, product, payloads = _inputs()
    arch, config = _mini_arch_and_config()
    called = False

    def unexpected(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("trainer must not run")

    monkeypatch.setattr(training, "train_flow_transformer_self_supervised", unexpected)
    with pytest.raises(ValueError, match="payload"):
        training.run_evidence_gated_flow_reconstruction(spec, product, {"velocity": b"bad"}, tmp_path / "bad.npz", arch, config)
    assert called is False
    existing = tmp_path / "existing.npz"
    existing.write_bytes(b"do not overwrite")
    with pytest.raises(FileExistsError):
        training.run_evidence_gated_flow_reconstruction(spec, product, payloads, existing, arch, config)
    assert called is False


def test_non_cpu_device_is_rejected_before_trainer_and_leaves_no_output(tmp_path: Path, monkeypatch) -> None:
    import tensorlbm.ml.torch_flow_training as training

    spec, product, payloads = _inputs()
    arch, config = _mini_arch_and_config()
    called = False

    def unexpected(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("trainer must not run")

    monkeypatch.setattr(training, "train_flow_transformer_self_supervised", unexpected)
    for device in ("cuda", "sdaa", "mps", " cpu ", "cpu:0"):
        out = tmp_path / f"{device}.npz"
        rejected = replace(config, device=device)
        with pytest.raises(ValueError, match="device='cpu'"):
            training.run_evidence_gated_flow_reconstruction(spec, product, payloads, out, arch, rejected)
        assert not out.exists()
        assert not Path(f"{out}.json").exists()
        assert not Path(f"{out}.provenance.json").exists()
    assert called is False


def test_writer_ast_delegates_to_existing_components_without_other_training_stacks() -> None:
    source = Path("src/tensorlbm/ml/torch_flow_training.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    }
    imported.update(
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    calls = {
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    forbidden_nodes = (ast.For, ast.While, ast.ClassDef)
    assert "materialize_torch_velocity_snapshots" in calls
    assert "train_flow_transformer_self_supervised" in calls
    assert not any(token in source.lower() for token in ("paddle", "mindspore", "cuda", "sdaa"))
    assert not any("optimizer" in name.lower() or name.lower().startswith("train_") for name in calls - {"train_flow_transformer_self_supervised"})
    assert not any(isinstance(node, ast.ClassDef) and node.name != "TrainingExecutionRecord" for node in ast.walk(tree))
    assert not imported & {"paddle", "mindspore"}
