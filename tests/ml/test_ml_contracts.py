"""Tests for framework-neutral, evidence-gated ML task contracts."""

import ast
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
from types import MappingProxyType

import pytest

from tensorlbm.data import DatasetManifest, FieldProduct
from tensorlbm.ml.contracts import (
    BACKEND_REGISTRY,
    BackendAvailability,
    EvaluationSpec,
    ModelArtifact,
    ModelArtifactStatus,
    ModelSignature,
    TaskKind,
    TrainingBackend,
    TrainingSpec,
    validate_evaluation_spec,
    validate_model_artifact,
    validate_training_spec,
)
from tensorlbm.runtime import ArtifactManifest, MetricEvidence, RunManifest, ValidationStatus

_CODE_SHA = "a" * 40


def _manifest(status: ValidationStatus = ValidationStatus.PASS) -> RunManifest:
    payload = json.dumps({"drag": 1.25}).encode("utf-8")
    artifact = ArtifactManifest.from_bytes("metrics-json", "application/json", payload)
    metric = MetricEvidence("drag", 1.25, "1", artifact.artifact_id, "/drag")
    return RunManifest(
        run_id="run-001",
        model_identity={"case": "reference"},
        config={"resolution": 64},
        code_sha=_CODE_SHA,
        environment={"backend": "recorded"},
        artifacts=(artifact,),
        metrics=(metric,),
        validation_status=status,
        validation_reason="runtime evidence reviewed",
    )


def _dataset(status: ValidationStatus = ValidationStatus.PASS) -> DatasetManifest:
    product = FieldProduct(
        product_id="velocity-field",
        run_manifest=_manifest(status),
        artifact_id="metrics-json",
        field_name="velocity",
        shape=(8, 4, 2),
        dtype="float32",
        units="m/s",
        quality_status=status,
        lineage={"source": {"campaign": "baseline"}},
    )
    return DatasetManifest(
        dataset_id="training-set",
        version="r1",
        products=(product,),
        task_name="field-reconstruction",
        group_splits={"train": ("velocity-field",), "val": (), "test": ()},
        lineage={"curation": {"owner": "data-governance"}},
    )


def _signature() -> ModelSignature:
    return ModelSignature(
        inputs=("velocity", "pressure"),
        outputs=("closure",),
        units={"velocity": "m/s", "pressure": "Pa", "closure": "m2/s2"},
    )


def _spec(dataset: DatasetManifest | None = None, backend: TrainingBackend = TrainingBackend.TORCH) -> TrainingSpec:
    return TrainingSpec(
        run_id="ml-run-001",
        dataset=dataset or _dataset(),
        task=TaskKind.FIELD_RECONSTRUCTION,
        signature=_signature(),
        backend=backend,
        hyperparameters={"schedule": {"epochs": 4}},
    )


def test_r1_registry_declares_only_torch_supported_without_hardware_claims() -> None:
    assert BACKEND_REGISTRY == {
        TrainingBackend.TORCH: BackendAvailability.SUPPORTED,
        TrainingBackend.PADDLE: BackendAvailability.NOT_SUPPORTED,
        TrainingBackend.MINDSPORE: BackendAvailability.NOT_SUPPORTED,
    }
    assert "gpu" not in repr(BACKEND_REGISTRY).lower()
    assert "sdaa" not in repr(BACKEND_REGISTRY).lower()


def test_pass_dataset_creates_torch_training_spec_but_never_trains() -> None:
    spec = _spec()

    assert validate_training_spec(spec) is spec
    artifact = ModelArtifact.from_training_spec(spec, artifact_id="model-001")
    assert artifact.status is ModelArtifactStatus.NOT_TRAINED
    assert artifact.training_run_id == "ml-run-001"
    assert validate_model_artifact(artifact) is artifact


def test_withheld_dataset_and_post_construction_evidence_tampering_are_rejected_at_use_time() -> None:
    with pytest.raises(ValueError, match="training ready"):
        validate_training_spec(_spec(_dataset(ValidationStatus.WITHHELD)))

    dataset = _dataset()
    spec = _spec(dataset)
    object.__setattr__(dataset.products[0].run_manifest.artifacts[0], "payload", b'{"drag": 99.0}')
    with pytest.raises(ValueError, match="training ready"):
        validate_training_spec(spec)


@pytest.mark.parametrize("backend", [TrainingBackend.PADDLE, TrainingBackend.MINDSPORE])
def test_unsupported_backends_are_rejected(backend: TrainingBackend) -> None:
    with pytest.raises(ValueError, match="not supported"):
        validate_training_spec(_spec(backend=backend))


def test_signature_requires_unique_nonempty_fields_and_complete_units() -> None:
    with pytest.raises(ValueError, match="unique"):
        ModelSignature(inputs=("velocity", "velocity"), outputs=("closure",), units={"velocity": "m/s", "closure": "1"})
    with pytest.raises(ValueError, match="non-empty"):
        ModelSignature(inputs=("",), outputs=("closure",), units={"": "m/s", "closure": "1"})
    with pytest.raises(ValueError, match="cover"):
        ModelSignature(inputs=("velocity",), outputs=("closure",), units={"velocity": "m/s"})
    with pytest.raises(ValueError, match="unique"):
        ModelSignature(inputs=("velocity",), outputs=("velocity",), units={"velocity": "m/s"})


def test_contracts_are_frozen_and_deeply_immutable() -> None:
    signature = _signature()
    spec = _spec()

    assert isinstance(signature.units, MappingProxyType)
    assert isinstance(spec.hyperparameters, MappingProxyType)
    assert isinstance(spec.hyperparameters["schedule"], MappingProxyType)
    with pytest.raises(TypeError):
        signature.units["velocity"] = "changed"
    with pytest.raises(TypeError):
        spec.hyperparameters["schedule"]["epochs"] = 9
    with pytest.raises(FrozenInstanceError):
        spec.run_id = "changed"


def test_evaluation_binds_training_run_dataset_signature_and_metrics_without_truth_claims() -> None:
    evaluation = EvaluationSpec(
        training_run_id="ml-run-001",
        dataset=_dataset(),
        signature=_signature(),
        metrics=("relative-l2",),
    )

    assert validate_evaluation_spec(evaluation) is evaluation
    with pytest.raises(ValueError, match="physical truth"):
        EvaluationSpec("ml-run-001", _dataset(), _signature(), ("physical truth",))


def test_model_artifact_never_accepts_metrics_or_weights_and_has_no_completed_status() -> None:
    spec = _spec()
    with pytest.raises(TypeError):
        ModelArtifact.from_training_spec(spec, artifact_id="model-001", metrics={})
    with pytest.raises(TypeError):
        ModelArtifact("model-001", "ml-run-001", _dataset(), _signature(), ModelArtifactStatus.NOT_TRAINED, weights=b"x")
    assert {status.value for status in ModelArtifactStatus} == {"NOT_TRAINED", "REGISTERED"}
    with pytest.raises(ValueError, match="only declare"):
        ModelArtifact("model-registered", "ml-run-001", _dataset(), _signature(), ModelArtifactStatus.REGISTERED)


def test_model_artifact_validator_rejects_post_construction_training_claim_or_bad_evidence() -> None:
    artifact = ModelArtifact.from_training_spec(_spec(), artifact_id="model-001")
    object.__setattr__(artifact, "status", "TRAINED")
    with pytest.raises(ValueError, match="status"):
        validate_model_artifact(artifact)

    artifact = ModelArtifact.from_training_spec(_spec(), artifact_id="model-002")
    object.__setattr__(artifact.dataset.products[0].run_manifest.artifacts[0], "payload", b'{"drag": 9.0}')
    with pytest.raises(ValueError, match="dataset"):
        validate_model_artifact(artifact)


def test_ml_contract_ast_has_no_framework_or_solver_imports_or_train_fit_api() -> None:
    contracts = Path("src/tensorlbm/ml/contracts.py")
    tree = ast.parse(contracts.read_text(encoding="utf-8"))
    imported_roots = set()
    definitions = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions.add(node.name.lower())

    assert imported_roots <= {"__future__", "dataclasses", "enum", "types", "typing", "tensorlbm"}
    assert not {"train", "fit"} & definitions
    assert not {"torch", "paddle", "mindspore", "solver", "loop"} & {
        root.lower() for root in imported_roots
    }
