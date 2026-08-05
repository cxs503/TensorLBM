"""Framework-neutral, evidence-gated ML task metadata contracts.

R1 records what an eventual training or evaluation operation is allowed to use.
It deliberately contains neither a framework import nor an execution API.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from tensorlbm.data import DatasetManifest


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
    raise TypeError(f"unsupported contract value: {type(value).__name__}")


def _immutable_mapping(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    frozen = _freeze(value)
    assert isinstance(frozen, Mapping)
    return frozen


def _field_names(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{name} must be a non-empty tuple")
    names = tuple(_require_text(field, f"{name} field") for field in value)
    if len(set(names)) != len(names):
        raise ValueError(f"{name} field names must be unique")
    return names


class TrainingBackend(str, Enum):
    TORCH = "TORCH"
    PADDLE = "PADDLE"
    MINDSPORE = "MINDSPORE"


class BackendAvailability(str, Enum):
    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"


BACKEND_REGISTRY = MappingProxyType(
    {
        TrainingBackend.TORCH: BackendAvailability.SUPPORTED,
        TrainingBackend.PADDLE: BackendAvailability.NOT_SUPPORTED,
        TrainingBackend.MINDSPORE: BackendAvailability.NOT_SUPPORTED,
    }
)


class TaskKind(str, Enum):
    FIELD_RECONSTRUCTION = "FIELD_RECONSTRUCTION"
    TURBULENCE_CLOSURE = "TURBULENCE_CLOSURE"
    SURROGATE = "SURROGATE"


@dataclass(frozen=True, slots=True)
class ModelSignature:
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    units: Mapping[str, str]

    def __post_init__(self) -> None:
        inputs = _field_names(self.inputs, "inputs")
        outputs = _field_names(self.outputs, "outputs")
        fields = inputs + outputs
        if len(set(fields)) != len(fields):
            raise ValueError("input and output field names must be unique")
        if not isinstance(self.units, Mapping):
            raise TypeError("units must be a mapping")
        unit_names = set(self.units)
        if unit_names != set(fields):
            raise ValueError("units must cover exactly every input and output field")
        frozen_units = {
            _require_text(field, "units field"): _require_text(unit, "units value")
            for field, unit in self.units.items()
        }
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "units", MappingProxyType(frozen_units))


@dataclass(frozen=True, slots=True)
class TrainingSpec:
    run_id: str
    dataset: DatasetManifest
    task: TaskKind
    signature: ModelSignature
    backend: TrainingBackend
    hyperparameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_text(self.run_id, "run_id")
        if not isinstance(self.dataset, DatasetManifest):
            raise TypeError("dataset must be a DatasetManifest")
        if not isinstance(self.task, TaskKind):
            raise TypeError("task must be a TaskKind")
        if not isinstance(self.signature, ModelSignature):
            raise TypeError("signature must be a ModelSignature")
        if not isinstance(self.backend, TrainingBackend):
            raise TypeError("backend must be a TrainingBackend")
        object.__setattr__(self, "hyperparameters", _immutable_mapping(self.hyperparameters, "hyperparameters"))


@dataclass(frozen=True, slots=True)
class EvaluationSpec:
    training_run_id: str
    dataset: DatasetManifest
    signature: ModelSignature
    metrics: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text(self.training_run_id, "training_run_id")
        if not isinstance(self.dataset, DatasetManifest):
            raise TypeError("dataset must be a DatasetManifest")
        if not isinstance(self.signature, ModelSignature):
            raise TypeError("signature must be a ModelSignature")
        metrics = _field_names(self.metrics, "metrics")
        if any("physical truth" in metric.lower() for metric in metrics):
            raise ValueError("metrics must not claim physical truth")
        object.__setattr__(self, "metrics", metrics)


class ModelArtifactStatus(str, Enum):
    NOT_TRAINED = "NOT_TRAINED"
    REGISTERED = "REGISTERED"


@dataclass(frozen=True, slots=True, init=False)
class ModelArtifact:
    artifact_id: str
    training_run_id: str
    dataset: DatasetManifest
    signature: ModelSignature
    status: ModelArtifactStatus

    def __init__(
        self,
        artifact_id: str,
        training_run_id: str,
        dataset: DatasetManifest,
        signature: ModelSignature,
        status: ModelArtifactStatus,
        **unsupported: Any,
    ) -> None:
        if unsupported:
            raise TypeError("model artifacts do not accept metrics or model weights")
        _require_text(artifact_id, "artifact_id")
        _require_text(training_run_id, "training_run_id")
        if not isinstance(dataset, DatasetManifest):
            raise TypeError("dataset must be a DatasetManifest")
        if not isinstance(signature, ModelSignature):
            raise TypeError("signature must be a ModelSignature")
        if not isinstance(status, ModelArtifactStatus):
            raise TypeError("status must be a ModelArtifactStatus")
        if status is not ModelArtifactStatus.NOT_TRAINED:
            raise ValueError("R1 artifacts may only declare NOT_TRAINED")
        object.__setattr__(self, "artifact_id", artifact_id)
        object.__setattr__(self, "training_run_id", training_run_id)
        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "signature", signature)
        object.__setattr__(self, "status", status)

    @classmethod
    def from_training_spec(cls, spec: TrainingSpec, artifact_id: str) -> ModelArtifact:
        validate_training_spec(spec)
        return cls(
            artifact_id=artifact_id,
            training_run_id=spec.run_id,
            dataset=spec.dataset,
            signature=spec.signature,
            status=ModelArtifactStatus.NOT_TRAINED,
        )



def validate_model_artifact(artifact: ModelArtifact) -> ModelArtifact:
    """Fail closed before any future artifact consumer may inspect a model record."""
    if not isinstance(artifact, ModelArtifact):
        raise TypeError("artifact must be a ModelArtifact")
    _require_text(artifact.artifact_id, "artifact_id")
    _require_text(artifact.training_run_id, "training_run_id")
    if not isinstance(artifact.status, ModelArtifactStatus):
        raise ValueError("artifact status must be a ModelArtifactStatus")
    if artifact.status is not ModelArtifactStatus.NOT_TRAINED:
        raise ValueError("R1 artifact is not an executable or trained model")
    if not isinstance(artifact.dataset, DatasetManifest) or not isinstance(artifact.signature, ModelSignature):
        raise ValueError("artifact dataset or signature is invalid")
    try:
        artifact.dataset.require_training_ready()
    except (TypeError, ValueError) as error:
        raise ValueError("artifact dataset is not training ready") from error
    return artifact


def validate_training_spec(spec: TrainingSpec) -> TrainingSpec:
    """Recheck data evidence immediately before any future use, without executing it."""
    if not isinstance(spec, TrainingSpec):
        raise TypeError("spec must be a TrainingSpec")
    if BACKEND_REGISTRY[spec.backend] is not BackendAvailability.SUPPORTED:
        raise ValueError(f"backend {spec.backend.value} is not supported in R1")
    try:
        spec.dataset.require_training_ready()
    except (TypeError, ValueError) as error:
        raise ValueError("dataset is not training ready") from error
    return spec


def validate_evaluation_spec(spec: EvaluationSpec) -> EvaluationSpec:
    """Recheck evidence at evaluation-use time; PASS remains a gate, not physical truth."""
    if not isinstance(spec, EvaluationSpec):
        raise TypeError("spec must be an EvaluationSpec")
    try:
        spec.dataset.require_training_ready()
    except (TypeError, ValueError) as error:
        raise ValueError("dataset is not training ready") from error
    return spec


__all__ = [
    "BACKEND_REGISTRY",
    "BackendAvailability",
    "EvaluationSpec",
    "ModelArtifact",
    "ModelArtifactStatus",
    "ModelSignature",
    "TaskKind",
    "TrainingBackend",
    "TrainingSpec",
    "validate_evaluation_spec",
    "validate_model_artifact",
    "validate_training_spec",
]
