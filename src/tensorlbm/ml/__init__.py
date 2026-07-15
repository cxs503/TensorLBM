"""Framework-neutral ML task-contract surface; this package has no training API."""

from .contracts import (
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
