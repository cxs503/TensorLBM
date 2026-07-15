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
from .torch_dataset_materialize import (
    DatasetMaterializationRecord,
    FieldSnapshotReference,
    materialize_torch_field_dataset,
)

__all__ = [
    "BACKEND_REGISTRY",
    "BackendAvailability",
    "DatasetMaterializationRecord",
    "EvaluationSpec",
    "FieldSnapshotReference",
    "ModelArtifact",
    "ModelArtifactStatus",
    "ModelSignature",
    "TaskKind",
    "TrainingBackend",
    "TrainingSpec",
    "materialize_torch_field_dataset",
    "validate_evaluation_spec",
    "validate_model_artifact",
    "validate_training_spec",
]
