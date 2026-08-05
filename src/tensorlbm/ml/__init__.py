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
from .torch_holdout_evaluation import (
    HoldoutEvaluationRecord,
    HoldoutSampleEvidence,
    evaluate_evidence_gated_holdout,
)
from .torch_dataset_flow_training import (
    DatasetTrainingExecutionRecord,
    run_evidence_gated_field_dataset_flow_reconstruction,
)
from .torch_flow_transformer_holdout_evaluation import (
    FlowTransformerHoldoutEvaluationRecord,
    evaluate_evidence_gated_flow_transformer_holdout,
)

__all__ = [
    "BACKEND_REGISTRY",
    "BackendAvailability",
    "DatasetMaterializationRecord",
    "DatasetTrainingExecutionRecord",
    "EvaluationSpec",
    "FieldSnapshotReference",
    "FlowTransformerHoldoutEvaluationRecord",
    "HoldoutEvaluationRecord",
    "HoldoutSampleEvidence",
    "ModelArtifact",
    "ModelArtifactStatus",
    "ModelSignature",
    "TaskKind",
    "TrainingBackend",
    "TrainingSpec",
    "materialize_torch_field_dataset",
    "evaluate_evidence_gated_holdout",
    "evaluate_evidence_gated_flow_transformer_holdout",
    "run_evidence_gated_field_dataset_flow_reconstruction",
    "validate_evaluation_spec",
    "validate_model_artifact",
    "validate_training_spec",
]
