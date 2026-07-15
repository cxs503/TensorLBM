"""Evidence-gated CPU execution over an already partitioned field dataset.

This is an orchestration boundary: it delegates evidence validation/materialization
and transformer optimization to the existing components.  It is smoke-only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from tensorlbm.ai.transformer import (
    FlowTransformerArch,
    FlowTransformerTrainConfig,
    train_flow_transformer_self_supervised,
)
from tensorlbm.data import FieldDatasetR2
from tensorlbm.ml.contracts import TaskKind, TrainingBackend, TrainingSpec
from tensorlbm.ml.torch_dataset_materialize import (
    DatasetMaterializationRecord,
    FieldSnapshotReference,
    materialize_torch_field_dataset,
)


@dataclass(frozen=True, slots=True)
class DatasetTrainingExecutionRecord:
    """Completed and file-verified dataset smoke-training evidence, not a model artifact."""

    status: str
    smoke_only: bool
    run_id: str
    dataset_fingerprint: str
    n_snapshots: int
    grid: tuple[int, int]
    weights_path: str
    metadata_path: str
    provenance_path: str


def _output_paths(out: Path) -> tuple[Path, Path, Path]:
    return out, Path(f"{out}.json"), Path(f"{out}.provenance.json")


def _canonical_sha256(document: object) -> str:
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _require_new_output(out: Path) -> None:
    if any(path.exists() for path in _output_paths(out)):
        raise FileExistsError("output, metadata, and provenance paths must all be new")
    out.parent.mkdir(parents=True, exist_ok=True)


def _remove_partial_outputs(out: Path) -> None:
    weights, metadata, provenance = _output_paths(out)
    if weights.exists():
        weights.unlink()
    if metadata.exists():
        metadata.unlink()
    if provenance.exists():
        provenance.unlink()


def _verified_trainer_files(out: Path) -> Mapping[str, Any]:
    _, metadata_path, _ = _output_paths(out)
    if not out.is_file() or out.stat().st_size == 0:
        raise ValueError("trainer did not write non-empty weights")
    if not metadata_path.is_file() or metadata_path.stat().st_size == 0:
        raise ValueError("trainer did not write non-empty metadata")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("trainer metadata is not valid JSON") from error
    if not isinstance(metadata, dict):
        raise ValueError("trainer metadata must be a JSON object")
    if metadata.get("family") != "flow_transformer_ssl" or metadata.get("backend") != "torch":
        raise ValueError("trainer metadata family/backend mismatch")
    return metadata


def _sample_provenance(reference: FieldSnapshotReference) -> dict[str, object]:
    field = reference.field_provenance
    return {
        "sample_id": reference.sample_id,
        "group_id": reference.group_id,
        "source_case_id": reference.source_case_id,
        "source_trajectory_id": reference.source_trajectory_id,
        "field_provenance": {
            "product_id": field.product_id,
            "run_id": field.run_id,
            "array_id": field.array_id,
            "blob_sha256": field.blob_sha256,
            "shape": list(field.shape),
            "dtype": field.dtype,
            "units": field.units,
            "order": field.order,
            "component_labels": list(field.component_labels),
        },
    }


def _split_provenance(record: DatasetMaterializationRecord, split: str) -> dict[str, object]:
    records = getattr(record, split)
    return {
        "sample_ids": list(record.split_ids[split]),
        "count": record.split_counts[split],
        "samples": [_sample_provenance(reference) for reference in records],
    }


def run_evidence_gated_field_dataset_flow_reconstruction(
    spec: TrainingSpec,
    dataset: FieldDatasetR2,
    payloads: Mapping[str, Mapping[str, bytes]],
    out_path: str | Path,
    arch: FlowTransformerArch | None = None,
    config: FlowTransformerTrainConfig | None = None,
) -> DatasetTrainingExecutionRecord:
    """Materialize every evidence-gated split, then train solely on train snapshots."""
    effective_config = config if config is not None else FlowTransformerTrainConfig()
    if effective_config.device.lower() != "cpu":
        raise ValueError("evidence-gated smoke execution requires device='cpu'")
    if not isinstance(spec, TrainingSpec):
        raise TypeError("spec must be a TrainingSpec")
    if spec.backend is not TrainingBackend.TORCH:
        raise ValueError("execution requires the TORCH backend")
    if spec.task is not TaskKind.FIELD_RECONSTRUCTION:
        raise ValueError("execution requires FIELD_RECONSTRUCTION")

    out = Path(out_path)
    _require_new_output(out)
    try:
        materialized = materialize_torch_field_dataset(spec, dataset, payloads)
        if len(materialized.train) < 2:
            raise ValueError("dataset requires at least 2 train snapshots for multi-snapshot execution")
        if dataset.training_input_fingerprint() != materialized.training_input_fingerprint:
            raise ValueError("dataset changed while processing")
        first = materialized.train[0].snapshot
        grid = (int(first[0].shape[0]), int(first[0].shape[1]))
        trainer_metadata = train_flow_transformer_self_supervised(
            list(reference.snapshot for reference in materialized.train),
            out,
            arch,
            effective_config,
            backend="torch",
        )
        if not isinstance(trainer_metadata, dict):
            raise ValueError("trainer returned invalid metadata")
        if trainer_metadata.get("family") != "flow_transformer_ssl" or trainer_metadata.get("backend") != "torch":
            raise ValueError("trainer return family/backend mismatch")
        if "n_snapshots" in trainer_metadata and trainer_metadata["n_snapshots"] != len(materialized.train):
            raise ValueError("trainer return snapshot count mismatch")
        if "grid" in trainer_metadata and trainer_metadata["grid"] != list(grid):
            raise ValueError("trainer return grid mismatch")
        if dataset.training_input_fingerprint() != materialized.training_input_fingerprint:
            raise ValueError("dataset changed while training")
        metadata = _verified_trainer_files(out)
        _, metadata_path, provenance_path = _output_paths(out)
        provenance: dict[str, object] = {
            "schema": "tensorlbm.dataset-training-provenance.r1",
            "training_spec": {"run_id": spec.run_id},
            "dataset_fingerprint": materialized.training_input_fingerprint,
            "splits": {split: _split_provenance(materialized, split) for split in ("train", "val", "test")},
            "trainer": {
                "family": metadata["family"],
                "backend": metadata["backend"],
                "n_snapshots": len(materialized.train),
                "grid": list(grid),
                "metadata_canonical_sha256": _canonical_sha256(metadata),
                "config": asdict(effective_config),
            },
            "files": {
                "weights": {"path": str(out), "sha256": _file_sha256(out)},
                "metadata": {"path": str(metadata_path), "sha256": _file_sha256(metadata_path)},
            },
            "smoke_only": True,
        }
        provenance["provenance_sha256"] = _canonical_sha256(provenance)
        provenance_path.write_text(json.dumps(provenance, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        written = json.loads(provenance_path.read_text(encoding="utf-8"))
        claimed = written.pop("provenance_sha256", None)
        if claimed != _canonical_sha256(written):
            raise ValueError("provenance self-digest verification failed")
    except Exception:
        _remove_partial_outputs(out)
        raise
    return DatasetTrainingExecutionRecord(
        status="training execution completed",
        smoke_only=True,
        run_id=spec.run_id,
        dataset_fingerprint=materialized.training_input_fingerprint,
        n_snapshots=len(materialized.train),
        grid=grid,
        weights_path=str(out),
        metadata_path=str(metadata_path),
        provenance_path=str(provenance_path),
    )


__all__ = ["DatasetTrainingExecutionRecord", "run_evidence_gated_field_dataset_flow_reconstruction"]
