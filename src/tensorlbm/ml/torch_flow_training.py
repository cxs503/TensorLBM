"""Evidence-gated CPU Torch flow-reconstruction smoke-training execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

from tensorlbm.ai.transformer import (
    FlowTransformerArch,
    FlowTransformerTrainConfig,
    train_flow_transformer_self_supervised,
)
from tensorlbm.data.field_r2 import FieldDataProductR2
from tensorlbm.ml.contracts import TaskKind, TrainingBackend, TrainingSpec
from tensorlbm.ml.torch_materialize import materialize_torch_velocity_snapshots


@dataclass(frozen=True, slots=True)
class TrainingExecutionRecord:
    """Completed, file-verified smoke-training evidence; never a ModelArtifact."""

    status: str
    smoke_only: bool
    run_id: str
    n_snapshots: int
    grid: tuple[int, int]
    weights_path: str
    metadata_path: str
    provenance_path: str


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(document: dict[str, object]) -> str:
    return sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _output_paths(out: Path) -> tuple[Path, Path, Path]:
    return out, Path(f"{out}.json"), Path(f"{out}.provenance.json")


def _require_new_output(out: Path) -> None:
    if any(candidate.exists() for candidate in _output_paths(out)):
        raise FileExistsError("output, metadata, and provenance paths must all be new")
    out.parent.mkdir(parents=True, exist_ok=True)


def _remove_incomplete_outputs(out: Path) -> None:
    for candidate in _output_paths(out):
        if candidate.exists():
            candidate.unlink()


def _verified_metadata(path: Path, n_snapshots: int, grid: tuple[int, int]) -> dict[str, object]:
    _, metadata_path, _ = _output_paths(path)
    if not path.is_file() or path.stat().st_size == 0:
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
    if metadata.get("n_snapshots") != n_snapshots or metadata.get("grid") != list(grid):
        raise ValueError("trainer metadata snapshot/grid mismatch")
    return metadata


def run_evidence_gated_flow_reconstruction(
    spec: TrainingSpec,
    product: FieldDataProductR2,
    payloads: Mapping[str, bytes],
    out_path: str | Path,
    arch: FlowTransformerArch | None = None,
    config: FlowTransformerTrainConfig | None = None,
) -> TrainingExecutionRecord:
    """Materialize verified field bytes, run the existing Torch trainer, then attest files."""
    if spec.backend is not TrainingBackend.TORCH:
        raise ValueError("execution requires the TORCH backend")
    if spec.task is not TaskKind.FIELD_RECONSTRUCTION:
        raise ValueError("execution requires FIELD_RECONSTRUCTION")
    out = Path(out_path)
    _require_new_output(out)
    effective_config = config if config is not None else FlowTransformerTrainConfig()
    if effective_config.device.lower() != "cpu":
        raise ValueError("evidence-gated smoke execution requires device='cpu'")
    snapshots, field = materialize_torch_velocity_snapshots(spec, product, payloads)
    n_snapshots = len(snapshots)
    grid = (int(snapshots[0][0].shape[0]), int(snapshots[0][0].shape[1]))
    _, metadata_path, provenance_path = _output_paths(out)
    try:
        metadata = train_flow_transformer_self_supervised(
            list(snapshots), out, arch, effective_config, backend="torch"
        )
        if not isinstance(metadata, dict):
            raise ValueError("trainer returned invalid metadata")
        if metadata.get("n_snapshots") != n_snapshots or metadata.get("grid") != list(grid):
            raise ValueError("trainer return snapshot/grid mismatch")
        saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(saved_metadata, dict):
            raise ValueError("trainer metadata must be a JSON object")
        saved_metadata.update({"n_snapshots": n_snapshots, "grid": list(grid)})
        metadata_path.write_text(json.dumps(saved_metadata, sort_keys=True), encoding="utf-8")
        verified_metadata = _verified_metadata(out, n_snapshots, grid)
        metadata_config = metadata.get("config")
        if not isinstance(metadata_config, dict):
            metadata_config = asdict(effective_config)
        provenance: dict[str, object] = {
            "schema": "tensorlbm.training-provenance.r1",
            "training_spec": {"run_id": spec.run_id},
            "dataset": {"id": spec.dataset.dataset_id, "version": spec.dataset.version},
            "field_data": {
                "product_id": field.product_id,
                "run_id": field.run_id,
                "array_id": field.array_id,
                "blob_sha256": field.blob_sha256,
                "shape": list(field.shape),
                "dtype": field.dtype,
                "units": field.units,
                "order": field.order,
                "components": list(field.component_labels),
            },
            "trainer": {
                "family": verified_metadata["family"],
                "backend": verified_metadata["backend"],
                "n_snapshots": n_snapshots,
                "grid": list(grid),
                "metadata_sha256": _canonical_sha256(verified_metadata),
                "config": metadata_config,
            },
            "files": {
                "weights": {"path": str(out), "sha256": _sha256_file(out)},
                "metadata": {"path": str(metadata_path), "sha256": _sha256_file(metadata_path)},
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
        _remove_incomplete_outputs(out)
        raise
    return TrainingExecutionRecord(
        status="training execution completed",
        smoke_only=True,
        run_id=spec.run_id,
        n_snapshots=n_snapshots,
        grid=grid,
        weights_path=str(out),
        metadata_path=str(metadata_path),
        provenance_path=str(provenance_path),
    )


__all__ = ["TrainingExecutionRecord", "run_evidence_gated_flow_reconstruction"]
