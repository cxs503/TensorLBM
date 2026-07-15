"""Evidence-gated CPU evaluation of smoke-trained Flow Transformer artifacts.

This evaluator measures only masked held-out velocity-token reconstruction.  It is
not a CFD physical-truth, accuracy, or generality evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from io import BytesIO
from math import ceil, isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import torch

from tensorlbm.ai.transformer import FlowFieldTransformer, FlowTransformerArch, flow_snapshot_to_tokens
from tensorlbm.data import FieldDatasetR2
from tensorlbm.ml.contracts import TaskKind, TrainingBackend, TrainingSpec
from tensorlbm.ml.torch_dataset_materialize import (
    DatasetMaterializationRecord,
    FieldSnapshotReference,
    materialize_torch_field_dataset,
)

_ALLOWED_SPLITS = ("val", "test")
_ARCH_FIELDS = {"in_features", "d_model", "n_heads", "n_layers", "ffn_dim", "dropout", "max_tokens"}


@dataclass(frozen=True, slots=True)
class FlowTransformerHoldoutEvaluationRecord:
    """Immutable evidence for held-out masked velocity-token reconstruction only."""

    status: str
    model_evaluation: bool
    data_only: bool
    physical_truth_evaluation: bool
    smoke_trained_model: bool
    metric_semantics: str
    split: str
    dataset_fingerprint: str
    model_family: str
    backend: str
    arch: Mapping[str, int | float]
    weights_sha256: str
    metadata_sha256: str
    provenance_sha256: str
    selected_blob_hashes: Mapping[str, str]
    sample_ids: tuple[str, ...]
    selected_group_ids: Mapping[str, str]
    selected_case_ids: Mapping[str, str]
    selected_trajectory_ids: Mapping[str, str]
    mask_ratio: float
    mask_seed: int
    masked_token_indices: Mapping[str, tuple[int, ...]]
    masked_token_count: int
    total_token_count: int
    prediction_finite_count: int
    prediction_nonfinite_count: int
    public_metrics: Mapping[str, float]


def _canonical_sha256(document: object) -> str:
    return sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _read_json_object(data: bytes, label: str) -> dict[str, Any]:
    try:
        document = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be valid JSON") from error
    if not isinstance(document, dict):
        raise ValueError(f"{label} must be a JSON object")
    return document


def _require_nonempty_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"{label} must exist and be non-empty")


def _require_evaluator_inputs(spec: TrainingSpec, split: object, mask_ratio: object, mask_seed: object) -> tuple[str, float, int]:
    if not isinstance(spec, TrainingSpec):
        raise TypeError("spec must be a TrainingSpec")
    if spec.backend is not TrainingBackend.TORCH:
        raise ValueError("evaluation requires the TORCH backend")
    if spec.task is not TaskKind.FIELD_RECONSTRUCTION:
        raise ValueError("evaluation requires FIELD_RECONSTRUCTION")
    if split not in _ALLOWED_SPLITS:
        raise ValueError("split must be val or test; train is never eligible")
    if isinstance(mask_ratio, bool) or not isinstance(mask_ratio, (int, float)):
        raise TypeError("mask_ratio must be a finite number in (0, 1]")
    ratio = float(mask_ratio)
    if not isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise ValueError("mask_ratio must be a finite number in (0, 1]")
    if isinstance(mask_seed, bool) or not isinstance(mask_seed, int):
        raise TypeError("mask_seed must be a non-bool int")
    return split, ratio, mask_seed


def _validate_arch(metadata: Mapping[str, Any]) -> dict[str, int | float]:
    if metadata.get("family") != "flow_transformer_ssl" or metadata.get("backend") != "torch":
        raise ValueError("metadata family/backend mismatch")
    if metadata.get("format_version") != 1:
        raise ValueError("metadata format_version must be 1")
    arch = metadata.get("arch")
    if not isinstance(arch, dict) or set(arch) != _ARCH_FIELDS:
        raise ValueError("metadata arch fields do not exactly match FlowTransformerArch")
    integer_fields = _ARCH_FIELDS - {"dropout"}
    for name in integer_fields:
        if isinstance(arch[name], bool) or not isinstance(arch[name], int) or arch[name] <= 0:
            raise ValueError(f"metadata arch {name} must be a positive int")
    dropout = arch["dropout"]
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0.0 <= float(dropout) < 1.0:
        raise ValueError("metadata arch dropout must be in [0, 1)")
    if arch["in_features"] != 2 or arch["d_model"] % arch["n_heads"] != 0:
        raise ValueError("metadata arch in_features/d_model/n_heads is invalid")
    return {name: float(value) if name == "dropout" else int(value) for name, value in arch.items()}


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


def _validate_provenance(
    provenance: Mapping[str, Any], metadata: Mapping[str, Any], materialized: DatasetMaterializationRecord,
    weights_path: Path, metadata_path: Path, weights_sha: str, metadata_sha: str,
) -> str:
    if provenance.get("schema") != "tensorlbm.dataset-training-provenance.r1":
        raise ValueError("provenance schema mismatch")
    without_digest = dict(provenance)
    claimed_digest = without_digest.pop("provenance_sha256", None)
    if not isinstance(claimed_digest, str) or claimed_digest != _canonical_sha256(without_digest):
        raise ValueError("provenance self-digest verification failed")
    if provenance.get("dataset_fingerprint") != materialized.training_input_fingerprint:
        raise ValueError("provenance dataset fingerprint mismatch")
    if provenance.get("smoke_only") is not True:
        raise ValueError("provenance smoke_only must be true")
    trainer = provenance.get("trainer")
    if not isinstance(trainer, dict) or trainer.get("family") != metadata["family"] or trainer.get("backend") != metadata["backend"]:
        raise ValueError("provenance trainer mismatch")
    if trainer.get("metadata_canonical_sha256") != _canonical_sha256(metadata):
        raise ValueError("provenance metadata canonical digest mismatch")
    files = provenance.get("files")
    expected_files = {
        "weights": {"path": str(weights_path), "sha256": weights_sha},
        "metadata": {"path": str(metadata_path), "sha256": metadata_sha},
    }
    if not isinstance(files, dict) or files != expected_files:
        raise ValueError("provenance artifact hashes mismatch")
    splits = provenance.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "val", "test"}:
        raise ValueError("provenance split evidence mismatch")
    for split in ("train", "val", "test"):
        evidence = splits[split]
        records = getattr(materialized, split)
        expected = {"sample_ids": list(materialized.split_ids[split]), "count": materialized.split_counts[split], "samples": [_sample_provenance(item) for item in records]}
        if evidence != expected:
            raise ValueError(f"provenance {split} split evidence mismatch")
    return claimed_digest


def _stable_generator_seed(mask_seed: int, sample_id: str) -> int:
    digest = sha256(f"{mask_seed}:{sample_id}".encode("utf-8")).digest()
    return (mask_seed + int.from_bytes(digest[:8], "big")) % (2**63 - 1)


def _masked_indices(token_count: int, ratio: float, seed: int, sample_id: str) -> torch.Tensor:
    count = max(1, ceil(ratio * token_count))
    generator = torch.Generator(device="cpu").manual_seed(_stable_generator_seed(seed, sample_id))
    return torch.randperm(token_count, generator=generator)[:count]


def _require_cpu_model(model: Any) -> torch.device:
    devices = {value.device for value in list(model.parameters()) + list(model.buffers())}
    if len(devices) != 1:
        raise ValueError("loaded model must have exactly one device")
    device = next(iter(devices))
    if device.type != "cpu":
        raise ValueError("evaluation requires a CPU model")
    return device


def _load_verified_cpu_model(weights_bytes: bytes, arch: Mapping[str, int | float]) -> FlowFieldTransformer:
    """Build Torch model from the exact verified NPZ bytes, never a mutable path."""
    normalized_arch = {
        "in_features": int(arch["in_features"]),
        "d_model": int(arch["d_model"]),
        "n_heads": int(arch["n_heads"]),
        "n_layers": int(arch["n_layers"]),
        "ffn_dim": int(arch["ffn_dim"]),
        "dropout": float(arch["dropout"]),
        "max_tokens": int(arch["max_tokens"]),
    }
    model = FlowFieldTransformer(FlowTransformerArch(**normalized_arch))
    try:
        with BytesIO(weights_bytes) as stream:
            archive = np.load(stream, allow_pickle=False)
            current = model.state_dict()
            loaded = {
                name: torch.from_numpy(archive[name]).to(dtype=current[name].dtype)
                for name in current
            }
    except (KeyError, OSError, ValueError) as error:
        raise ValueError("verified weight bytes cannot load Flow Transformer state") from error
    model.load_state_dict(loaded)
    model.eval()
    return model


def _require_artifacts_unchanged(
    weights: Path,
    metadata_path: Path,
    provenance_path: Path,
    weights_sha: str,
    metadata_sha: str,
    provenance_file_sha: str,
) -> None:
    """Reject a path artifact that changed after its evidence preflight."""
    current = (
        _file_sha256(weights),
        _file_sha256(metadata_path),
        _file_sha256(provenance_path),
    )
    if current != (weights_sha, metadata_sha, provenance_file_sha):
        raise ValueError("artifact changed after evidence preflight")


def evaluate_evidence_gated_flow_transformer_holdout(
    spec: TrainingSpec,
    dataset: FieldDatasetR2,
    payloads: Mapping[str, Mapping[str, bytes]],
    weights_path: str | Path,
    split: str = "test",
    mask_ratio: float = 0.5,
    mask_seed: int = 0,
) -> FlowTransformerHoldoutEvaluationRecord:
    """Evaluate a verified smoke artifact only on deterministic masked val/test tokens."""
    split, ratio, seed = _require_evaluator_inputs(spec, split, mask_ratio, mask_seed)
    if not isinstance(dataset, FieldDatasetR2):
        raise TypeError("dataset must be a FieldDatasetR2")
    materialized = materialize_torch_field_dataset(spec, dataset, payloads)
    if dataset.training_input_fingerprint() != materialized.training_input_fingerprint:
        raise ValueError("dataset changed while processing")
    selected = getattr(materialized, split)
    if not selected:
        raise ValueError(f"holdout split {split!r} must be non-empty")

    weights = Path(weights_path)
    metadata_path = Path(f"{weights}.json")
    provenance_path = Path(f"{weights}.provenance.json")
    for path, label in ((weights, "weights"), (metadata_path, "metadata"), (provenance_path, "provenance")):
        _require_nonempty_file(path, label)
    weights_bytes = weights.read_bytes()
    metadata_bytes = metadata_path.read_bytes()
    provenance_bytes = provenance_path.read_bytes()
    weights_sha = sha256(weights_bytes).hexdigest()
    metadata_sha = sha256(metadata_bytes).hexdigest()
    provenance_file_sha = sha256(provenance_bytes).hexdigest()
    metadata = _read_json_object(metadata_bytes, "metadata")
    arch = _validate_arch(metadata)
    provenance = _read_json_object(provenance_bytes, "provenance")
    provenance_digest = _validate_provenance(provenance, metadata, materialized, weights, metadata_path, weights_sha, metadata_sha)

    model = _load_verified_cpu_model(weights_bytes, arch)
    device = _require_cpu_model(model)
    model.eval()
    masked_index_evidence: dict[str, tuple[int, ...]] = {}
    all_squared: list[torch.Tensor] = []
    all_absolute: list[torch.Tensor] = []
    component_squared: list[list[torch.Tensor]] = [[], []]
    component_absolute: list[list[torch.Tensor]] = [[], []]
    finite_count = 0
    nonfinite_count = 0
    total_tokens = 0
    with torch.no_grad():
        for reference in selected:
            tokens = flow_snapshot_to_tokens(*reference.snapshot).to(device=device, dtype=torch.float32)
            if int(tokens.shape[0]) > int(arch["max_tokens"]):
                raise ValueError("selected token count exceeds metadata max_tokens")
            indices = _masked_indices(int(tokens.shape[0]), ratio, seed, reference.sample_id)
            masked_index_evidence[reference.sample_id] = tuple(int(index) for index in indices.tolist())
            mask = torch.zeros((1, int(tokens.shape[0])), dtype=torch.bool, device=device)
            mask[0, indices.to(device=device)] = True
            target = tokens.unsqueeze(0)
            masked = torch.where(mask.unsqueeze(-1), model.mask_token.expand_as(target), target)
            prediction = model(masked)
            selected_prediction = prediction[mask]
            selected_target = target[mask]
            finite = torch.isfinite(selected_prediction)
            finite_count += int(finite.sum().item())
            nonfinite_count += int((~finite).sum().item())
            if nonfinite_count:
                raise ValueError("model produced non-finite masked prediction")
            difference = selected_prediction - selected_target
            squared = difference.square()
            absolute = difference.abs()
            all_squared.append(squared.reshape(-1))
            all_absolute.append(absolute.reshape(-1))
            for component in range(2):
                component_squared[component].append(squared[:, component])
                component_absolute[component].append(absolute[:, component])
            total_tokens += int(tokens.shape[0])
    if dataset.training_input_fingerprint() != materialized.training_input_fingerprint:
        raise ValueError("dataset changed while processing")
    flattened_squared = torch.cat(all_squared)
    flattened_absolute = torch.cat(all_absolute)
    metrics = {
        "masked_token_mean_squared_error": float(flattened_squared.mean().item()),
        "masked_token_mean_absolute_error": float(flattened_absolute.mean().item()),
        "masked_token_maximum_absolute_error": float(flattened_absolute.max().item()),
    }
    for component, name in enumerate(("u_x", "u_y")):
        squared = torch.cat(component_squared[component])
        absolute = torch.cat(component_absolute[component])
        metrics[f"masked_{name}_mean_squared_error"] = float(squared.mean().item())
        metrics[f"masked_{name}_mean_absolute_error"] = float(absolute.mean().item())
        metrics[f"masked_{name}_maximum_absolute_error"] = float(absolute.max().item())
    if not all(isfinite(value) for value in metrics.values()):
        raise ValueError("masked reconstruction metrics must be finite")
    return FlowTransformerHoldoutEvaluationRecord(
        status="flow transformer holdout evaluation completed",
        model_evaluation=True,
        data_only=False,
        physical_truth_evaluation=False,
        smoke_trained_model=True,
        metric_semantics="evidence-gated held-out masked velocity-token reconstruction",
        split=split,
        dataset_fingerprint=materialized.training_input_fingerprint,
        model_family="flow_transformer_ssl",
        backend="torch",
        arch=MappingProxyType(arch),
        weights_sha256=weights_sha,
        metadata_sha256=metadata_sha,
        provenance_sha256=provenance_file_sha,
        selected_blob_hashes=MappingProxyType({item.sample_id: item.field_provenance.blob_sha256 for item in selected}),
        sample_ids=tuple(item.sample_id for item in selected),
        selected_group_ids=MappingProxyType({item.sample_id: item.group_id for item in selected}),
        selected_case_ids=MappingProxyType({item.sample_id: item.source_case_id for item in selected}),
        selected_trajectory_ids=MappingProxyType({item.sample_id: item.source_trajectory_id for item in selected}),
        mask_ratio=ratio,
        mask_seed=seed,
        masked_token_indices=MappingProxyType(masked_index_evidence),
        masked_token_count=sum(len(indices) for indices in masked_index_evidence.values()),
        total_token_count=total_tokens,
        prediction_finite_count=finite_count,
        prediction_nonfinite_count=nonfinite_count,
        public_metrics=MappingProxyType(metrics),
    )


__all__ = ["FlowTransformerHoldoutEvaluationRecord", "evaluate_evidence_gated_flow_transformer_holdout"]
