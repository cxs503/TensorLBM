"""TDD coverage for evidence-gated masked holdout Flow Transformer evaluation."""

from __future__ import annotations

import ast
from hashlib import sha256
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest
import torch


def _training_fixtures() -> ModuleType:
    path = Path("tests/ml/test_torch_dataset_flow_training.py")
    spec = importlib.util.spec_from_file_location("dataset_flow_training_fixtures", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trained_artifacts(tmp_path: Path):
    fixtures = _training_fixtures()
    from tensorlbm.ml.torch_dataset_flow_training import (
        run_evidence_gated_field_dataset_flow_reconstruction,
    )

    spec, dataset, payloads = fixtures._inputs()
    arch, config = fixtures._mini_arch_and_config()
    path = tmp_path / "flow-smoke.npz"
    run_evidence_gated_field_dataset_flow_reconstruction(spec, dataset, payloads, path, arch, config)
    return spec, dataset, payloads, path


def _evaluate(tmp_path: Path, **kwargs):
    from tensorlbm.ml.torch_flow_transformer_holdout_evaluation import (
        evaluate_evidence_gated_flow_transformer_holdout,
    )

    spec, dataset, payloads, path = _trained_artifacts(tmp_path)
    return evaluate_evidence_gated_flow_transformer_holdout(spec, dataset, payloads, path, **kwargs)


def test_real_smoke_trained_model_evaluates_only_test_masked_velocity_tokens(tmp_path: Path) -> None:
    record = _evaluate(tmp_path, mask_ratio=0.5, mask_seed=23)

    assert record.model_evaluation is True
    assert record.data_only is False and record.physical_truth_evaluation is False
    assert record.smoke_trained_model is True
    assert record.split == "test" and record.sample_ids == ("sample-test",)
    assert record.selected_group_ids == {"sample-test": "group-test"}
    assert record.selected_case_ids == {"sample-test": "case-test"}
    assert record.selected_trajectory_ids == {"sample-test": "trajectory-test"}
    assert record.selected_blob_hashes["sample-test"]
    assert record.mask_ratio == 0.5 and record.mask_seed == 23
    assert record.masked_token_count == 2 and record.total_token_count == 4
    assert record.metric_semantics == "evidence-gated held-out masked velocity-token reconstruction"
    assert record.model_family == "flow_transformer_ssl" and record.backend == "torch"
    assert record.arch["in_features"] == 2
    assert set(record.public_metrics) == {
        "masked_token_mean_squared_error",
        "masked_token_mean_absolute_error",
        "masked_token_maximum_absolute_error",
        "masked_u_x_mean_squared_error",
        "masked_u_y_mean_squared_error",
        "masked_u_x_mean_absolute_error",
        "masked_u_y_mean_absolute_error",
        "masked_u_x_maximum_absolute_error",
        "masked_u_y_maximum_absolute_error",
    }
    assert all(torch.isfinite(torch.tensor(value)) for value in record.public_metrics.values())
    assert record.prediction_nonfinite_count == 0


def test_mask_selection_is_deterministic_nonzero_and_val_is_allowed(tmp_path: Path) -> None:
    first = _evaluate(tmp_path / "one", mask_ratio=0.01, mask_seed=9)
    second = _evaluate(tmp_path / "two", mask_ratio=0.01, mask_seed=9)
    val = _evaluate(tmp_path / "three", split="val", mask_ratio=0.5, mask_seed=9)

    assert first.masked_token_count == second.masked_token_count == 1
    assert first.masked_token_indices == second.masked_token_indices
    assert val.sample_ids == ("sample-val",)
    assert "sample-test" not in val.sample_ids


@pytest.mark.parametrize("split", ("train", "", "TEST"))
def test_rejects_train_or_invalid_split_before_artifact_loader(tmp_path: Path, monkeypatch, split: str) -> None:
    import tensorlbm.ml.torch_flow_transformer_holdout_evaluation as evaluation

    fixtures = _training_fixtures()
    spec, dataset, payloads = fixtures._inputs()
    called: list[object] = []
    monkeypatch.setattr(evaluation, "_load_verified_cpu_model", lambda *args, **kwargs: called.append(args))
    with pytest.raises(ValueError, match="val or test"):
        evaluation.evaluate_evidence_gated_flow_transformer_holdout(spec, dataset, payloads, tmp_path / "missing.npz", split=split)
    assert called == []


@pytest.mark.parametrize("mask_ratio", (0.0, -0.1, 1.1, float("inf"), float("nan"), True))
def test_rejects_invalid_mask_ratio_before_artifact_loader(tmp_path: Path, monkeypatch, mask_ratio: float) -> None:
    import tensorlbm.ml.torch_flow_transformer_holdout_evaluation as evaluation

    fixtures = _training_fixtures()
    spec, dataset, payloads = fixtures._inputs()
    monkeypatch.setattr(evaluation, "_load_verified_cpu_model", lambda *args, **kwargs: pytest.fail("loader called"))
    with pytest.raises((TypeError, ValueError), match="mask_ratio"):
        evaluation.evaluate_evidence_gated_flow_transformer_holdout(spec, dataset, payloads, tmp_path / "missing.npz", mask_ratio=mask_ratio)


@pytest.mark.parametrize("mask_seed", (True, 1.2, "0"))
def test_rejects_invalid_mask_seed_before_artifact_loader(tmp_path: Path, monkeypatch, mask_seed: object) -> None:
    import tensorlbm.ml.torch_flow_transformer_holdout_evaluation as evaluation

    fixtures = _training_fixtures()
    spec, dataset, payloads = fixtures._inputs()
    monkeypatch.setattr(evaluation, "_load_verified_cpu_model", lambda *args, **kwargs: pytest.fail("loader called"))
    with pytest.raises((TypeError, ValueError), match="mask_seed"):
        evaluation.evaluate_evidence_gated_flow_transformer_holdout(spec, dataset, payloads, tmp_path / "missing.npz", mask_seed=mask_seed)


@pytest.mark.parametrize("target", ("weights", "metadata", "provenance", "dataset_fingerprint", "split_ids", "blob_sha"))
def test_tampered_artifacts_reject_before_loader(tmp_path: Path, monkeypatch, target: str) -> None:
    import tensorlbm.ml.torch_flow_transformer_holdout_evaluation as evaluation

    spec, dataset, payloads, path = _trained_artifacts(tmp_path)
    metadata_path = Path(f"{path}.json")
    provenance_path = Path(f"{path}.provenance.json")
    if target == "weights":
        path.write_bytes(path.read_bytes() + b"tamper")
    elif target == "metadata":
        metadata_path.write_text("{}", encoding="utf-8")
    else:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance.pop("provenance_sha256")
        if target == "provenance":
            provenance["smoke_only"] = False
        elif target == "dataset_fingerprint":
            provenance["dataset_fingerprint"] = "0" * 64
        elif target == "split_ids":
            provenance["splits"]["test"]["sample_ids"] = ["sample-val"]
        else:
            provenance["splits"]["test"]["samples"][0]["field_provenance"]["blob_sha256"] = "0" * 64
        provenance["provenance_sha256"] = sha256(
            json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    monkeypatch.setattr(evaluation, "_load_verified_cpu_model", lambda *args, **kwargs: pytest.fail("loader called"))
    with pytest.raises(ValueError):
        evaluation.evaluate_evidence_gated_flow_transformer_holdout(spec, dataset, payloads, path)



def test_artifact_swap_during_loader_is_rejected_after_load(tmp_path: Path, monkeypatch) -> None:
    import tensorlbm.ml.torch_flow_transformer_holdout_evaluation as evaluation

    spec, dataset, payloads, path = _trained_artifacts(tmp_path)
    real_loader = evaluation._load_verified_cpu_model

    original = path.read_bytes()

    def swapping_loader(weights_bytes, arch):
        path.write_bytes(original + b"swap")
        return real_loader(weights_bytes, arch)

    monkeypatch.setattr(evaluation, "_load_verified_cpu_model", swapping_loader)
    record = evaluation.evaluate_evidence_gated_flow_transformer_holdout(spec, dataset, payloads, path)
    assert record.weights_sha256 == sha256(original).hexdigest()


def test_cpu_model_required_nonfinite_prediction_and_toctou_fail_closed(tmp_path: Path, monkeypatch) -> None:
    import tensorlbm.ml.torch_flow_transformer_holdout_evaluation as evaluation

    spec, dataset, payloads, path = _trained_artifacts(tmp_path)
    real_loader = evaluation._load_verified_cpu_model
    model = real_loader(path.read_bytes(), json.loads(Path(f"{path}.json").read_text(encoding="utf-8"))["arch"])
    assert next(model.parameters()).device.type == "cpu"

    class NonfiniteModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, 2))

        def forward(self, value):
            return torch.full_like(value, float("nan"))

    monkeypatch.setattr(evaluation, "_load_verified_cpu_model", lambda *args, **kwargs: NonfiniteModel())
    with pytest.raises(ValueError, match="non-finite"):
        evaluation.evaluate_evidence_gated_flow_transformer_holdout(spec, dataset, payloads, path)

    monkeypatch.setattr(evaluation, "_load_verified_cpu_model", real_loader)
    real_fingerprint = type(dataset).training_input_fingerprint
    calls = 0

    def changing_fingerprint(instance):
        nonlocal calls
        calls += 1
        return real_fingerprint(instance) if calls == 1 else "changed"

    monkeypatch.setattr(type(dataset), "training_input_fingerprint", changing_fingerprint)
    with pytest.raises(ValueError, match="dataset changed"):
        evaluation.evaluate_evidence_gated_flow_transformer_holdout(spec, dataset, payloads, path)


def test_boundary_ast_forbids_trainer_writer_endpoint_and_full_field_reconstruction() -> None:
    source = Path("src/tensorlbm/ml/torch_flow_transformer_holdout_evaluation.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert {"materialize_torch_field_dataset", "_load_verified_cpu_model"} <= names
    forbidden = {"train_flow_transformer_self_supervised", "save_flow_transformer_model", "reconstruct_flow_field", "write_text", "write_bytes", "optimizer", "backward"}
    assert not names & forbidden
    assert "torch" in source
    assert "torch.Generator(device=\"cpu\")" in source
