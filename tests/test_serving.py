"""Tests for the model serving layer (clean-room reimplementation)."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from tensorlbm.ai.model import EddyViscosityMLP, ModelArch, save_model
from tensorlbm.ai.transformer import (
    FlowFieldTransformer,
    FlowTransformerArch,
    save_flow_transformer_model,
)
from tensorlbm.ml.serving import (
    FAMILY_EDDY_MLP,
    FAMILY_FLOW_TRANSFORMER,
    InferenceService,
    ModelMetadata,
    ModelNotFoundError,
    ModelRegistry,
    OnnxUnavailableError,
    export_onnx,
    infer_io_shapes,
    onnx_available,
)


@pytest.fixture
def registry(tmp_path):
    reg = ModelRegistry.open(tmp_path / "serving.db")
    yield reg
    reg.close()


def _save_mlp(tmp_path: Path) -> Path:
    model = EddyViscosityMLP(
        ModelArch(in_features=3, hidden_features=8, n_hidden_layers=1)
    )
    return save_model(model, tmp_path / "eddy.pt")


def _register_mlp(registry: ModelRegistry, path: Path, **kwargs: Any) -> int:
    params: dict[str, Any] = {
        "name": "eddy-mlp",
        "path": str(path),
        "arch": {"in_features": 3, "hidden_features": 8, "n_hidden_layers": 1},
        "input_shapes": {"input": [None, 3]},
        "output_shapes": {"output": [None, 1]},
        "lineage": {"training_run_id": "run-1"},
        "version": "1",
        "family": FAMILY_EDDY_MLP,
    }
    params.update(kwargs)
    return registry.register_model(**params)


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------

def test_register_and_get_model(registry, tmp_path):
    path = _save_mlp(tmp_path)
    model_id = _register_mlp(registry, path, metrics={"final_val_loss": 0.001})

    record = registry.get_model(model_id)
    assert record is not None
    assert record["name"] == "eddy-mlp"
    assert record["path"] == str(path)
    assert record["metrics"] == {"final_val_loss": 0.001}


def test_get_missing_model_returns_none(registry):
    assert registry.get_model(9999) is None
    assert registry.get_model_metadata(9999) is None


def test_list_models_newest_first(registry, tmp_path):
    path = _save_mlp(tmp_path)
    first = _register_mlp(registry, path, name="first")
    second = _register_mlp(registry, path, name="second")

    rows = registry.list_models()
    ids = [r["id"] for r in rows]
    assert ids[0] == second and ids[1] == first


def test_register_model_metadata_roundtrip(registry, tmp_path):
    path = _save_mlp(tmp_path)
    model_id = _register_mlp(
        registry,
        path,
        lineage={"dataset_id": 7, "training_run_id": "run-42"},
        version="3",
    )

    meta = registry.get_model_metadata(model_id)
    assert isinstance(meta, ModelMetadata)
    assert meta.version == "3"
    assert meta.family == FAMILY_EDDY_MLP
    assert meta.framework == "torch"
    assert meta.input_shapes == {"input": [None, 3]}
    assert meta.output_shapes == {"output": [None, 1]}
    assert meta.lineage["training_run_id"] == "run-42"

    # The descriptor survives a full reload from the database.
    record = registry.get_model(model_id)
    assert ModelMetadata.from_model_record(record) == meta


# ---------------------------------------------------------------------------
# InferenceService
# ---------------------------------------------------------------------------

def test_inference_service_load_and_predict(registry, tmp_path):
    path = _save_mlp(tmp_path)
    model_id = _register_mlp(registry, path)

    service = InferenceService(registry)
    x = np.random.randn(8, 3).astype(np.float32)
    y = service.predict(model_id, x)

    assert isinstance(y, np.ndarray)
    assert y.dtype == np.float32
    assert y.shape == (8, 1)
    # Softplus output guarantees ν_t >= 0.
    assert np.all(y >= 0.0)


def test_inference_service_accepts_tensor_and_caches(registry, tmp_path):
    path = _save_mlp(tmp_path)
    model_id = _register_mlp(registry, path)

    service = InferenceService(registry)
    x = torch.randn(4, 3)

    model_a = service.load_model(model_id)
    model_b = service.load_model(model_id)
    assert model_a is model_b  # cached

    y = service.predict(model_id, x)
    assert y.shape == (4, 1)
    assert service.list_loaded() == [model_id]

    service.unload_model(model_id)
    assert service.list_loaded() == []
    # Reloading from disk creates a fresh object but still predicts correctly.
    model_c = service.load_model(model_id)
    assert model_c is not model_a
    assert service.predict(model_id, x).shape == (4, 1)


def test_inference_service_missing_model(registry):
    service = InferenceService(registry)
    with pytest.raises(ModelNotFoundError):
        service.load_model(9999)


def test_inference_service_transformer_family(registry, tmp_path):
    arch = FlowTransformerArch(
        in_features=2, d_model=8, n_heads=2, n_layers=1, ffn_dim=16, max_tokens=64
    )
    model = FlowFieldTransformer(arch)
    path = save_flow_transformer_model(model, tmp_path / "flow.pt")

    model_id = registry.register_model(
        name="flow-transformer",
        path=str(path),
        arch={"in_features": 2, "d_model": 8, "n_heads": 2, "n_layers": 1},
        family=FAMILY_FLOW_TRANSFORMER,
        input_shapes={"input": [None, 32, 2]},
        output_shapes={"output": [None, 32, 2]},
    )

    service = InferenceService(registry)
    x = torch.randn(2, 32, 2)
    y = service.predict(model_id, x)
    assert y.shape == (2, 32, 2)

    meta = registry.get_model_metadata(model_id)
    assert meta.family == FAMILY_FLOW_TRANSFORMER


def test_inference_service_unsupported_family(registry, tmp_path):
    path = _save_mlp(tmp_path)
    model_id = registry.register_model(
        name="weird",
        path=str(path),
        arch={},
        family="some_unknown_family",
    )
    service = InferenceService(registry)
    with pytest.raises(ValueError):
        service.load_model(model_id)


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def test_infer_io_shapes(tmp_path):
    model = EddyViscosityMLP(ModelArch(in_features=3, hidden_features=8, n_hidden_layers=1))
    in_shapes, out_shapes = infer_io_shapes(model, torch.randn(5, 3))
    assert in_shapes == {"input_0": [None, 3]}
    assert out_shapes == {"output_0": [None, 1]}


def test_export_onnx_smoke(tmp_path):
    model = EddyViscosityMLP(ModelArch(in_features=3, hidden_features=8, n_hidden_layers=1))
    out = tmp_path / "eddy.onnx"

    if onnx_available():
        exported = export_onnx(
            model,
            torch.randn(2, 3),
            out,
            input_names=["input"],
            output_names=["output"],
        )
        assert Path(exported).exists()
        assert Path(exported).stat().st_size > 0
    else:
        # Without the optional dependency the export path must fail loudly
        # rather than fabricate a file.
        with pytest.raises(OnnxUnavailableError):
            export_onnx(model, torch.randn(2, 3), out)
        assert not out.exists()


def test_export_onnx_rejects_non_module(tmp_path):
    with pytest.raises(TypeError):
        export_onnx("not a module", torch.randn(2, 3), tmp_path / "x.onnx")
