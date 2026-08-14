"""Tests for :class:`tensorlbm.apps.flow_transformer_app.FlowTransformerApp`.

Verifies the Flow Transformer refactor into an ``AI4SApplication`` framework
instance: the inherited ``run()`` full-stack loop (produce -> register ->
train -> serve -> lineage), the ``flow_transformer_ssl`` serving family, and
the five developer-implemented methods.
"""

from __future__ import annotations

import torch
import pytest

from tensorlbm.ai.transformer import FlowFieldTransformer
from tensorlbm.apps.base import DataProduct, Prediction, TrainingResult
from tensorlbm.apps.flow_transformer_app import FlowTransformerApp


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _fake_train(snapshots, out_path, arch=None, config=None):
    """Mock of ``train_flow_transformer_self_supervised`` for fast tests."""
    return {
        "path": str(out_path),
        "family": "flow_transformer_ssl",
        "arch": {"in_features": 2, "d_model": 32, "n_layers": 2},
        "config": {"epochs": 2, "batch_size": 4},
        "backend": "torch",
        "n_snapshots": len(snapshots),
        "n_tokens": 64,
        "grid": [8, 8],
        "final_train_loss": 0.41,
        "final_val_loss": 0.45,
    }


def _snapshots(n=4, shape=(8, 8), seed=0):
    g = torch.Generator().manual_seed(seed)
    return [
        (torch.rand(shape, generator=g), torch.rand(shape, generator=g))
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# run() full-stack loop
# ---------------------------------------------------------------------------

def test_run_full_loop(tmp_path):
    app = FlowTransformerApp(train_fn=_fake_train)
    db_path = tmp_path / "flowtf.db"
    produce_cfg = {"snapshots": _snapshots()}
    train_cfg = {
        "out_path": str(tmp_path / "flow_transformer.pt"),
        "arch": {"d_model": 32, "n_layers": 2},
    }

    report = app.run(db_path, produce_cfg, train_cfg, name_prefix="flow_tf")

    # identity / family
    assert report.name == "flow_transformer"
    assert report.family == "flow_transformer_ssl"

    # platform identifiers
    assert report.data_asset_id == "flow_tf:u"
    assert report.dataset_asset_id == "flow_tf:dataset"
    assert report.job_id.startswith("job_")
    assert report.model_id > 0

    # metrics recorded from TrainingResult
    assert report.metrics == {
        "final_train_loss": 0.41,
        "final_val_loss": 0.45,
    }

    # lineage: data -> dataset -> job (transitive upstream)
    upstream = set(report.lineage_upstream)
    assert report.data_asset_id in upstream
    assert report.dataset_asset_id in upstream

    # training job reached completed status
    from tensorlbm.ml.training_job import TrainingJobRegistry
    training = TrainingJobRegistry.open(db_path)
    try:
        job = training.get_job(report.job_id)
        assert job is not None
        assert job.status == "completed"
        assert job.metrics.get("final_train_loss") == 0.41
    finally:
        training.close()

    # serving registered the model under the supported family
    from tensorlbm.ml.serving import ModelRegistry
    serving = ModelRegistry.open(db_path)
    try:
        model = serving.get_model(report.model_id)
        assert model is not None
        arch = (model or {}).get("arch") or {}
        assert arch.get("family") == "flow_transformer_ssl"
    finally:
        serving.close()


def test_run_with_generated_data(tmp_path):
    """produce_data generates random snapshots when cfg has none."""
    app = FlowTransformerApp(train_fn=_fake_train)
    db_path = tmp_path / "gen.db"
    produce_cfg = {"n_snapshots": 6, "grid": (4, 4), "seed": 0}
    train_cfg = {"out_path": str(tmp_path / "model.pt")}

    report = app.run(db_path, produce_cfg, train_cfg)

    assert report.data_asset_id == "flow_transformer:u"
    assert report.family == "flow_transformer_ssl"
    assert "flow_transformer:dataset" in report.lineage_upstream


# ---------------------------------------------------------------------------
# Individual developer-implemented methods
# ---------------------------------------------------------------------------

def test_produce_data_returns_dataproduct():
    app = FlowTransformerApp()
    product = app.produce_data({"snapshots": _snapshots(3)})
    assert isinstance(product, DataProduct)
    assert product.field_name == "u"
    assert product.shape == (8, 8)
    assert product.metadata["n_snapshots"] == 3


def test_produce_data_rejects_empty():
    app = FlowTransformerApp()
    with pytest.raises(ValueError):
        app.produce_data({"snapshots": []})


def test_build_model():
    app = FlowTransformerApp()
    model = app.build_model({"d_model": 16, "n_layers": 1, "max_tokens": 64})
    assert isinstance(model, FlowFieldTransformer)
    assert model.arch.d_model == 16
    assert model.arch.n_layers == 1


def test_make_dataset_shapes():
    app = FlowTransformerApp()
    product = app.produce_data({"snapshots": _snapshots(4, shape=(8, 8))})
    dataset = app.make_dataset(product)
    assert dataset["batch"].shape == (4, 64, 2)
    assert dataset["grid"] == (8, 8)


def test_train_returns_trainingresult(tmp_path):
    app = FlowTransformerApp(train_fn=_fake_train)
    product = app.produce_data({"snapshots": _snapshots(4)})
    dataset = app.make_dataset(product)
    model = app.build_model({"d_model": 32, "n_layers": 2})
    cfg = {"out_path": str(tmp_path / "m.pt"), "epochs": 2}

    result = app.train(dataset, model, cfg)

    assert isinstance(result, TrainingResult)
    assert result.model_path == str(tmp_path / "m.pt")
    assert result.metrics["final_train_loss"] == 0.41
    assert result.metrics["final_val_loss"] == 0.45
    assert result.arch.get("d_model") == 32


def test_infer_reconstructs():
    app = FlowTransformerApp()
    model = app.build_model({"d_model": 16, "n_layers": 1, "max_tokens": 64})
    ux, uy = torch.rand(8, 8), torch.rand(8, 8)

    pred = app.infer(model, (ux, uy))

    assert isinstance(pred, Prediction)
    ux_rec, uy_rec = pred.output
    assert ux_rec.shape == (8, 8)
    assert uy_rec.shape == (8, 8)
    assert "mse" in pred.metadata
    assert "max_abs_error" in pred.metadata


def test_default_train_fn_is_real(tmp_path):
    """Without injection, the real self-supervised trainer is used."""
    app = FlowTransformerApp()
    assert app._train_fn.__module__.startswith("tensorlbm.ai.transformer")
