"""Tests for :class:`tensorlbm.apps.mesh_gnn_flow.MeshGNNFlow`.

Verifies the MeshGraphNet-style message-passing GNN as an
:class:`AI4SApplication` framework instance: the inherited ``run()`` full-stack
loop (produce → register → train → serve → lineage), the ``gnn`` serving
family, and the five developer-implemented methods.  A mocked training loop
(``train_fn``) keeps the closed-loop test fast and deterministic; a second test
exercises the real Adam+MSE loop on a tiny graph, and a third proves the
serving layer loads a ``gnn`` model end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import torch
import pytest

from tensorlbm.apps.base import TrainingResult
from tensorlbm.apps.mesh_gnn_flow import (
    MeshGNNFlow,
    MeshGraphNet,
    load_mesh_gnn,
    save_mesh_gnn,
)
from tensorlbm.ml.serving import FAMILY_GNN, InferenceService, ModelRegistry
from tensorlbm.ml.training_job import TrainingJobRegistry


# ---------------------------------------------------------------------------
# Mock collaborators
# ---------------------------------------------------------------------------

def _mock_train_fn(dataset, model, cfg):
    """Save the (already built) model and return a fixed TrainingResult."""
    out_path = cfg.get("out_path", "/tmp/mock_mesh_gnn.pt")
    path = save_mesh_gnn(model, out_path)
    return TrainingResult(
        model_path=str(path),
        metrics={"train_loss": 0.0123},
        arch={
            "node_dim": 2,
            "edge_dim": 5,
            "out_dim": 2,
            "hidden_dim": 16,
            "n_layers": 2,
        },
    )


@pytest.fixture
def app():
    return MeshGNNFlow(train_fn=_mock_train_fn)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_class_identity():
    assert MeshGNNFlow.name == "mesh_gnn_flow"
    assert MeshGNNFlow.family == "gnn"
    assert MeshGNNFlow.version == "1.0"


def test_run_closed_loop_and_lineage(app, tmp_path):
    db_path = tmp_path / "gnn_app.db"
    produce_cfg = {"grid_size": 8, "n_graphs": 2, "seed": 0}
    train_cfg = {
        "arch": {"node_dim": 2, "edge_dim": 5, "out_dim": 2, "hidden_dim": 16, "n_layers": 2},
        "epochs": 1,
        "out_path": str(tmp_path / "model.pt"),
    }

    report = app.run(str(db_path), produce_cfg, train_cfg)

    # run() closed-loop identifiers
    assert report.name == "mesh_gnn_flow"
    assert report.family == "gnn"
    assert report.data_asset_id == "mesh_gnn_flow:u"
    assert report.dataset_asset_id == "mesh_gnn_flow:dataset"
    assert report.job_id.startswith("job_")
    assert report.model_id > 0
    assert report.metrics.get("train_loss") == 0.0123

    # lineage: job -> dataset -> data product (full transitive upstream)
    assert set(report.lineage_upstream) >= {
        "mesh_gnn_flow:dataset",
        "mesh_gnn_flow:u",
    }

    # serving model registered under the gnn family
    serving = ModelRegistry.open(db_path)
    try:
        model = serving.get_model(report.model_id)
        assert model is not None
        arch = model.get("arch") or {}
        assert arch.get("family") == "gnn"
        # raw arch hyper-parameters preserved alongside the family
        assert arch.get("arch", {}).get("hidden_dim") == 16
    finally:
        serving.close()

    # training job reached completed with metrics recorded
    training = TrainingJobRegistry.open(db_path)
    try:
        job = training.get_job(report.job_id)
        assert job is not None
        assert job.status == "completed"
        assert (job.metrics or {}).get("train_loss") == 0.0123
    finally:
        training.close()


def test_make_dataset_shapes():
    app = MeshGNNFlow()
    product = app.produce_data({"grid_size": 8, "n_graphs": 3})
    assert product.metadata["n_nodes"] == 64
    assert product.metadata["n_edges"] == 224  # 4 * gs * (gs - 1)

    dataset = app.make_dataset(product)
    assert dataset["n_samples"] == 3
    assert dataset["node_dim"] == 2
    assert dataset["edge_dim"] == 5
    assert dataset["out_dim"] == 2

    g = dataset["graphs"][0]
    assert g["x"].shape == (64, 2)
    assert g["y"].shape == (64, 2)
    assert g["pos"].shape == (64, 2)
    assert g["edge_index"].shape == (2, 224)
    assert g["edge_attr"].shape == (224, 5)


def test_build_model_and_infer(app):
    model = app.build_model(
        {"node_dim": 2, "edge_dim": 5, "out_dim": 2, "hidden_dim": 16, "n_layers": 2},
    )
    assert isinstance(model, MeshGraphNet)
    assert model.n_layers == 2
    assert model.hidden_dim == 16

    # a graph dict sample
    product = app.produce_data({"grid_size": 8, "n_graphs": 1})
    g = app.make_dataset(product)["graphs"][0]
    pred = app.infer(model, g)
    assert pred.output.shape == (64, 2)
    assert pred.metadata["field_name"] == "u"

    # a (x, edge_index, edge_attr) tuple sample is also accepted
    pred2 = app.infer(model, (g["x"], g["edge_index"], g["edge_attr"]))
    assert pred2.output.shape == (64, 2)

    # edge_attr may be omitted (defaults to zeros)
    pred3 = app.infer(model, {"x": g["x"], "edge_index": g["edge_index"]})
    assert pred3.output.shape == (64, 2)


def test_train_default_fn_writes_real_checkpoint(tmp_path):
    """Without injection, ``train`` runs the real Adam+MSE loop on CPU."""
    torch.manual_seed(0)
    app = MeshGNNFlow()
    dataset = app.make_dataset(
        app.produce_data({"grid_size": 6, "n_graphs": 3}),
    )
    model = app.build_model(
        {"node_dim": 2, "edge_dim": 5, "out_dim": 2, "hidden_dim": 8, "n_layers": 2},
    )
    out_path = tmp_path / "gnn_model.pt"
    result = app.train(
        dataset,
        model,
        {"epochs": 2, "batch_size": 2, "learning_rate": 1e-3, "out_path": str(out_path)},
    )

    assert isinstance(result, TrainingResult)
    assert result.model_path.endswith("gnn_model.pt")
    assert "train_loss" in result.metrics
    assert float(result.metrics["train_loss"]) >= 0.0
    assert result.arch.get("hidden_dim") == 8

    # checkpoint actually written and loadable
    assert Path(result.model_path).exists()
    loaded = load_mesh_gnn(result.model_path)
    assert isinstance(loaded, MeshGraphNet)
    assert loaded.hidden_dim == 8


def test_serving_loads_gnn_family(tmp_path):
    """InferenceService loads and predicts a model registered under ``gnn``."""
    app = MeshGNNFlow()
    model = app.build_model(
        {"node_dim": 2, "edge_dim": 5, "out_dim": 2, "hidden_dim": 8, "n_layers": 2},
    )
    assert isinstance(model, MeshGraphNet)
    path = save_mesh_gnn(model, tmp_path / "gnn.pt")

    product = app.produce_data({"grid_size": 8, "n_graphs": 1})
    g = app.make_dataset(product)["graphs"][0]

    registry = ModelRegistry.open(tmp_path / "serving.db")
    try:
        model_id = registry.register_model(
            name="mesh-gnn",
            path=str(path),
            arch={"node_dim": 2, "edge_dim": 5, "out_dim": 2, "hidden_dim": 8, "n_layers": 2},
            family=FAMILY_GNN,
            input_shapes={"x": [None, 2], "edge_index": [2, None], "edge_attr": [None, 5]},
            output_shapes={"output": [None, 2]},
        )

        service = InferenceService(registry)
        y = service.predict(model_id, (g["x"], g["edge_index"], g["edge_attr"]))
        assert y.shape == (64, 2)

        meta = registry.get_model_metadata(model_id)
        assert meta is not None
        assert meta.family == FAMILY_GNN
    finally:
        registry.close()
