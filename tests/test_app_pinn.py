"""Tests for :class:`tensorlbm.apps.physics_informed_lbm.PhysicsInformedLBM`.

Verifies the Physics-Informed Neural Network as an :class:`AI4SApplication`
framework instance: the inherited ``run()`` full-stack loop (produce →
register → train → serve → lineage), the ``pinn`` serving family, the five
developer-implemented methods, and the physics residual itself (which must
vanish for the exact Taylor–Green reference field).  A mocked training loop
keeps the closed-loop test fast; a couple of tests exercise the real
Adam + (data + λ·physics) loop and the serving loader on a tiny model.
"""

from __future__ import annotations

from pathlib import Path

import torch
import pytest

from tensorlbm.apps.base import DataProduct, TrainingResult
from tensorlbm.apps.physics_informed_lbm import (
    PINNMLP,
    PhysicsInformedLBM,
    _taylor_green,
    load_pinn_model,
    pde_residuals,
    save_pinn_model,
)
from tensorlbm.ml.serving import FAMILY_PINN, InferenceService, ModelRegistry
from tensorlbm.ml.training_job import TrainingJobRegistry


# ---------------------------------------------------------------------------
# Mock collaborators
# ---------------------------------------------------------------------------

def _mock_train_fn(dataset, model, cfg):
    """Return a fixed TrainingResult without running the training loop."""
    return TrainingResult(
        model_path="/tmp/mock_pinn_model.pt",
        metrics={"train_loss": 0.123, "physics_loss": 0.045, "data_loss": 0.078},
        arch={"hidden_dim": 8, "n_layers": 2, "activation": "tanh"},
    )


@pytest.fixture
def app():
    return PhysicsInformedLBM(train_fn=_mock_train_fn)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_class_identity():
    assert PhysicsInformedLBM.name == "physics_informed_lbm"
    assert PhysicsInformedLBM.family == "pinn"
    assert PhysicsInformedLBM.version == "1.0"


def test_run_closed_loop_and_lineage(app, tmp_path):
    db_path = tmp_path / "pinn_app.db"
    produce_cfg = {"n_points": 64, "n_collocation": 32, "seed": 0}
    train_cfg = {
        "arch": {"hidden_dim": 8, "n_layers": 2},
        "epochs": 1,
        "out_path": str(tmp_path / "model.pt"),
    }

    report = app.run(str(db_path), produce_cfg, train_cfg)

    # run() closed-loop identifiers
    assert report.name == "physics_informed_lbm"
    assert report.family == "pinn"
    assert report.data_asset_id == "physics_informed_lbm:flow_field"
    assert report.dataset_asset_id == "physics_informed_lbm:dataset"
    assert report.job_id.startswith("job_")
    assert report.model_id > 0
    assert report.metrics.get("train_loss") == 0.123
    assert report.metrics.get("physics_loss") == 0.045
    assert report.metrics.get("data_loss") == 0.078

    # lineage: job -> dataset -> data product (full transitive upstream)
    assert set(report.lineage_upstream) >= {
        "physics_informed_lbm:dataset",
        "physics_informed_lbm:flow_field",
    }

    # serving model registered with the pinn family
    serving = ModelRegistry.open(db_path)
    try:
        model = serving.get_model(report.model_id)
        assert model is not None
        arch = model.get("arch") or {}
        assert arch.get("family") == "pinn"
        assert arch.get("arch", {}).get("hidden_dim") == 8
    finally:
        serving.close()

    # training job reached completed with metrics recorded
    training = TrainingJobRegistry.open(db_path)
    try:
        job = training.get_job(report.job_id)
        assert job is not None
        assert job.status == "completed"
        assert (job.metrics or {}).get("train_loss") == 0.123
        assert (job.metrics or {}).get("physics_loss") == 0.045
    finally:
        training.close()


def test_make_dataset_from_product(app):
    product = app.produce_data({"n_points": 50, "n_collocation": 30, "seed": 0})
    assert product.field_name == "flow_field"

    dataset = app.make_dataset(product)
    assert dataset["points"].shape == (50, 2)
    assert dataset["labels"].shape == (50, 3)
    assert dataset["collocation"].shape == (30, 2)
    assert dataset["nu"] == 0.0
    assert len(dataset["domain"]) == 4


def test_build_model_and_infer(app):
    model = app.build_model({"hidden_dim": 8, "n_layers": 2})
    assert isinstance(model, PINNMLP)

    # single coordinate -> (3,)
    pred = app.infer(model, torch.tensor([0.5, 0.5]))
    assert pred.output.shape == (3,)
    assert pred.metadata["channels"] == ["u", "v", "p"]

    # batched coordinates -> (N, 3)
    pred = app.infer(model, torch.rand(10, 2))
    assert pred.output.shape == (10, 3)

    # (x, y) pair of 1-D tensors -> (N, 3)
    pred = app.infer(
        model,
        (torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.4, 0.5, 0.6])),
    )
    assert pred.output.shape == (3, 3)


def test_pde_residual_zero_for_taylor_green():
    """The physics residual must vanish for the exact Euler reference field."""
    class PerfectField(torch.nn.Module):
        def forward(self, xy: torch.Tensor) -> torch.Tensor:
            return _taylor_green(xy)

    coords = torch.rand(128, 2) * (2.0 * 3.141592653589793)
    cont, mom_x, mom_y = pde_residuals(PerfectField(), coords, nu=0.0)
    assert cont.abs().max().item() < 1e-4
    assert mom_x.abs().max().item() < 1e-4
    assert mom_y.abs().max().item() < 1e-4


def test_train_default_fn_writes_real_checkpoint(tmp_path):
    """Without injection, ``train`` runs the real PINN loop on CPU."""
    torch.manual_seed(0)
    app = PhysicsInformedLBM()
    product = app.produce_data({"n_points": 100, "n_collocation": 100, "seed": 0})
    dataset = app.make_dataset(product)
    model = app.build_model({"hidden_dim": 16, "n_layers": 2})

    out_path = tmp_path / "pinn_model.pt"
    result = app.train(
        dataset,
        model,
        {
            "epochs": 3,
            "learning_rate": 1e-3,
            "lambda_physics": 1.0,
            "out_path": str(out_path),
        },
    )

    assert isinstance(result, TrainingResult)
    assert result.model_path.endswith("pinn_model.pt")
    for key in ("train_loss", "physics_loss", "data_loss"):
        assert key in result.metrics
    assert result.arch.get("hidden_dim") == 16

    # checkpoint actually written and loadable by the pinn loader
    assert Path(result.model_path).exists()
    loaded = load_pinn_model(result.model_path)
    assert isinstance(loaded, PINNMLP)
    assert loaded.arch.hidden_dim == 16


def test_serving_loads_pinn_family(tmp_path):
    """InferenceService loads and predicts a model registered under ``pinn``."""
    app = PhysicsInformedLBM()
    model = app.build_model({"hidden_dim": 8, "n_layers": 2})
    path = save_pinn_model(model, tmp_path / "pinn.pt")

    registry = ModelRegistry.open(tmp_path / "serving.db")
    try:
        model_id = registry.register_model(
            name="pinn-flow",
            path=str(path),
            arch={"hidden_dim": 8, "n_layers": 2},
            family=FAMILY_PINN,
            input_shapes={"input": [None, 2]},
            output_shapes={"output": [None, 3]},
        )

        service = InferenceService(registry)
        x = torch.randn(5, 2)
        y = service.predict(model_id, x)
        assert y.shape == (5, 3)

        meta = registry.get_model_metadata(model_id)
        assert meta is not None
        assert meta.family == FAMILY_PINN
    finally:
        registry.close()
