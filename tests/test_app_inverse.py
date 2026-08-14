"""Tests for :class:`tensorlbm.apps.inverse_problem.InverseProblem`.

Verifies the inverse problem as an :class:`AI4SApplication` framework instance:
the inherited ``run()`` full-stack loop (produce → register → train → serve →
lineage), the ``inverse`` serving family, the five developer-implemented
methods, and — crucially — that the gradient-descent inversion actually
recovers the true physical parameters from the observed velocity field
(both a single-parameter ``nu`` inversion and a two-parameter ``nu`` +
``u_wall`` inversion).  A mocked training loop keeps the closed-loop test fast;
the real Adam inversion loop and the serving loader are exercised on tiny
CPU-only models.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tensorlbm.apps.base import DataProduct, TrainingResult
from tensorlbm.apps.inverse_problem import (
    InverseProblem,
    ParametricChannelFlow,
    load_inverse_model,
    save_inverse_model,
)
from tensorlbm.ml.serving import FAMILY_INVERSE, InferenceService, ModelRegistry
from tensorlbm.ml.training_job import TrainingJobRegistry


# ---------------------------------------------------------------------------
# Mock collaborators
# ---------------------------------------------------------------------------

def _mock_train_fn(dataset, model, cfg):
    """Return a fixed TrainingResult without running the inversion loop."""
    return TrainingResult(
        model_path="/tmp/mock_inverse_model.pt",
        metrics={
            "final_loss": 1e-6,
            "recovered_nu": 0.1,
            "recovered_u_wall": 0.3,
            "nu_error": 0.0,
            "u_wall_error": 0.0,
        },
        arch={"nu": 0.1, "u_wall": 0.3, "G": 1.0, "H": 1.0, "invert": ["nu", "u_wall"]},
    )


@pytest.fixture
def app():
    return InverseProblem(train_fn=_mock_train_fn)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_class_identity():
    assert InverseProblem.name == "inverse_problem"
    assert InverseProblem.family == "inverse"
    assert InverseProblem.version == "1.0"


def test_run_closed_loop_and_lineage(app, tmp_path):
    db_path = tmp_path / "inverse_app.db"
    produce_cfg = {"nu_true": 0.1, "u_wall_true": 0.3, "n_points": 32, "seed": 0}
    train_cfg = {
        "arch": {"nu_init": 0.5, "u_wall_init": 0.0},
        "epochs": 1,
        "out_path": str(tmp_path / "model.pt"),
    }

    report = app.run(str(db_path), produce_cfg, train_cfg)

    # run() closed-loop identifiers
    assert report.name == "inverse_problem"
    assert report.family == "inverse"
    assert report.data_asset_id == "inverse_problem:velocity_field"
    assert report.dataset_asset_id == "inverse_problem:dataset"
    assert report.job_id.startswith("job_")
    assert report.model_id > 0
    assert report.metrics.get("final_loss") == 1e-6
    assert report.metrics.get("recovered_nu") == 0.1
    assert report.metrics.get("recovered_u_wall") == 0.3

    # lineage: job -> dataset -> data product (full transitive upstream)
    assert set(report.lineage_upstream) >= {
        "inverse_problem:dataset",
        "inverse_problem:velocity_field",
    }

    # serving model registered with the inverse family
    serving = ModelRegistry.open(db_path)
    try:
        model = serving.get_model(report.model_id)
        assert model is not None
        arch = model.get("arch") or {}
        assert arch.get("family") == "inverse"
        assert arch.get("arch", {}).get("nu") == 0.1
    finally:
        serving.close()

    # training job reached completed with metrics recorded
    training = TrainingJobRegistry.open(db_path)
    try:
        job = training.get_job(report.job_id)
        assert job is not None
        assert job.status == "completed"
        assert (job.metrics or {}).get("final_loss") == 1e-6
        assert (job.metrics or {}).get("recovered_nu") == 0.1
    finally:
        training.close()


def test_make_dataset_from_product(app):
    product = app.produce_data({"nu_true": 0.1, "u_wall_true": 0.3, "n_points": 40, "seed": 0})
    assert product.field_name == "velocity_field"

    dataset = app.make_dataset(product)
    assert dataset["coords"].shape == (40, 1)
    assert dataset["observations"].shape == (40, 1)
    assert dataset["nu_true"] == 0.1
    assert dataset["u_wall_true"] == 0.3
    assert dataset["G"] == 1.0
    assert dataset["H"] == 1.0
    assert len(dataset["domain"]) == 2


def test_build_model_and_infer(app):
    model = app.build_model({"nu_init": 0.5, "u_wall_init": 0.0})
    assert isinstance(model, ParametricChannelFlow)

    # scalar -> scalar velocity
    pred = app.infer(model, torch.tensor(0.5))
    assert pred.output.shape == ()
    assert pred.metadata["nu"] == pytest.approx(0.5)

    # batched (N,) -> (N,)
    pred = app.infer(model, torch.linspace(0.0, 1.0, 7))
    assert pred.output.shape == (7,)

    # (N, 1) -> (N, 1)
    pred = app.infer(model, torch.linspace(0.0, 1.0, 5).unsqueeze(-1))
    assert pred.output.shape == (5, 1)


def test_inversion_recovers_nu():
    """Real gradient-descent inversion of a single parameter (viscosity)."""
    app = InverseProblem()
    product = app.produce_data(
        {"G": 1.0, "H": 1.0, "nu_true": 0.1, "u_wall_true": 0.0, "n_points": 64, "seed": 0}
    )
    dataset = app.make_dataset(product)
    model = app.build_model({"nu_init": 0.5, "u_wall_init": 0.0, "invert": ("nu",)})

    result = app.train(
        dataset,
        model,
        {"epochs": 300, "learning_rate": 5e-2, "out_path": "/tmp/inv_nu.pt", "seed": 0},
    )

    assert isinstance(result, TrainingResult)
    assert result.metrics["nu_error"] < 1e-3
    assert result.metrics["recovered_nu"] == pytest.approx(0.1, rel=1e-2)
    # the frozen wall velocity stays pinned at its true value
    assert result.metrics["recovered_u_wall"] == pytest.approx(0.0, abs=1e-6)


def test_inversion_recovers_two_params():
    """Real gradient-descent inversion of two parameters (nu + wall velocity)."""
    app = InverseProblem()
    product = app.produce_data(
        {"G": 1.0, "H": 1.0, "nu_true": 0.1, "u_wall_true": 0.3, "n_points": 64, "seed": 0}
    )
    dataset = app.make_dataset(product)
    model = app.build_model({"nu_init": 0.5, "u_wall_init": 0.0, "invert": ("nu", "u_wall")})

    result = app.train(
        dataset,
        model,
        {"epochs": 300, "learning_rate": 5e-2, "out_path": "/tmp/inv_both.pt", "seed": 0},
    )

    assert result.metrics["nu_error"] < 1e-3
    assert result.metrics["u_wall_error"] < 1e-3
    assert result.metrics["recovered_nu"] == pytest.approx(0.1, rel=1e-2)
    assert result.metrics["recovered_u_wall"] == pytest.approx(0.3, rel=1e-2)


def test_train_default_fn_writes_real_checkpoint(tmp_path):
    """Without injection, ``train`` runs the real inversion and saves params."""
    torch.manual_seed(0)
    app = InverseProblem()
    product = app.produce_data(
        {"G": 1.0, "H": 1.0, "nu_true": 0.1, "u_wall_true": 0.3, "n_points": 64, "seed": 0}
    )
    dataset = app.make_dataset(product)
    model = app.build_model({"nu_init": 0.5, "u_wall_init": 0.0})

    out_path = tmp_path / "inverse_model.pt"
    result = app.train(
        dataset,
        model,
        {"epochs": 300, "learning_rate": 5e-2, "out_path": str(out_path)},
    )

    assert isinstance(result, TrainingResult)
    assert result.model_path.endswith("inverse_model.pt")
    for key in ("final_loss", "recovered_nu", "recovered_u_wall", "nu_error", "u_wall_error"):
        assert key in result.metrics
    assert result.arch.get("G") == 1.0

    # checkpoint actually written and loadable by the inverse loader
    assert Path(result.model_path).exists()
    loaded = load_inverse_model(result.model_path)
    assert isinstance(loaded, ParametricChannelFlow)
    assert loaded.nu() == pytest.approx(0.1, rel=1e-2)


def test_serving_loads_inverse_family(tmp_path):
    """InferenceService loads and predicts a model registered under ``inverse``."""
    app = InverseProblem()
    model = app.build_model({"nu_init": 0.1, "u_wall_init": 0.3})
    path = save_inverse_model(model, tmp_path / "inverse.pt")

    registry = ModelRegistry.open(tmp_path / "serving.db")
    try:
        model_id = registry.register_model(
            name="inverse-flow",
            path=str(path),
            arch=model.arch_dict(),
            family=FAMILY_INVERSE,
            input_shapes={"input": [None, 1]},
            output_shapes={"output": [None, 1]},
        )

        service = InferenceService(registry)
        y = torch.linspace(0.0, 1.0, 9).unsqueeze(-1)
        out = service.predict(model_id, y)
        assert out.shape == (9, 1)

        meta = registry.get_model_metadata(model_id)
        assert meta is not None
        assert meta.family == FAMILY_INVERSE
    finally:
        registry.close()
