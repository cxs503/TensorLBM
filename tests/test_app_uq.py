"""Tests for :class:`tensorlbm.apps.uncertainty_quantification.UncertaintyQuantification`.

Verifies the MC-dropout UQ application as an :class:`AI4SApplication` framework
instance: the inherited ``run()`` full-stack loop (produce → register → train →
serve → lineage), the ``uq`` serving family, and the five developer-implemented
methods.  A mocked velocity field (``run_les_fn``) and a mocked training loop
(``train_fn``) keep the closed-loop test fast and deterministic; other tests
exercise the real Adam+MSE loop and the real MC-dropout sampling on tiny data.
"""

from __future__ import annotations

from pathlib import Path

import torch
import pytest

from tensorlbm.apps.base import DataProduct, TrainingResult
from tensorlbm.apps.uncertainty_quantification import (
    MCDropoutMLP,
    UncertaintyQuantification,
    load_uq_mlp,
)
from tensorlbm.ml.serving import FAMILY_UQ, InferenceService, ModelRegistry
from tensorlbm.ml.training_job import TrainingJobRegistry


# ---------------------------------------------------------------------------
# Mock collaborators
# ---------------------------------------------------------------------------

def _mock_velocity_snapshots(nx: int, ny: int, seed: int):
    """Deterministic turbulent-looking velocity fields (no solver run)."""
    torch.manual_seed(seed)
    ux = 0.05 + 0.02 * torch.randn(ny, nx)
    uy = 0.02 * torch.randn(ny, nx)
    return [(ux, uy), (0.5 * ux, 0.5 * uy)]


def _mock_run_les(nx, ny, tau, c_s, n_steps, sample_every, seed, device):
    """Drop-in replacement for ``_run_les_smoke`` returning fixed snapshots."""
    return _mock_velocity_snapshots(int(nx), int(ny), int(seed))


def _mock_train_fn(dataset, model, cfg):
    """Return a fixed TrainingResult without running the training loop."""
    return TrainingResult(
        model_path="/tmp/mock_uq_mlp_model.pt",
        metrics={"train_loss": 0.0042},
        arch={
            "in_features": 3,
            "hidden_features": 16,
            "n_hidden_layers": 2,
            "dropout_p": 0.1,
            "activation": "gelu",
            "out_features": 1,
        },
    )


@pytest.fixture
def app():
    return UncertaintyQuantification(
        train_fn=_mock_train_fn,
        run_les_fn=_mock_run_les,
        n_mc_samples=20,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_class_identity():
    assert UncertaintyQuantification.name == "uncertainty_quantification"
    assert UncertaintyQuantification.family == "uq"
    assert UncertaintyQuantification.version == "1.0"


def test_run_closed_loop_and_lineage(app, tmp_path):
    db_path = tmp_path / "uq_app.db"
    produce_cfg = {"nx": 16, "ny": 16, "n_steps": 4, "sample_every": 2, "seed": 0}
    train_cfg = {
        "arch": {"in_features": 3, "hidden_features": 16, "n_hidden_layers": 2},
        "epochs": 1,
        "out_path": str(tmp_path / "model.pt"),
    }

    report = app.run(str(db_path), produce_cfg, train_cfg)

    # run() closed-loop identifiers
    assert report.name == "uncertainty_quantification"
    assert report.family == "uq"
    assert report.data_asset_id == "uncertainty_quantification:nu_t"
    assert report.dataset_asset_id == "uncertainty_quantification:dataset"
    assert report.job_id.startswith("job_")
    assert report.model_id > 0
    assert report.metrics.get("train_loss") == 0.0042

    # lineage: job -> dataset -> data product (full transitive upstream)
    assert set(report.lineage_upstream) >= {
        "uncertainty_quantification:dataset",
        "uncertainty_quantification:nu_t",
    }

    # serving model registered with the uq family
    serving = ModelRegistry.open(db_path)
    try:
        model = serving.get_model(report.model_id)
        assert model is not None
        arch = model.get("arch") or {}
        assert arch.get("family") == "uq"
        # the raw arch hyper-parameters are preserved alongside the family
        assert arch.get("arch", {}).get("hidden_features") == 16
    finally:
        serving.close()

    # training job reached completed with metrics recorded
    training = TrainingJobRegistry.open(db_path)
    try:
        job = training.get_job(report.job_id)
        assert job is not None
        assert job.status == "completed"
        assert (job.metrics or {}).get("train_loss") == 0.0042
    finally:
        training.close()


def test_make_dataset_shapes(app):
    nx, ny = 16, 16
    product = DataProduct(
        name="velocity",
        field_name="nu_t",
        shape=(ny, nx),
        dtype="torch.float32",
        units="lu",
        metadata={
            "snapshots": _mock_velocity_snapshots(nx, ny, 0),
            "c_s": 0.1,
        },
    )
    dataset = app.make_dataset(product)
    # two 16x16 snapshots -> 2 * 256 cells, 3 strain features, 1 target
    assert dataset["inputs"].shape == (2 * nx * ny, 3)
    assert dataset["targets"].shape == (2 * nx * ny, 1)
    assert dataset["n_samples"] == 2 * nx * ny
    assert dataset["in_features"] == 3


def test_build_model_and_infer_mc_dropout():
    torch.manual_seed(0)
    app = UncertaintyQuantification(n_mc_samples=30)
    model = app.build_model(
        {"in_features": 3, "hidden_features": 16, "n_hidden_layers": 2, "dropout_p": 0.5},
    )
    assert isinstance(model, MCDropoutMLP)

    x = torch.randn(64, 3)
    pred = app.infer(model, x)

    mean = pred.output["mean"]
    std = pred.output["std"]
    samples = pred.output["samples"]

    # output shapes: mean/std collapse the MC axis, samples keep it
    assert mean.shape == (64, 1)
    assert std.shape == (64, 1)
    assert samples.shape == (30, 64, 1)

    # MC-dropout must yield a non-degenerate predictive distribution
    assert pred.metadata["method"] == "mc_dropout"
    assert pred.metadata["n_samples"] == 30
    assert float(std.mean()) > 0.0
    assert float(std.max()) > 0.0


def test_infer_accepts_velocity_tuple():
    torch.manual_seed(1)
    app = UncertaintyQuantification(n_mc_samples=15)
    model = app.build_model(
        {"in_features": 3, "hidden_features": 8, "n_hidden_layers": 1, "dropout_p": 0.5},
    )
    ux = 0.05 + 0.02 * torch.randn(16, 16)
    uy = 0.02 * torch.randn(16, 16)
    pred = app.infer(model, (ux, uy))
    # 16x16 = 256 cells -> (256, 1) mean/std
    assert pred.output["mean"].shape == (256, 1)
    assert pred.output["std"].shape == (256, 1)
    assert pred.output["samples"].shape == (15, 256, 1)
    assert float(pred.output["std"].max()) > 0.0


def test_train_default_fn_writes_real_checkpoint(tmp_path):
    """Without injection, ``train`` runs the real Adam+MSE loop on CPU."""
    torch.manual_seed(0)
    app = UncertaintyQuantification(n_mc_samples=20)
    dataset = app.make_dataset(
        DataProduct(
            name="velocity",
            field_name="nu_t",
            shape=(16, 16),
            dtype="torch.float32",
            metadata={"snapshots": _mock_velocity_snapshots(16, 16, 0), "c_s": 0.1},
        ),
    )
    model = app.build_model(
        {"in_features": 3, "hidden_features": 16, "n_hidden_layers": 2, "dropout_p": 0.1},
    )
    out_path = tmp_path / "uq_model.pt"
    result = app.train(
        dataset,
        model,
        {"epochs": 2, "batch_size": 128, "learning_rate": 1e-3, "out_path": str(out_path)},
    )

    assert isinstance(result, TrainingResult)
    assert result.model_path.endswith("uq_model.pt")
    assert "train_loss" in result.metrics
    assert result.arch.get("hidden_features") == 16

    # checkpoint actually written and loadable by the uq loader
    assert Path(result.model_path).exists()
    loaded = load_uq_mlp(result.model_path)
    assert isinstance(loaded, MCDropoutMLP)


def test_serving_loads_uq_family(tmp_path):
    """InferenceService loads and predicts a model registered under uq."""
    torch.manual_seed(0)
    app = UncertaintyQuantification(n_mc_samples=20)
    model = app.build_model(
        {"in_features": 3, "hidden_features": 8, "n_hidden_layers": 1, "dropout_p": 0.2},
    )
    from tensorlbm.apps.uncertainty_quantification import save_uq_mlp

    assert isinstance(model, MCDropoutMLP)
    path = save_uq_mlp(model, tmp_path / "uq.pt")

    registry = ModelRegistry.open(tmp_path / "serving.db")
    try:
        model_id = registry.register_model(
            name="uq-mlp",
            path=str(path),
            arch={"in_features": 3, "hidden_features": 8, "n_hidden_layers": 1},
            family=FAMILY_UQ,
            input_shapes={"input": [None, 3]},
            output_shapes={"output": [None, 1]},
        )

        service = InferenceService(registry)
        x = torch.randn(4, 3)
        y = service.predict(model_id, x)
        assert y.shape == (4, 1)

        meta = registry.get_model_metadata(model_id)
        assert meta is not None
        assert meta.family == FAMILY_UQ
    finally:
        registry.close()
