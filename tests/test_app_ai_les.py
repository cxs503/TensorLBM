"""Tests for the AI-LES case as an :class:`AI4SApplication` framework instance.

Verifies the full :meth:`AILesApp.run` loop — data production → dataset →
training job → model serving — together with lineage and the serving model
``family``, using a mocked velocity field (``run_les_fn``) and a mocked
training loop (``train_fn``) so the test stays fast and deterministic.
"""

from __future__ import annotations

import torch
import pytest

from tensorlbm.apps.ai_les_app import AILesApp
from tensorlbm.apps.base import DataProduct, TrainingResult
from tensorlbm.ml.serving import ModelRegistry
from tensorlbm.ml.training_job import TrainingJobRegistry

# ---------------------------------------------------------------------------
# Mock collaborators
# ---------------------------------------------------------------------------


def _mock_velocity_snapshots(nx: int, ny: int, seed: int):
    """Deterministic turbulent-looking velocity field (no solver run)."""
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
        model_path="/tmp/mock_ai_les_model.pt",
        metrics={"final_train_mse": 0.0042},
        arch={
            "in_features": 3,
            "hidden_features": 8,
            "n_hidden_layers": 1,
            "activation": "tanh",
        },
    )


@pytest.fixture
def app():
    return AILesApp(train_fn=_mock_train_fn, run_les_fn=_mock_run_les)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_class_identity():
    assert AILesApp.name == "ai_les"
    assert AILesApp.family == "eddy_viscosity_mlp"
    assert AILesApp.version == "1.0"


def test_run_closed_loop_and_lineage(app, tmp_path):
    db_path = tmp_path / "ai_les_app.db"
    produce_cfg = {"nx": 8, "ny": 8, "n_steps": 4, "sample_every": 2, "seed": 0}
    train_cfg = {
        "arch": {"hidden_features": 8, "n_hidden_layers": 1},
        "epochs": 1,
        "out_path": str(tmp_path / "model.pt"),
    }

    report = app.run(str(db_path), produce_cfg, train_cfg)

    # run() closed-loop identifiers
    assert report.name == "ai_les"
    assert report.family == "eddy_viscosity_mlp"
    assert report.data_asset_id == "ai_les:velocity"
    assert report.dataset_asset_id == "ai_les:dataset"
    assert report.job_id.startswith("job_")
    assert report.model_id > 0
    assert report.metrics.get("final_train_mse") == 0.0042

    # lineage: job -> dataset -> data product (full transitive upstream)
    assert set(report.lineage_upstream) >= {"ai_les:dataset", "ai_les:velocity"}

    # serving model registered with the expected family
    serving = ModelRegistry.open(db_path)
    try:
        model = serving.get_model(report.model_id)
        assert model is not None
        arch = model.get("arch") or {}
        assert arch.get("family") == "eddy_viscosity_mlp"
        # the raw arch hyper-parameters are preserved alongside the family
        assert arch.get("arch", {}).get("hidden_features") == 8
    finally:
        serving.close()

    # training job reached completed with metrics recorded
    training = TrainingJobRegistry.open(db_path)
    try:
        job = training.get_job(report.job_id)
        assert job is not None
        assert job.status == "completed"
        assert job.metrics.get("final_train_mse") == 0.0042
    finally:
        training.close()


def test_run_registers_field_product_asset(app, tmp_path):
    from tensorlbm.data.catalog import FieldDataCatalog

    db_path = tmp_path / "ai_les_assets.db"
    report = app.run(
        str(db_path),
        {"nx": 8, "ny": 8, "n_steps": 2, "sample_every": 1, "seed": 0},
        {"arch": {}, "out_path": str(tmp_path / "model.pt")},
    )

    catalog = FieldDataCatalog.open(db_path)
    try:
        product = catalog.get_asset(report.data_asset_id)
        assert product is not None
        assert product.kind == "field_product"
        assert product.field_name == "velocity"
    finally:
        catalog.close()


def test_make_dataset_from_product(app):
    nx, ny = 4, 4
    torch.manual_seed(0)
    ux = 0.05 + 0.02 * torch.randn(ny, nx)
    uy = 0.02 * torch.randn(ny, nx)
    product = DataProduct(
        name="velocity",
        field_name="velocity",
        shape=(ny, nx),
        dtype="torch.float32",
        units="lu",
        metadata={"snapshots": [(ux, uy)], "c_s": 0.1},
    )
    dataset = app.make_dataset(product)
    assert len(dataset) == nx * ny
    assert dataset.features.shape == (nx * ny, 3)
    assert dataset.targets.shape == (nx * ny, 1)


def test_build_model_and_infer(app):
    model = app.build_model({"hidden_features": 8, "n_hidden_layers": 1})
    assert isinstance(model, torch.nn.Module)

    ux = torch.rand(4, 4)
    uy = torch.rand(4, 4)
    pred = app.infer(model, (ux, uy))
    assert pred.output.shape == (4, 4)
    assert pred.metadata["field_name"] == "nu_t"


def test_train_default_fn_uses_real_trainer(tmp_path):
    """With no mock injected, ``train`` wires the real trainer end-to-end."""
    torch.manual_seed(0)
    ux = 0.05 + 0.02 * torch.randn(8, 8)
    uy = 0.02 * torch.randn(8, 8)
    app = AILesApp()  # no injections
    dataset = app.make_dataset(
        DataProduct(
            name="velocity",
            field_name="velocity",
            shape=(8, 8),
            dtype="torch.float32",
            metadata={"snapshots": [(ux, uy)], "c_s": 0.1},
        ),
    )
    model = app.build_model({"hidden_features": 8, "n_hidden_layers": 1})
    result = app.train(
        dataset,
        model,
        {"epochs": 2, "hidden_features": 8, "out_path": str(tmp_path / "model.pt")},
    )
    assert isinstance(result, TrainingResult)
    assert result.model_path.endswith("model.pt")
    assert "final_train_mse" in result.metrics
    assert result.arch.get("hidden_features") == 8
    # the checkpoint was actually written to disk
    from pathlib import Path
    assert Path(result.model_path).exists()
