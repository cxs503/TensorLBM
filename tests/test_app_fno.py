"""Tests for :class:`tensorlbm.apps.neural_operator_fno.NeuralOperatorFNO`.

Verifies the FNO2d neural operator as an :class:`AI4SApplication` framework
instance: the inherited ``run()`` full-stack loop (produce → register → train
→ serve → lineage), the ``fno2d`` serving family, and the five
developer-implemented methods.  A mocked velocity field (``run_les_fn``) and a
mocked training loop (``train_fn``) keep the closed-loop test fast and
deterministic; one test exercises the real Adam+MSE loop on a tiny grid.
"""

from __future__ import annotations

import pytest
import torch

from tensorlbm.ai.fno import FNO2d, load_fno2d
from tensorlbm.apps.base import DataProduct, TrainingResult
from tensorlbm.apps.neural_operator_fno import NeuralOperatorFNO
from tensorlbm.ml.serving import FAMILY_FNO, InferenceService, ModelRegistry
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
        model_path="/tmp/mock_fno2d_model.pt",
        metrics={"train_loss": 0.0037},
        arch={
            "in_channels": 2,
            "out_channels": 2,
            "width": 16,
            "n_layers": 2,
            "modes_x": 8,
            "modes_y": 8,
        },
    )


@pytest.fixture
def app():
    return NeuralOperatorFNO(train_fn=_mock_train_fn, run_les_fn=_mock_run_les)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_class_identity():
    assert NeuralOperatorFNO.name == "neural_operator_fno"
    assert NeuralOperatorFNO.family == "fno2d"
    assert NeuralOperatorFNO.version == "1.0"


def test_run_closed_loop_and_lineage(app, tmp_path):
    db_path = tmp_path / "fno_app.db"
    produce_cfg = {"nx": 16, "ny": 16, "n_steps": 4, "sample_every": 2, "seed": 0}
    train_cfg = {
        "arch": {"in_channels": 2, "out_channels": 2, "width": 16, "n_layers": 2},
        "epochs": 1,
        "out_path": str(tmp_path / "model.pt"),
    }

    report = app.run(str(db_path), produce_cfg, train_cfg)

    # run() closed-loop identifiers
    assert report.name == "neural_operator_fno"
    assert report.family == "fno2d"
    assert report.data_asset_id == "neural_operator_fno:u"
    assert report.dataset_asset_id == "neural_operator_fno:dataset"
    assert report.job_id.startswith("job_")
    assert report.model_id > 0
    assert report.metrics.get("train_loss") == 0.0037

    # lineage: job -> dataset -> data product (full transitive upstream)
    assert set(report.lineage_upstream) >= {
        "neural_operator_fno:dataset",
        "neural_operator_fno:u",
    }

    # serving model registered with the fno2d family
    serving = ModelRegistry.open(db_path)
    try:
        model = serving.get_model(report.model_id)
        assert model is not None
        arch = model.get("arch") or {}
        assert arch.get("family") == "fno2d"
        # the raw arch hyper-parameters are preserved alongside the family
        assert arch.get("arch", {}).get("width") == 16
    finally:
        serving.close()

    # training job reached completed with metrics recorded
    training = TrainingJobRegistry.open(db_path)
    try:
        job = training.get_job(report.job_id)
        assert job is not None
        assert job.status == "completed"
        assert (job.metrics or {}).get("train_loss") == 0.0037
    finally:
        training.close()


def test_make_dataset_shapes(app):
    nx, ny = 16, 16
    torch.manual_seed(0)
    ux = 0.05 + 0.02 * torch.randn(ny, nx)
    uy = 0.02 * torch.randn(ny, nx)
    product = DataProduct(
        name="velocity",
        field_name="u",
        shape=(ny, nx),
        dtype="torch.float32",
        units="lu",
        metadata={
            "snapshots": [(ux, uy), (0.5 * ux, 0.5 * uy)],
            "downsample_factor": 2,
        },
    )
    dataset = app.make_dataset(product)
    assert dataset["inputs"].shape == (2, 2, ny, nx)
    assert dataset["targets"].shape == (2, 2, ny, nx)
    assert dataset["grid"] == (ny, nx)
    assert dataset["n_samples"] == 2


def test_build_model_and_infer(app):
    model = app.build_model(
        {
            "in_channels": 2,
            "out_channels": 2,
            "width": 8,
            "n_layers": 2,
            "modes_x": 8,
            "modes_y": 8,
        },
    )
    assert isinstance(model, FNO2d)

    ux = torch.rand(16, 16)
    uy = torch.rand(16, 16)
    pred = app.infer(model, (ux, uy))
    assert pred.output.shape == (2, 16, 16)
    assert pred.metadata["field_name"] == "u"

    # a batched (C, ny, nx) tensor sample is also accepted
    pred2 = app.infer(model, torch.stack([ux, uy], dim=0))
    assert pred2.output.shape == (2, 16, 16)


def test_train_default_fn_writes_real_checkpoint(tmp_path):
    """Without injection, ``train`` runs the real Adam+MSE loop on CPU."""
    torch.manual_seed(0)
    nx, ny = 16, 16
    ux = 0.05 + 0.02 * torch.randn(ny, nx)
    uy = 0.02 * torch.randn(ny, nx)
    app = NeuralOperatorFNO()
    dataset = app.make_dataset(
        DataProduct(
            name="velocity",
            field_name="u",
            shape=(ny, nx),
            dtype="torch.float32",
            metadata={"snapshots": [(ux, uy), (0.5 * ux, 0.5 * uy)], "downsample_factor": 2},
        ),
    )
    model = app.build_model(
        {
            "in_channels": 2,
            "out_channels": 2,
            "width": 8,
            "n_layers": 2,
            "modes_x": 8,
            "modes_y": 8,
        },
    )
    out_path = tmp_path / "fno_model.pt"
    result = app.train(
        dataset,
        model,
        {"epochs": 2, "batch_size": 4, "learning_rate": 1e-3, "out_path": str(out_path)},
    )

    assert isinstance(result, TrainingResult)
    assert result.model_path.endswith("fno_model.pt")
    assert "train_loss" in result.metrics
    assert result.arch.get("width") == 8

    # checkpoint actually written and loadable by the ai.fno loader
    from pathlib import Path

    assert Path(result.model_path).exists()
    loaded = load_fno2d(result.model_path)
    assert isinstance(loaded, FNO2d)


def test_serving_loads_fno_family(tmp_path):
    """InferenceService loads and predicts a model registered under fno2d."""
    app = NeuralOperatorFNO()
    model = app.build_model(
        {
            "in_channels": 2,
            "out_channels": 2,
            "width": 8,
            "n_layers": 2,
            "modes_x": 8,
            "modes_y": 8,
        },
    )
    from tensorlbm.ai.fno import save_fno2d

    assert isinstance(model, FNO2d)
    path = save_fno2d(model, tmp_path / "fno.pt")

    registry = ModelRegistry.open(tmp_path / "serving.db")
    try:
        model_id = registry.register_model(
            name="fno-op",
            path=str(path),
            arch={"in_channels": 2, "out_channels": 2, "width": 8, "n_layers": 2},
            family=FAMILY_FNO,
            input_shapes={"input": [None, 2, 16, 16]},
            output_shapes={"output": [None, 2, 16, 16]},
        )

        service = InferenceService(registry)
        x = torch.randn(1, 2, 16, 16)
        y = service.predict(model_id, x)
        assert y.shape == (1, 2, 16, 16)

        meta = registry.get_model_metadata(model_id)
        assert meta is not None
        assert meta.family == FAMILY_FNO
    finally:
        registry.close()
