"""Tests for :class:`tensorlbm.apps.generative_flow.GenerativeFlow`.

Verifies the Denoising Diffusion Probabilistic Model (DDPM) as an
:class:`AI4SApplication` framework instance: the inherited ``run()`` full-stack
loop (produce → register → train → serve → lineage), the ``diffusion`` serving
family, and the five developer-implemented methods.  A mocked training loop
(``train_fn``) keeps the closed-loop test fast and deterministic; a second test
exercises the real DDPM noise-prediction loop on a tiny 16×16 field, a third
proves the reverse (sampling) loop generates fields of the right shape, and a
fourth proves the serving layer loads a ``diffusion`` model end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tensorlbm.apps.base import TrainingResult
from tensorlbm.apps.generative_flow import (
    DDPM,
    DenoiseCNN,
    GenerativeFlow,
    load_diffusion_model,
    save_diffusion_model,
)
from tensorlbm.ml.serving import (
    FAMILY_DIFFUSION,
    InferenceService,
    ModelRegistry,
)
from tensorlbm.ml.training_job import TrainingJobRegistry

# ---------------------------------------------------------------------------
# Mock collaborators
# ---------------------------------------------------------------------------


def _mock_train_fn(dataset, model, cfg):
    """Save the (already built) DDPM and return a fixed TrainingResult."""
    out_path = cfg.get("out_path", "/tmp/mock_diffusion.pt")
    path = save_diffusion_model(model, out_path)
    return TrainingResult(
        model_path=str(path),
        metrics={"train_loss": 0.0234},
        arch={
            "in_channels": 2,
            "out_channels": 2,
            "hidden_dim": 16,
            "n_layers": 2,
            "time_emb_dim": 16,
            "timesteps": 5,
        },
    )


@pytest.fixture
def app():
    return GenerativeFlow(train_fn=_mock_train_fn)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_class_identity():
    assert GenerativeFlow.name == "generative_flow"
    assert GenerativeFlow.family == "diffusion"
    assert GenerativeFlow.version == "1.0"


def test_run_closed_loop_and_lineage(app, tmp_path):
    db_path = tmp_path / "diffusion_app.db"
    produce_cfg = {"nx": 16, "ny": 16, "n_snapshots": 2, "data_source": "vortex", "seed": 0}
    train_cfg = {
        "arch": {
            "in_channels": 2,
            "out_channels": 2,
            "hidden_dim": 16,
            "n_layers": 2,
            "time_emb_dim": 16,
            "timesteps": 5,
        },
        "epochs": 1,
        "out_path": str(tmp_path / "model.pt"),
    }

    report = app.run(str(db_path), produce_cfg, train_cfg)

    # run() closed-loop identifiers
    assert report.name == "generative_flow"
    assert report.family == "diffusion"
    assert report.data_asset_id == "generative_flow:u"
    assert report.dataset_asset_id == "generative_flow:dataset"
    assert report.job_id.startswith("job_")
    assert report.model_id > 0
    assert report.metrics.get("train_loss") == 0.0234

    # lineage: job -> dataset -> data product (full transitive upstream)
    assert set(report.lineage_upstream) >= {
        "generative_flow:dataset",
        "generative_flow:u",
    }

    # serving model registered under the diffusion family
    serving = ModelRegistry.open(db_path)
    try:
        model = serving.get_model(report.model_id)
        assert model is not None
        arch = model.get("arch") or {}
        assert arch.get("family") == "diffusion"
        # raw arch hyper-parameters preserved alongside the family
        assert arch.get("arch", {}).get("hidden_dim") == 16
        assert arch.get("arch", {}).get("timesteps") == 5
    finally:
        serving.close()

    # training job reached completed with metrics recorded
    training = TrainingJobRegistry.open(db_path)
    try:
        job = training.get_job(report.job_id)
        assert job is not None
        assert job.status == "completed"
        assert (job.metrics or {}).get("train_loss") == 0.0234
    finally:
        training.close()


def test_produce_and_make_dataset_normalization():
    app = GenerativeFlow()
    product = app.produce_data(
        {"nx": 16, "ny": 16, "n_snapshots": 4, "data_source": "vortex", "seed": 1},
    )
    assert product.metadata["n_snapshots"] == 4
    assert product.metadata["grid"] == (16, 16)
    assert product.metadata["channels"] == ["u", "v"]

    dataset = app.make_dataset(product)
    samples = dataset["samples"]
    assert samples.shape == (4, 2, 16, 16)
    # normalised to [-1, 1]
    assert float(samples.min()) >= -1.0
    assert float(samples.max()) <= 1.0
    assert dataset["n_samples"] == 4


def test_build_model_and_infer_shapes():
    app = GenerativeFlow()
    model = app.build_model(
        {
            "in_channels": 2,
            "out_channels": 2,
            "hidden_dim": 16,
            "n_layers": 2,
            "time_emb_dim": 16,
            "timesteps": 4,
        },
    )
    assert isinstance(model, DDPM)
    assert isinstance(model.denoiser, DenoiseCNN)
    assert model.timesteps == 4
    assert model.denoiser.hidden_dim == 16
    assert model.denoiser.n_layers == 2

    # mapping sample -> (2, 16, 16) generated field
    pred = app.infer(model, {"shape": (16, 16), "seed": 0})
    assert pred.output.shape == (2, 16, 16)
    assert pred.metadata["family"] == "diffusion"
    assert pred.metadata["field_name"] == "u"

    # tuple sample -> same shape
    pred2 = app.infer(model, (16, 16))
    assert pred2.output.shape == (2, 16, 16)

    # explicit (C, ny, nx) shape
    pred3 = app.infer(model, (2, 8, 8))
    assert pred3.output.shape == (2, 8, 8)

    # batched generation
    pred4 = app.infer(model, {"shape": (16, 16), "n_samples": 3, "seed": 0})
    assert pred4.output.shape == (3, 2, 16, 16)

    # a tensor sample is reverse-diffused in place
    x_t = torch.randn(2, 16, 16)
    pred5 = app.infer(model, x_t)
    assert pred5.output.shape == (2, 16, 16)

    # generated fields are finite
    assert torch.isfinite(pred.output).all()


def test_train_default_fn_writes_real_checkpoint(tmp_path):
    """Without injection, ``train`` runs the real DDPM loop on CPU."""
    torch.manual_seed(0)
    app = GenerativeFlow()
    dataset = app.make_dataset(
        app.produce_data(
            {"nx": 16, "ny": 16, "n_snapshots": 4, "data_source": "vortex", "seed": 2}
        ),
    )
    model = app.build_model(
        {
            "in_channels": 2,
            "out_channels": 2,
            "hidden_dim": 8,
            "n_layers": 2,
            "time_emb_dim": 8,
            "timesteps": 10,
        },
    )
    out_path = tmp_path / "diffusion_model.pt"
    result = app.train(
        dataset,
        model,
        {"epochs": 2, "batch_size": 2, "learning_rate": 1e-3, "out_path": str(out_path)},
    )

    assert isinstance(result, TrainingResult)
    assert result.model_path.endswith("diffusion_model.pt")
    assert "train_loss" in result.metrics
    assert float(result.metrics["train_loss"]) >= 0.0
    assert result.arch.get("hidden_dim") == 8
    assert result.arch.get("timesteps") == 10

    # checkpoint actually written and loadable
    assert Path(result.model_path).exists()
    loaded = load_diffusion_model(result.model_path)
    assert isinstance(loaded, DDPM)
    assert loaded.denoiser.hidden_dim == 8
    assert loaded.timesteps == 10

    # the loaded model can still generate a field
    pred = app.infer(loaded, {"shape": (16, 16), "seed": 0})
    assert pred.output.shape == (2, 16, 16)


def test_serving_loads_diffusion_family(tmp_path):
    """InferenceService loads and predicts a model registered under ``diffusion``."""
    app = GenerativeFlow()
    model = app.build_model(
        {
            "in_channels": 2,
            "out_channels": 2,
            "hidden_dim": 8,
            "n_layers": 2,
            "time_emb_dim": 8,
            "timesteps": 4,
        },
    )
    assert isinstance(model, DDPM)
    path = save_diffusion_model(model, tmp_path / "diffusion.pt")

    registry = ModelRegistry.open(tmp_path / "serving.db")
    try:
        model_id = registry.register_model(
            name="generative-flow",
            path=str(path),
            arch={
                "in_channels": 2,
                "out_channels": 2,
                "hidden_dim": 8,
                "n_layers": 2,
                "time_emb_dim": 8,
                "timesteps": 4,
            },
            family=FAMILY_DIFFUSION,
            input_shapes={"noise": [None, 2, 16, 16]},
            output_shapes={"output": [None, 2, 16, 16]},
        )

        service = InferenceService(registry)
        # run one reverse-diffusion step through the served model's denoiser
        noise = torch.randn(1, 2, 16, 16)
        loaded = service.load_model(model_id)
        assert isinstance(loaded, DDPM)
        y = loaded.sample_from(noise)
        assert y.shape == (1, 2, 16, 16)

        meta = registry.get_model_metadata(model_id)
        assert meta is not None
        assert meta.family == FAMILY_DIFFUSION
    finally:
        registry.close()
