"""Focused tests for the SUBOFF AI inference and error endpoints.

Tests verify that the router correctly delegates to library functions
(build_suboff_model, predict_suboff, error_analysis_suboff) rather than
inline logic.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from pathlib import Path

    from _pytest.monkeypatch import MonkeyPatch


def test_suboff_predict_calls_library(monkeypatch: MonkeyPatch) -> None:
    """Verify that /predict endpoint calls predict_suboff library function."""
    from backend.routers import ai_suboff  # type: ignore[import-not-found]

    # Mock predict_suboff to return a known result
    fake_result = {
        "coords": np.zeros((500000, 3)),
        "real": np.zeros((500000, 5)),
        "pred": np.ones((500000, 5)) * 0.5,
        "error": np.ones((500000, 5)) * 0.5,
        "input": None,
        "mape": 3.14,
        "rel_l2_avg": 15.0,
        "mse_avg": 10.0,
        "checkpoint": "/fake/ckpt",
        "snap_idx": 55,
    }

    monkeypatch.setattr(ai_suboff, "predict_suboff", lambda cfg: fake_result)

    # Mock CKPT_DIR to have a checkpoint
    import os
    import tempfile
    tmp = tempfile.mkdtemp()
    ckpt_path = os.path.join(tmp, "model_checkpoint0.ckpt")
    torch.save({"encoder": {}, "decoder": {}}, ckpt_path)
    monkeypatch.setattr(ai_suboff, "CKPT_DIR", __import__("pathlib").Path(tmp))

    # Mock data_dir existence
    monkeypatch.setattr(ai_suboff.os.path, "isdir", lambda d: "p" in d)

    response = ai_suboff.suboff_predict_api(ai_suboff.SuboffPredictRequest(n_points=2000))

    assert response["status"] == "ok"
    assert response["mape"] == 3.14
    assert response["rel_l2_avg_1e4"] == 15.0

    # Cleanup
    import os
    os.remove(ckpt_path)
    os.rmdir(tmp)


def test_suboff_error_calls_library(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Verify that /error endpoint calls error_analysis_suboff library function."""
    from backend.routers import ai_suboff  # type: ignore[import-not-found]

    fake_result = {
        "status": "ok",
        "n_snapshots": 10,
        "n_points": 5000,
        "time_ms": 123.4,
        "checkpoint": "/fake/ckpt",
        "summary": {
            "vx": {"rel_l2_mean": 0.05, "rel_l2_max": 0.12},
            "vy": {"rel_l2_mean": 0.03, "rel_l2_max": 0.08},
        },
        "per_snapshot": [],
    }

    monkeypatch.setattr(ai_suboff, "error_analysis_suboff", lambda cfg: fake_result)

    response = ai_suboff.suboff_error_api(
        ai_suboff.SuboffErrorRequest(data_dir=str(tmp_path), n_points=5000, device="cpu")
    )

    assert response["status"] == "ok"
    assert response["n_snapshots"] == 10
    assert response["summary"]["vx"]["rel_l2_mean"] == 0.05


def test_suboff_train_config_creation():
    """Verify SuboffTrainConfig dataclass works."""
    from tensorlbm.ai.suboff_train import SuboffTrainConfig

    cfg = SuboffTrainConfig(lr=1e-3, iters=100, batch_size=2, data_dir="/tmp/data")
    assert cfg.lr == 1e-3
    assert cfg.iters == 100
    assert cfg.batch_size == 2
    assert cfg.data_dir == "/tmp/data"


def test_suboff_predict_config_creation():
    """Verify SuboffPredictConfig dataclass works."""
    from tensorlbm.ai.suboff_inference import SuboffPredictConfig

    cfg = SuboffPredictConfig(checkpoint_path="/tmp/ckpt", data_dir="/tmp/data", snap_idx=55)
    assert cfg.checkpoint_path == "/tmp/ckpt"
    assert cfg.data_dir == "/tmp/data"
    assert cfg.snap_idx == 55


def test_build_suboff_model_cpu():
    """Verify model builds and forward pass works on CPU."""
    from tensorlbm.ai import build_suboff_model

    enc, dec = build_suboff_model("cpu")
    x = torch.randn(1, 1, 100, 4)
    pos = torch.randn(1, 100, 3)
    z = enc(x, pos)
    pred = dec(z, pos, pos)
    assert pred.shape == (1, 100, 4)


def test_pointwise_rel_loss():
    """Verify loss function computes correctly."""
    from tensorlbm.ai import pointwise_rel_loss

    x = torch.ones(1, 50, 4) * 0.9
    y = torch.ones(1, 50, 4)
    loss = pointwise_rel_loss(x, y, p=2)
    assert loss.item() > 0
    # Same input should give ~0 loss
    loss_same = pointwise_rel_loss(y, y, p=2)
    assert loss_same.item() < 1e-6


def test_checkpoint_save_load(tmp_path: Path):
    """Verify checkpoint save/load roundtrip."""
    from tensorlbm.ai import save_checkpoint, load_checkpoint, build_suboff_model

    enc, dec = build_suboff_model("cpu")
    ckpt_path = str(tmp_path / "test.ckpt")
    state = {
        "encoder": enc.state_dict(),
        "decoder": dec.state_dict(),
        "n_iter": 42,
    }
    save_checkpoint(state, ckpt_path)
    loaded = load_checkpoint(ckpt_path)
    assert loaded["n_iter"] == 42
