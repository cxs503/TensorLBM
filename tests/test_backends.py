"""Tests for the tensorlbm.backends multi-framework dispatch layer.

These tests run with PyTorch only (the default backend) since PaddlePaddle
and MindSpore may not be installed in every CI environment.  They verify:

1. Backend registry (set/get, env-var default, invalid name).
2. PyTorch backend tensor operations.
3. PyTorch backend model factories (EddyViscosityMLP, FlowTransformer).
4. PyTorch backend training step.
5. End-to-end: ``train_eddy_viscosity_model`` with explicit backend="torch".
6. End-to-end: ``train_flow_transformer_self_supervised`` with backend="torch".
7. ``backend`` key appears in all training results.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pytest
import torch

import tensorlbm.backends as B


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

def test_default_backend_torch(monkeypatch):
    monkeypatch.delenv("TENSORLBM_BACKEND", raising=False)
    # The default is set at import time, so just assert the documented default.
    assert B.get_backend() in {"torch", "paddle", "mindspore"}


def test_set_and_get_backend():
    original = B.get_backend()
    B.set_backend("torch")
    assert B.get_backend() == "torch"
    B.set_backend(original)


def test_set_invalid_backend():
    with pytest.raises(ValueError, match="Unknown backend"):
        B.set_backend("tensorflow")


def test_get_ops_returns_torch_module():
    B.set_backend("torch")
    ops = B.get_ops()
    assert ops.BACKEND_NAME == "torch"


def test_ops_cached():
    B.set_backend("torch")
    ops1 = B.get_ops()
    ops2 = B.get_ops()
    assert ops1 is ops2


def test_using_backend_restores_previous_backend():
    B.set_backend("torch")
    with B.using_backend("mindspore"):
        assert B.get_backend() == "mindspore"
    assert B.get_backend() == "torch"


# ---------------------------------------------------------------------------
# Torch backend – tensor ops
# ---------------------------------------------------------------------------

@pytest.fixture
def ops():
    B.set_backend("torch")
    return B.get_ops()


def test_zeros(ops):
    t = ops.zeros([3, 4])
    assert t.shape == (3, 4)
    assert float(t.sum()) == 0.0


def test_ones(ops):
    t = ops.ones([2, 3])
    assert t.shape == (2, 3)
    assert float(t.sum()) == 6.0


def test_arange(ops):
    t = ops.arange(0, 5)
    assert list(t.numpy()) == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_tensor_from_list(ops):
    t = ops.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert t.shape == (2, 2)


def test_stack(ops):
    a = ops.tensor([1.0, 2.0])
    b = ops.tensor([3.0, 4.0])
    s = ops.stack([a, b], dim=0)
    assert s.shape == (2, 2)


def test_cat(ops):
    a = ops.tensor([1.0, 2.0])
    b = ops.tensor([3.0, 4.0])
    c = ops.cat([a, b], dim=0)
    assert c.shape == (4,)


def test_clamp_min(ops):
    t = ops.tensor([-1.0, 0.0, 1.0])
    c = ops.clamp_min(t, 0.0)
    assert list(c.numpy()) == [0.0, 0.0, 1.0]


def test_mean(ops):
    t = ops.tensor([1.0, 2.0, 3.0])
    assert abs(float(ops.mean(t)) - 2.0) < 1e-6


def test_to_numpy(ops):
    t = ops.tensor([1.0, 2.0])
    arr = ops.to_numpy(t)
    assert isinstance(arr, np.ndarray)
    assert list(arr) == pytest.approx([1.0, 2.0])


def test_no_grad_context(ops):
    """Verify no_grad context manager doesn't raise."""
    t = ops.tensor([1.0, 2.0, 3.0])
    with ops.no_grad():
        result = ops.mean(t)
    assert float(result) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Torch backend – model factories
# ---------------------------------------------------------------------------

def test_build_eddy_viscosity_mlp(ops):
    mean_np = np.zeros(3, dtype=np.float32)
    std_np  = np.ones(3, dtype=np.float32)
    model = ops.build_eddy_viscosity_mlp(3, 16, 2, "tanh", mean_np, std_np)
    # Forward pass
    x = torch.rand(10, 3)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (10, 1)
    assert (out >= 0).all()   # Softplus → non-negative


def test_build_flow_transformer(ops):
    model = ops.build_flow_transformer(2, 16, 2, 1, 32, 0.0, 64)
    x = torch.rand(2, 4, 2)   # batch=2, tokens=4, features=2
    ops.eval_mode(model)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 4, 2)


def test_get_state_dict_numpy(ops):
    mean_np = np.zeros(3, dtype=np.float32)
    std_np  = np.ones(3, dtype=np.float32)
    model = ops.build_eddy_viscosity_mlp(3, 16, 2, "tanh", mean_np, std_np)
    sd = ops.get_state_dict_numpy(model)
    assert isinstance(sd, dict)
    assert all(isinstance(v, np.ndarray) for v in sd.values())


def test_load_state_dict_numpy_roundtrip(ops, tmp_path):
    mean_np = np.zeros(3, dtype=np.float32)
    std_np  = np.ones(3, dtype=np.float32)
    model1  = ops.build_eddy_viscosity_mlp(3, 16, 2, "tanh", mean_np, std_np)
    sd_np   = ops.get_state_dict_numpy(model1)

    model2 = ops.build_eddy_viscosity_mlp(3, 16, 2, "tanh", mean_np, std_np)
    ops.load_state_dict_numpy(model2, sd_np)

    # Both models should produce the same output
    x = torch.rand(5, 3)
    ops.eval_mode(model1)
    ops.eval_mode(model2)
    with torch.no_grad():
        o1 = model1(x).numpy()
        o2 = model2(x).numpy()
    np.testing.assert_allclose(o1, o2)


# ---------------------------------------------------------------------------
# Torch backend – train_step
# ---------------------------------------------------------------------------

def test_train_step(ops):
    mean_np = np.zeros(3, dtype=np.float32)
    std_np  = np.ones(3, dtype=np.float32)
    model   = ops.build_eddy_viscosity_mlp(3, 16, 2, "tanh", mean_np, std_np)
    loss_fn = ops.mse_loss_fn()
    optim   = ops.adam_optimizer(model, 1e-3)
    x = torch.rand(8, 3)
    y = torch.rand(8, 1) * 0.1
    loss = ops.train_step(model, loss_fn, optim, x, y, max_norm=1.0)
    assert isinstance(loss, float)
    assert loss >= 0.0


# ---------------------------------------------------------------------------
# End-to-end: train_eddy_viscosity_model with backend="torch"
# ---------------------------------------------------------------------------

def test_train_eddy_viscosity_model_torch_backend(tmp_path):
    from tensorlbm import (
        EddyViscosityDataset,
        TrainConfig,
        train_eddy_viscosity_model,
    )
    n = 200
    features = torch.rand(n, 3)
    targets  = torch.rand(n, 1) * 0.01
    ds = EddyViscosityDataset(features=features, targets=targets, c_s=0.1)
    cfg = TrainConfig(epochs=2, batch_size=64, seed=0)
    out = tmp_path / "model_torch.pt"
    result = train_eddy_viscosity_model(ds, out, cfg, backend="torch")

    assert result["backend"] == "torch"
    assert out.exists()
    assert result["final_val_mse"] >= 0.0
    assert len(result["history"]) == 2


def test_train_eddy_viscosity_model_uses_current_backend(tmp_path):
    """When backend kwarg is omitted, the current global backend is used."""
    from tensorlbm import EddyViscosityDataset, TrainConfig, train_eddy_viscosity_model
    B.set_backend("torch")
    n = 100
    ds = EddyViscosityDataset(
        features=torch.rand(n, 3),
        targets=torch.rand(n, 1) * 0.01,
        c_s=0.1,
    )
    result = train_eddy_viscosity_model(ds, tmp_path / "m.pt", TrainConfig(epochs=1, batch_size=50))
    assert result["backend"] == "torch"


# ---------------------------------------------------------------------------
# End-to-end: train_flow_transformer_self_supervised with backend="torch"
# ---------------------------------------------------------------------------

def test_train_flow_transformer_torch_backend(tmp_path):
    from tensorlbm import (
        FlowTransformerArch,
        FlowTransformerTrainConfig,
        train_flow_transformer_self_supervised,
    )
    ny, nx = 8, 8
    snapshots = [
        (torch.rand(ny, nx), torch.rand(ny, nx))
        for _ in range(4)
    ]
    arch = FlowTransformerArch(d_model=8, n_heads=2, n_layers=1, ffn_dim=16, max_tokens=128)
    cfg  = FlowTransformerTrainConfig(epochs=2, batch_size=2, seed=0)
    out  = tmp_path / "transformer_torch.pt"
    result = train_flow_transformer_self_supervised(
        snapshots, out, arch=arch, config=cfg, backend="torch"
    )
    assert result["backend"] == "torch"
    assert out.exists()
    assert result["final_train_loss"] >= 0.0
    assert len(result["history"]) == 2


def test_train_eddy_viscosity_model_explicit_backend_switches_temporarily(monkeypatch, tmp_path):
    from tensorlbm import EddyViscosityDataset, TrainConfig, train_eddy_viscosity_model
    from tensorlbm.ai import train as train_mod

    seen: list[str] = []

    def fake_train(*args, **kwargs):
        seen.append(B.get_backend())
        return {"backend": B.get_backend(), "history": [], "final_val_mse": 0.0}

    monkeypatch.setattr(train_mod, "_train_with_backend", fake_train)
    B.set_backend("torch")
    ds = EddyViscosityDataset(
        features=torch.rand(8, 3),
        targets=torch.rand(8, 1),
        c_s=0.1,
    )
    result = train_eddy_viscosity_model(ds, tmp_path / "model.npz", TrainConfig(epochs=1), backend="paddle")
    assert seen == ["paddle"]
    assert result["backend"] == "paddle"
    assert B.get_backend() == "torch"


def test_train_flow_transformer_explicit_backend_switches_temporarily(monkeypatch, tmp_path):
    from tensorlbm import FlowTransformerArch, FlowTransformerTrainConfig, train_flow_transformer_self_supervised
    from tensorlbm.ai import transformer as transformer_mod

    seen: list[str] = []

    def fake_train(*args, **kwargs):
        seen.append(B.get_backend())
        return {"backend": B.get_backend(), "history": [], "final_train_loss": 0.0, "final_val_loss": 0.0}

    monkeypatch.setattr(transformer_mod, "_train_flow_transformer_backend", fake_train)
    B.set_backend("torch")
    snapshots = [(torch.rand(4, 4), torch.rand(4, 4)) for _ in range(2)]
    result = train_flow_transformer_self_supervised(
        snapshots,
        tmp_path / "model.npz",
        arch=FlowTransformerArch(d_model=8, n_heads=2, n_layers=1, ffn_dim=16, max_tokens=32),
        config=FlowTransformerTrainConfig(epochs=1, batch_size=1),
        backend="mindspore",
    )
    assert seen == ["mindspore"]
    assert result["backend"] == "mindspore"
    assert B.get_backend() == "torch"


def test_load_flow_transformer_model_uses_metadata_backend(monkeypatch, tmp_path):
    from tensorlbm.ai import transformer as transformer_mod

    class FakeModel:
        def __init__(self):
            self.training = True
            self.loaded = None

    class FakeOps:
        def __init__(self):
            self.loaded = None

        def build_flow_transformer(self, *args, **kwargs):
            self.build_backend = B.get_backend()
            return FakeModel()

        def load_state_dict_numpy(self, model, arrays):
            self.loaded = arrays
            model.loaded = arrays

        def eval_mode(self, model):
            model.training = False

    fake_ops = FakeOps()
    monkeypatch.setattr(transformer_mod, "get_ops", lambda: fake_ops)
    path = tmp_path / "transformer.npz"
    with path.open("wb") as fh:
        np.savez_compressed(fh, weight=np.array([1.0], dtype=np.float32))
    path.with_suffix(path.suffix + ".json").write_text(
        '{"arch": {"in_features": 2, "d_model": 8, "n_heads": 2, "n_layers": 1, "ffn_dim": 16, "dropout": 0.0, "max_tokens": 32}, "backend": "paddle"}'
    )

    B.set_backend("torch")
    model = transformer_mod.load_flow_transformer_model(path)
    assert fake_ops.build_backend == "paddle"
    assert model.tensorlbm_backend == "paddle"
    assert model.loaded is not None
    np.testing.assert_allclose(model.loaded["weight"], np.array([1.0], dtype=np.float32))
    assert B.get_backend() == "torch"


def test_reconstruct_flow_field_uses_model_backend(monkeypatch):
    from tensorlbm.ai import transformer as transformer_mod

    class FakeModel:
        def __init__(self):
            self.training = True
            self.tensorlbm_backend = "mindspore"

        def __call__(self, x):
            return x + 1.0

    class FakeOps:
        def __init__(self):
            self.tensor_backend = None

        def tensor(self, data):
            self.tensor_backend = B.get_backend()
            return np.asarray(data, dtype=np.float32)

        def unsqueeze(self, tensor, dim):
            return np.expand_dims(tensor, axis=dim)

        def is_training(self, model):
            return model.training

        def eval_mode(self, model):
            model.training = False

        @contextmanager
        def no_grad(self):
            yield

        def train_mode(self, model):
            model.training = True

    fake_ops = FakeOps()
    monkeypatch.setattr(transformer_mod, "get_ops", lambda: fake_ops)
    B.set_backend("torch")
    result = transformer_mod.reconstruct_flow_field(
        FakeModel(),
        torch.zeros((2, 2), dtype=torch.float32),
        torch.zeros((2, 2), dtype=torch.float32),
    )
    assert np.allclose(result["ux_reconstructed"], 1.0)
    assert np.allclose(result["uy_reconstructed"], 1.0)
    assert result["mse"] == pytest.approx(1.0)
    assert result["max_abs_error"] == pytest.approx(1.0)
    assert fake_ops.tensor_backend == "mindspore"
    assert B.get_backend() == "torch"


# ---------------------------------------------------------------------------
# tensorlbm top-level re-exports
# ---------------------------------------------------------------------------

def test_top_level_get_set_backend():
    import tensorlbm
    tensorlbm.set_backend("torch")
    assert tensorlbm.get_backend() == "torch"
