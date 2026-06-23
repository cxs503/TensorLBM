"""MindSpore backend for TensorLBM multi-backend AI layer.

Requires MindSpore ≥ 2.0 running in **PyNative** mode.  Install with::

    pip install mindspore        # CPU
    pip install mindspore-gpu    # GPU (CUDA)

PyNative mode is set automatically when this module is first imported.

All public functions mirror :mod:`tensorlbm.backends.torch_backend` exactly.
"""
from __future__ import annotations

import contextlib
import math
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Lazy import guard
# ---------------------------------------------------------------------------

def _ms():
    try:
        import mindspore as ms  # noqa: PLC0415
        ms.set_context(mode=ms.PYNATIVE_MODE)  # ensure eager execution
        return ms
    except ImportError as exc:
        raise ImportError(
            "MindSpore is not installed.  "
            "Install with `pip install mindspore` (CPU) or "
            "`pip install mindspore-gpu` (GPU)."
        ) from exc


def _ms_nn():
    return _ms().nn


def _ms_ops():
    return _ms().ops


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def pi() -> float:
    return math.pi


def float32_dtype():
    return _ms().float32


# ---------------------------------------------------------------------------
# Tensor creation
# ---------------------------------------------------------------------------

def zeros(shape, dtype=None, device: str = "cpu"):
    ms = _ms()
    dt = dtype if dtype is not None else ms.float32
    return ms.ops.zeros(shape, dtype=dt)


def ones(shape, dtype=None, device: str = "cpu"):
    ms = _ms()
    dt = dtype if dtype is not None else ms.float32
    return ms.ops.ones(shape, dtype=dt)


def arange(start, stop=None, step=1, dtype=None, device: str = "cpu"):
    ms = _ms()
    dt = dtype if dtype is not None else ms.float32
    if stop is None:
        return ms.numpy.arange(start, dtype=dt)
    return ms.numpy.arange(start, stop, step, dtype=dt)


def tensor(data, dtype=None, device: str = "cpu"):
    ms = _ms()
    dt = dtype if dtype is not None else ms.float32
    return ms.Tensor(data, dtype=dt)


def from_numpy(arr: np.ndarray):
    return _ms().Tensor(arr)


def meshgrid(*tensors, indexing: str = "ij"):
    return _ms().numpy.meshgrid(*tensors, indexing=indexing)


# ---------------------------------------------------------------------------
# Tensor operations
# ---------------------------------------------------------------------------

def stack(tensors, dim: int = 0):
    return _ms().ops.stack(list(tensors), axis=dim)


def cat(tensors, dim: int = 0):
    return _ms().ops.cat(list(tensors), axis=dim)


def where(cond, x, y):
    return _ms().ops.where(cond, x, y)


def clamp(x, min_val=None, max_val=None):
    return _ms().ops.clamp(x, min=min_val, max=max_val)


def clamp_min(x, min_val: float):
    return _ms().ops.clamp(x, min=float(min_val))


def sqrt(x):
    return _ms().ops.sqrt(x)


def sin(x):
    return _ms().ops.sin(x)


def cos(x):
    return _ms().ops.cos(x)


def abs_val(x):
    return _ms().ops.abs(x)


def mean(x, dim=None):
    ms = _ms()
    return ms.ops.mean(x) if dim is None else ms.ops.mean(x, dim)


def sum_val(x, dim=None):
    ms = _ms()
    return ms.ops.sum(x) if dim is None else ms.ops.sum(x, dim)


def max_val(x):
    return _ms().ops.max(x)


def var(x, unbiased: bool = False):
    return _ms().ops.var(x, unbiased=unbiased)


def std(x, dim=None, unbiased: bool = False):
    ms = _ms()
    return ms.ops.std(x) if dim is None else ms.ops.std(x, axis=dim)


def rand(shape, device: str = "cpu"):
    return _ms().ops.uniform(shape, _ms().Tensor(0.0), _ms().Tensor(1.0))


def randperm(n: int, device: str = "cpu"):
    return _ms().ops.randperm(n)


def make_generator(seed: int, device: str = "cpu") -> int:
    """Return seed integer (MindSpore uses global seed)."""
    return int(seed)


def randperm_with_gen(n: int, generator: int):
    ms = _ms()
    ms.set_seed(int(generator))
    return ms.ops.randperm(n)


def rand_shape_with_gen(shape, device: str, generator: int):
    ms = _ms()
    ms.set_seed(int(generator))
    return ms.ops.uniform(shape, ms.Tensor(0.0), ms.Tensor(1.0))


def manual_seed(seed: int) -> None:
    _ms().set_seed(int(seed))


def isfinite(x):
    return _ms().ops.isfinite(x)


def index_select(t, dim: int, idx):
    return _ms().ops.index_select(t, dim, idx)


def reshape(t, shape):
    return _ms().ops.reshape(t, shape)


def unsqueeze(t, dim: int):
    return _ms().ops.expand_dims(t, dim)


def expand_as(t, other):
    return t.broadcast_to(other.shape)


def contiguous(t):
    return t


def roll(t, shifts, dims):
    return _ms().ops.roll(t, shifts=shifts, dims=dims)


def to_numpy(t) -> np.ndarray:
    return t.asnumpy()


def to_device(t, device_str: str):
    # MindSpore uses context-level device selection; return as-is
    return t


def clone(t):
    return t.copy()


def detach(t):
    return _ms().ops.stop_gradient(t)


def float_scalar(t) -> float:
    return float(t.asnumpy())


def bool_scalar(t) -> bool:
    return bool(t.asnumpy())


def any_true(t) -> bool:
    return bool(_ms().ops.any(t).asnumpy())


def all_true(t) -> bool:
    return bool(_ms().ops.all(t).asnumpy())


@contextlib.contextmanager
def no_grad():
    """Context manager: disable gradient computation (PyNative stop_gradient)."""
    # In MindSpore PyNative mode, wrap with a no-grad inference context.
    ms = _ms()
    with ms._context.stop_gradient_cell():  # type: ignore[attr-defined]
        yield


# ---------------------------------------------------------------------------
# NN helpers – activation
# ---------------------------------------------------------------------------

def _activation_layer(name: str):
    nn = _ms_nn()
    n = name.lower()
    if n == "tanh":
        return nn.Tanh()
    if n == "relu":
        return nn.ReLU()
    if n == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name!r}")


# ---------------------------------------------------------------------------
# EddyViscosityMLP (MindSpore)
# ---------------------------------------------------------------------------

def build_eddy_viscosity_mlp(
    in_features: int,
    hidden_features: int,
    n_hidden_layers: int,
    activation: str,
    feature_mean_np: np.ndarray,
    feature_std_np: np.ndarray,
    device: str = "cpu",
):
    """Build an eddy-viscosity MLP as a ``mindspore.nn.Cell``."""
    ms = _ms()
    nn = _ms_nn()

    mean_t = ms.Tensor(feature_mean_np, dtype=ms.float32)
    std_t = ms.Tensor(np.maximum(feature_std_np, 1e-6), dtype=ms.float32)

    class _EddyViscosityMLP(nn.Cell):  # type: ignore[name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.in_features = in_features
            # Store as non-trainable parameters
            self.feature_mean = ms.Parameter(mean_t, requires_grad=False)
            self.feature_std = ms.Parameter(std_t, requires_grad=False)
            layers: list[Any] = []
            in_dim = in_features
            for _ in range(n_hidden_layers):
                layers.append(nn.Dense(in_dim, hidden_features))
                layers.append(_activation_layer(activation))
                in_dim = hidden_features
            layers.append(nn.Dense(in_dim, 1))
            layers.append(nn.Softplus())
            self.net = nn.SequentialCell(*layers)

        def construct(self, x):  # noqa: D401 – MindSpore uses 'construct' not 'forward'
            if x.shape[-1] != self.in_features:
                raise ValueError(
                    f"Expected {self.in_features} features, got {tuple(x.shape)}"
                )
            m = ms.ops.cast(self.feature_mean, x.dtype)
            s = ms.ops.cast(self.feature_std, x.dtype)
            return self.net((x - m) / s)

    return _EddyViscosityMLP()


# ---------------------------------------------------------------------------
# FlowFieldTransformer (MindSpore)
# ---------------------------------------------------------------------------

def build_flow_transformer(
    in_features: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    ffn_dim: int,
    dropout: float,
    max_tokens: int,
    device: str = "cpu",
):
    """Build a flow transformer as a ``mindspore.nn.Cell``."""
    ms = _ms()
    nn = _ms_nn()

    class _FlowTransformer(nn.Cell):  # type: ignore[name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.in_features = in_features
            self.max_tokens = max_tokens
            self.mask_token = ms.Parameter(
                ms.ops.zeros([1, 1, in_features], dtype=ms.float32),
                requires_grad=True,
            )
            self.input_proj = nn.Dense(in_features, d_model)
            self.pos_embedding = ms.Parameter(
                ms.ops.zeros([1, max_tokens, d_model], dtype=ms.float32),
                requires_grad=True,
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.head = nn.Dense(d_model, in_features)

        def construct(self, x):  # noqa: D401
            n_tok = int(x.shape[1])
            if n_tok > self.max_tokens:
                raise ValueError(f"Token count {n_tok} > max_tokens={self.max_tokens}")
            h = self.input_proj(x) + self.pos_embedding[:, :n_tok, :]
            h = self.encoder(h)
            return self.head(h)

        def apply_mask_token(self, x, mask):
            ms_mod = _ms()
            mask_exp = ms_mod.ops.expand_dims(mask, -1).broadcast_to(x.shape)
            mt = self.mask_token.broadcast_to(x.shape)
            return ms_mod.ops.where(mask_exp, mt, x)

    return _FlowTransformer()


# ---------------------------------------------------------------------------
# Loss, optimizer, scheduler
# ---------------------------------------------------------------------------

def mse_loss_fn():
    return _ms_nn().MSELoss()


def adam_optimizer(model, lr: float):
    return _ms_nn().Adam(model.trainable_params(), learning_rate=float(lr))


def cosine_lr_scheduler(optimizer, T_max: int):
    ms = _ms()
    return ms.nn.CosineDecayLR(
        min_lr=0.0,
        max_lr=float(optimizer.learning_rate.data.asnumpy()),
        decay_steps=max(1, int(T_max)),
    )


def plateau_lr_scheduler(optimizer):
    # MindSpore doesn't have a direct ReduceLROnPlateau; return None and skip
    return None


def is_plateau_scheduler(scheduler) -> bool:
    return False  # simplified – MindSpore uses fixed schedules


def zero_grad(optimizer) -> None:
    """No-op for MindSpore: grads cleared by the optimizer after each step."""


def optimizer_step(optimizer) -> None:
    """No-op: in MindSpore the optimizer is called with grads directly."""


def clip_grad_norm_(model, max_norm: float) -> None:
    """Stored for use in _train_step; actual clipping happens inside value_and_grad."""
    # Clipping is applied in the training step wrapper (see _ms_train_step).


def scheduler_step(scheduler, val: float | None = None) -> None:
    if scheduler is not None:
        scheduler.step()


def get_lr(optimizer) -> float:
    try:
        return float(optimizer.learning_rate.data.asnumpy())
    except Exception:
        return float(optimizer.learning_rate)


def backward(loss) -> None:
    """No-op: MindSpore uses value_and_grad functional API for gradients."""


# ---------------------------------------------------------------------------
# MindSpore training step helper
# ---------------------------------------------------------------------------

def ms_train_step(model, loss_fn, optimizer, x_batch, y_batch, max_norm: float | None):
    """Run one gradient step; returns scalar loss value."""
    ms = _ms()

    def _forward(xb, yb):
        pred = model(xb)
        return loss_fn(pred, yb)

    grad_fn = ms.value_and_grad(_forward, None, model.trainable_params())
    loss_val, grads = grad_fn(x_batch, y_batch)
    if max_norm is not None:
        grads = ms.ops.clip_by_global_norm(grads, max_norm)
    optimizer(grads)
    return loss_val


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def get_state_dict_numpy(model) -> dict[str, np.ndarray]:
    """Return model state dict as dict of numpy arrays."""
    return {
        name: param.asnumpy()
        for name, param in model.parameters_and_names()
    }


def load_state_dict_numpy(model, arrays: dict[str, np.ndarray]) -> None:
    """Load model weights from numpy-array state dict."""
    ms = _ms()
    params = {name: ms.Parameter(ms.Tensor(v)) for name, v in arrays.items()}
    ms.load_param_into_net(model, params)


def get_feature_stats_numpy(model) -> tuple[np.ndarray, np.ndarray]:
    return model.feature_mean.asnumpy(), model.feature_std.asnumpy()


def compute_feature_stats(x_train) -> tuple[np.ndarray, np.ndarray]:
    ms = _ms()
    mean = ms.ops.mean(x_train, 0)
    std = ms.ops.clamp(ms.ops.std(x_train, axis=0), min=1e-6)
    return to_numpy(mean), to_numpy(std)


# ---------------------------------------------------------------------------
# Model state helpers
# ---------------------------------------------------------------------------

def eval_mode(model) -> None:
    model.set_train(False)


def train_mode(model) -> None:
    model.set_train(True)


def is_training(model) -> bool:
    return model.training


# ---------------------------------------------------------------------------
# Unified training step (one minibatch)
# ---------------------------------------------------------------------------

def train_step(model, loss_fn, optimizer, x_batch, y_batch, max_norm: float | None) -> float:
    """Run one gradient step via MindSpore's functional value_and_grad."""
    return float_scalar(ms_train_step(model, loss_fn, optimizer, x_batch, y_batch, max_norm))


# ---------------------------------------------------------------------------
# Identifier
# ---------------------------------------------------------------------------

BACKEND_NAME: str = "mindspore"
