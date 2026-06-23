"""PaddlePaddle backend for TensorLBM multi-backend AI layer.

Requires PaddlePaddle ≥ 2.4.  Install with::

    pip install paddlepaddle        # CPU
    pip install paddlepaddle-gpu    # GPU (CUDA)

All public functions mirror :mod:`tensorlbm.backends.torch_backend` exactly.
"""
from __future__ import annotations

import contextlib
import math
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Lazy import guard – paddle is only needed when this backend is active.
# ---------------------------------------------------------------------------

def _paddle():
    try:
        import paddle  # noqa: PLC0415
        return paddle
    except ImportError as exc:
        raise ImportError(
            "PaddlePaddle is not installed.  "
            "Install with `pip install paddlepaddle` (CPU) or "
            "`pip install paddlepaddle-gpu` (GPU)."
        ) from exc


def _paddle_nn():
    return _paddle().nn


def _paddle_optim():
    return _paddle().optimizer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def pi() -> float:
    return math.pi


def float32_dtype():
    return _paddle().float32


# ---------------------------------------------------------------------------
# Tensor creation
# ---------------------------------------------------------------------------

def zeros(shape, dtype=None, device: str = "cpu"):
    pd = _paddle()
    dt = dtype if dtype is not None else pd.float32
    with _device_scope(device):
        return pd.zeros(shape, dtype=dt)


def ones(shape, dtype=None, device: str = "cpu"):
    pd = _paddle()
    dt = dtype if dtype is not None else pd.float32
    with _device_scope(device):
        return pd.ones(shape, dtype=dt)


def arange(start, stop=None, step=1, dtype=None, device: str = "cpu"):
    pd = _paddle()
    with _device_scope(device):
        if stop is None:
            return pd.arange(start, dtype=dtype or pd.float32)
        return pd.arange(start, stop, step, dtype=dtype or pd.float32)


def tensor(data, dtype=None, device: str = "cpu"):
    pd = _paddle()
    dt = dtype if dtype is not None else pd.float32
    with _device_scope(device):
        return pd.to_tensor(data, dtype=dt)


def from_numpy(arr: np.ndarray):
    return _paddle().to_tensor(arr)


def meshgrid(*tensors, indexing: str = "ij"):
    return _paddle().meshgrid(*tensors)


# ---------------------------------------------------------------------------
# Tensor operations
# ---------------------------------------------------------------------------

def stack(tensors, dim: int = 0):
    return _paddle().stack(list(tensors), axis=dim)


def cat(tensors, dim: int = 0):
    return _paddle().concat(list(tensors), axis=dim)


def where(cond, x, y):
    return _paddle().where(cond, x, y)


def clamp(x, min_val=None, max_val=None):
    return _paddle().clip(x, min=min_val, max=max_val)


def clamp_min(x, min_val: float):
    return _paddle().clip(x, min=float(min_val))


def sqrt(x):
    return _paddle().sqrt(x)


def sin(x):
    return _paddle().sin(x)


def cos(x):
    return _paddle().cos(x)


def abs_val(x):
    return _paddle().abs(x)


def mean(x, dim=None):
    pd = _paddle()
    return pd.mean(x) if dim is None else pd.mean(x, axis=dim)


def sum_val(x, dim=None):
    pd = _paddle()
    return pd.sum(x) if dim is None else pd.sum(x, axis=dim)


def max_val(x):
    return _paddle().max(x)


def var(x, unbiased: bool = False):
    return _paddle().var(x, unbiased=unbiased)


def std(x, dim=None, unbiased: bool = False):
    pd = _paddle()
    return pd.std(x, unbiased=unbiased) if dim is None else pd.std(x, axis=dim, unbiased=unbiased)


def rand(shape, device: str = "cpu"):
    with _device_scope(device):
        return _paddle().rand(shape)


def randperm(n: int, device: str = "cpu"):
    with _device_scope(device):
        return _paddle().randperm(n)


def make_generator(seed: int, device: str = "cpu"):
    """Return a seed integer (Paddle uses global seed per call site)."""
    return int(seed)


def randperm_with_gen(n: int, generator: int):
    pd = _paddle()
    pd.seed(int(generator))
    return pd.randperm(n)


def rand_shape_with_gen(shape, device: str, generator: int):
    pd = _paddle()
    pd.seed(int(generator))
    with _device_scope(device):
        return pd.rand(shape)


def manual_seed(seed: int) -> None:
    _paddle().seed(int(seed))


def isfinite(x):
    return _paddle().isfinite(x)


def index_select(t, dim: int, idx):
    return _paddle().index_select(t, idx, axis=dim)


def reshape(t, shape):
    return _paddle().reshape(t, shape)


def unsqueeze(t, dim: int):
    return _paddle().unsqueeze(t, axis=dim)


def expand_as(t, other):
    return t.expand_as(other)


def contiguous(t):
    return t.contiguous() if hasattr(t, "contiguous") else t


def roll(t, shifts, dims):
    return _paddle().roll(t, shifts=shifts, axis=dims)


def to_numpy(t) -> np.ndarray:
    return t.numpy()


def to_device(t, device_str: str):
    return t.cuda() if "cuda" in device_str else t.cpu()


def clone(t):
    return t.clone()


def detach(t):
    return t.detach()


def float_scalar(t) -> float:
    return float(t)


def bool_scalar(t) -> bool:
    return bool(t)


def any_true(t) -> bool:
    return bool(_paddle().any(t))


def all_true(t) -> bool:
    return bool(_paddle().all(t))


@contextlib.contextmanager
def no_grad():
    with _paddle().no_grad():
        yield


# ---------------------------------------------------------------------------
# Device scope helper
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _device_scope(device: str):
    """Null context – Paddle uses global device selection."""
    yield


# ---------------------------------------------------------------------------
# NN helpers – activation
# ---------------------------------------------------------------------------

def _activation_layer(name: str):
    nn = _paddle_nn()
    n = name.lower()
    if n == "tanh":
        return nn.Tanh()
    if n == "relu":
        return nn.ReLU()
    if n == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name!r}")


# ---------------------------------------------------------------------------
# EddyViscosityMLP (PaddlePaddle)
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
    """Build an eddy-viscosity MLP as a ``paddle.nn.Layer``."""
    pd = _paddle()
    nn = _paddle_nn()

    feature_mean = pd.to_tensor(feature_mean_np, dtype=pd.float32)
    feature_std = pd.to_tensor(
        np.maximum(feature_std_np, 1e-6), dtype=pd.float32
    )

    class _EddyViscosityMLP(nn.Layer):  # type: ignore[name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.in_features = in_features
            # Store as non-trainable buffers via create_parameter with trainable=False
            self.feature_mean = pd.create_parameter(
                shape=feature_mean.shape,
                dtype="float32",
                default_initializer=nn.initializer.Assign(feature_mean),
            )
            self.feature_mean.trainable = False
            self.feature_std = pd.create_parameter(
                shape=feature_std.shape,
                dtype="float32",
                default_initializer=nn.initializer.Assign(feature_std),
            )
            self.feature_std.trainable = False
            layers: list[Any] = []
            in_dim = in_features
            for _ in range(n_hidden_layers):
                layers.append(nn.Linear(in_dim, hidden_features))
                layers.append(_activation_layer(activation))
                in_dim = hidden_features
            layers.append(nn.Linear(in_dim, 1))
            layers.append(nn.Softplus(beta=10.0))
            self.net = nn.Sequential(*layers)

        def forward(self, x):  # noqa: D401
            if x.shape[-1] != self.in_features:
                raise ValueError(
                    f"Expected {self.in_features} features, got {tuple(x.shape)}"
                )
            m = pd.cast(self.feature_mean, x.dtype)
            s = pd.cast(self.feature_std, x.dtype)
            return self.net((x - m) / s)

    model = _EddyViscosityMLP()
    if "cuda" in device:
        model = model.cuda()
    return model


# ---------------------------------------------------------------------------
# FlowFieldTransformer (PaddlePaddle)
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
    """Build a flow transformer as a ``paddle.nn.Layer``."""
    pd = _paddle()
    nn = _paddle_nn()

    class _FlowTransformer(nn.Layer):  # type: ignore[name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.in_features = in_features
            self.max_tokens = max_tokens
            self.mask_token = pd.create_parameter(
                shape=[1, 1, in_features],
                dtype="float32",
                default_initializer=nn.initializer.Constant(0.0),
            )
            self.input_proj = nn.Linear(in_features, d_model)
            self.pos_embedding = pd.create_parameter(
                shape=[1, max_tokens, d_model],
                dtype="float32",
                default_initializer=nn.initializer.Constant(0.0),
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.head = nn.Linear(d_model, in_features)

        def forward(self, x):  # noqa: D401
            n_tok = int(x.shape[1])
            if n_tok > self.max_tokens:
                raise ValueError(f"Token count {n_tok} > max_tokens={self.max_tokens}")
            h = self.input_proj(x) + self.pos_embedding[:, :n_tok, :]
            h = self.encoder(h)
            return self.head(h)

        def apply_mask_token(self, x, mask):
            mask_expanded = pd.unsqueeze(mask, axis=-1)
            mask_expanded = pd.expand_as(mask_expanded, x)
            mt = pd.expand_as(self.mask_token, x)
            return pd.where(mask_expanded, mt, x)

    model = _FlowTransformer()
    if "cuda" in device:
        model = model.cuda()
    return model


# ---------------------------------------------------------------------------
# Loss, optimizer, scheduler
# ---------------------------------------------------------------------------

def mse_loss_fn():
    return _paddle_nn().MSELoss()


def adam_optimizer(model, lr: float):
    return _paddle_optim().Adam(
        parameters=model.parameters(), learning_rate=float(lr)
    )


def cosine_lr_scheduler(optimizer, T_max: int):
    pd = _paddle()
    return pd.optimizer.lr.CosineAnnealingDecay(
        learning_rate=optimizer.get_lr(), T_max=max(1, int(T_max))
    )


def plateau_lr_scheduler(optimizer):
    pd = _paddle()
    return pd.optimizer.lr.ReduceOnPlateau(
        learning_rate=optimizer.get_lr(), mode="min", factor=0.5, patience=2
    )


def is_plateau_scheduler(scheduler) -> bool:
    pd = _paddle()
    return isinstance(scheduler, pd.optimizer.lr.ReduceOnPlateau)


def zero_grad(optimizer) -> None:
    optimizer.clear_grad()


def optimizer_step(optimizer) -> None:
    optimizer.step()


def clip_grad_norm_(model, max_norm: float) -> None:
    _paddle_nn().utils.clip_grad_norm_(model.parameters(), max_norm=float(max_norm))


def scheduler_step(scheduler, val: float | None = None) -> None:
    pd = _paddle()
    if isinstance(scheduler, pd.optimizer.lr.ReduceOnPlateau):
        scheduler.step(metrics=val)
    else:
        scheduler.step()


def get_lr(optimizer) -> float:
    return float(optimizer.get_lr())


def backward(loss) -> None:
    loss.backward()


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def get_state_dict_numpy(model) -> dict[str, np.ndarray]:
    """Return model state dict as dict of numpy arrays."""
    return {k: v.numpy() for k, v in model.state_dict().items()}


def load_state_dict_numpy(model, arrays: dict[str, np.ndarray]) -> None:
    """Load model weights from numpy-array state dict."""
    pd = _paddle()
    state = {k: pd.to_tensor(v) for k, v in arrays.items()}
    model.set_state_dict(state)


def get_feature_stats_numpy(model) -> tuple[np.ndarray, np.ndarray]:
    return model.feature_mean.numpy(), model.feature_std.numpy()


def compute_feature_stats(x_train) -> tuple[np.ndarray, np.ndarray]:
    pd = _paddle()
    mean = pd.mean(x_train, axis=0)
    std = pd.clip(pd.std(x_train, axis=0, unbiased=False), min=1e-6)
    return to_numpy(mean), to_numpy(std)


# ---------------------------------------------------------------------------
# Model state helpers
# ---------------------------------------------------------------------------

def eval_mode(model) -> None:
    model.eval()


def train_mode(model) -> None:
    model.train()


def is_training(model) -> bool:
    return model.training


# ---------------------------------------------------------------------------
# Unified training step (one minibatch)
# ---------------------------------------------------------------------------

def train_step(model, loss_fn, optimizer, x_batch, y_batch, max_norm: float | None) -> float:
    """Run one forward/backward/update step; return scalar loss."""
    pred = model(x_batch)
    loss = loss_fn(pred, y_batch)
    backward(loss)
    if max_norm is not None:
        clip_grad_norm_(model, max_norm)
    optimizer_step(optimizer)
    zero_grad(optimizer)
    return float_scalar(loss.detach())


# ---------------------------------------------------------------------------
# Identifier
# ---------------------------------------------------------------------------

BACKEND_NAME: str = "paddle"
