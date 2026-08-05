"""PyTorch backend for TensorLBM multi-backend AI layer.

This module provides the canonical reference implementation.  Every
function and class here is mirrored in :mod:`paddle_backend` and
:mod:`mindspore_backend` with identical signatures.
"""
from __future__ import annotations

import contextlib
import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..core.lattice import D3Q19, LatticeDescriptor
from ..models.contracts import ModelComposition
from ..models.torch_execution import TorchD3Q19MRTPlan, compile_torch_d3q19_mrt_plan
from .contracts import BackendCapabilities, BackendId, BackendSupport, DeviceSpec


# ---------------------------------------------------------------------------
# Constants & dtypes
# ---------------------------------------------------------------------------

def pi() -> float:
    return math.pi


def float32_dtype() -> torch.dtype:
    return torch.float32


# ---------------------------------------------------------------------------
# Tensor creation
# ---------------------------------------------------------------------------

def zeros(shape, dtype=None, device: str = "cpu") -> torch.Tensor:
    dt = dtype if dtype is not None else torch.float32
    return torch.zeros(shape, dtype=dt, device=device)


def ones(shape, dtype=None, device: str = "cpu") -> torch.Tensor:
    dt = dtype if dtype is not None else torch.float32
    return torch.ones(shape, dtype=dt, device=device)


def arange(start, stop=None, step=1, dtype=None, device: str = "cpu") -> torch.Tensor:
    if stop is None:
        return torch.arange(start, dtype=dtype, device=device)
    return torch.arange(start, stop, step, dtype=dtype, device=device)


def tensor(data, dtype=None, device: str = "cpu") -> torch.Tensor:
    dt = dtype if dtype is not None else torch.float32
    return torch.tensor(data, dtype=dt, device=device)


def from_numpy(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr)


def meshgrid(*tensors: torch.Tensor, indexing: str = "ij"):
    return torch.meshgrid(*tensors, indexing=indexing)


# ---------------------------------------------------------------------------
# Tensor operations
# ---------------------------------------------------------------------------

def stack(tensors, dim: int = 0) -> torch.Tensor:
    return torch.stack(list(tensors), dim=dim)


def cat(tensors, dim: int = 0) -> torch.Tensor:
    return torch.cat(list(tensors), dim=dim)


def where(cond: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.where(cond, x, y)


def clamp(x: torch.Tensor, min_val=None, max_val=None) -> torch.Tensor:
    return torch.clamp(x, min=min_val, max=max_val)


def clamp_min(x: torch.Tensor, min_val: float) -> torch.Tensor:
    return torch.clamp_min(x, min_val)


def sqrt(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(x)


def sin(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x)


def cos(x: torch.Tensor) -> torch.Tensor:
    return torch.cos(x)


def abs_val(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(x)


def mean(x: torch.Tensor, dim=None) -> torch.Tensor:
    return torch.mean(x) if dim is None else torch.mean(x, dim=dim)


def sum_val(x: torch.Tensor, dim=None) -> torch.Tensor:
    return torch.sum(x) if dim is None else torch.sum(x, dim=dim)


def max_val(x: torch.Tensor) -> torch.Tensor:
    return torch.max(x)


def var(x: torch.Tensor, unbiased: bool = False) -> torch.Tensor:
    return torch.var(x, unbiased=unbiased)


def std(x: torch.Tensor, dim=None, unbiased: bool = False) -> torch.Tensor:
    return torch.std(x, unbiased=unbiased) if dim is None else torch.std(x, dim=dim, unbiased=unbiased)


def rand(shape, device: str = "cpu") -> torch.Tensor:
    return torch.rand(shape, device=device)


def randperm(n: int, device: str = "cpu") -> torch.Tensor:
    return torch.randperm(n, device=device)


def make_generator(seed: int, device: str = "cpu") -> torch.Generator:
    return torch.Generator(device=device).manual_seed(int(seed))


def randperm_with_gen(n: int, generator: torch.Generator) -> torch.Tensor:
    return torch.randperm(n, generator=generator)


def rand_shape_with_gen(shape, device: str, generator: torch.Generator) -> torch.Tensor:
    return torch.rand(shape, device=device, generator=generator)


def manual_seed(seed: int) -> None:
    torch.manual_seed(int(seed))


def isfinite(x: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(x)


def index_select(t: torch.Tensor, dim: int, idx: torch.Tensor) -> torch.Tensor:
    return t.index_select(dim, idx)


def reshape(t: torch.Tensor, shape) -> torch.Tensor:
    return t.reshape(shape)


def unsqueeze(t: torch.Tensor, dim: int) -> torch.Tensor:
    return t.unsqueeze(dim)


def expand_as(t: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    return t.expand_as(other)


def contiguous(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous()


def roll(t: torch.Tensor, shifts, dims) -> torch.Tensor:
    return torch.roll(t, shifts=shifts, dims=dims)


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def to_device(t: torch.Tensor, device_str: str) -> torch.Tensor:
    return t.to(device=torch.device(device_str))


def clone(t: torch.Tensor) -> torch.Tensor:
    return t.clone()


def detach(t: torch.Tensor) -> torch.Tensor:
    return t.detach()


def float_scalar(t: torch.Tensor) -> float:
    return float(t)


def bool_scalar(t: torch.Tensor) -> bool:
    return bool(t)


def any_true(t: torch.Tensor) -> bool:
    return bool(t.any())


def all_true(t: torch.Tensor) -> bool:
    return bool(t.all())


# ---------------------------------------------------------------------------
# No-grad context
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def no_grad():
    with torch.no_grad():
        yield


# ---------------------------------------------------------------------------
# NN helpers – activation
# ---------------------------------------------------------------------------

def _activation_layer(name: str) -> nn.Module:
    n = name.lower()
    if n == "tanh":
        return nn.Tanh()
    if n == "relu":
        return nn.ReLU()
    if n == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name!r}")


# ---------------------------------------------------------------------------
# EddyViscosityMLP (PyTorch)
# ---------------------------------------------------------------------------

class _EddyViscosityMLP(nn.Module):
    """PyTorch implementation of the eddy-viscosity MLP."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        n_hidden_layers: int,
        activation: str,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.register_buffer("feature_mean", feature_mean.clone().to(dtype=torch.float32))
        self.register_buffer("feature_std", feature_std.clone().clamp_min(1e-6).to(dtype=torch.float32))
        layers: list[nn.Module] = []
        in_dim = in_features
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_features))
            layers.append(_activation_layer(activation))
            in_dim = hidden_features
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Softplus(beta=10.0))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected {self.in_features} features, got shape {tuple(x.shape)}"
            )
        mean = self.feature_mean.to(device=x.device, dtype=x.dtype)
        std = self.feature_std.to(device=x.device, dtype=x.dtype)
        return self.net((x - mean) / std)


def build_eddy_viscosity_mlp(
    in_features: int,
    hidden_features: int,
    n_hidden_layers: int,
    activation: str,
    feature_mean_np: np.ndarray,
    feature_std_np: np.ndarray,
    device: str = "cpu",
) -> _EddyViscosityMLP:
    """Build and return an EddyViscosityMLP on *device*."""
    mean = torch.tensor(feature_mean_np, dtype=torch.float32)
    std = torch.tensor(feature_std_np, dtype=torch.float32)
    model = _EddyViscosityMLP(in_features, hidden_features, n_hidden_layers, activation, mean, std)
    return model.to(torch.device(device))


# ---------------------------------------------------------------------------
# FlowFieldTransformer (PyTorch)
# ---------------------------------------------------------------------------

class _FlowTransformer(nn.Module):
    """PyTorch masked-token reconstruction transformer."""

    def __init__(
        self,
        in_features: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ffn_dim: int,
        dropout: float,
        max_tokens: int,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.max_tokens = max_tokens
        self.mask_token = nn.Parameter(torch.zeros(1, 1, in_features))
        self.input_proj = nn.Linear(in_features, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_tokens, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        n_tokens = int(x.shape[1])
        if n_tokens > self.max_tokens:
            raise ValueError(f"Token count {n_tokens} > max_tokens={self.max_tokens}")
        h = self.input_proj(x) + self.pos_embedding[:, :n_tokens, :]
        h = self.encoder(h)
        return self.head(h)

    def apply_mask_token(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(x), x)


def build_flow_transformer(
    in_features: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    ffn_dim: int,
    dropout: float,
    max_tokens: int,
    device: str = "cpu",
) -> _FlowTransformer:
    """Build and return a FlowFieldTransformer on *device*."""
    model = _FlowTransformer(in_features, d_model, n_heads, n_layers, ffn_dim, dropout, max_tokens)
    return model.to(torch.device(device))


# ---------------------------------------------------------------------------
# Loss, optimizer, scheduler
# ---------------------------------------------------------------------------

def mse_loss_fn():
    """Return an MSE loss callable ``(pred, target) -> scalar``."""
    criterion = nn.MSELoss()
    return criterion


def adam_optimizer(model: nn.Module, lr: float) -> optim.Adam:
    return optim.Adam(model.parameters(), lr=float(lr))


def cosine_lr_scheduler(optimizer: optim.Optimizer, T_max: int):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(T_max)))


def plateau_lr_scheduler(optimizer: optim.Optimizer):
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)


def is_plateau_scheduler(scheduler) -> bool:
    return isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau)


def zero_grad(optimizer: optim.Optimizer) -> None:
    optimizer.zero_grad()


def optimizer_step(optimizer: optim.Optimizer) -> None:
    optimizer.step()


def clip_grad_norm_(model: nn.Module, max_norm: float) -> None:
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(max_norm))


def scheduler_step(scheduler, val: float | None = None) -> None:
    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(val)
    else:
        scheduler.step()


def get_lr(optimizer: optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def backward(loss: torch.Tensor) -> None:
    loss.backward()


# ---------------------------------------------------------------------------
# Model persistence (backend-portable: numpy arrays)
# ---------------------------------------------------------------------------

def get_state_dict_numpy(model: nn.Module) -> dict[str, np.ndarray]:
    """Return model state dict as a dict of numpy arrays (portable format)."""
    return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}


def load_state_dict_numpy(model: nn.Module, arrays: dict[str, np.ndarray]) -> None:
    """Restore model weights from numpy-array state dict."""
    current = model.state_dict()
    loaded = {
        name: torch.from_numpy(arr).to(dtype=current[name].dtype)
        for name, arr in arrays.items()
    }
    model.load_state_dict(loaded)


def get_feature_stats_numpy(model: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return (feature_mean, feature_std) as numpy arrays from an MLP model."""
    return (
        model.feature_mean.detach().cpu().numpy(),
        model.feature_std.detach().cpu().numpy(),
    )


def compute_feature_stats(x_train: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean/std of training features; return as numpy."""
    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    return to_numpy(mean), to_numpy(std)


# ---------------------------------------------------------------------------
# Model state helpers
# ---------------------------------------------------------------------------

def eval_mode(model: nn.Module) -> None:
    model.eval()


def train_mode(model: nn.Module) -> None:
    model.train()


def is_training(model: nn.Module) -> bool:
    return model.training


# ---------------------------------------------------------------------------
# Unified training step (one minibatch)
# ---------------------------------------------------------------------------

def train_step(
    model: nn.Module,
    loss_fn,
    optimizer: optim.Optimizer,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    max_norm: float | None,
) -> float:
    """Run one forward/backward/update step; return scalar loss."""
    zero_grad(optimizer)
    pred = model(x_batch)
    loss = loss_fn(pred, y_batch)
    backward(loss)
    if max_norm is not None:
        clip_grad_norm_(model, max_norm)
    optimizer_step(optimizer)
    return float(loss.detach())


# ---------------------------------------------------------------------------
# Identifier
# ---------------------------------------------------------------------------

BACKEND_NAME: str = "torch"


# R1 LBM binding: setup-only.  It deliberately does not participate in the
# legacy AI backend dispatch above, and it never owns a solver step.
_R1_CAPABILITIES = BackendCapabilities(
    backend_id=BackendId.TORCH,
    support=BackendSupport.SUPPORTED,
    supported_devices=("cpu",),
    supported_dtypes=("float32",),
    notes="R1 supports the existing direct D3Q19 MRT PyTorch execution plan on CPU float32 only.",
)
_R1_DEVICE = DeviceSpec(device="cpu", dtype_name="float32")


@dataclass(frozen=True, slots=True)
class TorchBackend:
    """Cold-path binder; stepping remains exclusively on ``TorchD3Q19MRTPlan``."""

    @property
    def capabilities(self) -> BackendCapabilities:
        """Return the deliberately narrow, actually tested R1 support declaration."""
        return _R1_CAPABILITIES

    def validate_device(self, spec: DeviceSpec) -> None:
        """Reject every device or dtype outside the R1 CPU float32 contract."""
        if spec != _R1_DEVICE:
            raise ValueError("Torch backend R1 supports only device='cpu', dtype_name='float32'")

    def compile_d3q19_mrt(
        self, composition: ModelComposition, tau: float, device_spec: DeviceSpec
    ) -> TorchD3Q19MRTPlan:
        """Bind validated cold-path metadata to the existing PyTorch-only MRT plan."""
        self.validate_device(device_spec)
        return compile_torch_d3q19_mrt_plan(composition, tau)


def build_torch_lattice_constants(
    descriptor: LatticeDescriptor, device_spec: DeviceSpec
) -> dict[str, torch.Tensor]:
    """Adapt the core D3Q19 tuple descriptor to CPU tensors during setup only."""
    TorchBackend().validate_device(device_spec)
    if descriptor != D3Q19:
        raise ValueError("Torch backend R1 lattice constants support only the core D3Q19 descriptor")
    return {
        "directions": torch.tensor(descriptor.directions, dtype=torch.int64, device=device_spec.device),
        "weights": torch.tensor(descriptor.weights, dtype=torch.float32, device=device_spec.device),
        "opposite": torch.tensor(descriptor.opposite, dtype=torch.int64, device=device_spec.device),
    }