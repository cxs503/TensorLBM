"""Neural-network turbulence-model architecture.

A compact multi-layer perceptron that maps the local strain-rate tensor of
a 2-D LBM velocity field to a non-negative eddy viscosity.  The output
non-negativity is enforced via :class:`torch.nn.Softplus` so the trained
model is safe to plug into ``τ_eff = τ + 3 ν_t`` collision operators.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class ModelArch:
    """Hyper-parameters describing an :class:`EddyViscosityMLP`."""

    in_features: int = 3
    hidden_features: int = 16
    n_hidden_layers: int = 2
    activation: str = "tanh"   # "tanh" or "relu"


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name!r}")


class EddyViscosityMLP(nn.Module):
    """MLP that predicts non-negative eddy viscosity from strain features.

    The input tensor has shape ``(N, in_features)`` (default 3 features:
    ``S_xx, S_yy, S_xy``) and the output has shape ``(N, 1)`` representing
    the per-cell eddy viscosity.
    """

    def __init__(self, arch: ModelArch | None = None) -> None:
        super().__init__()
        self.arch = arch or ModelArch()
        self.register_buffer(
            "feature_mean",
            torch.zeros(self.arch.in_features, dtype=torch.float32),
        )
        self.register_buffer(
            "feature_std",
            torch.ones(self.arch.in_features, dtype=torch.float32),
        )
        layers: list[nn.Module] = []
        in_dim = self.arch.in_features
        for _ in range(self.arch.n_hidden_layers):
            layers.append(nn.Linear(in_dim, self.arch.hidden_features))
            layers.append(_activation(self.arch.activation))
            in_dim = self.arch.hidden_features
        layers.append(nn.Linear(in_dim, 1))
        # Softplus guarantees ν_t > 0 (numerically stable, smooth).
        layers.append(nn.Softplus(beta=10.0))
        self.net = nn.Sequential(*layers)

    def set_feature_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Persist feature-normalization statistics on the model."""
        if mean.shape != (self.arch.in_features,) or std.shape != (self.arch.in_features,):
            raise ValueError(
                "Normalization stats must have shape "
                f"({self.arch.in_features},), got {tuple(mean.shape)} and {tuple(std.shape)}",
            )
        self.feature_mean.copy_(
            mean.detach().to(device=self.feature_mean.device, dtype=torch.float32),
        )
        self.feature_std.copy_(
            std.detach().clamp_min(1e-6).to(device=self.feature_std.device, dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        if x.shape[-1] != self.arch.in_features:
            raise ValueError(
                f"Expected input with {self.arch.in_features} features, "
                f"got tensor of shape {tuple(x.shape)}",
            )
        mean = self.feature_mean.to(device=x.device, dtype=x.dtype)
        std = self.feature_std.to(device=x.device, dtype=x.dtype)
        return self.net((x - mean) / std)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model: EddyViscosityMLP, path: str | Path) -> Path:
    """Serialize a model to a tensor-only ``.pt`` file plus JSON metadata."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), p)
    # Companion JSON for tooling that doesn't want to load a torch blob.
    meta = {
        "arch": asdict(model.arch),
        "normalization": {
            "feature_mean": model.feature_mean.detach().cpu().tolist(),
            "feature_std": model.feature_std.detach().cpu().tolist(),
        },
        "format_version": 2,
    }
    meta_path = p.with_suffix(p.suffix + ".json")
    meta_path.write_text(json.dumps(meta, indent=2))
    return p


def load_model(path: str | Path) -> EddyViscosityMLP:
    """Inverse of :func:`save_model`."""
    p = Path(path)
    blob = torch.load(p, map_location="cpu", weights_only=True)
    meta_path = p.with_suffix(p.suffix + ".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        arch_dict = meta.get("arch") or {}
    elif isinstance(blob, dict) and "arch" in blob:
        arch_dict = blob.get("arch") or {}
    else:
        arch_dict = {}
    arch = ModelArch(**arch_dict) if arch_dict else ModelArch()
    model = EddyViscosityMLP(arch)
    state_dict = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported model payload in {p}")
    model.load_state_dict(state_dict)
    model.eval()
    return model
