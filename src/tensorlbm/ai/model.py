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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        if x.shape[-1] != self.arch.in_features:
            raise ValueError(
                f"Expected input with {self.arch.in_features} features, "
                f"got tensor of shape {tuple(x.shape)}",
            )
        return self.net(x)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model: EddyViscosityMLP, path: str | Path) -> Path:
    """Serialize a model and its architecture to a single ``.pt`` file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "arch": asdict(model.arch),
            "format_version": 1,
        },
        p,
    )
    # Companion JSON for tooling that doesn't want to load a torch blob.
    meta_path = p.with_suffix(p.suffix + ".json")
    meta_path.write_text(json.dumps({"arch": asdict(model.arch)}, indent=2))
    return p


def load_model(path: str | Path) -> EddyViscosityMLP:
    """Inverse of :func:`save_model`."""
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    arch_dict = blob.get("arch") or {}
    arch = ModelArch(**arch_dict) if arch_dict else ModelArch()
    model = EddyViscosityMLP(arch)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model
