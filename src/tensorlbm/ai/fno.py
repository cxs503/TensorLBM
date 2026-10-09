"""Fourier Neural Operator (FNO2d) for 2-D flow-field surrogate modelling.

Implementation follows Li et al. (2021) "Fourier Neural Operator for
Parametric Partial Differential Equations" (arXiv:2010.08895).

The operator maps an input function discretized on a 2-D grid
(e.g. an LBM velocity snapshot of shape ``(batch, in_channels, ny, nx)``)
to an output function of the same spatial resolution.  Typical uses in
TensorLBM:

- **Turbulence closure**: replace the Smagorinsky algebraic model with a
  global, non-local operator that captures long-range correlations.
- **Flow-field super-resolution**: upsample coarse LBM output to a finer
  grid without re-running the solver.
- **Multi-geometry generalization**: train once on a library of
  geometries and infer on new shapes at near-zero marginal cost.

Architecture overview
---------------------
::

    x  →  P  →  ┌──────────────────────┐(×n_layers)→  Q  →  y
                 │  SpectralConv2d      │
                 │  + PointwiseConv2d   │
                 │  + activation        │
                 └──────────────────────┘

- **P** (lifting): 1×1 conv from ``in_channels`` to ``width``
- **FNO blocks**: ``n_layers`` Fourier layers; each adds a local
  pointwise (1×1) convolution to the global spectral branch
- **Q** (projection): two-layer MLP from ``width`` to ``out_channels``

All tensors are ``(batch, channels, ny, nx)`` throughout.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Architecture hyper-parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FNO2dArch:
    """Hyper-parameters for :class:`FNO2d`."""

    in_channels: int = 3
    """Number of input channels (e.g. ux, uy, rho)."""

    out_channels: int = 1
    """Number of output channels (e.g. eddy viscosity ν_t)."""

    width: int = 32
    """Number of channels in the Fourier lifting space."""

    n_layers: int = 4
    """Number of Fourier layers."""

    modes_x: int = 12
    """Number of Fourier modes retained along x (≤ nx // 2 + 1)."""

    modes_y: int = 12
    """Number of Fourier modes retained along y (≤ ny // 2 + 1)."""

    mlp_hidden: int = 128
    """Hidden dimension of the final projection MLP."""

    activation: str = "gelu"
    """Pointwise activation: 'gelu', 'relu', or 'tanh'."""


# ---------------------------------------------------------------------------
# Spectral convolution layer
# ---------------------------------------------------------------------------


class SpectralConv2d(nn.Module):
    """Global convolution in Fourier space (one Fourier layer).

    For an input ``u`` of shape ``(B, in_channels, ny, nx)`` it:

    1. Computes the 2-D rFFT: ``u_hat = rfft2(u)``  → ``(B, in_c, ny, nx//2+1)``
    2. Multiplies the retained ``(modes_y × modes_x)`` low-frequency
       corner by a learned complex weight tensor.
    3. Inverse rFFT back to physical space.

    The weight tensors are stored as real tensors of shape
    ``(in_channels, out_channels, modes_y, modes_x, 2)`` where the last
    dimension holds the real and imaginary parts — this avoids PyTorch's
    complex-number handling issues on some backends.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_y: int,
        modes_x: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_y = modes_y
        self.modes_x = modes_x

        scale = 1.0 / (in_channels * out_channels)
        # Shape: (in_c, out_c, modes_y, modes_x, 2) — real + imag stored together
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_y, modes_x, 2)
        )

    def _complex_mul2d(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
    ) -> torch.Tensor:
        """Batched complex multiply: (B, in_c, my, mx) × (in_c, out_c, my, mx) → (B, out_c, my, mx).

        ``x`` is a complex tensor; ``w`` stores real/imag in the last dim.
        """
        w_c = torch.view_as_complex(w)  # (in_c, out_c, modes_y, modes_x)
        # einsum: b i y x, i o y x -> b o y x
        return torch.einsum("biyx,ioyx->boyx", x, w_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spectral convolution.

        Args:
            x: ``(B, in_channels, ny, nx)`` real tensor.

        Returns:
            ``(B, out_channels, ny, nx)`` real tensor.
        """
        batch, _, ny, nx = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")  # (B, in_c, ny, nx//2+1) complex

        # Zero-initialised output in Fourier space
        out_ft = torch.zeros(
            batch,
            self.out_channels,
            ny,
            nx // 2 + 1,
            dtype=x_ft.dtype,
            device=x.device,
        )

        # Retain only the low-frequency corner of the Fourier spectrum
        my, mx = self.modes_y, self.modes_x
        out_ft[:, :, :my, :mx] = self._complex_mul2d(x_ft[:, :, :my, :mx], self.weight)

        return torch.fft.irfft2(out_ft, s=(ny, nx), norm="ortho")  # (B, out_c, ny, nx)


# ---------------------------------------------------------------------------
# FNO2d main model
# ---------------------------------------------------------------------------


def _get_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    name = name.lower()
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
    if name == "tanh":
        return torch.tanh
    raise ValueError(f"Unsupported activation: {name!r}")


class FNO2d(nn.Module):
    """2-D Fourier Neural Operator for flow-field surrogate modelling.

    Maps a spatial input field ``(B, in_channels, ny, nx)`` to an output
    field ``(B, out_channels, ny, nx)`` through a series of Fourier
    layers that capture both global (spectral) and local (pointwise)
    features.

    Example::

        arch = FNO2dArch(in_channels=3, out_channels=1, width=32, n_layers=4)
        model = FNO2d(arch)
        x = torch.randn(8, 3, 64, 64)   # batch of 8 snapshots
        y = model(x)                     # (8, 1, 64, 64)
    """

    def __init__(self, arch: FNO2dArch | None = None) -> None:
        super().__init__()
        self.arch = arch or FNO2dArch()
        a = self.arch
        self._act = _get_activation(a.activation)

        # Lifting layer P: in_channels → width
        self.lift = nn.Conv2d(a.in_channels, a.width, kernel_size=1)

        # Fourier layers
        self.spectral = nn.ModuleList(
            [SpectralConv2d(a.width, a.width, a.modes_y, a.modes_x) for _ in range(a.n_layers)]
        )
        # Pointwise (local) bypass per Fourier layer
        self.pointwise = nn.ModuleList(
            [nn.Conv2d(a.width, a.width, kernel_size=1) for _ in range(a.n_layers)]
        )

        # Projection MLP Q: width → mlp_hidden → out_channels
        self.proj = nn.Sequential(
            nn.Conv2d(a.width, a.mlp_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(a.mlp_hidden, a.out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, in_channels, ny, nx)``.

        Returns:
            Output tensor of shape ``(B, out_channels, ny, nx)``.
        """
        # Lifting
        x = self.lift(x)  # (B, width, ny, nx)

        # Fourier layers
        for spec, pw in zip(self.spectral, self.pointwise):
            x = self._act(spec(x) + pw(x))

        # Projection
        return self.proj(x)  # (B, out_channels, ny, nx)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def save_fno2d(model: FNO2d, path: str | Path) -> Path:
    """Serialize a :class:`FNO2d` to a ``.pt`` file plus JSON metadata.

    Args:
        model: Trained FNO2d instance.
        path: Destination path (``*.pt``).

    Returns:
        Resolved path of the saved weight file.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), p)
    meta = {
        "arch": asdict(model.arch),
        "model_class": "FNO2d",
        "format_version": 1,
    }
    meta_path = p.with_suffix(p.suffix + ".json")
    meta_path.write_text(json.dumps(meta, indent=2))
    return p


def load_fno2d(path: str | Path) -> FNO2d:
    """Load a :class:`FNO2d` model saved by :func:`save_fno2d`.

    Args:
        path: Path to the ``.pt`` weight file.  A companion ``.pt.json``
              metadata file is expected alongside it.

    Returns:
        Loaded model in eval mode.
    """
    p = Path(path)
    blob = torch.load(p, map_location="cpu", weights_only=True)
    meta_path = p.with_suffix(p.suffix + ".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        arch_dict = meta.get("arch") or {}
    else:
        arch_dict = {}
    arch = FNO2dArch(**arch_dict) if arch_dict else FNO2dArch()
    model = FNO2d(arch)
    model.load_state_dict(blob)
    model.eval()
    return model
