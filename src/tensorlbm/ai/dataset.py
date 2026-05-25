"""Training-dataset generation for the AI turbulence sub-package.

The training signal mirrors the Smagorinsky eddy-viscosity closure that the
rest of TensorLBM already uses for LES.  For a 2-D velocity field
``(u_x, u_y)`` we compute the symmetric strain-rate tensor

.. math::

    S_{ij} = \\tfrac{1}{2}\\left(\\partial_j u_i + \\partial_i u_j\\right)

and the magnitude ``|S| = sqrt(2 S_{ij} S_{ij})``.  The Smagorinsky eddy
viscosity (``ν_t = (C_s Δ)^2 |S|`` with the lattice spacing ``Δ = 1``)
serves as the regression target.  A neural network is therefore trained to
reproduce the algebraic Smagorinsky closure from local strain features —
this is the simplest non-trivial AI turbulence model and matches what is
commonly used as a baseline in *a-priori* SGS-model studies.

The per-cell features fed to the network are the three independent
components ``(S_xx, S_yy, S_xy)``.  Solid / masked cells (e.g. the
cylinder interior) can be excluded via the optional ``mask`` argument.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Iterable


# ---------------------------------------------------------------------------
# Strain-rate computation
# ---------------------------------------------------------------------------

def strain_rate_tensor_2d(
    ux: torch.Tensor,
    uy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the symmetric strain-rate tensor of a 2-D velocity field.

    Central finite differences on a periodic grid (``torch.roll``) are used,
    which matches the conventions of the existing turbulence module.

    Args:
        ux: x-velocity field, shape ``(ny, nx)``.
        uy: y-velocity field, shape ``(ny, nx)``.

    Returns:
        Tuple ``(S_xx, S_yy, S_xy)`` of tensors with shape ``(ny, nx)``.
    """
    if ux.shape != uy.shape or ux.ndim != 2:
        raise ValueError(
            f"ux and uy must be 2-D tensors of equal shape, got "
            f"{tuple(ux.shape)} and {tuple(uy.shape)}",
        )

    # d/dx along the last axis (x), d/dy along the second-to-last (y).
    dudx = 0.5 * (torch.roll(ux, -1, dims=-1) - torch.roll(ux, 1, dims=-1))
    dudy = 0.5 * (torch.roll(ux, -1, dims=-2) - torch.roll(ux, 1, dims=-2))
    dvdx = 0.5 * (torch.roll(uy, -1, dims=-1) - torch.roll(uy, 1, dims=-1))
    dvdy = 0.5 * (torch.roll(uy, -1, dims=-2) - torch.roll(uy, 1, dims=-2))

    s_xx = dudx
    s_yy = dvdy
    s_xy = 0.5 * (dudy + dvdx)
    return s_xx, s_yy, s_xy


def _strain_magnitude(
    s_xx: torch.Tensor,
    s_yy: torch.Tensor,
    s_xy: torch.Tensor,
) -> torch.Tensor:
    """``|S| = sqrt(2 S_{ij} S_{ij})`` for a symmetric 2x2 tensor."""
    return torch.sqrt(2.0 * (s_xx * s_xx + s_yy * s_yy + 2.0 * s_xy * s_xy))


# ---------------------------------------------------------------------------
# Sample extraction
# ---------------------------------------------------------------------------

def extract_les_samples_2d(
    ux: torch.Tensor,
    uy: torch.Tensor,
    c_s: float = 0.1,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a 2-D velocity snapshot into (features, target) tensors.

    Args:
        ux, uy: Velocity fields, shape ``(ny, nx)``.
        c_s: Smagorinsky constant used to build the regression target.
        mask: Optional boolean tensor of shape ``(ny, nx)``.  Cells where
            ``mask`` is *True* are treated as solid/invalid and removed
            from the returned samples.

    Returns:
        ``(features, target)`` with shapes ``(N, 3)`` and ``(N, 1)``
        respectively, where ``N`` is the number of valid cells.
    """
    s_xx, s_yy, s_xy = strain_rate_tensor_2d(ux, uy)
    s_mag = _strain_magnitude(s_xx, s_yy, s_xy)
    nu_t = (c_s * c_s) * s_mag  # Δ = 1 lattice unit

    feats = torch.stack([s_xx, s_yy, s_xy], dim=-1)  # (ny, nx, 3)
    target = nu_t.unsqueeze(-1)                       # (ny, nx, 1)

    feats_flat = feats.reshape(-1, 3)
    target_flat = target.reshape(-1, 1)

    if mask is not None:
        if mask.shape != ux.shape:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} must match velocity "
                f"shape {tuple(ux.shape)}",
            )
        keep = (~mask.reshape(-1)).nonzero(as_tuple=False).squeeze(-1)
        feats_flat = feats_flat.index_select(0, keep)
        target_flat = target_flat.index_select(0, keep)

    return feats_flat.contiguous(), target_flat.contiguous()


def extract_les_samples_2d_multi(
    snapshots: Iterable[tuple[torch.Tensor, torch.Tensor]],
    c_s: float = 0.1,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate samples from many ``(ux, uy)`` snapshots."""
    feats_list: list[torch.Tensor] = []
    target_list: list[torch.Tensor] = []
    for ux, uy in snapshots:
        f, t = extract_les_samples_2d(ux, uy, c_s=c_s, mask=mask)
        feats_list.append(f)
        target_list.append(t)
    if not feats_list:
        return torch.zeros(0, 3), torch.zeros(0, 1)
    return torch.cat(feats_list, dim=0), torch.cat(target_list, dim=0)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

@dataclass
class EddyViscosityDataset:
    """In-memory dataset of (features, target) tensors plus metadata."""

    features: torch.Tensor   # (N, 3)
    targets: torch.Tensor    # (N, 1)
    c_s: float = 0.1
    description: str = ""

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def split(self, val_fraction: float = 0.1, seed: int = 0) -> tuple[
        EddyViscosityDataset, EddyViscosityDataset,
    ]:
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must be in (0, 1)")
        n = len(self)
        n_val = max(1, int(round(n * val_fraction)))
        g = torch.Generator().manual_seed(int(seed))
        perm = torch.randperm(n, generator=g)
        idx_val = perm[:n_val]
        idx_train = perm[n_val:]
        return (
            EddyViscosityDataset(
                features=self.features.index_select(0, idx_train),
                targets=self.targets.index_select(0, idx_train),
                c_s=self.c_s,
                description=self.description + " [train]",
            ),
            EddyViscosityDataset(
                features=self.features.index_select(0, idx_val),
                targets=self.targets.index_select(0, idx_val),
                c_s=self.c_s,
                description=self.description + " [val]",
            ),
        )


def save_dataset_pt(dataset: EddyViscosityDataset, path: str | Path) -> Path:
    """Serialize the dataset to a ``.pt`` file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": dataset.features,
            "targets": dataset.targets,
            "c_s": float(dataset.c_s),
            "description": dataset.description,
            "format_version": 1,
        },
        p,
    )
    return p


def load_dataset_pt(path: str | Path) -> EddyViscosityDataset:
    """Load a previously saved ``EddyViscosityDataset``."""
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    return EddyViscosityDataset(
        features=blob["features"],
        targets=blob["targets"],
        c_s=float(blob.get("c_s", 0.1)),
        description=str(blob.get("description", "")),
    )
