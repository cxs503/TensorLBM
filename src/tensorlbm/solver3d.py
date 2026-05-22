from __future__ import annotations

import torch

from .d3q19 import C, equilibrium3d, macroscopic3d


def collide_bgk3d(f: torch.Tensor, tau: float) -> torch.Tensor:
    """Single-relaxation-time BGK collision step for D3Q19."""
    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    return f - (f - feq) / tau


def stream3d(f: torch.Tensor) -> torch.Tensor:
    """Streaming by shifting each discrete direction for D3Q19.

    Args:
        f: distribution tensor of shape (19, nz, ny, nx).

    Returns:
        Streamed tensor of the same shape.
    """
    streamed = torch.empty_like(f)
    for i in range(19):
        cx, cy, cz = int(C[i, 0].item()), int(C[i, 1].item()), int(C[i, 2].item())
        # f[i] has shape (nz, ny, nx); dims=(0,1,2) correspond to z, y, x
        streamed[i] = torch.roll(f[i], shifts=(cz, cy, cx), dims=(0, 1, 2))
    return streamed


__all__ = [
    "collide_bgk3d",
    "stream3d",
]
