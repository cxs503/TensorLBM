from __future__ import annotations

import torch

from .boundaries import apply_simple_channel_boundaries, bounce_back_cells, cylinder_mask, make_channel_wall_mask
from .d2q9 import C, equilibrium, macroscopic


def collide_bgk(f: torch.Tensor, tau: float) -> torch.Tensor:
    """Single-relaxation-time BGK collision step."""
    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)
    return f - (f - feq) / tau


def stream(f: torch.Tensor) -> torch.Tensor:
    """Streaming by shifting each discrete direction."""
    streamed = torch.empty_like(f)
    for i in range(9):
        cx, cy = int(C[i, 0].item()), int(C[i, 1].item())
        streamed[i] = torch.roll(f[i], shifts=(cy, cx), dims=(0, 1))
    return streamed


__all__ = [
    "cylinder_mask",
    "make_channel_wall_mask",
    "bounce_back_cells",
    "apply_simple_channel_boundaries",
    "collide_bgk",
    "stream",
]
