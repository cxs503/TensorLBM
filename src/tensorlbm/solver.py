from __future__ import annotations

"""Core LBM solver steps for a minimal D2Q9 flow simulation."""

import torch

from .d2q9 import C, OPPOSITE, equilibrium, macroscopic


def cylinder_mask(nx: int, ny: int, cx: float, cy: float, radius: float, device: torch.device) -> torch.Tensor:
    """Boolean mask for circular obstacle in a 2D grid."""
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2


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


def bounce_back_cells(f: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Bounce-back reflection on selected cells (obstacle/walls)."""
    bounced = f.clone()
    for i in range(9):
        bounced[i, mask] = f[OPPOSITE[i], mask]
    return bounced


def apply_simple_channel_boundaries(
    f: torch.Tensor,
    u_in: float,
    wall_mask: torch.Tensor,
    obstacle_mask: torch.Tensor,
) -> torch.Tensor:
    """Minimal boundary treatment: equilibrium inlet, zero-gradient outlet, bounce-back walls/obstacle."""
    rho, ux, uy = macroscopic(f)

    # Inlet (left): reset to equilibrium with fixed velocity
    ux[:, 0] = u_in
    uy[:, 0] = 0.0
    rho[:, 0] = rho[:, 1]
    feq_in = equilibrium(rho[:, 0:1], ux[:, 0:1], uy[:, 0:1])
    f[:, :, 0] = feq_in[:, :, 0]

    # Outlet (right): simple zero-gradient copy
    f[:, :, -1] = f[:, :, -2]

    # No-slip bounce-back on walls and obstacle
    f = bounce_back_cells(f, wall_mask)
    f = bounce_back_cells(f, obstacle_mask)
    return f
