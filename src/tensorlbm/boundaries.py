from __future__ import annotations

import torch

from .d2q9 import OPPOSITE, equilibrium, macroscopic


def cylinder_mask(nx: int, ny: int, cx: float, cy: float, radius: float, device: torch.device) -> torch.Tensor:
    """Boolean mask for circular obstacle in a 2D grid."""
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2


def make_channel_wall_mask(ny: int, nx: int, obstacle_mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Top/bottom wall mask excluding obstacle cells."""
    wall_mask = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall_mask[0, :] = True
    wall_mask[-1, :] = True
    wall_mask[obstacle_mask] = False
    return wall_mask


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

    ux[:, 0] = u_in
    uy[:, 0] = 0.0
    rho[:, 0] = rho[:, 1]
    feq_in = equilibrium(rho[:, 0:1], ux[:, 0:1], uy[:, 0:1])
    f[:, :, 0] = feq_in[:, :, 0]

    f[:, :, -1] = f[:, :, -2]

    f = bounce_back_cells(f, wall_mask)
    f = bounce_back_cells(f, obstacle_mask)
    return f
