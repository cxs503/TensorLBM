from __future__ import annotations

import torch

from .d3q19 import OPPOSITE, equilibrium3d, macroscopic3d


def sphere_mask(
    nx: int,
    ny: int,
    nz: int,
    cx: float,
    cy: float,
    cz: float,
    radius: float,
    device: torch.device,
) -> torch.Tensor:
    """Boolean mask for a spherical obstacle in a 3D grid.

    Returns a tensor of shape (nz, ny, nx).
    """
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2 <= radius ** 2


def make_channel_wall_mask_3d(
    nz: int,
    ny: int,
    nx: int,
    obstacle_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Wall mask for a 3D channel: top/bottom (±y) and front/back (±z) faces.

    Returns a tensor of shape (nz, ny, nx).
    """
    wall_mask = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    wall_mask[:, 0, :] = True   # bottom (y=0)
    wall_mask[:, -1, :] = True  # top    (y=ny-1)
    wall_mask[0, :, :] = True   # front  (z=0)
    wall_mask[-1, :, :] = True  # back   (z=nz-1)
    wall_mask[obstacle_mask] = False
    return wall_mask


def bounce_back_cells_3d(f: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Bounce-back reflection on selected cells (obstacle/walls) for D3Q19."""
    bounced = f.clone()
    for i in range(19):
        bounced[i, mask] = f[OPPOSITE[i], mask]
    return bounced


def apply_simple_channel_boundaries_3d(
    f: torch.Tensor,
    u_in: float,
    wall_mask: torch.Tensor,
    obstacle_mask: torch.Tensor,
) -> torch.Tensor:
    """Minimal boundary treatment for a 3D channel.

    - Equilibrium inlet at x=0 with uniform x-velocity u_in.
    - Zero-gradient outlet at x=nx-1.
    - Bounce-back on walls and obstacle.

    Args:
        f: distribution tensor of shape (19, nz, ny, nx).
        u_in: inlet x-velocity.
        wall_mask: boolean tensor of shape (nz, ny, nx).
        obstacle_mask: boolean tensor of shape (nz, ny, nx).

    Returns:
        Updated distribution tensor.
    """
    rho, ux, uy, uz = macroscopic3d(f)

    # Inlet: x=0
    ux[:, :, 0] = u_in
    uy[:, :, 0] = 0.0
    uz[:, :, 0] = 0.0
    rho[:, :, 0] = rho[:, :, 1]
    feq_in = equilibrium3d(rho[:, :, 0:1], ux[:, :, 0:1], uy[:, :, 0:1], uz[:, :, 0:1])
    f[:, :, :, 0] = feq_in[:, :, :, 0]

    # Outlet: x=nx-1 (zero gradient)
    f[:, :, :, -1] = f[:, :, :, -2]

    f = bounce_back_cells_3d(f, wall_mask)
    f = bounce_back_cells_3d(f, obstacle_mask)
    return f
