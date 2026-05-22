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
    opp = OPPOSITE.to(f.device)  # (19,)
    bounced[:, mask] = f[opp][:, mask]
    return bounced


def zou_he_inlet_velocity_3d(
    f: torch.Tensor,
    u_in: float,
    uy_in: float = 0.0,
    uz_in: float = 0.0,
) -> torch.Tensor:
    """Zou/He inlet velocity boundary condition at the left face (x=0) for D3Q19.

    Prescribes *ux = u_in*, *uy = uy_in*, *uz = uz_in* at every cell of the
    inlet plane.  The density at the inlet is derived from mass conservation
    and the unknown in-flowing populations are reconstructed with the
    non-equilibrium bounce-back method (Latt & Chopard 2008).

    Args:
        f: Distribution tensor of shape ``(19, nz, ny, nx)``.
        u_in: Prescribed x-velocity at the inlet.
        uy_in: Prescribed y-velocity at the inlet (default 0).
        uz_in: Prescribed z-velocity at the inlet (default 0).

    Returns:
        Updated distribution tensor (same shape).
    """
    device = f.device
    # Directions with cx > 0 (unknown after streaming from outside):
    #   1:(1,0,0), 7:(1,1,0), 9:(1,-1,0), 11:(1,0,1), 13:(1,0,-1)
    # Their opposites (cx < 0, known):
    #   2:(-1,0,0), 8:(-1,-1,0), 10:(-1,1,0), 12:(-1,0,-1), 14:(-1,0,1)
    #
    # Step 1: Determine rho from mass + x-momentum balance
    sum_cx0 = (
        f[0] + f[3] + f[4] + f[5] + f[6]
        + f[15] + f[16] + f[17] + f[18]
    )  # all directions with cx=0
    sum_cx_neg = f[2] + f[8] + f[10] + f[12] + f[14]  # directions with cx=-1
    rho = (sum_cx0 + 2.0 * sum_cx_neg) / (1.0 - u_in)

    # Step 2: Compute equilibrium at (rho, u_in, uy_in, uz_in)
    ux_field = torch.full_like(rho, u_in)
    uy_field = torch.full_like(rho, uy_in)
    uz_field = torch.full_like(rho, uz_in)
    feq = equilibrium3d(rho, ux_field, uy_field, uz_field, device=device)

    # Step 3: Non-equilibrium bounce-back for each cx>0 direction k:
    #   f[k] = feq[k] - feq[OPPOSITE[k]] + f[OPPOSITE[k]]
    f_new = f.clone()
    for k in (1, 7, 9, 11, 13):
        opp = int(OPPOSITE[k].item())
        f_new[k, :, :, 0] = feq[k, :, :, 0] - feq[opp, :, :, 0] + f[opp, :, :, 0]
    return f_new


def zou_he_outlet_pressure_3d(f: torch.Tensor, rho_out: float = 1.0) -> torch.Tensor:
    """Zou/He pressure boundary condition at the right face (x=nx-1) for D3Q19.

    Prescribes *rho = rho_out* at the outlet plane.  The unknown populations
    (cx < 0) are reconstructed with non-equilibrium bounce-back.

    Args:
        f: Distribution tensor of shape ``(19, nz, ny, nx)``.
        rho_out: Prescribed outlet density (default 1.0).

    Returns:
        Updated distribution tensor (same shape).
    """
    device = f.device
    # Directions with cx < 0 (unknown at outlet): 2,8,10,12,14
    # Their opposites (cx > 0, known): 1,7,9,11,13
    sum_cx0 = (
        f[0, :, :, -1] + f[3, :, :, -1] + f[4, :, :, -1]
        + f[5, :, :, -1] + f[6, :, :, -1]
        + f[15, :, :, -1] + f[16, :, :, -1]
        + f[17, :, :, -1] + f[18, :, :, -1]
    )
    sum_cx_pos = (
        f[1, :, :, -1] + f[7, :, :, -1] + f[9, :, :, -1]
        + f[11, :, :, -1] + f[13, :, :, -1]
    )
    ux_out = -1.0 + (sum_cx0 + 2.0 * sum_cx_pos) / rho_out

    rho_field = torch.full_like(f[0, :, :, -1], rho_out)  # (nz, ny)
    ux_field = ux_out                                       # (nz, ny)
    uy_field = torch.zeros_like(rho_field)
    uz_field = torch.zeros_like(rho_field)
    feq = equilibrium3d(rho_field, ux_field, uy_field, uz_field, device=device)  # (19, nz, ny)

    f_new = f.clone()
    for k in (2, 8, 10, 12, 14):
        opp = int(OPPOSITE[k].item())
        f_new[k, :, :, -1] = feq[k] - feq[opp] + f[opp, :, :, -1]
    return f_new


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


def apply_zou_he_channel_boundaries_3d(
    f: torch.Tensor,
    u_in: float,
    wall_mask: torch.Tensor,
    obstacle_mask: torch.Tensor,
) -> torch.Tensor:
    """Channel boundaries using Zou/He inlet and pressure outlet for D3Q19.

    Drop-in replacement for :func:`apply_simple_channel_boundaries_3d`.

    Args:
        f: Distribution tensor of shape ``(19, nz, ny, nx)``.
        u_in: Inlet x-velocity.
        wall_mask: Boolean tensor of shape ``(nz, ny, nx)``.
        obstacle_mask: Boolean tensor of shape ``(nz, ny, nx)``.

    Returns:
        Updated distribution tensor.
    """
    f = zou_he_inlet_velocity_3d(f, u_in)
    f = zou_he_outlet_pressure_3d(f)
    f = bounce_back_cells_3d(f, wall_mask)
    f = bounce_back_cells_3d(f, obstacle_mask)
    return f
