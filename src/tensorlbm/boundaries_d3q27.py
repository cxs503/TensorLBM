"""Boundary conditions for D3Q27 channel-flow simulations.

Provides:
- :func:`bounce_back_cells_27`             – full bounce-back on solid cells
- :func:`make_channel_wall_mask_27`        – top/bottom + front/back wall mask
- :func:`zou_he_inlet_velocity_27`         – Zou/He inlet velocity BC (x=0)
- :func:`zou_he_outlet_pressure_27`        – Zou/He pressure BC (x=nx-1)
- :func:`apply_zou_he_channel_boundaries_27` – combined channel BC helper
"""

from __future__ import annotations

import torch

from .d3q27 import OPPOSITE, C, equilibrium27


def bounce_back_cells_27(f: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Full bounce-back reflection on selected cells for D3Q27.

    Args:
        f: Distribution tensor of shape ``(27, nz, ny, nx)``.
        mask: Boolean tensor of shape ``(nz, ny, nx)`` marking solid cells.

    Returns:
        Updated distribution tensor.
    """
    bounced = f.clone()
    opp = OPPOSITE.to(f.device)  # (27,)
    bounced[:, mask] = f[opp][:, mask]
    return bounced


def make_channel_wall_mask_27(
    nz: int,
    ny: int,
    nx: int,
    obstacle_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Wall mask for a 3D channel: ±y and ±z faces.

    Args:
        nz, ny, nx: Grid dimensions.
        obstacle_mask: Boolean tensor of shape ``(nz, ny, nx)``; obstacle
            cells are excluded from the wall mask.
        device: Target device.

    Returns:
        Boolean tensor of shape ``(nz, ny, nx)``.
    """
    wall_mask = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    wall_mask[:, 0, :] = True    # bottom (y=0)
    wall_mask[:, -1, :] = True   # top    (y=ny-1)
    wall_mask[0, :, :] = True    # front  (z=0)
    wall_mask[-1, :, :] = True   # back   (z=nz-1)
    wall_mask[obstacle_mask] = False
    return wall_mask


def zou_he_inlet_velocity_27(
    f: torch.Tensor,
    u_in: float,
    uy_in: float = 0.0,
    uz_in: float = 0.0,
) -> torch.Tensor:
    """Zou/He inlet velocity boundary condition at the left face (x=0) for D3Q27.

    Prescribes *ux = u_in*, *uy = uy_in*, *uz = uz_in* at the inlet plane
    using the non-equilibrium bounce-back method (Latt & Chopard 2008).

    Args:
        f: Distribution tensor of shape ``(27, nz, ny, nx)``.
        u_in: Prescribed x-velocity at the inlet.
        uy_in: Prescribed y-velocity at the inlet (default 0).
        uz_in: Prescribed z-velocity at the inlet (default 0).

    Returns:
        Updated distribution tensor.
    """
    device = f.device

    # cx > 0 directions (incoming at x=0): 1, 7, 9, 11, 13, 19, 21, 23, 25
    # cx = 0 directions: 0, 3, 4, 5, 6, 15, 16, 17, 18
    # cx < 0 directions: 2, 8, 10, 12, 14, 20, 22, 24, 26

    cx_pos = [1, 7, 9, 11, 13, 19, 21, 23, 25]
    cx_zero = [0, 3, 4, 5, 6, 15, 16, 17, 18]
    cx_neg = [2, 8, 10, 12, 14, 20, 22, 24, 26]

    sum_cx0 = sum(f[k, :, :, 0] for k in cx_zero)
    sum_cx_neg = sum(f[k, :, :, 0] for k in cx_neg)
    rho = (sum_cx0 + 2.0 * sum_cx_neg) / (1.0 - u_in)  # (nz, ny)

    ux_field = torch.full_like(rho, u_in)
    uy_field = torch.full_like(rho, uy_in)
    uz_field = torch.full_like(rho, uz_in)
    feq = equilibrium27(rho, ux_field, uy_field, uz_field, device=device)  # (27, nz, ny)

    f_new = f.clone()
    for k in cx_pos:
        opp = int(OPPOSITE[k].item())
        f_new[k, :, :, 0] = feq[k] - feq[opp] + f[opp, :, :, 0]
    return f_new


def zou_he_outlet_pressure_27(f: torch.Tensor, rho_out: float = 1.0) -> torch.Tensor:
    """Zou/He pressure boundary condition at the right face (x=nx-1) for D3Q27.

    Prescribes *rho = rho_out* at the outlet plane; the unknown (cx<0)
    populations are reconstructed with non-equilibrium bounce-back.

    Args:
        f: Distribution tensor of shape ``(27, nz, ny, nx)``.
        rho_out: Prescribed outlet density (default 1.0).

    Returns:
        Updated distribution tensor.
    """
    device = f.device

    cx_pos = [1, 7, 9, 11, 13, 19, 21, 23, 25]
    cx_zero = [0, 3, 4, 5, 6, 15, 16, 17, 18]
    cx_neg = [2, 8, 10, 12, 14, 20, 22, 24, 26]

    sum_cx0 = sum(f[k, :, :, -1] for k in cx_zero)
    sum_cx_pos = sum(f[k, :, :, -1] for k in cx_pos)
    ux_out = -1.0 + (sum_cx0 + 2.0 * sum_cx_pos) / rho_out  # (nz, ny)

    rho_field = torch.full_like(ux_out, rho_out)
    uy_field = torch.zeros_like(ux_out)
    uz_field = torch.zeros_like(ux_out)
    feq = equilibrium27(rho_field, ux_out, uy_field, uz_field, device=device)  # (27, nz, ny)

    f_new = f.clone()
    for k in cx_neg:
        opp = int(OPPOSITE[k].item())
        f_new[k, :, :, -1] = feq[k] - feq[opp] + f[opp, :, :, -1]
    return f_new


def apply_zou_he_channel_boundaries_27(
    f: torch.Tensor,
    u_in: float,
    wall_mask: torch.Tensor,
    obstacle_mask: torch.Tensor,
) -> torch.Tensor:
    """Full D3Q27 channel BC: Zou/He inlet + pressure outlet + bounce-back walls.

    Drop-in replacement for the D3Q19 equivalent.

    Args:
        f: Distribution tensor of shape ``(27, nz, ny, nx)``.
        u_in: Inlet x-velocity.
        wall_mask: Boolean tensor of shape ``(nz, ny, nx)``.
        obstacle_mask: Boolean tensor of shape ``(nz, ny, nx)``.

    Returns:
        Updated distribution tensor.
    """
    f = zou_he_inlet_velocity_27(f, u_in)
    f = zou_he_outlet_pressure_27(f)
    f = bounce_back_cells_27(f, wall_mask)
    f = bounce_back_cells_27(f, obstacle_mask)
    return f


def compute_obstacle_forces_27(
    f: torch.Tensor,
    obstacle_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Momentum-exchange drag, lift, and side forces for D3Q27.

    Implements the Ladd (1994) momentum-exchange method for D3Q27.
    Must be called **after streaming but before bounce-back**.

    Args:
        f: Distribution tensor of shape ``(27, nz, ny, nx)`` *after* streaming.
        obstacle_mask: Boolean tensor of shape ``(nz, ny, nx)`` marking solid cells.

    Returns:
        Tuple ``(fx, fy, fz)`` — scalar force tensors.
    """
    device = f.device
    c = C.to(device).float()  # (27, 3)

    cx = c[:, 0].view(27, 1, 1, 1)
    cy = c[:, 1].view(27, 1, 1, 1)
    cz = c[:, 2].view(27, 1, 1, 1)

    mask_4d = obstacle_mask.unsqueeze(0)
    f_solid = f * mask_4d

    fx = 2.0 * (cx * f_solid).sum()
    fy = 2.0 * (cy * f_solid).sum()
    fz = 2.0 * (cz * f_solid).sum()
    return fx, fy, fz


__all__ = [
    "bounce_back_cells_27",
    "make_channel_wall_mask_27",
    "zou_he_inlet_velocity_27",
    "zou_he_outlet_pressure_27",
    "apply_zou_he_channel_boundaries_27",
    "compute_obstacle_forces_27",
]
