"""Boundary conditions for the D3Q27 lattice.

Provides bounce-back and Zou/He non-equilibrium bounce-back (NEBB) boundary
conditions for the D3Q27 lattice, following the same conventions as
:mod:`tensorlbm.boundaries3d` for D3Q19.

D3Q27 direction index → (cx, cy, cz):
  0: (0,0,0)   rest
  1: (+1,0,0)  2: (-1,0,0)   3: (0,+1,0)  4: (0,-1,0)
  5: (0,0,+1)  6: (0,0,-1)
  7: (+1,+1,0) 8: (-1,+1,0)  9: (+1,-1,0) 10: (-1,-1,0)
  11:(+1,0,+1) 12:(-1,0,+1)  13:(+1,0,-1) 14:(-1,0,-1)
  15:(0,+1,+1) 16:(0,-1,+1)  17:(0,+1,-1) 18:(0,-1,-1)
  19:(+1,+1,+1) 20:(-1,+1,+1) 21:(+1,-1,+1) 22:(-1,-1,+1)
  23:(+1,+1,-1) 24:(-1,+1,-1) 25:(+1,-1,-1) 26:(-1,-1,-1)

Directions with cx > 0 (unknown at x=0 inlet):
  1, 7, 9, 11, 13, 19, 21, 23, 25

Directions with cx < 0 (unknown at x=nx-1 outlet):
  2, 8, 10, 12, 14, 20, 22, 24, 26
"""
from __future__ import annotations

import torch

from .d3q27 import OPPOSITE as OPPOSITE27
from .d3q27 import equilibrium27

# ---------------------------------------------------------------------------
# Module-level direction-index constants (derived from the fixed D3Q27 lattice).
# Using Python lists avoids .item() GPU→CPU sync inside hot BC functions.
# ---------------------------------------------------------------------------

# Directions with cx > 0 (unknown at x=0 inlet) and their cx<0 opposites
_D3Q27_INLET_DIRS: list[int] = [1, 7, 9, 11, 13, 19, 21, 23, 25]
_D3Q27_INLET_OPP: list[int] = [2, 8, 10, 12, 14, 20, 22, 24, 26]   # OPPOSITE[inlet]

# Directions with cx < 0 (unknown at x=nx-1 outlet) and their cx>0 opposites
_D3Q27_OUTLET_DIRS: list[int] = [2, 8, 10, 12, 14, 20, 22, 24, 26]
_D3Q27_OUTLET_OPP: list[int] = [1, 7, 9, 11, 13, 19, 21, 23, 25]   # OPPOSITE[outlet]


def bounce_back_cells_27(f: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Bounce-back reflection on selected cells for D3Q27.

    Uses ``torch.where`` instead of clone + scatter to reduce the number of
    GPU kernel launches and avoid an intermediate boolean-indexed allocation.

    Args:
        f: Distribution tensor of shape ``(27, nz, ny, nx)``.
        mask: Boolean tensor of shape ``(nz, ny, nx)`` marking solid cells.

    Returns:
        Updated distribution tensor with bounce-back applied to solid cells.
    """
    opp = OPPOSITE27.to(f.device)  # (27,)
    # mask.unsqueeze(0) broadcasts (1, nz, ny, nx) → (27, nz, ny, nx)
    return torch.where(mask.unsqueeze(0), f[opp], f)


def zou_he_inlet_velocity_27(
    f: torch.Tensor,
    u_in: float,
    uy_in: float = 0.0,
    uz_in: float = 0.0,
) -> torch.Tensor:
    """Zou/He inlet velocity BC at the left face (x=0) for D3Q27.

    Prescribes *ux = u_in*, *uy = uy_in*, *uz = uz_in* at every cell of the
    inlet plane.  The density at the inlet is derived from mass conservation
    and the unknown in-flowing populations (cx > 0) are reconstructed with
    the non-equilibrium bounce-back method.

    Directions with cx > 0 (unknown): 1, 7, 9, 11, 13, 19, 21, 23, 25
    Directions with cx < 0 (known):   2, 8, 10, 12, 14, 20, 22, 24, 26
    Directions with cx = 0:           0, 3, 4, 5, 6, 15, 16, 17, 18

    Args:
        f: Distribution tensor of shape ``(27, nz, ny, nx)``.
        u_in: Prescribed x-velocity at the inlet.
        uy_in: Prescribed y-velocity at the inlet (default 0).
        uz_in: Prescribed z-velocity at the inlet (default 0).

    Returns:
        Updated distribution tensor (same shape).
    """
    device = f.device

    # cx=0 directions: 0, 3, 4, 5, 6, 15, 16, 17, 18
    sum_cx0 = (
        f[0, :, :, 0] + f[3, :, :, 0] + f[4, :, :, 0]
        + f[5, :, :, 0] + f[6, :, :, 0]
        + f[15, :, :, 0] + f[16, :, :, 0]
        + f[17, :, :, 0] + f[18, :, :, 0]
    )
    # cx<0 directions: 2, 8, 10, 12, 14, 20, 22, 24, 26
    sum_cx_neg = (
        f[2, :, :, 0] + f[8, :, :, 0] + f[10, :, :, 0]
        + f[12, :, :, 0] + f[14, :, :, 0]
        + f[20, :, :, 0] + f[22, :, :, 0]
        + f[24, :, :, 0] + f[26, :, :, 0]
    )

    rho = (sum_cx0 + 2.0 * sum_cx_neg) / (1.0 - u_in)  # (nz, ny)

    # Wrap to 3-D to satisfy equilibrium27 which expects (nz, ny, nx)
    rho3 = rho.unsqueeze(-1)          # (nz, ny, 1)
    ux3 = torch.full_like(rho3, u_in)
    uy3 = torch.full_like(rho3, uy_in)
    uz3 = torch.full_like(rho3, uz_in)
    feq3 = equilibrium27(rho3, ux3, uy3, uz3, device=device)  # (27, nz, ny, 1)

    f_new = f.clone()
    # Vectorised update: no Python loop, no .item()
    f_new[_D3Q27_INLET_DIRS, :, :, 0] = (
        feq3[_D3Q27_INLET_DIRS, :, :, 0]
        - feq3[_D3Q27_INLET_OPP, :, :, 0]
        + f[_D3Q27_INLET_OPP, :, :, 0]
    )
    return f_new


def zou_he_outlet_pressure_27(f: torch.Tensor, rho_out: float = 1.0) -> torch.Tensor:
    """Zou/He pressure outlet BC at the right face (x=nx-1) for D3Q27.

    Prescribes *rho = rho_out* at the outlet plane.  Unknown populations
    (cx < 0) are reconstructed with non-equilibrium bounce-back.

    Directions with cx < 0 (unknown at outlet): 2, 8, 10, 12, 14, 20, 22, 24, 26
    Directions with cx > 0 (known):             1, 7, 9, 11, 13, 19, 21, 23, 25

    Args:
        f: Distribution tensor of shape ``(27, nz, ny, nx)``.
        rho_out: Prescribed outlet density (default 1.0).

    Returns:
        Updated distribution tensor (same shape).
    """
    device = f.device

    sum_cx0 = (
        f[0, :, :, -1] + f[3, :, :, -1] + f[4, :, :, -1]
        + f[5, :, :, -1] + f[6, :, :, -1]
        + f[15, :, :, -1] + f[16, :, :, -1]
        + f[17, :, :, -1] + f[18, :, :, -1]
    )
    sum_cx_pos = (
        f[1, :, :, -1] + f[7, :, :, -1] + f[9, :, :, -1]
        + f[11, :, :, -1] + f[13, :, :, -1]
        + f[19, :, :, -1] + f[21, :, :, -1]
        + f[23, :, :, -1] + f[25, :, :, -1]
    )
    ux_out = -1.0 + (sum_cx0 + 2.0 * sum_cx_pos) / rho_out

    rho_field = torch.full_like(f[0, :, :, -1], rho_out)  # (nz, ny)
    ux_field = ux_out                                       # (nz, ny)
    uy_field = torch.zeros_like(rho_field)
    uz_field = torch.zeros_like(rho_field)

    # Wrap to 3-D to satisfy equilibrium27 which expects (nz, ny, nx)
    rho3 = rho_field.unsqueeze(-1)  # (nz, ny, 1)
    ux3 = ux_field.unsqueeze(-1)
    uy3 = uy_field.unsqueeze(-1)
    uz3 = uz_field.unsqueeze(-1)
    feq3 = equilibrium27(rho3, ux3, uy3, uz3, device=device)  # (27, nz, ny, 1)

    f_new = f.clone()
    # Vectorised update: no Python loop, no .item()
    f_new[_D3Q27_OUTLET_DIRS, :, :, -1] = (
        feq3[_D3Q27_OUTLET_DIRS, :, :, 0]
        - feq3[_D3Q27_OUTLET_OPP, :, :, 0]
        + f[_D3Q27_OUTLET_OPP, :, :, -1]
    )
    return f_new


def make_channel_wall_mask_27(
    nz: int,
    ny: int,
    nx: int,
    obstacle_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Wall mask for a 3D channel (D3Q27): top/bottom (±y) and front/back (±z) faces.

    Args:
        nz: Grid depth.
        ny: Grid height.
        nx: Grid width.
        obstacle_mask: Boolean tensor of shape ``(nz, ny, nx)``; obstacle cells
            are excluded from the wall mask.
        device: Target PyTorch device.

    Returns:
        Boolean tensor of shape ``(nz, ny, nx)``.
    """
    wall_mask = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    wall_mask[:, 0, :] = True   # bottom (y=0)
    wall_mask[:, -1, :] = True  # top    (y=ny-1)
    wall_mask[0, :, :] = True   # front  (z=0)
    wall_mask[-1, :, :] = True  # back   (z=nz-1)
    wall_mask[obstacle_mask] = False
    return wall_mask


def far_field_bc_27(
    f: torch.Tensor,
    u_in: float,
    obstacle_mask: torch.Tensor | None = None,
    uy: float = 0.0,
    uz: float = 0.0,
) -> torch.Tensor:
    """Free-stream (Dirichlet) far-field boundary condition for D3Q27.

    D3Q27 analogue of :func:`tensorlbm.boundaries3d.far_field_bc_3d`.
    Imposes the free-stream equilibrium on the inlet and **all four lateral
    faces** (y±, z±), zero-gradient outlet at x=nx-1.  Removes blockage so
    the body sees effectively unbounded flow.
    """
    rho1 = torch.ones((f.shape[1], f.shape[2], f.shape[3]), dtype=f.dtype, device=f.device)
    feq = equilibrium27(
        rho1, torch.full_like(rho1, u_in), torch.full_like(rho1, uy),
        torch.full_like(rho1, uz), device=f.device,
    )
    f = f.clone()
    f[:, :, :, 0] = feq[:, :, :, 0]          # inlet (free stream)
    f[:, :, :, -1] = f[:, :, :, -2]          # outlet (zero gradient)
    f[:, 0, :, :] = feq[:, 0, :, :]          # y- lateral
    f[:, -1, :, :] = feq[:, -1, :, :]        # y+ lateral
    f[:, :, 0, :] = feq[:, :, 0, :]          # z- lateral
    f[:, :, -1, :] = feq[:, :, -1, :]        # z+ lateral
    if obstacle_mask is not None:
        f = bounce_back_cells_27(f, obstacle_mask)
    return f


def apply_zou_he_channel_boundaries_27(
    f: torch.Tensor,
    u_in: float,
    wall_mask: torch.Tensor,
    obstacle_mask: torch.Tensor,
) -> torch.Tensor:
    """Channel boundaries using Zou/He inlet and pressure outlet for D3Q27.

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


__all__ = [
    "bounce_back_cells_27",
    "zou_he_inlet_velocity_27",
    "zou_he_outlet_pressure_27",
    "make_channel_wall_mask_27",
    "apply_zou_he_channel_boundaries_27",
    "far_field_bc_27",
]
