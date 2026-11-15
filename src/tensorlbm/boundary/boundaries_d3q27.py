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

from tensorlbm.d3q27 import OPPOSITE as OPPOSITE27
from tensorlbm.d3q27 import equilibrium27
from ..input_contract import check_positive_scalar
from .zou_he_3d import plane_field

# ---------------------------------------------------------------------------
# Module-level direction-index constants (derived from the fixed D3Q27 lattice).
# Using Python lists avoids .item() GPU→CPU sync inside hot BC functions.
# ---------------------------------------------------------------------------

# Directions with cx > 0 (unknown at x=0 inlet) and their cx<0 opposites
_D3Q27_INLET_DIRS: list[int] = [1, 7, 9, 11, 13, 19, 21, 23, 25]
_D3Q27_INLET_OPP: list[int] = [2, 8, 10, 12, 14, 20, 22, 24, 26]  # OPPOSITE[inlet]

# Directions with cx < 0 (unknown at x=nx-1 outlet) and their cx>0 opposites
_D3Q27_OUTLET_DIRS: list[int] = [2, 8, 10, 12, 14, 20, 22, 24, 26]
_D3Q27_OUTLET_OPP: list[int] = [1, 7, 9, 11, 13, 19, 21, 23, 25]  # OPPOSITE[outlet]


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


def far_field_bc_27(
    f: torch.Tensor,
    u_in: float,
    obstacle_mask: torch.Tensor | None = None,
    uy: float = 0.0,
    uz: float = 0.0,
) -> torch.Tensor:
    """Free-stream (Dirichlet) far-field boundary condition for D3Q27.

    Mirrors :func:`tensorlbm.boundaries3d.far_field_bc_3d` for the D3Q27
    lattice.  Imposes the free-stream equilibrium on the inlet and all
    four lateral faces (y±, z±), zero-gradient outlet at x=nx-1.

    Args:
        f: Distribution tensor of shape ``(27, nz, ny, nx)``.
        u_in: Free-stream x-velocity.
        obstacle_mask: Optional solid mask; if provided, bounce-back is
            applied to solid cells after the far-field update.
        uy: Free-stream y-velocity (default 0).
        uz: Free-stream z-velocity (default 0).

    Returns:
        Updated distribution tensor (same shape).
    """
    rho1 = torch.ones((f.shape[1], f.shape[2], f.shape[3]), dtype=f.dtype, device=f.device)
    feq = equilibrium27(
        rho1,
        torch.full_like(rho1, u_in),
        torch.full_like(rho1, uy),
        torch.full_like(rho1, uz),
        device=f.device,
    )
    f = f.clone()
    f[:, :, :, 0] = feq[:, :, :, 0]  # inlet (free stream)
    f[:, :, :, -1] = f[:, :, :, -2]  # outlet (zero gradient)
    f[:, 0, :, :] = feq[:, 0, :, :]  # y- lateral
    f[:, -1, :, :] = feq[:, -1, :, :]  # y+ lateral
    f[:, :, 0, :] = feq[:, :, 0, :]  # z- lateral
    f[:, :, -1, :] = feq[:, :, -1, :]  # z+ lateral
    if obstacle_mask is not None:
        f = bounce_back_cells_27(f, obstacle_mask)
    return f


# ---------------------------------------------------------------------------
# Zou/He x-face inlet/outlet reconstruction (D3Q27).
#
# Implementations live in ``zou_he_27`` (shared closure core in ``zou_he_3d``);
# re-exported here because this module is the historical import site.
# ---------------------------------------------------------------------------
from .zou_he_27 import (  # noqa: E402  (re-export after the module body)
    zou_he_inlet_pressure_27,
    zou_he_inlet_velocity_27,
    zou_he_outlet_pressure_27,
    zou_he_outlet_velocity_27,
)


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
    wall_mask[:, 0, :] = True  # bottom (y=0)
    wall_mask[:, -1, :] = True  # top    (y=ny-1)
    wall_mask[0, :, :] = True  # front  (z=0)
    wall_mask[-1, :, :] = True  # back   (z=nz-1)
    wall_mask[obstacle_mask] = False
    return wall_mask


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
]
