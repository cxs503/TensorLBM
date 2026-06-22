from __future__ import annotations

import torch

from .d3q19 import OPPOSITE, equilibrium3d, macroscopic3d

# ---------------------------------------------------------------------------
# Module-level direction-index constants (derived from the fixed D3Q19 lattice).
# Using Python lists here avoids .item() GPU→CPU sync inside hot BC functions.
# ---------------------------------------------------------------------------

# Directions with cx > 0 (unknown at x=0 inlet) and their cx<0 opposites
_D3Q19_INLET_DIRS: list[int] = [1, 7, 9, 11, 13]
_D3Q19_INLET_OPP: list[int] = [2, 8, 10, 12, 14]   # OPPOSITE[inlet_dirs]

# Directions with cx < 0 (unknown at x=nx-1 outlet) and their cx>0 opposites
_D3Q19_OUTLET_DIRS: list[int] = [2, 8, 10, 12, 14]
_D3Q19_OUTLET_OPP: list[int] = [1, 7, 9, 11, 13]   # OPPOSITE[outlet_dirs]

# Directions with cz > 0 (unknown at z=0 bottom inlet) and their cz<0 opposites
_D3Q19_ZINLET_DIRS: list[int] = [5, 11, 14, 15, 18]
_D3Q19_ZINLET_OPP: list[int] = [6, 12, 13, 16, 17]  # OPPOSITE[z-inlet_dirs]

# Directions with cz < 0 (unknown at z=nz-1 top outlet) and their cz>0 opposites
_D3Q19_ZOUTLET_DIRS: list[int] = [6, 12, 13, 16, 17]
_D3Q19_ZOUTLET_OPP: list[int] = [5, 11, 14, 15, 18]  # OPPOSITE[z-outlet_dirs]


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


# ---------------------------------------------------------------------------
# D3Q19 mirror tables for FreeSlip boundary (specular reflection)
# ---------------------------------------------------------------------------
# For each direction i, MIRROR_X[i] gives the direction with (cx→-cx, cy, cz).
# Similarly for MIRROR_Y (cx, -cy, cz) and MIRROR_Z (cx, cy, -cz).
#
# Computed from the D3Q19 stencil:
#   dir  0: ( 0, 0, 0) → mx: 0, my: 0, mz: 0
#   dir  1: ( 1, 0, 0) → mx: 2, my: 1, mz: 1
#   dir  2: (-1, 0, 0) → mx: 1, my: 2, mz: 2
#   dir  3: ( 0, 1, 0) → mx: 3, my: 4, mz: 3
#   dir  4: ( 0,-1, 0) → mx: 4, my: 3, mz: 4
#   dir  5: ( 0, 0, 1) → mx: 5, my: 5, mz: 6
#   dir  6: ( 0, 0,-1) → mx: 6, my: 6, mz: 5
#   dir  7: ( 1, 1, 0) → mx:10, my: 9, mz: 7
#   dir  8: (-1,-1, 0) → mx: 9, my: 7, mz: 8
#   dir  9: ( 1,-1, 0) → mx: 8, my:10, mz: 9
#   dir 10: (-1, 1, 0) → mx: 7, my: 8, mz:10
#   dir 11: ( 1, 0, 1) → mx:14, my:11, mz:13
#   dir 12: (-1, 0,-1) → mx:13, my:12, mz:14
#   dir 13: ( 1, 0,-1) → mx:12, my:13, mz:11
#   dir 14: (-1, 0, 1) → mx:11, my:14, mz:12
#   dir 15: ( 0, 1, 1) → mx:15, my:18, mz:17
#   dir 16: ( 0,-1,-1) → mx:16, my:17, mz:18
#   dir 17: ( 0, 1,-1) → mx:17, my:16, mz:15
#   dir 18: ( 0,-1, 1) → mx:18, my:15, mz:16
_D3Q19_MIRROR_X = torch.tensor(
    [0, 2, 1, 3, 4, 5, 6, 10, 9, 8, 7, 14, 13, 12, 11, 15, 16, 17, 18],
    dtype=torch.int64,
)
_D3Q19_MIRROR_Y = torch.tensor(
    [0, 1, 2, 4, 3, 5, 6, 9, 10, 7, 8, 11, 12, 13, 14, 18, 17, 16, 15],
    dtype=torch.int64,
)
_D3Q19_MIRROR_Z = torch.tensor(
    [0, 1, 2, 3, 4, 6, 5, 7, 8, 9, 10, 13, 14, 11, 12, 17, 18, 15, 16],
    dtype=torch.int64,
)


def bounce_back_cells_3d(f: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Bounce-back reflection on selected cells (obstacle/walls) for D3Q19.

    Uses ``torch.where`` instead of clone + scatter to reduce the number of
    GPU kernel launches and avoid an intermediate boolean-indexed allocation.
    """
    opp = OPPOSITE.to(f.device)  # (19,)
    # mask.unsqueeze(0) broadcasts (1, nz, ny, nx) → (19, nz, ny, nx)
    return torch.where(mask.unsqueeze(0), f[opp], f)


def free_slip_cells_3d(
    f: torch.Tensor,
    mask: torch.Tensor,
    axis: int = 0,
) -> torch.Tensor:
    """Specular (free-slip) reflection for D3Q19 walls.

    Unlike bounce-back which reverses ALL velocity components (no-slip),
    free-slip only reverses the wall-normal component, preserving tangential
    momentum.  This is the standard approach used by waLBerla's FreeSlip
    boundary (``lbm::FreeSlip``) for dam-break walls.

    Implemented via pre-computed mirror tables:
      - ``axis=0`` (x-wall): mirror_x — flip cx, keep cy, cz
      - ``axis=1`` (y-wall): mirror_y — flip cy, keep cx, cz
      - ``axis=2`` (z-wall): mirror_z — flip cz, keep cx, cy

    Args:
        f:     Distribution tensor, shape ``(19, nz, ny, nx)``.
        mask:  Boolean mask ``(nz, ny, nx)`` of wall cells.
        axis:  Which coordinate axis is wall-normal (0=x, 1=y, 2=z).

    Returns:
        Updated distribution tensor with specular reflection at wall cells.

    References
    ----------
    waLBerla ``src/lbm/boundary/FreeSlip.h`` (specular reflection pattern).
    """
    _mirrors = {0: _D3Q19_MIRROR_X, 1: _D3Q19_MIRROR_Y, 2: _D3Q19_MIRROR_Z}
    mirror = _mirrors[axis].to(f.device)  # (19,)
    return torch.where(mask.unsqueeze(0), f[mirror], f)


def free_slip_y_walls_3d(
    f: torch.Tensor,
    wall_mask: torch.Tensor,
) -> torch.Tensor:
    """Convenience wrapper: FreeSlip on y-faces (top and bottom).

    Uses ``free_slip_cells_3d(f, wall_mask, axis=1)``.  This is the
    recommended boundary for dam-break top/bottom walls per waLBerla's
    ``DamBreakRectangular.prm`` (all walls FreeSlip).

    Args:
        f:          Distribution tensor, shape ``(19, nz, ny, nx)``.
        wall_mask:  Boolean mask ``(nz, ny, nx)`` — ``True`` at wall cells.

    Returns:
        Updated distribution tensor.
    """
    return free_slip_cells_3d(f, wall_mask, axis=1)


def free_slip_x_walls_3d(
    f: torch.Tensor,
    wall_mask: torch.Tensor,
) -> torch.Tensor:
    """Convenience wrapper: FreeSlip on x-faces (left and right)."""
    return free_slip_cells_3d(f, wall_mask, axis=0)


def free_slip_z_walls_3d(
    f: torch.Tensor,
    wall_mask: torch.Tensor,
) -> torch.Tensor:
    """Convenience wrapper: FreeSlip on z-faces (front and back)."""
    return free_slip_cells_3d(f, wall_mask, axis=2)


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
    # Step 1: Determine rho from mass + x-momentum balance at x=0 only
    sum_cx0 = (
        f[0, :, :, 0] + f[3, :, :, 0] + f[4, :, :, 0]
        + f[5, :, :, 0] + f[6, :, :, 0]
        + f[15, :, :, 0] + f[16, :, :, 0] + f[17, :, :, 0] + f[18, :, :, 0]
    )  # cx=0 directions at x=0  → shape (nz, ny)
    sum_cx_neg = (
        f[2, :, :, 0] + f[8, :, :, 0] + f[10, :, :, 0]
        + f[12, :, :, 0] + f[14, :, :, 0]
    )  # cx<0 directions at x=0 → shape (nz, ny)
    rho = (sum_cx0 + 2.0 * sum_cx_neg) / (1.0 - u_in)  # (nz, ny)

    # Step 2: Compute equilibrium at (rho, u_in, uy_in, uz_in) for the inlet plane only.
    # Unsqueeze to (nz, ny, 1) so equilibrium3d produces (19, nz, ny, 1).
    rho3 = rho.unsqueeze(-1)               # (nz, ny, 1)
    ux_field = torch.full_like(rho3, u_in)
    uy_field = torch.full_like(rho3, uy_in)
    uz_field = torch.full_like(rho3, uz_in)
    feq = equilibrium3d(rho3, ux_field, uy_field, uz_field, device=device)  # (19, nz, ny, 1)

    # Step 3: Non-equilibrium bounce-back (vectorised, no Python loop, no .item())
    #   f[k, :, :, 0] = feq[k, :, :, 0] - feq[opp_k, :, :, 0] + f[opp_k, :, :, 0]
    f_new = f
    f_new[_D3Q19_INLET_DIRS, :, :, 0] = (
        feq[_D3Q19_INLET_DIRS, :, :, 0]
        - feq[_D3Q19_INLET_OPP, :, :, 0]
        + f[_D3Q19_INLET_OPP, :, :, 0]
    )
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
    # Unsqueeze to (nz, ny, 1) so equilibrium3d produces (19, nz, ny, 1)
    feq = equilibrium3d(
        rho_field.unsqueeze(-1),
        ux_field.unsqueeze(-1),
        uy_field.unsqueeze(-1),
        uz_field.unsqueeze(-1),
        device=device,
    )  # (19, nz, ny, 1)

    # Vectorised update: no Python loop, no .item()
    f_new = f
    f_new[_D3Q19_OUTLET_DIRS, :, :, -1] = (
        feq[_D3Q19_OUTLET_DIRS, :, :, 0]
        - feq[_D3Q19_OUTLET_OPP, :, :, 0]
        + f[_D3Q19_OUTLET_OPP, :, :, -1]
    )
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


def zou_he_inlet_velocity_z(
    f: torch.Tensor,
    uz_in: float,
    ux_in: float = 0.0,
    uy_in: float = 0.0,
) -> torch.Tensor:
    """Zou/He inlet velocity BC at bottom face (z=0) for D3Q19.

    Prescribes upward z-velocity *uz = uz_in* at every cell of the inlet plane.
    Unknown populations (cz > 0, directions 5, 11, 14, 15, 18) are
    reconstructed using the non-equilibrium bounce-back method
    (Latt & Chopard 2008).  The local density is inferred from the
    mass + z-momentum balance:

    .. math::

        \\rho = \\frac{\\sum_{c_z=0} f_k + 2 \\sum_{c_z<0} f_k}{1 - u_z}

    Args:
        f: Distribution tensor of shape ``(19, nz, ny, nx)``.
        uz_in: Prescribed z-velocity at the inlet (positive = upward).
        ux_in: Prescribed x-velocity at the inlet (default 0).
        uy_in: Prescribed y-velocity at the inlet (default 0).

    Returns:
        Updated distribution tensor (same shape).
    """
    device = f.device
    # At z=0: f[q, 0, :, :] has shape (ny, nx) for each direction q.
    # cz=0 directions: 0,1,2,3,4,7,8,9,10
    sum_cz0 = (
        f[0, 0] + f[1, 0] + f[2, 0] + f[3, 0] + f[4, 0]
        + f[7, 0] + f[8, 0] + f[9, 0] + f[10, 0]
    )  # (ny, nx)
    # cz<0 directions: 6,12,13,16,17
    sum_cz_neg = f[6, 0] + f[12, 0] + f[13, 0] + f[16, 0] + f[17, 0]  # (ny, nx)

    rho = (sum_cz0 + 2.0 * sum_cz_neg) / (1.0 - uz_in)  # (ny, nx)

    # Equilibrium at (rho, ux_in, uy_in, uz_in); shape (1, ny, nx) for the z=0 slice
    rho3 = rho.unsqueeze(0)  # (1, ny, nx)
    ux3 = torch.full_like(rho3, ux_in)
    uy3 = torch.full_like(rho3, uy_in)
    uz3 = torch.full_like(rho3, uz_in)
    feq = equilibrium3d(rho3, ux3, uy3, uz3, device=device)  # (19, 1, ny, nx)

    # Vectorised update: no Python loop, no .item()
    f_new = f
    f_new[_D3Q19_ZINLET_DIRS, 0, :, :] = (
        feq[_D3Q19_ZINLET_DIRS, 0]
        - feq[_D3Q19_ZINLET_OPP, 0]
        + f[_D3Q19_ZINLET_OPP, 0]
    )
    return f_new


def zou_he_outlet_pressure_z(f: torch.Tensor, rho_out: float = 1.0) -> torch.Tensor:
    """Zou/He pressure outlet BC at top face (z=nz-1) for D3Q19.

    Prescribes *rho = rho_out* at the outlet plane.  Unknown populations
    (cz < 0, directions 6, 12, 13, 16, 17) are reconstructed with
    non-equilibrium bounce-back.

    Args:
        f: Distribution tensor of shape ``(19, nz, ny, nx)``.
        rho_out: Prescribed outlet density (default 1.0).

    Returns:
        Updated distribution tensor (same shape).
    """
    device = f.device
    # At z=nz-1: cz>0 directions are known (streaming outward).
    sum_cz0 = (
        f[0, -1] + f[1, -1] + f[2, -1] + f[3, -1] + f[4, -1]
        + f[7, -1] + f[8, -1] + f[9, -1] + f[10, -1]
    )  # (ny, nx)
    sum_cz_pos = f[5, -1] + f[11, -1] + f[14, -1] + f[15, -1] + f[18, -1]  # (ny, nx)

    uz_out = -1.0 + (sum_cz0 + 2.0 * sum_cz_pos) / rho_out  # (ny, nx)

    rho3 = torch.full_like(uz_out.unsqueeze(0), rho_out)  # (1, ny, nx)
    ux3 = torch.zeros_like(rho3)
    uy3 = torch.zeros_like(rho3)
    uz3 = uz_out.unsqueeze(0)  # (1, ny, nx)
    feq = equilibrium3d(rho3, ux3, uy3, uz3, device=device)  # (19, 1, ny, nx)

    # Vectorised update: no Python loop, no .item()
    f_new = f
    f_new[_D3Q19_ZOUTLET_DIRS, -1, :, :] = (
        feq[_D3Q19_ZOUTLET_DIRS, 0]
        - feq[_D3Q19_ZOUTLET_OPP, 0]
        + f[_D3Q19_ZOUTLET_OPP, -1]
    )
    return f_new


def make_tank_wall_mask_3d(
    nz: int,
    ny: int,
    nx: int,
    obstacle_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Wall mask for a 3-D vertical water-entry tank.

    Marks the four lateral faces (x = 0, x = nx−1, y = 0, y = ny−1) as
    no-slip walls.  The bottom face (z = 0) and top face (z = nz−1) are
    intentionally excluded; they are handled by :func:`zou_he_inlet_velocity_z`
    and :func:`zou_he_outlet_pressure_z` respectively.

    Args:
        nz: Vertical extent (flow direction).
        ny: Lateral extent (y direction).
        nx: Lateral extent (x direction).
        obstacle_mask: Boolean tensor of shape ``(nz, ny, nx)``; obstacle
            cells are excluded from the wall mask to prevent double
            bounce-back.
        device: Target PyTorch device.

    Returns:
        Boolean tensor of shape ``(nz, ny, nx)``.
    """
    wall_mask = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    wall_mask[:, :, 0] = True   # left wall  (x = 0)
    wall_mask[:, :, -1] = True  # right wall (x = nx−1)
    wall_mask[:, 0, :] = True   # front wall (y = 0)
    wall_mask[:, -1, :] = True  # back wall  (y = ny−1)
    wall_mask[obstacle_mask] = False
    return wall_mask


def apply_water_entry_boundaries_3d(
    f: torch.Tensor,
    v_entry: float,
    wall_mask: torch.Tensor,
    obstacle_mask: torch.Tensor,
    rho_out: float = 1.0,
) -> torch.Tensor:
    """Combined boundary treatment for the sphere water-entry simulation.

    Applies in order:

    1. Zou/He velocity inlet at z = 0 with upward velocity uz = *v_entry*.
    2. Zou/He pressure outlet at z = nz−1 with ρ = *rho_out*.
    3. Bounce-back on the four lateral walls (x = 0, x = nx−1, y = 0,
       y = ny−1).
    4. Bounce-back on the sphere obstacle.

    Args:
        f: Distribution tensor of shape ``(19, nz, ny, nx)``.
        v_entry: Prescribed upward z-velocity at the inlet.
        wall_mask: Boolean lateral-wall mask of shape ``(nz, ny, nx)``.
        obstacle_mask: Boolean sphere mask of shape ``(nz, ny, nx)``.
        rho_out: Outlet density (default 1.0).

    Returns:
        Updated distribution tensor.
    """
    f = zou_he_inlet_velocity_z(f, v_entry)
    f = zou_he_outlet_pressure_z(f, rho_out=rho_out)
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



# ---------------------------------------------------------------------------
# P1.3 New boundary conditions — 3-D (D3Q19)
# ---------------------------------------------------------------------------

def porous_jump_3d(
    f: torch.Tensor,
    jump_slice: int,
    alpha: float,
    beta: float,
    thickness: float = 1.0,
    axis: str = "x",
) -> torch.Tensor:
    """Porous jump boundary condition (3-D, D3Q19 / Ergun equation).

    Args:
        f:         Distribution tensor (19, nz, ny, nx).
        jump_slice: Slice index along *axis* where the porous interface sits.
        alpha:     Viscous resistance [1/lu²].
        beta:      Inertial resistance [1/lu].
        thickness: Porous medium thickness in lattice units.
        axis:      Flow-normal axis ``'x'``, ``'y'``, or ``'z'``.

    Returns:
        Updated distribution tensor.
    """
    from .d3q19 import equilibrium3d, macroscopic3d

    rho, ux, uy, uz = macroscopic3d(f)

    axis_map = {"x": (ux, -1), "y": (uy, -2), "z": (uz, -3)}
    if axis not in axis_map:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")
    u_normal, dim = axis_map[axis]

    # Extract velocity at the interface slice
    if axis == "x":
        u_face = u_normal[:, :, jump_slice]  # (nz, ny)
        rho_face = rho[:, :, jump_slice]
        ux_f = ux[:, :, jump_slice]
        uy_f = uy[:, :, jump_slice]
        uz_f = uz[:, :, jump_slice]
    elif axis == "y":
        u_face = u_normal[:, jump_slice, :]
        rho_face = rho[:, jump_slice, :]
        ux_f = ux[:, jump_slice, :]
        uy_f = uy[:, jump_slice, :]
        uz_f = uz[:, jump_slice, :]
    else:
        u_face = u_normal[jump_slice, :, :]
        rho_face = rho[jump_slice, :, :]
        ux_f = ux[jump_slice, :, :]
        uy_f = uy[jump_slice, :, :]
        uz_f = uz[jump_slice, :, :]

    delta_rho = -(alpha * u_face + beta * u_face.abs() * u_face) * thickness
    rho_down = (rho_face + delta_rho).clamp(min=1e-6)

    f_new = f.clone()
    if axis == "x" and jump_slice + 1 < f.shape[3]:
        fe = equilibrium3d(rho_down, ux_f, uy_f, uz_f)
        f_new[:, :, :, jump_slice + 1] = fe
    elif axis == "y" and jump_slice + 1 < f.shape[2]:
        fe = equilibrium3d(rho_down, ux_f, uy_f, uz_f)
        f_new[:, :, jump_slice + 1, :] = fe
    elif axis == "z" and jump_slice + 1 < f.shape[1]:
        fe = equilibrium3d(rho_down, ux_f, uy_f, uz_f)
        f_new[:, jump_slice + 1, :, :] = fe
    return f_new


def fan_model_3d(
    f: torch.Tensor,
    fan_slice: int,
    pressure_rise_fn: object,
    axis: str = "x",
) -> torch.Tensor:
    """Simplified fan / actuator-plane boundary condition (3-D, D3Q19).

    Args:
        f:               Distribution tensor (19, nz, ny, nx).
        fan_slice:       Slice index along *axis* for the fan plane.
        pressure_rise_fn: Callable ``(Q: float) -> float`` giving ΔP.
        axis:            ``'x'``, ``'y'``, or ``'z'``.

    Returns:
        Updated distribution tensor.
    """
    from .d3q19 import equilibrium3d, macroscopic3d

    rho, ux, uy, uz = macroscopic3d(f)

    if axis == "x":
        u_col = ux[:, :, fan_slice]
        rho_face = rho[:, :, fan_slice]
        ux_f = ux[:, :, fan_slice]
        uy_f = uy[:, :, fan_slice]
        uz_f = uz[:, :, fan_slice]
    elif axis == "y":
        u_col = uy[:, fan_slice, :]
        rho_face = rho[:, fan_slice, :]
        ux_f = ux[:, fan_slice, :]
        uy_f = uy[:, fan_slice, :]
        uz_f = uz[:, fan_slice, :]
    else:
        u_col = uz[fan_slice, :, :]
        rho_face = rho[fan_slice, :, :]
        ux_f = ux[fan_slice, :, :]
        uy_f = uy[fan_slice, :, :]
        uz_f = uz[fan_slice, :, :]

    flow_rate = float(u_col.sum().item())
    try:
        delta_p = float(pressure_rise_fn(flow_rate))  # type: ignore[operator]
    except Exception:
        delta_p = 0.0

    rho_down = (rho_face + delta_p).clamp(min=1e-6)
    f_new = f.clone()
    if axis == "x" and fan_slice + 1 < f.shape[3]:
        f_new[:, :, :, fan_slice + 1] = equilibrium3d(rho_down, ux_f, uy_f, uz_f)
    elif axis == "y" and fan_slice + 1 < f.shape[2]:
        f_new[:, :, fan_slice + 1, :] = equilibrium3d(rho_down, ux_f, uy_f, uz_f)
    elif axis == "z" and fan_slice + 1 < f.shape[1]:
        f_new[:, fan_slice + 1, :, :] = equilibrium3d(rho_down, ux_f, uy_f, uz_f)
    return f_new


def nscbc_outlet_3d(
    f: torch.Tensor,
    rho_target: float = 1.0,
    sigma: float = 0.25,
    c_s: float = 1.0 / 3.0 ** 0.5,
) -> torch.Tensor:
    """Non-Reflecting (NSCBC) outlet boundary condition (3-D, D3Q19).

    Applies soft pressure relaxation at the x = nx−1 outlet plane to
    suppress spurious acoustic reflections.

    Args:
        f:          Distribution tensor (19, nz, ny, nx).
        rho_target: Target outlet density (default 1.0).
        sigma:      Relaxation factor [0, 1].  0 = fully non-reflecting.
        c_s:        Lattice speed of sound.

    Returns:
        Updated distribution tensor.
    """
    from .d3q19 import equilibrium3d, macroscopic3d

    rho, ux, uy, uz = macroscopic3d(f)

    rho_out = rho[:, :, -1]
    ux_out  = ux[:, :, -1]
    uy_out  = uy[:, :, -1]
    uz_out  = uz[:, :, -1]

    L1 = sigma * c_s * (rho_out - rho_target)
    rho_corrected = (rho_out - L1).clamp(min=1e-6)

    f_new = f.clone()
    f_new[:, :, :, -1] = equilibrium3d(rho_corrected, ux_out, uy_out, uz_out)
    return f_new
