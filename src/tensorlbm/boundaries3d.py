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


def bounce_back_cells_3d(f: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Bounce-back reflection on selected cells (obstacle/walls) for D3Q19.

    Uses ``torch.where`` instead of clone + scatter to reduce the number of
    GPU kernel launches and avoid an intermediate boolean-indexed allocation.
    """
    opp = OPPOSITE.to(f.device)  # (19,)
    # mask.unsqueeze(0) broadcasts (1, nz, ny, nx) → (19, nz, ny, nx)
    return torch.where(mask.unsqueeze(0), f[opp], f)


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
    f_new = f.clone()
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
    f_new = f.clone()
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
    f_new = f.clone()
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
    f_new = f.clone()
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
