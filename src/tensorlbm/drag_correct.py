"""Correct drag computation for LBM with bounce-back.

Two independent methods that should give the same result:

1. Momentum exchange (Ladd 1994):
   F = Σ (f_i + f_{opp(i)}) * c_i    for links crossing into solid

2. Pressure integration:
   F = Σ p * n_x * dA                 for all solid-fluid interfaces

Both methods require knowing which directions cross the wall for each
individual near-wall cell — the wall normal varies across the surface.
"""
import torch
from typing import Tuple


def get_wall_normal(solid: torch.Tensor, near: torch.Tensor) -> torch.Tensor:
    """Compute outward wall normal (solid → fluid) at each near-wall cell.

    For each near-wall fluid cell, the normal is the average direction
    to the adjacent solid cells.

    Args:
        solid: (nz, ny, nx) boolean, True = solid.
        near:  (nz, ny, nx) boolean mask of near-wall fluid cells.

    Returns:
        normal: (nz, ny, nx, 3) float, wall normal at each cell (zero elsewhere).
    """
    nz, ny, nx = solid.shape
    normal = torch.zeros(nz, ny, nx, 3, device=solid.device, dtype=torch.float32)

    # Check all 6 face directions
    for ax, sgn in [(0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)]:
        k_shift = sgn if ax == 0 else 0
        j_shift = sgn if ax == 1 else 0
        i_shift = sgn if ax == 2 else 0
        # Solid neighbor in direction (k_shift, j_shift, i_shift)
        shifted = torch.roll(solid, (-k_shift, -j_shift, -i_shift), dims=(0, 1, 2))
        has_solid = shifted & near
        # Outward normal: from solid TO fluid = opposite of sgn
        normal[:, :, :, ax][has_solid] += -sgn

    # Normalize (avoid division by zero)
    norm = normal.norm(dim=3, keepdim=True)
    normal = normal / norm.clamp(min=1e-10)
    return normal


def momentum_exchange_drag(
    f: torch.Tensor,
    solid: torch.Tensor,
    near: torch.Tensor,
    wall_normal: torch.Tensor,
    velocities: torch.Tensor,  # (19, 3) D3Q19 C matrix
) -> Tuple[float, float]:
    """Compute drag via Ladd (1994) momentum exchange method.

    For each near-wall cell, sum (f_i + f_opp) * c_i only for
    directions i that cross from fluid INTO solid (c_i · n < 0).

    Args:
        f: (19, nz, ny, nx) distribution.
        solid, near, wall_normal: masks and normals.
        velocities: (19, 3) lattice velocities.

    Returns:
        drag_fric: friction (momentum exchange) in x-direction.
        drag_pres: pressure (momentum exchange) in x-direction.
    """
    fx = torch.zeros(1, device=f.device)
    fy = torch.zeros(1, device=f.device)
    fz = torch.zeros(1, device=f.device)

    c = velocities.to(f.device).float()  # (19, 3)
    nz, ny, nx = solid.shape

    for i in range(19):
        opp_i = (i + 9) % 18  # opposite direction (skip rest particle 18)
        if i == 18:
            continue  # rest particle has no momentum

        ci_x = c[i, 0]
        ci_y = c[i, 1]
        ci_z = c[i, 2]

        # Direction i crosses from fluid INTO solid if c_i · n < 0
        # (normal points from solid TO fluid, so INTO solid means opposite)
        c_dot_n = (ci_x * wall_normal[:, :, :, 0]
                 + ci_y * wall_normal[:, :, :, 1]
                 + ci_z * wall_normal[:, :, :, 2])

        # Link crosses from fluid into solid
        crossing = (c_dot_n < -0.01) & near

        if crossing.any():
            # Force contribution: (f_i + f_opp) * c_i
            df = (f[i] + f[opp_i]) * crossing.float()
            fx += (df * ci_x).sum()
            fy += (df * ci_y).sum()
            fz += (df * ci_z).sum()

    return float(fx.item()), float(fy.item()), float(fz.item())


def pressure_integration_drag(
    f: torch.Tensor,
    solid: torch.Tensor,
    near: torch.Tensor,
    wall_normal: torch.Tensor,
) -> float:
    """Compute drag by integrating pressure over solid-fluid interfaces.

    For each near-wall cell, the pressure at the wall surface (half-way
    between fluid and solid cell) is p = (ρ-1)/3.  The force on the
    solid is p * n * dA, where dA = 1 for 2D, = nz for 3D.

    Drag is the x-component: Σ p * n_x * dA over all interfaces.

    Returns:
        drag_pressure_x: float.
    """
    from .d3q19 import macroscopic3d
    rho, _, _, _ = macroscopic3d(f)
    p = (rho - 1.0) / 3.0

    # Pressure force on solid = p * (-normal) * area
    # (normal points INTO fluid, wall force on solid is opposite)
    nx = wall_normal[:, :, :, 0]
    force_x = (p * nx * near.float()).sum()

    # Area factor: for 2D (nz=1 or 2), area = 1.0; for 3D, area = nz per cell
    # Actually, each surface cell has unit area in lattice units
    return float(force_x.item())


def total_drag_bounce_back(
    f: torch.Tensor,
    solid: torch.Tensor,
) -> dict:
    """Compute total drag on solid via bounce-back methods.

    Two independent approaches that should match:
      - momentum exchange (Ladd 1994)
      - pressure integration

    Args:
        f: (19, nz, ny, nx) post-stream distribution.
        solid: (nz, ny, nx) boolean solid mask.

    Returns:
        dict with keys: Cd_momentum, Cd_pressure, Cd_total, convergence_ratio.
    """
    fluid = ~solid
    near = torch.zeros_like(solid)
    for ax, sgn in [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]:
        near |= torch.roll(solid, sgn, dims=ax) & fluid

    wall_n = get_wall_normal(solid, near)

    from .d3q19 import C
    fx_m, fy_m, fz_m = momentum_exchange_drag(f, solid, near, wall_n, C)
    fx_p = pressure_integration_drag(f, solid, near, wall_n)

    return {
        'Fx_momentum': fx_m,
        'Fy_momentum': fy_m,
        'Fz_momentum': fz_m,
        'Fx_pressure': fx_p,
        'ratio': fx_m / fx_p if abs(fx_p) > 1e-10 else float('inf'),
    }


# ── Test ──
if __name__ == '__main__':
    import time
    print("Drag computation module compiled OK.")
    print("Key fix: wall-normal aware momentum exchange + pressure integration.")
    print("Previous: uniform sum over 9 pairs → over-counted + wrong sign.")
