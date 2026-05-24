"""Immersed Boundary Method (IBM) for lattice Boltzmann simulations.

Implements the direct-forcing immersed boundary method (Peskin 1972, Uhlmann
2005) for coupling moving or fixed Lagrangian boundary markers to the Eulerian
LBM grid.  The method consists of two operations:

1. **Velocity interpolation** (:func:`ibm_velocity_interpolate`): map the
   Eulerian fluid velocity field onto Lagrangian boundary markers using a
   discrete delta kernel.

2. **Force spreading** (:func:`ibm_force_spread`): spread Lagrangian point
   forces (the IBM body force) back onto the Eulerian grid using the same
   delta kernel.

The direct-forcing formulation computes the force required to drive each
Lagrangian marker to the target (desired) velocity in one step:

    F_L = (u_target − u_interpolated) / dt

where ``dt = 1`` in lattice units.  The force is then spread to the fluid
grid and added to the distribution function via a Guo-type body-force
correction.

Delta kernel
------------
The standard Peskin 2-point (hat) kernel and 4-point kernel are both
provided.  The 4-point kernel produces smoother forces at the cost of a
wider support stencil.

References
----------
Peskin, C. S. (1972). Flow patterns around heart valves: a numerical method.
    *J. Comput. Phys.* 10(2), 252–271.
Uhlmann, M. (2005). An immersed boundary method with direct forcing for
    the simulation of particulate flows.
    *J. Comput. Phys.* 209(2), 448–476.
Guo, Z., Zheng, C., & Shi, B. (2002). Discrete lattice effects on the
    forcing term in the lattice Boltzmann method.
    *Phys. Rev. E* 65, 046308.
"""
from __future__ import annotations

import math

import torch

__all__ = [
    "ibm_delta_hat",
    "ibm_delta_4pt",
    "ibm_velocity_interpolate",
    "ibm_force_spread",
    "ibm_direct_forcing",
    "ibm_apply_body_force_2d",
    "ibm_velocity_interpolate_3d",
    "ibm_force_spread_3d",
    "ibm_direct_forcing_3d",
    "ibm_apply_body_force_3d",
]


# ---------------------------------------------------------------------------
# Delta kernels
# ---------------------------------------------------------------------------


def ibm_delta_hat(r: torch.Tensor) -> torch.Tensor:
    """Peskin 2-point (hat / triangle) delta kernel.

    φ(r) = max(0, 1 − |r|)  for |r| ≤ 1, else 0.

    This is a first-order kernel with a support width of 2 cells.

    Args:
        r: Signed distance tensor (any shape), in lattice units.

    Returns:
        Delta weights of the same shape as *r*.
    """
    return torch.clamp(1.0 - r.abs(), min=0.0)


def ibm_delta_4pt(r: torch.Tensor) -> torch.Tensor:
    """Peskin 4-point (cosine) delta kernel.

    The piecewise function with support in [−2, 2] that satisfies the
    smoothness and moment conditions described in Peskin (2002):

    φ(r) =
        (3 − 2|r| + √(1 + 4|r| − 4r²)) / 8  if 0 ≤ |r| ≤ 1
        (5 − 2|r| − √(−7 + 12|r| − 4r²)) / 8 if 1 ≤ |r| ≤ 2
        0                                       if |r| > 2

    Args:
        r: Signed distance tensor (any shape), in lattice units.

    Returns:
        Delta weights of the same shape as *r*.
    """
    ra = r.abs()
    # Region 1: 0 ≤ |r| ≤ 1
    disc1 = (1.0 + 4.0 * ra - 4.0 * ra * ra).clamp(min=0.0)
    phi1 = (3.0 - 2.0 * ra + torch.sqrt(disc1)) / 8.0

    # Region 2: 1 < |r| ≤ 2
    disc2 = (-7.0 + 12.0 * ra - 4.0 * ra * ra).clamp(min=0.0)
    phi2 = (5.0 - 2.0 * ra - torch.sqrt(disc2)) / 8.0

    mask1 = ra <= 1.0
    mask2 = (ra > 1.0) & (ra <= 2.0)

    return torch.where(mask1, phi1, torch.where(mask2, phi2, torch.zeros_like(r)))


# ---------------------------------------------------------------------------
# Core IBM operations
# ---------------------------------------------------------------------------


def ibm_velocity_interpolate(
    ux: torch.Tensor,
    uy: torch.Tensor,
    marker_x: torch.Tensor,
    marker_y: torch.Tensor,
    kernel: str = "hat",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolate the Eulerian velocity field onto Lagrangian markers.

    For each Lagrangian marker at position (marker_x[k], marker_y[k]) the
    fluid velocity is reconstructed as:

        U_k = Σ_{i,j} u(i, j) · φ(i − x_k) · φ(j − y_k)

    where φ is the chosen delta kernel.

    Args:
        ux:        x-velocity field, shape ``(ny, nx)``.
        uy:        y-velocity field, shape ``(ny, nx)``.
        marker_x:  x-coordinates of Lagrangian markers, shape ``(N,)``
                   (floating-point, lattice units).
        marker_y:  y-coordinates of Lagrangian markers, shape ``(N,)``
                   (floating-point, lattice units).
        kernel:    Delta kernel: ``"hat"`` (2-point) or ``"4pt"`` (4-point).

    Returns:
        Tuple ``(u_marker_x, u_marker_y)`` — interpolated velocity components
        for each marker, shape ``(N,)``.
    """
    ny, nx = ux.shape
    device = ux.device
    n_markers = marker_x.shape[0]

    delta_fn = ibm_delta_hat if kernel == "hat" else ibm_delta_4pt
    support = 2 if kernel == "hat" else 4
    half_s = support // 2

    u_mx = torch.zeros(n_markers, dtype=ux.dtype, device=device)
    u_my = torch.zeros(n_markers, dtype=uy.dtype, device=device)

    for k in range(n_markers):
        xk = float(marker_x[k].item())
        yk = float(marker_y[k].item())

        ix0 = math.floor(xk) - half_s + 1
        iy0 = math.floor(yk) - half_s + 1

        for di in range(support):
            ix = (ix0 + di) % nx
            rx = torch.tensor(ix0 + di - xk, dtype=ux.dtype, device=device)
            wx = delta_fn(rx)
            for dj in range(support):
                iy = (iy0 + dj) % ny
                ry = torch.tensor(iy0 + dj - yk, dtype=uy.dtype, device=device)
                wy = delta_fn(ry)
                w = wx * wy
                u_mx[k] += w * ux[iy, ix]
                u_my[k] += w * uy[iy, ix]

    return u_mx, u_my


def ibm_force_spread(
    marker_fx: torch.Tensor,
    marker_fy: torch.Tensor,
    marker_x: torch.Tensor,
    marker_y: torch.Tensor,
    ny: int,
    nx: int,
    kernel: str = "hat",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Spread Lagrangian forces onto the Eulerian grid.

    For each Lagrangian marker at position (marker_x[k], marker_y[k]) with
    force (marker_fx[k], marker_fy[k]), the force is distributed to the
    surrounding Eulerian nodes:

        F(i, j) += F_k · φ(i − x_k) · φ(j − y_k)

    Args:
        marker_fx:  x-force at each marker, shape ``(N,)``.
        marker_fy:  y-force at each marker, shape ``(N,)``.
        marker_x:   x-coordinates of markers, shape ``(N,)``.
        marker_y:   y-coordinates of markers, shape ``(N,)``.
        ny:         Eulerian grid height.
        nx:         Eulerian grid width.
        kernel:     Delta kernel: ``"hat"`` or ``"4pt"``.

    Returns:
        Tuple ``(fx_grid, fy_grid)`` — Eulerian force field, each of shape
        ``(ny, nx)``.
    """
    device = marker_fx.device
    n_markers = marker_x.shape[0]

    delta_fn = ibm_delta_hat if kernel == "hat" else ibm_delta_4pt
    support = 2 if kernel == "hat" else 4
    half_s = support // 2

    fx_grid = torch.zeros((ny, nx), dtype=marker_fx.dtype, device=device)
    fy_grid = torch.zeros((ny, nx), dtype=marker_fy.dtype, device=device)

    for k in range(n_markers):
        xk = float(marker_x[k].item())
        yk = float(marker_y[k].item())
        fxk = marker_fx[k]
        fyk = marker_fy[k]

        ix0 = math.floor(xk) - half_s + 1
        iy0 = math.floor(yk) - half_s + 1

        for di in range(support):
            ix = (ix0 + di) % nx
            rx = torch.tensor(ix0 + di - xk, dtype=marker_fx.dtype, device=device)
            wx = delta_fn(rx)
            for dj in range(support):
                iy = (iy0 + dj) % ny
                ry = torch.tensor(iy0 + dj - yk, dtype=marker_fy.dtype, device=device)
                wy = delta_fn(ry)
                w = wx * wy
                fx_grid[iy, ix] += w * fxk
                fy_grid[iy, ix] += w * fyk

    return fx_grid, fy_grid


def ibm_direct_forcing(
    ux: torch.Tensor,
    uy: torch.Tensor,
    marker_x: torch.Tensor,
    marker_y: torch.Tensor,
    u_target_x: torch.Tensor,
    u_target_y: torch.Tensor,
    kernel: str = "hat",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute direct-forcing IBM body force for 2D flows.

    Calculates the force required to drive each Lagrangian marker to its
    target velocity in one time step, then spreads it to the Eulerian grid:

        F_L = u_target − u_interpolated   (in lattice units, dt=1)

    Args:
        ux:          x-velocity field, shape ``(ny, nx)``.
        uy:          y-velocity field, shape ``(ny, nx)``.
        marker_x:    x-positions of markers, shape ``(N,)``.
        marker_y:    y-positions of markers, shape ``(N,)``.
        u_target_x:  Target x-velocity for each marker, shape ``(N,)``.
        u_target_y:  Target y-velocity for each marker, shape ``(N,)``.
        kernel:      Delta kernel: ``"hat"`` or ``"4pt"``.

    Returns:
        Tuple ``(fx_grid, fy_grid)`` — Eulerian IBM body-force field,
        each of shape ``(ny, nx)``.
    """
    ny, nx = ux.shape
    u_mx, u_my = ibm_velocity_interpolate(ux, uy, marker_x, marker_y, kernel=kernel)
    marker_fx = u_target_x - u_mx
    marker_fy = u_target_y - u_my
    return ibm_force_spread(marker_fx, marker_fy, marker_x, marker_y, ny, nx, kernel=kernel)


def ibm_apply_body_force_2d(
    f: torch.Tensor,
    fx_grid: torch.Tensor,
    fy_grid: torch.Tensor,
) -> torch.Tensor:
    """Apply a 2D Eulerian body force to the D2Q9 distribution function.

    Uses the Guo (2002) first-order forcing scheme:

        f_i ← f_i + w_i · 3 · (c_ix F_x + c_iy F_y)

    This is a first-order correction; the Guo second-order scheme (which
    also subtracts the force contribution from f before collision) gives
    better accuracy but requires the force to be known before the collision
    step.

    Args:
        f:        Distribution tensor, shape ``(9, ny, nx)``.
        fx_grid:  x-body force per lattice node, shape ``(ny, nx)``.
        fy_grid:  y-body force per lattice node, shape ``(ny, nx)``.

    Returns:
        Updated distribution tensor of the same shape.
    """
    from .d2q9 import C, W

    device = f.device
    c = C.to(device).float()
    w = W.to(device).float()

    cx = c[:, 0].view(9, 1, 1)
    cy = c[:, 1].view(9, 1, 1)
    w_view = w.view(9, 1, 1)

    forcing = w_view * 3.0 * (cx * fx_grid.unsqueeze(0) + cy * fy_grid.unsqueeze(0))
    return f + forcing


def ibm_velocity_interpolate_3d(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    marker_x: torch.Tensor,
    marker_y: torch.Tensor,
    marker_z: torch.Tensor,
    kernel: str = "hat",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Interpolate a 3D Eulerian velocity field onto Lagrangian markers.

    Args:
        ux: x-velocity field of shape ``(nz, ny, nx)``.
        uy: y-velocity field of shape ``(nz, ny, nx)``.
        uz: z-velocity field of shape ``(nz, ny, nx)``.
        marker_x: Marker x-coordinates of shape ``(N,)``.
        marker_y: Marker y-coordinates of shape ``(N,)``.
        marker_z: Marker z-coordinates of shape ``(N,)``.
        kernel: Delta kernel name, ``"hat"`` or ``"4pt"``.

    Returns:
        Tuple of interpolated marker velocities ``(u_mx, u_my, u_mz)``.
    """
    nz, ny, nx = ux.shape
    device = ux.device
    n_markers = marker_x.shape[0]

    delta_fn = ibm_delta_hat if kernel == "hat" else ibm_delta_4pt
    support = 2 if kernel == "hat" else 4
    half_s = support // 2

    u_mx = torch.zeros(n_markers, dtype=ux.dtype, device=device)
    u_my = torch.zeros(n_markers, dtype=uy.dtype, device=device)
    u_mz = torch.zeros(n_markers, dtype=uz.dtype, device=device)

    for k in range(n_markers):
        xk = float(marker_x[k].item())
        yk = float(marker_y[k].item())
        zk = float(marker_z[k].item())

        ix0 = math.floor(xk) - half_s + 1
        iy0 = math.floor(yk) - half_s + 1
        iz0 = math.floor(zk) - half_s + 1

        for di in range(support):
            ix = (ix0 + di) % nx
            rx = torch.tensor(ix0 + di - xk, dtype=ux.dtype, device=device)
            wx = delta_fn(rx)
            for dj in range(support):
                iy = (iy0 + dj) % ny
                ry = torch.tensor(iy0 + dj - yk, dtype=uy.dtype, device=device)
                wy = delta_fn(ry)
                for dk in range(support):
                    iz = (iz0 + dk) % nz
                    rz = torch.tensor(iz0 + dk - zk, dtype=uz.dtype, device=device)
                    wz = delta_fn(rz)
                    weight = wx * wy * wz
                    u_mx[k] += weight * ux[iz, iy, ix]
                    u_my[k] += weight * uy[iz, iy, ix]
                    u_mz[k] += weight * uz[iz, iy, ix]

    return u_mx, u_my, u_mz


def ibm_force_spread_3d(
    marker_fx: torch.Tensor,
    marker_fy: torch.Tensor,
    marker_fz: torch.Tensor,
    marker_x: torch.Tensor,
    marker_y: torch.Tensor,
    marker_z: torch.Tensor,
    nz: int,
    ny: int,
    nx: int,
    kernel: str = "hat",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Spread 3D Lagrangian marker forces onto the Eulerian grid.

    Args:
        marker_fx: Marker x-forces of shape ``(N,)``.
        marker_fy: Marker y-forces of shape ``(N,)``.
        marker_fz: Marker z-forces of shape ``(N,)``.
        marker_x: Marker x-coordinates of shape ``(N,)``.
        marker_y: Marker y-coordinates of shape ``(N,)``.
        marker_z: Marker z-coordinates of shape ``(N,)``.
        nz: Number of z-cells.
        ny: Number of y-cells.
        nx: Number of x-cells.
        kernel: Delta kernel name, ``"hat"`` or ``"4pt"``.

    Returns:
        Tuple ``(fx_grid, fy_grid, fz_grid)`` of shape ``(nz, ny, nx)``.
    """
    device = marker_fx.device
    n_markers = marker_x.shape[0]

    delta_fn = ibm_delta_hat if kernel == "hat" else ibm_delta_4pt
    support = 2 if kernel == "hat" else 4
    half_s = support // 2

    fx_grid = torch.zeros((nz, ny, nx), dtype=marker_fx.dtype, device=device)
    fy_grid = torch.zeros((nz, ny, nx), dtype=marker_fy.dtype, device=device)
    fz_grid = torch.zeros((nz, ny, nx), dtype=marker_fz.dtype, device=device)

    for k in range(n_markers):
        xk = float(marker_x[k].item())
        yk = float(marker_y[k].item())
        zk = float(marker_z[k].item())
        fxk = marker_fx[k]
        fyk = marker_fy[k]
        fzk = marker_fz[k]

        ix0 = math.floor(xk) - half_s + 1
        iy0 = math.floor(yk) - half_s + 1
        iz0 = math.floor(zk) - half_s + 1

        for di in range(support):
            ix = (ix0 + di) % nx
            rx = torch.tensor(ix0 + di - xk, dtype=marker_fx.dtype, device=device)
            wx = delta_fn(rx)
            for dj in range(support):
                iy = (iy0 + dj) % ny
                ry = torch.tensor(iy0 + dj - yk, dtype=marker_fy.dtype, device=device)
                wy = delta_fn(ry)
                for dk in range(support):
                    iz = (iz0 + dk) % nz
                    rz = torch.tensor(iz0 + dk - zk, dtype=marker_fz.dtype, device=device)
                    wz = delta_fn(rz)
                    weight = wx * wy * wz
                    fx_grid[iz, iy, ix] += weight * fxk
                    fy_grid[iz, iy, ix] += weight * fyk
                    fz_grid[iz, iy, ix] += weight * fzk

    return fx_grid, fy_grid, fz_grid


def ibm_direct_forcing_3d(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    marker_x: torch.Tensor,
    marker_y: torch.Tensor,
    marker_z: torch.Tensor,
    u_target_x: torch.Tensor,
    u_target_y: torch.Tensor,
    u_target_z: torch.Tensor,
    kernel: str = "hat",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the 3D direct-forcing IBM body force field.

    Args:
        ux: x-velocity field of shape ``(nz, ny, nx)``.
        uy: y-velocity field of shape ``(nz, ny, nx)``.
        uz: z-velocity field of shape ``(nz, ny, nx)``.
        marker_x: Marker x-coordinates of shape ``(N,)``.
        marker_y: Marker y-coordinates of shape ``(N,)``.
        marker_z: Marker z-coordinates of shape ``(N,)``.
        u_target_x: Target marker x-velocity of shape ``(N,)``.
        u_target_y: Target marker y-velocity of shape ``(N,)``.
        u_target_z: Target marker z-velocity of shape ``(N,)``.
        kernel: Delta kernel name, ``"hat"`` or ``"4pt"``.

    Returns:
        Tuple ``(fx_grid, fy_grid, fz_grid)`` of shape ``(nz, ny, nx)``.
    """
    nz, ny, nx = ux.shape
    u_mx, u_my, u_mz = ibm_velocity_interpolate_3d(
        ux, uy, uz, marker_x, marker_y, marker_z, kernel=kernel
    )
    marker_fx = u_target_x - u_mx
    marker_fy = u_target_y - u_my
    marker_fz = u_target_z - u_mz
    return ibm_force_spread_3d(
        marker_fx,
        marker_fy,
        marker_fz,
        marker_x,
        marker_y,
        marker_z,
        nz,
        ny,
        nx,
        kernel=kernel,
    )


def ibm_apply_body_force_3d(
    f: torch.Tensor,
    fx_grid: torch.Tensor,
    fy_grid: torch.Tensor,
    fz_grid: torch.Tensor,
) -> torch.Tensor:
    """Apply a 3D Guo body-force correction to a D3Q19 distribution.

    Args:
        f: Distribution tensor of shape ``(19, nz, ny, nx)``.
        fx_grid: Eulerian x-force field of shape ``(nz, ny, nx)``.
        fy_grid: Eulerian y-force field of shape ``(nz, ny, nx)``.
        fz_grid: Eulerian z-force field of shape ``(nz, ny, nx)``.

    Returns:
        Updated D3Q19 distribution tensor of the same shape.
    """
    from .d3q19 import C as C_D3Q19
    from .d3q19 import W as W_D3Q19

    device = f.device
    c = C_D3Q19.to(device).float()
    w = W_D3Q19.to(device).float()

    cx = c[:, 0].view(19, 1, 1, 1)
    cy = c[:, 1].view(19, 1, 1, 1)
    cz = c[:, 2].view(19, 1, 1, 1)
    w_view = w.view(19, 1, 1, 1)
    forcing = w_view * 3.0 * (
        cx * fx_grid.unsqueeze(0) + cy * fy_grid.unsqueeze(0) + cz * fz_grid.unsqueeze(0)
    )
    return f + forcing
