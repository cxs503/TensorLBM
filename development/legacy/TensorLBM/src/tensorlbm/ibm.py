"""Immersed Boundary Method (IBM) for static obstacles in TensorLBM.

Implements Peskin's immersed boundary method with the classic 4-point delta
function for spreading Lagrangian marker forces onto the Eulerian grid and
interpolating grid velocities back to the markers.

References
----------
- Peskin, C.S. (2002). "The immersed boundary method." Acta Numerica 11:479-517.
- Goldstein, D., Handler, R. & Sirovich, L. (1993). "Modeling a no-slip flow
  boundary with an external force field." J. Comput. Phys. 105:354-366.
- Uhlmann, M. (2005). "An immersed boundary method with direct forcing for
  the simulation of particulate flows." J. Comput. Phys. 209:448-476.
"""

from __future__ import annotations

import torch
from typing import Tuple, Optional


def peskin_delta_4pt(r: torch.Tensor) -> torch.Tensor:
    """Peskin's classic 4-point delta function (support |r| <= 2, C^1 continuous)."""
    r = torch.abs(r)
    delta = torch.zeros_like(r)
    mask1 = r <= 1.0
    r1 = r[mask1]
    delta[mask1] = (1.0 / 8.0) * (3.0 - 2.0 * r1 + torch.sqrt(1.0 + 4.0 * r1 - 4.0 * r1 * r1))
    mask2 = (r > 1.0) & (r <= 2.0)
    r2 = r[mask2]
    delta[mask2] = (1.0 / 8.0) * (5.0 - 2.0 * r2 - torch.sqrt(-7.0 + 12.0 * r2 - 4.0 * r2 * r2))
    return delta


def build_stencil_2d(
    marker_x: torch.Tensor,
    marker_y: torch.Tensor,
    ny: int,
    nx: int,
    support: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build tensor-product stencil for 2-D IBM markers."""
    device = marker_x.device
    M = marker_x.shape[0]
    half = support // 2

    base_x = torch.floor(marker_x).to(torch.int64)
    base_y = torch.floor(marker_y).to(torch.int64)

    offsets = torch.arange(-half + 1, half + 1, device=device)
    ox, oy = torch.meshgrid(offsets, offsets, indexing='ij')
    ox, oy = ox.flatten(), oy.flatten()

    gx = base_x[:, None] + ox[None, :]
    gy = base_y[:, None] + oy[None, :]

    dx = marker_x[:, None] - gx.to(marker_x.dtype)
    dy = marker_y[:, None] - gy.to(marker_y.dtype)

    weights = peskin_delta_4pt(dx) * peskin_delta_4pt(dy)

    valid = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny)
    gx_c = gx.clamp(min=0, max=nx - 1)
    gy_c = gy.clamp(min=0, max=ny - 1)
    indices = gy_c * nx + gx_c
    indices[~valid] = -1
    weights = weights * valid.to(weights.dtype)

    return weights, indices


def spread_force_2d(
    marker_fx: torch.Tensor,
    marker_fy: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    ny: int,
    nx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Spread marker forces to grid (vectorized)."""
    device = marker_fx.device
    dtype = marker_fx.dtype
    M, n_pts = weights.shape

    w_flat = weights.flatten()
    idx_flat = indices.flatten()
    valid = idx_flat >= 0
    w_valid = w_flat[valid]
    idx_valid = idx_flat[valid]

    fx_rep = marker_fx.repeat_interleave(n_pts)[valid]
    fy_rep = marker_fy.repeat_interleave(n_pts)[valid]

    force_x = torch.zeros(ny * nx, dtype=dtype, device=device)
    force_y = torch.zeros(ny * nx, dtype=dtype, device=device)
    force_x.index_add_(0, idx_valid, fx_rep * w_valid)
    force_y.index_add_(0, idx_valid, fy_rep * w_valid)

    return force_x.view(ny, nx), force_y.view(ny, nx)


def interpolate_velocity_2d(
    grid_ux: torch.Tensor,
    grid_uy: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Interpolate grid velocity to markers (vectorized)."""
    M, n_pts = weights.shape
    device = grid_ux.device
    dtype = grid_ux.dtype

    grid_fx = grid_ux.view(-1)
    grid_fy = grid_uy.view(-1)
    idx_flat = indices.flatten()
    valid = idx_flat >= 0
    idx_valid = idx_flat[valid]
    w_valid = weights.flatten()[valid].to(dtype)
    marker_idx = torch.arange(M, device=device).repeat_interleave(n_pts)[valid]

    mux = torch.zeros(M, dtype=dtype, device=device)
    muy = torch.zeros(M, dtype=dtype, device=device)
    mux.index_add_(0, marker_idx, grid_fx[idx_valid] * w_valid)
    muy.index_add_(0, marker_idx, grid_fy[idx_valid] * w_valid)

    return mux, muy


def direct_forcing_static(
    marker_ux: torch.Tensor,
    marker_uy: torch.Tensor,
    target_ux: torch.Tensor,
    target_uy: torch.Tensor,
    rho_grid: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    ny: int,
    nx: int,
    n_iter: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Iterative direct forcing for static obstacles (Uhlmann 2005)."""
    device = marker_ux.device
    dtype = marker_ux.dtype
    M = marker_ux.shape[0]
    n_pts = indices.shape[1]

    # Interpolate density
    rho_flat = rho_grid.view(-1).to(dtype)
    idx_flat = indices.flatten()
    valid = idx_flat >= 0
    w_valid = weights.flatten()[valid].to(dtype)
    idx_valid = idx_flat[valid]
    marker_idx = torch.arange(M, device=device).repeat_interleave(n_pts)[valid]

    rho_m = torch.zeros(M, dtype=dtype, device=device)
    rho_m.index_add_(0, marker_idx, rho_flat[idx_valid] * w_valid)

    f_mx = torch.zeros(M, dtype=dtype, device=device)
    f_my = torch.zeros(M, dtype=dtype, device=device)
    uc_x = marker_ux.clone()
    uc_y = marker_uy.clone()

    for _ in range(n_iter):
        df_x = 2.0 * rho_m * (target_ux - uc_x)
        df_y = 2.0 * rho_m * (target_uy - uc_y)
        f_mx += df_x
        f_my += df_y

        sfx, sfy = spread_force_2d(df_x, df_y, weights, indices, ny, nx)
        dugx = 0.5 * sfx / torch.clamp(rho_grid, min=1e-12)
        dugy = 0.5 * sfy / torch.clamp(rho_grid, min=1e-12)
        duix, duiy = interpolate_velocity_2d(dugx, dugy, weights, indices)
        uc_x += duix
        uc_y += duiy

    return spread_force_2d(f_mx, f_my, weights, indices, ny, nx)


def build_cylinder_markers_2d(
    center_x: float,
    center_y: float,
    radius: float,
    n_markers: Optional[int] = None,
    marker_spacing: float = 0.5,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate Lagrangian markers on a circle."""
    if n_markers is None:
        n_markers = max(16, int(2 * 3.14159265358979 * radius / marker_spacing))
    theta = torch.linspace(0, 2 * 3.14159265358979, n_markers + 1, device=device)[:-1]
    return center_x + radius * torch.cos(theta), center_y + radius * torch.sin(theta)
