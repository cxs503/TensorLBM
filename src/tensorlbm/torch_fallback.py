"""Pure PyTorch equivalents for Triton kernels (M2+M3 deliverable).

This module provides pure-PyTorch implementations of the Triton kernels
in triton_fused, triton_fused_distributed, triton_suboff_step_distributed,
and _voxel_kernels. These equivalents enable:
  1. CPU-only environments to run the same algorithms
  2. Gradient-based optimization through the LBM step
  3. Fallback when Triton is unavailable

All implementations are numerically equivalent to their Triton counterparts
within fp32 precision tolerances.
"""

from __future__ import annotations

import torch
from typing import Tuple, Optional

__all__ = [
    "torch_collide",
    "torch_stream_periodic",
    "torch_fused_collide_stream",
    "torch_parity_solid_mask",
    "torch_link_q_min_t",
]


# ---------------------------------------------------------------------------
# D3Q19 lattice constants
# ---------------------------------------------------------------------------
_D3Q19_C = torch.tensor(
    [
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
        [1, 1, 0],
        [-1, -1, 0],
        [1, -1, 0],
        [-1, 1, 0],
        [1, 0, 1],
        [-1, 0, -1],
        [1, 0, -1],
        [-1, 0, 1],
        [0, 1, 1],
        [0, -1, -1],
        [0, 1, -1],
        [0, -1, 1],
    ],
    dtype=torch.int64,
)

_D3Q19_W = torch.tensor(
    [1.0 / 3.0] + [1.0 / 18.0] * 6 + [1.0 / 36.0] * 12,
    dtype=torch.float32,
)


# ---------------------------------------------------------------------------
# Collision operator (BGK)
# ---------------------------------------------------------------------------


def torch_collide(
    f: torch.Tensor,
    tau: float,
    *,
    c: Optional[torch.Tensor] = None,
    w: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """BGK collision in pure PyTorch.

    Args:
        f: (Q, nz, ny, nx) distribution function
        tau: relaxation time
        c: (Q, 3) lattice velocities (default: D3Q19)
        w: (Q,) weights (default: D3Q19)

    Returns:
        Post-collision distribution (Q, nz, ny, nx)
    """
    if c is None:
        c = _D3Q19_C.to(device=f.device, dtype=torch.float32)
    if w is None:
        w = _D3Q19_W.to(device=f.device)

    Q, nz, ny, nx = f.shape
    
    # Compute macroscopic quantities
    rho = f.sum(dim=0)  # (nz, ny, nx)
    rho_safe = torch.where(rho > 1e-12, rho, 1e-12)
    
    # Velocity components
    ux = (c[:, 0:1, None, None] * f).sum(dim=0) / rho_safe
    uy = (c[:, 1:2, None, None] * f).sum(dim=0) / rho_safe
    uz = (c[:, 2:3, None, None] * f).sum(dim=0) / rho_safe
    
    usq = ux**2 + uy**2 + uz**2  # (nz, ny, nx)
    
    # Equilibrium distribution
    cu = c[:, 0:1, None, None] * ux[None, ...] + \
         c[:, 1:2, None, None] * uy[None, ...] + \
         c[:, 2:3, None, None] * uz[None, ...]
    
    feq = rho_safe[None, ...] * w[:, None, None, None] * (
        1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * usq[None, ...]
    )
    
    # BGK collision
    tau_inv = 1.0 / tau
    return f - tau_inv * (f - feq)


# ---------------------------------------------------------------------------
# Periodic streaming (pull scheme)
# ---------------------------------------------------------------------------


def torch_stream_periodic(
    f: torch.Tensor,
    *,
    c: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Periodic pull-streaming in pure PyTorch.

    Args:
        f: (Q, nz, ny, nx) distribution function
        c: (Q, 3) lattice velocities (default: D3Q19)

    Returns:
        Streamed distribution (Q, nz, ny, nx)
    """
    if c is None:
        c = _D3Q19_C.to(device=f.device)
    
    Q, nz, ny, nx = f.shape
    fnew = torch.empty_like(f)
    
    for q in range(Q):
        cx, cy, cz = int(c[q, 0]), int(c[q, 1]), int(c[q, 2])
        # Pull from neighbor: f_new[x] = f_old[x - c]
        # torch.roll shifts in the opposite direction, so we roll by -c
        fnew[q] = torch.roll(f[q], shifts=(-cz, -cy, -cx), dims=(0, 1, 2))
    
    return fnew


# ---------------------------------------------------------------------------
# Fused collide+stream
# ---------------------------------------------------------------------------


def torch_fused_collide_stream(
    f: torch.Tensor,
    tau: float,
    *,
    c: Optional[torch.Tensor] = None,
    w: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused BGK collision + periodic streaming in pure PyTorch.

    This is equivalent to calling torch_collide followed by torch_stream_periodic,
    but expressed as a single operation for clarity.

    Args:
        f: (Q, nz, ny, nx) distribution function
        tau: relaxation time
        c: (Q, 3) lattice velocities (default: D3Q19)
        w: (Q,) weights (default: D3Q19)

    Returns:
        Post-collision, streamed distribution (Q, nz, ny, nx)
    """
    f_post_collide = torch_collide(f, tau, c=c, w=w)
    return torch_stream_periodic(f_post_collide, c=c)


# ---------------------------------------------------------------------------
# Voxel kernels (ray-triangle intersection)
# ---------------------------------------------------------------------------


def torch_parity_solid_mask(
    tri9: torch.Tensor,
    shape: Tuple[int, int, int],
    origin: Tuple[float, float, float],
    spacing: Tuple[float, float, float],
) -> torch.Tensor:
    """Pure PyTorch solid mask via ray parity (reference implementation).

    This mirrors the Triton kernel in _voxel_kernels but uses vectorized
    PyTorch operations. Slower but works on any device.

    Args:
        tri9: (T, 9) triangle table [v0x v0y v0z v1x v1y v1z v2x v2y v2z]
        shape: (nz, ny, nx) grid shape
        origin: (x0, y0, z0) physical lower corner
        spacing: (dx, dy, dz) cell sizes

    Returns:
        Bool tensor of shape, True inside the closed surface
    """
    nz, ny, nx = shape
    x0, y0, z0 = origin
    dx, dy, dz = spacing
    
    device = tri9.device
    tri9 = tri9.to(dtype=torch.float64)
    
    # Reshape triangles to (T, 3, 3)
    tri = tri9.reshape(-1, 3, 3)
    v0, v1, v2 = tri[:, 0, :], tri[:, 1, :], tri[:, 2, :]
    
    # Edge vectors and normal
    e1 = v1 - v0
    e2 = v2 - v0
    normal = torch.linalg.cross(e1, e2, dim=1)
    nx_n = normal[:, 0]
    
    # Cull triangles parallel to ray direction (x-axis)
    ok = nx_n.abs() > 1e-12
    if not ok.any():
        return torch.zeros(shape, dtype=torch.bool, device=device)
    
    tri = tri[ok]
    v0, v1, v2 = tri[:, 0, :], tri[:, 1, :], tri[:, 2, :]
    e1 = v1 - v0
    e2 = v2 - v0
    normal = torch.linalg.cross(e1, e2, dim=1)
    nx_n, ny_n, nz_n = normal[:, 0], normal[:, 1], normal[:, 2]
    safe_nx = torch.where(nx_n.abs() > 1e-12, nx_n, torch.ones_like(nx_n))
    
    # Grid coordinates
    ox = x0 + (torch.arange(nx, device=device, dtype=torch.float64) + 0.5) * dx
    oy = y0 + (torch.arange(ny, device=device, dtype=torch.float64) + 0.5) * dy
    oz = z0 + (torch.arange(nz, device=device, dtype=torch.float64) + 0.5) * dz
    
    # Create meshgrid
    oz_grid, oy_grid, ox_grid = torch.meshgrid(oz, oy, ox, indexing='ij')
    
    # Ray-triangle intersection for each cell
    count = torch.zeros(shape, dtype=torch.int32, device=device)
    
    # Vectorized over triangles
    for t_idx in range(tri.shape[0]):
        v0_t = v0[t_idx]
        v1_t = v1[t_idx]
        v2_t = v2[t_idx]
        
        # Edge functions in (y, z) plane
        py0 = oy_grid - v0_t[1]
        pz0 = oz_grid - v0_t[2]
        d0 = (v1_t[1] - v0_t[1]) * pz0 - (v1_t[2] - v0_t[2]) * py0
        
        py1 = oy_grid - v1_t[1]
        pz1 = oz_grid - v1_t[2]
        d1 = (v2_t[1] - v1_t[1]) * pz1 - (v2_t[2] - v1_t[2]) * py1
        
        py2 = oy_grid - v2_t[1]
        pz2 = oz_grid - v2_t[2]
        d2 = (v0_t[1] - v2_t[1]) * pz2 - (v0_t[2] - v2_t[2]) * py2
        
        # Point-in-triangle test
        has_neg = (d0 < 0) | (d1 < 0) | (d2 < 0)
        has_pos = (d0 > 0) | (d1 > 0) | (d2 > 0)
        inside = ~(has_neg & has_pos)
        
        # Plane intersection
        x_isect = v0_t[0] - (ny_n[t_idx] * py0 + nz_n[t_idx] * pz0) / safe_nx[t_idx]
        
        # Hit test: inside triangle and intersection ahead of ray origin
        hit = inside & (x_isect > ox_grid)
        count += hit.to(torch.int32)
    
    # Odd parity = inside
    return (count % 2) == 1


def torch_link_q_min_t(
    tri9: torch.Tensor,
    cell_kji: torch.Tensor,
    direction: Tuple[int, int, int],
    origin: Tuple[float, float, float],
    spacing: Tuple[float, float, float],
) -> torch.Tensor:
    """Pure PyTorch Möller-Trumbore link intersection.

    Args:
        tri9: (T, 9) triangle table
        cell_kji: (N, 3) boundary cell indices (k, j, i)
        direction: (cx, cy, cz) lattice velocity
        origin: (x0, y0, z0) physical lower corner
        spacing: (dx, dy, dz) cell sizes

    Returns:
        (N,) fp32 tensor of minimum link fractions t in (0, 1],
        or sentinel 2.0 where no intersection
    """
    cxx, cyy, czz = direction
    x0, y0, z0 = origin
    dx, dy, dz = spacing
    
    device = tri9.device
    tri9 = tri9.to(dtype=torch.float64)
    tri = tri9.reshape(-1, 3, 3)
    
    n_cells = cell_kji.shape[0]
    if n_cells == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    
    # Physical ray origins
    k, j, i = cell_kji[:, 0], cell_kji[:, 1], cell_kji[:, 2]
    ox = x0 + (i.to(torch.float64) + 0.5) * dx
    oy = y0 + (j.to(torch.float64) + 0.5) * dy
    oz = z0 + (k.to(torch.float64) + 0.5) * dz
    
    # Physical ray direction
    dir_x = float(cxx) * dx
    dir_y = float(cyy) * dy
    dir_z = float(czz) * dz
    D = torch.tensor([dir_x, dir_y, dir_z], dtype=torch.float64, device=device)
    
    best = torch.full((n_cells,), 2.0, dtype=torch.float32, device=device)
    
    v0, v1, v2 = tri[:, 0, :], tri[:, 1, :], tri[:, 2, :]
    e1 = v1 - v0
    e2 = v2 - v0
    
    # Möller-Trumbore for each cell against all triangles
    for cell_idx in range(n_cells):
        ray_origin = torch.tensor([ox[cell_idx], oy[cell_idx], oz[cell_idx]], 
                                  dtype=torch.float64, device=device)
        
        # h = cross(D, e2)
        h = torch.linalg.cross(D.unsqueeze(0).expand_as(e2), e2, dim=1)
        a = (e1 * h).sum(dim=1)
        
        # Parallel rays
        ok = a.abs() > 1e-12
        if not ok.any():
            continue
        
        safe_a = torch.where(ok, a, torch.ones_like(a))
        f = torch.where(ok, 1.0 / safe_a, torch.zeros_like(a))
        
        s = ray_origin.unsqueeze(0) - v0
        u = f * (s * h).sum(dim=1)
        
        q = torch.linalg.cross(s, e1, dim=1)
        v = f * (D.unsqueeze(0) * q).sum(dim=1)
        t = f * (e2 * q).sum(dim=1)
        
        # Hit test
        hit = (
            ok &
            (u >= -1e-6) &
            (v >= -1e-6) &
            (u + v <= 1.0 + 1e-6) &
            (t > 1e-6) &
            (t <= 1.0 + 1e-6)
        )
        
        if hit.any():
            best[cell_idx] = t[hit].min().to(torch.float32)
    
    return best
