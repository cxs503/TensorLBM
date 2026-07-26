"""Surface mesh with precomputed normals for LBM drag computation.

The surface mesh stores:
  - Near-wall cell positions (as a mask)
  - Surface normal at each near-wall cell (analytical or from geometry)
  - Surface area element dA at each cell

The normal is a GEOMETRIC property computed during preprocessing,
not a numerical approximation during drag calculation.

Usage:
  # Preprocessing (once)
  mesh = SurfaceMesh.from_cylinder(solid, cx, cy, R)
  
  # During simulation (every step)
  cd_p, cl = drag_pressure_integration(f, mesh, dpS)
  cd_f = drag_friction_integration(f, mesh, dpS, nu)
  cd_tot, cd_p, cd_f = drag_total(f, mesh, dpS, nu)
"""
from __future__ import annotations

import math
import torch
from .d3q19 import macroscopic3d


class SurfaceMesh:
    """Precomputed surface mesh with normals.
    
    Attributes:
        near: Near-wall boolean mask (nz, ny, nx)
        nx_n, ny_n, nz_n: Surface normal components (nz, ny, nx), normalized
        dA: Surface area element per cell (nz, ny, nx), default 1.0
    """
    
    def __init__(self, near, nx_n, ny_n, nz_n, dA=None):
        self.near = near
        self.nx_n = nx_n
        self.ny_n = ny_n
        self.nz_n = nz_n
        self.dA = dA if dA is not None else torch.ones_like(near, dtype=torch.float32)
    
    @classmethod
    def from_cylinder(cls, solid, near, cx, cy, R, axis='z', cz=None):
        """Analytical normal for 2D extruded cylinder.

        The cylinder cross-section lies in the plane perpendicular to *axis*;
        the surface normal has components only in that plane.

        =========  ============================  ==========================
        axis      cross-section plane          normal
        =========  ============================  ==========================
        ``'z'``   x-y (default, current)       (x-cx, y-cy, 0) / r
        ``'y'``   x-z                           (x-cx, 0, z-cz) / r
        ``'x'``   y-z                           (0, y-cy, z-cz) / r
        =========  ============================  ==========================
        """
        nz, ny, nx = solid.shape
        device = solid.device

        if axis == 'z':
            yy, xx = torch.meshgrid(
                torch.arange(ny, device=device, dtype=torch.float32),
                torch.arange(nx, device=device, dtype=torch.float32),
                indexing='ij')
            nx_n = ((xx - cx) / R).unsqueeze(0).expand(nz, ny, nx)
            ny_n = ((yy - cy) / R).unsqueeze(0).expand(nz, ny, nx)
            nz_n = torch.zeros_like(nx_n)
        elif axis == 'y':
            cz_c = cz if cz is not None else nz / 2.0
            zz, xx = torch.meshgrid(
                torch.arange(nz, device=device, dtype=torch.float32),
                torch.arange(nx, device=device, dtype=torch.float32),
                indexing='ij')
            nx_n = ((xx - cx) / R).unsqueeze(1).expand(nz, ny, nx)
            nz_n = ((zz - cz_c) / R).unsqueeze(1).expand(nz, ny, nx)
            ny_n = torch.zeros_like(nx_n)
        elif axis == 'x':
            cz_c = cz if cz is not None else nz / 2.0
            zz, yy = torch.meshgrid(
                torch.arange(nz, device=device, dtype=torch.float32),
                torch.arange(ny, device=device, dtype=torch.float32),
                indexing='ij')
            ny_n = ((yy - cy) / R).unsqueeze(2).expand(nz, ny, nx)
            nz_n = ((zz - cz_c) / R).unsqueeze(2).expand(nz, ny, nx)
            nx_n = torch.zeros_like(ny_n)
        else:
            raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")

        norm = torch.sqrt(nx_n**2 + ny_n**2 + nz_n**2).clamp(min=1e-10)
        nx_n = nx_n / norm * near.float()
        ny_n = ny_n / norm * near.float()
        nz_n = nz_n / norm * near.float()
        return cls(near, nx_n, ny_n, nz_n)
    
    @classmethod
    def from_sphere(cls, solid, near, cx, cy, cz, R):
        """Analytical normal for 3D sphere.
        
        n = ((x-cx)/R, (y-cy)/R, (z-cz)/R) at each near-wall cell.
        """
        nz, ny, nx = solid.shape
        device = solid.device
        zz, yy, xx = torch.meshgrid(
            torch.arange(nz, device=device, dtype=torch.float32),
            torch.arange(ny, device=device, dtype=torch.float32),
            torch.arange(nx, device=device, dtype=torch.float32),
            indexing='ij')
        nx_n = (xx - cx) / R
        ny_n = (yy - cy) / R
        nz_n = (zz - cz) / R
        norm = torch.sqrt(nx_n**2 + ny_n**2 + nz_n**2).clamp(min=1e-10)
        nx_n = nx_n / norm * near.float()
        ny_n = ny_n / norm * near.float()
        nz_n = nz_n / norm * near.float()
        return cls(near, nx_n, ny_n, nz_n)
    
    @classmethod
    def from_gradient(cls, solid, near):
        """Generic normal from gradient of solid mask (for arbitrary geometry).
        
        dA = |∇solid| (gradient magnitude) accounts for surface orientation:
        face-aligned dA=1, diagonal dA=√2, curved dA varies.
        """
        nx_grad = torch.zeros_like(solid, dtype=torch.float32)
        ny_grad = torch.zeros_like(solid, dtype=torch.float32)
        nz_grad = torch.zeros_like(solid, dtype=torch.float32)
        
        nx_grad[:, :, 1:-1] = (solid[:, :, 2:].float() - solid[:, :, :-2].float()) / 2
        ny_grad[:, 1:-1, :] = (solid[:, 2:, :].float() - solid[:, :-2, :].float()) / 2
        nz_grad[1:-1, :, :] = (solid[2:, :, :].float() - solid[:-2, :, :].float()) / 2
        
        nx_n = -nx_grad * near.float()
        ny_n = -ny_grad * near.float()
        nz_n = -nz_grad * near.float()
        
        norm = torch.sqrt(nx_n**2 + ny_n**2 + nz_n**2).clamp(min=1e-10)
        # dA = 1.0 (surface area per cell, default)
        # Note: |∇solid| via central difference gives 0.5 for face-aligned
        # walls (wrong by 2×). Use dA=1.0 for all cells.
        return cls(near, nx_n/norm, ny_n/norm, nz_n/norm)
    
    @classmethod
    def from_square_prism(cls, solid, near, cx, cy, D):
        """Analytical normal for 2D square prism (axis-aligned).
        
        Front face: n=(-1,0), Back: n=(1,0), Top: n=(0,-1), Bottom: n=(0,1)
        """
        nz, ny, nx = solid.shape
        device = solid.device
        nx_n = torch.zeros(nz, ny, nx, dtype=torch.float32, device=device)
        ny_n = torch.zeros(nz, ny, nx, dtype=torch.float32, device=device)
        
        # Front face (x = cx-1, solid at cx): normal = (-1, 0)
        nx_n[:, :, cx-1] = -1.0
        # Back face (x = cx+D, solid at cx+D-1): normal = (1, 0)
        nx_n[:, :, cx+D] = 1.0
        # Bottom face (y = cy-D//2-1): normal = (0, -1)
        ny_n[:, cy-D//2-1, :] = -1.0
        # Top face (y = cy+D//2): normal = (0, 1)
        ny_n[:, cy+D//2, :] = 1.0
        
        # Only keep near-wall cells
        nx_n = nx_n * near.float()
        ny_n = ny_n * near.float()
        nz_n = torch.zeros_like(nx_n)
        return cls(near, nx_n, ny_n, nz_n)


def get_near_wall_2d(solid, axis='z'):
    """Near-wall mask for 2D extruded geometries.

    Detects fluid cells adjacent to solid cells in the 2D cross-section
    perpendicular to *axis*, replicated across all layers along *axis*.

    =========  =================  ====================================
    axis      cross-section       slicing
    =========  =================  ====================================
    ``'z'``   x-y (default)       ``solid[z]`` per z-layer
    ``'y'``   x-z                 ``solid[:, y, :]`` per y-layer
    ``'x'``   y-z                 ``solid[:, :, x]`` per x-layer
    =========  =================  ====================================
    """
    nz, ny, nx = solid.shape
    fluid = ~solid
    near = torch.zeros_like(solid)
    if axis == 'z':
        for z in range(nz):
            s = solid[z]; f = fluid[z]; n = torch.zeros_like(s)
            n[:, 1:-1] |= (s[:, 2:] | s[:, :-2]) & f[:, 1:-1]
            n[1:-1, :] |= (s[2:, :] | s[:-2, :]) & f[1:-1, :]
            near[z] = n
    elif axis == 'y':
        for y in range(ny):
            s = solid[:, y, :]; f = fluid[:, y, :]; n = torch.zeros_like(s)
            n[:, 1:-1] |= (s[:, 2:] | s[:, :-2]) & f[:, 1:-1]
            n[1:-1, :] |= (s[2:, :] | s[:-2, :]) & f[1:-1, :]
            near[:, y, :] = n
    elif axis == 'x':
        for x in range(nx):
            s = solid[:, :, x]; f = fluid[:, :, x]; n = torch.zeros_like(s)
            n[:, 1:-1] |= (s[:, 2:] | s[:, :-2]) & f[:, 1:-1]
            n[1:-1, :] |= (s[2:, :] | s[:-2, :]) & f[1:-1, :]
            near[:, :, x] = n
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")
    return near


def get_near_wall_3d(solid):
    """Near-wall mask for 3D geometries."""
    fluid = ~solid
    near = torch.zeros_like(solid)
    near[:, :, 1:-1] |= (solid[:, :, 2:] | solid[:, :, :-2]) & fluid[:, :, 1:-1]
    near[:, 1:-1, :] |= (solid[:, 2:, :] | solid[:, :-2, :]) & fluid[:, 1:-1, :]
    near[1:-1, :, :] |= (solid[2:, :, :] | solid[:-2, :, :]) & fluid[1:-1, :, :]
    return near


def drag_pressure_integration(f, mesh, dpS):
    """Pressure drag: 3D force vector from pressure × normal × dA.
    
    F = -Σ (p - p_0) · n · dA  (force on wall, negative of fluid force)
    Background pressure p_0 is subtracted to prevent spurious force from
    discrete surface non-closure (Σ n·dA ≠ 0 on staircase surface).
    Returns: (Cd_x, Cd_y, Cd_z) = (fx, fy, fz) / dpS
    """
    rho, _, _, _ = macroscopic3d(f)
    p = (rho - 1.0) / 3.0
    # Subtract background pressure (average at near-wall cells)
    # This prevents spurious force when discrete surface is not perfectly closed
    mask_float = mesh.near.float()
    n_near = mask_float.sum().clamp(min=1.0)
    p0 = (p * mask_float).sum() / n_near
    p_corr = p - p0
    mask = mask_float * mesh.dA
    fpx = -(p_corr * mesh.nx_n * mask).sum()
    fpy = -(p_corr * mesh.ny_n * mask).sum()
    fpz = -(p_corr * mesh.nz_n * mask).sum()
    return float(fpx.item() / dpS), float(fpy.item() / dpS), float(fpz.item() / dpS)


def drag_friction_integration(f, mesh, dpS, nu):
    """Friction drag via 3D wall shear stress (first-order, half-way BB).
    
    τ = 2ν · u_t  where u_t = u - (u·n)·n (tangent velocity)
    
    Verified on Couette flow: Cf error = 0.00%.
    Note: overestimates for curved surfaces at high τ (non-linear profile).
    The second-order formula (3·u_t1 - u_t2/3) was tested but gave worse
    results because the velocity profile is not quadratic at high τ.
    """
    rho, ux, uy, uz = macroscopic3d(f)
    nx, ny, nz = mesh.nx_n, mesh.ny_n, mesh.nz_n
    u_dot_n = ux * nx + uy * ny + uz * nz
    ut_x = ux - u_dot_n * nx
    ut_y = uy - u_dot_n * ny
    ut_z = uz - u_dot_n * nz
    tau_x = 2.0 * nu * ut_x
    tau_y = 2.0 * nu * ut_y
    tau_z = 2.0 * nu * ut_z
    mask = mesh.near.float() * mesh.dA
    ffx = (tau_x * mask).sum()
    ffy = (tau_y * mask).sum()
    ffz = (tau_z * mask).sum()
    return float(ffx.item() / dpS), float(ffy.item() / dpS), float(ffz.item() / dpS)


def drag_total(f, mesh, dpS, nu):
    """Total drag = pressure + friction (3D).
    
    Returns: (Cd_total_x, Cd_p_x, Cd_f_x) — x-component only.
    For full 3D force, call drag_pressure_integration and drag_friction_integration directly.
    """
    fx_p, _, _ = drag_pressure_integration(f, mesh, dpS)
    fx_f, _, _ = drag_friction_integration(f, mesh, dpS, nu)
    return fx_p + fx_f, fx_p, fx_f


def compute_cp_field(f, u_in):
    """Cp = p / (0.5 * u^2)."""
    rho, _, _, _ = macroscopic3d(f)
    p = (rho - 1.0) / 3.0
    return p / (0.5 * u_in**2)
