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
        dA = 1.0 (default; 1/max|n| overestimates staircase area).
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
    def from_suboff(cls, solid, near, cx, cy, cz, length, radius, config=None):
        """Analytical outward normal for SUBOFF axisymmetric bare hull.

        The hull is a body of revolution about the x-axis.  For a surface
        point (x, r) with normalised axial coordinate ``xi = (x - x_bow)/L``:

          r(xi) = suboff_radius_profile(xi) * R_max

        The surface is ``F(x, y, z) = sqrt((y-cy)^2 + (z-cz)^2) - r(x) = 0``.
        The outward normal (pointing from body into fluid) is:

          n = (-dr/dx, cos θ, sin θ) / |n|

        where ``θ = atan2(z - cz, y - cy)`` is the azimuthal angle and
        ``dr/dx = (dr_norm/dxi) * R_max / L``.

        Sign check:
          - Bow (dr/dx > 0): n_x = -dr/dx < 0 → normal points upstream ✓
          - Stern (dr/dx < 0): n_x = -dr/dx > 0 → normal points downstream ✓
          - Midbody (dr/dx = 0): n = (0, cos θ, sin θ) → purely radial ✓

        dA = 1.0 (default; consistent with from_sphere / from_cylinder).
        """
        import numpy as np
        from .suboff_cad import suboff_radius_profile, SuboffConfig

        if config is None:
            config = SuboffConfig()

        nz, ny, nx = solid.shape
        device = solid.device

        zz, yy, xx = torch.meshgrid(
            torch.arange(nz, device=device, dtype=torch.float32),
            torch.arange(ny, device=device, dtype=torch.float32),
            torch.arange(nx, device=device, dtype=torch.float32),
            indexing='ij')

        x_bow = cx - length / 2.0
        xi_t = (xx - x_bow) / length  # 0 at bow, 1 at stern

        # Numerical derivative of normalised radius profile (central diff)
        xi_np = xi_t.cpu().numpy()
        eps = 1e-4
        r_plus = suboff_radius_profile(np.clip(xi_np + eps, 0.0, 1.0), config)
        r_minus = suboff_radius_profile(np.clip(xi_np - eps, 0.0, 1.0), config)
        dr_dxi = (r_plus - r_minus) / (2.0 * eps)

        # dr/dx in lattice units = (dr/dxi) * R_max / L
        dr_dx = torch.tensor(
            dr_dxi * radius / length, device=device, dtype=torch.float32)

        # Azimuthal angle components
        dy = yy - cy
        dz = zz - cz
        r_cell = torch.sqrt(dy ** 2 + dz ** 2).clamp(min=1e-10)
        cos_theta = dy / r_cell
        sin_theta = dz / r_cell

        # Outward normal: n = (-dr/dx, cos θ, sin θ) / norm
        nx_n = -dr_dx
        ny_n = cos_theta
        nz_n = sin_theta

        # End-cap fix: cells outside hull axial extent (xi<0 or xi>1) have
        # dr/dx=0 (profile is clipped), giving |n|=0 on the axis.  These
        # are the bow/stern cap cells — set normal to purely axial.
        xi_field = (xx - x_bow) / length
        outside = (xi_field < 0.0) | (xi_field > 1.0)
        # Bow cap (xi<0): n=(-1,0,0); Stern cap (xi>1): n=(1,0,0)
        nx_n = torch.where(outside, torch.where(xi_field < 0, -1.0, 1.0), nx_n)
        ny_n = torch.where(outside, torch.zeros_like(ny_n), ny_n)
        nz_n = torch.where(outside, torch.zeros_like(nz_n), nz_n)

        norm = torch.sqrt(nx_n ** 2 + ny_n ** 2 + nz_n ** 2).clamp(min=1e-10)
        near_f = near.float()
        nx_n = nx_n / norm * near_f
        ny_n = ny_n / norm * near_f
        nz_n = nz_n / norm * near_f
        return cls(near, nx_n, ny_n, nz_n)

    @classmethod
    def from_ellipsoid(cls, solid, near, cx, cy, cz, a, b, c):
        """Analytical normal for 3D ellipsoid.
        
        Ellipsoid: (x/a)² + (y/b)² + (z/c)² = 1
        Normal: n = (x/a², y/b², z/c²) / |n|
        """
        nz, ny, nx = solid.shape
        device = solid.device
        zz, yy, xx = torch.meshgrid(
            torch.arange(nz, device=device, dtype=torch.float32),
            torch.arange(ny, device=device, dtype=torch.float32),
            torch.arange(nx, device=device, dtype=torch.float32),
            indexing='ij')
        nx_n = (xx - cx) / (a * a)
        ny_n = (yy - cy) / (b * b)
        nz_n = (zz - cz) / (c * c)
        norm = torch.sqrt(nx_n**2 + ny_n**2 + nz_n**2).clamp(min=1e-10)
        nx_n = nx_n / norm * near.float()
        ny_n = ny_n / norm * near.float()
        nz_n = nz_n / norm * near.float()
        return cls(near, nx_n, ny_n, nz_n)
    
    @classmethod
    def from_naca(cls, solid, near, x_le, y_c, chord):
        """Analytical normal for NACA 4-digit airfoil (2D extruded).
        
        NACA surface: y_t = 0.6*(0.2969*sqrt(x) - 0.1260*x - 0.3516*x² + 0.2843*x³ - 0.1015*x⁴)
        Tangent: dy/dx = 0.6*(0.2969/(2*sqrt(x)) - 0.1260 - 0.7032*x + 0.8529*x² - 0.4060*x³)
        Normal: n = (-dy/dx, sign, 0) / |n|  (sign=+1 upper, -1 lower)
        """
        nz, ny, nx = solid.shape
        device = solid.device
        zz, yy, xx = torch.meshgrid(
            torch.arange(nz, device=device, dtype=torch.float32),
            torch.arange(ny, device=device, dtype=torch.float32),
            torch.arange(nx, device=device, dtype=torch.float32),
            indexing='ij')
        # NACA x coordinate (normalized 0-1)
        xc = (xx - x_le) / chord
        xc = xc.clamp(min=1e-6, max=1.0)
        # Derivative of NACA thickness equation
        dydx = 0.6 * (0.2969 / (2.0 * torch.sqrt(xc)) - 0.1260 - 0.7032 * xc
                       + 0.8529 * xc**2 - 0.4060 * xc**3)
        # Determine upper/lower surface from solid mask
        # Upper: y > y_c, Lower: y < y_c
        sign = torch.where(yy > y_c, 1.0, -1.0)
        # Normal: (-dydx, sign, 0) normalized
        nx_n = -dydx * sign
        ny_n = sign.float()
        nz_n = torch.zeros_like(nx_n)
        norm = torch.sqrt(nx_n**2 + ny_n**2).clamp(min=1e-10)
        nx_n = nx_n / norm * near.float()
        ny_n = ny_n / norm * near.float()
        nz_n = nz_n / norm * near.float()
        return cls(near, nx_n, ny_n, nz_n)
    
    @classmethod
    def from_stl(cls, solid, near, vertices, faces, face_normals, origin, spacing):
        """STL-derived surface normals for arbitrary geometry.

        Thin wrapper around :func:`tensorlbm.stl_geometry.SurfaceMesh_from_stl`
        that finds the nearest STL triangle for each near-wall cell and
        uses its face normal (flipped to point outward).

        ``dA = 1.0`` (default, consistent with from_sphere / from_cylinder).
        """
        from .stl_geometry import SurfaceMesh_from_stl

        return SurfaceMesh_from_stl(
            solid, near, vertices, faces, face_normals, origin, spacing
        )

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


def _shift_along_normal_dominant(field, mesh, steps):
    """Shift *field* by *steps* lattice cells along the dominant normal
    direction at each near-wall cell.

    For each cell the axis (x/y/z) with the largest |n_component| is chosen
    and the field is sampled at ``±steps`` along that axis (sign taken from
    the normal component, so the sample lies further into the fluid).

    Uses ``torch.roll`` to build six shifted copies (±x, ±y, ±z) and selects
    per-cell with ``torch.where``.  Near-wall cells are interior, so
    wrap-around at domain boundaries does not affect them.

    Returns a tensor the same shape as *field*.
    """
    nx_n, ny_n, nz_n = mesh.nx_n, mesh.ny_n, mesh.nz_n
    abs_nx = nx_n.abs()
    abs_ny = ny_n.abs()
    abs_nz = nz_n.abs()

    # Dominant axis masks (break ties: x > y > z)
    dom_x = (abs_nx >= abs_ny) & (abs_nx >= abs_nz)
    dom_y = (~dom_x) & (abs_ny >= abs_nz)
    dom_z = (~dom_x) & (~dom_y)

    # Six shifted copies.
    # +x direction (nx_n>0): sample at i+steps → roll by -steps in dims=2
    fx_pos = torch.roll(field, -steps, dims=2)
    fx_neg = torch.roll(field,  steps, dims=2)
    fy_pos = torch.roll(field, -steps, dims=1)
    fy_neg = torch.roll(field,  steps, dims=1)
    fz_pos = torch.roll(field, -steps, dims=0)
    fz_neg = torch.roll(field,  steps, dims=0)

    x_pos_mask = dom_x & (nx_n > 0)
    x_neg_mask = dom_x & (nx_n < 0)
    y_pos_mask = dom_y & (ny_n > 0)
    y_neg_mask = dom_y & (ny_n < 0)
    z_pos_mask = dom_z & (nz_n > 0)
    z_neg_mask = dom_z & (nz_n < 0)

    result = torch.zeros_like(field)
    result = torch.where(x_pos_mask, fx_pos, result)
    result = torch.where(x_neg_mask, fx_neg, result)
    result = torch.where(y_pos_mask, fy_pos, result)
    result = torch.where(y_neg_mask, fy_neg, result)
    result = torch.where(z_pos_mask, fz_pos, result)
    result = torch.where(z_neg_mask, fz_neg, result)
    return result


def drag_pressure_integration(f, mesh, dpS, extrap='none'):
    """Pressure drag: 3D force vector from pressure × normal × dA.

    F = -Σ (p_wall - p_0) · n · dA  (force on wall, negative of fluid force)
    Background pressure p_0 is subtracted to prevent spurious force from
    discrete surface non-closure (Σ n·dA ≠ 0 on staircase surface).

    Parameters
    ----------
    extrap : {'none', 'linear', 'quadratic'}
        Wall-pressure extrapolation from near-wall cells along the dominant
        normal direction::

            'none'      → p_wall = p1                 (cell 1, current)
            'linear'    → p_wall = 2·p1 - p2          (1st-order extrap)
            'quadratic' → p_wall = 3·p1 - 3·p2 + p3   (2nd-order extrap)

        where p1 is the pressure at the near-wall cell, p2/p3 are pressures
        1/2 cells further into the fluid along the dominant normal axis.

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

    if extrap == 'none':
        p_wall = p_corr
    elif extrap == 'linear':
        p2 = _shift_along_normal_dominant(p_corr, mesh, steps=1)
        p_wall = 2.0 * p_corr - p2
    elif extrap == 'quadratic':
        p2 = _shift_along_normal_dominant(p_corr, mesh, steps=1)
        p3 = _shift_along_normal_dominant(p_corr, mesh, steps=2)
        p_wall = 3.0 * p_corr - 3.0 * p2 + p3
    else:
        raise ValueError(f"extrap must be 'none', 'linear', or 'quadratic', got '{extrap}'")

    mask = mask_float * mesh.dA
    fpx = -(p_wall * mesh.nx_n * mask).sum()
    fpy = -(p_wall * mesh.ny_n * mask).sum()
    fpz = -(p_wall * mesh.nz_n * mask).sum()
    return float(fpx.item() / dpS), float(fpy.item() / dpS), float(fpz.item() / dpS)


def drag_friction_integration(f, mesh, dpS, nu, q_wall=None):
    """Friction drag via 3D wall shear stress.

    Standard half-way BB (q_wall=None):
        τ = 2ν · u_t = ν · u_t / 0.5
    where u_t = u - (u·n)·n (tangent velocity) and the wall is at 0.5
    cell distance from the near-wall fluid cell.

    BFL corrected (q_wall provided):
        τ = ν · u_t / q
    where q is the fractional wall distance from the BFL q-field, averaged
    over boundary directions at each near-wall cell.  When q=0.5 this
    reduces to the standard formula τ = 2ν·u_t.

    Parameters
    ----------
    q_wall : torch.Tensor or None, shape (nz, ny, nx)
        Effective fractional wall distance at each near-wall cell.
        When None, uses the standard half-way BB formula (q=0.5).

    Verified on Couette flow: Cf error = 0.00% (standard, q_wall=None).
    """
    rho, ux, uy, uz = macroscopic3d(f)
    nx, ny, nz = mesh.nx_n, mesh.ny_n, mesh.nz_n
    u_dot_n = ux * nx + uy * ny + uz * nz
    ut_x = ux - u_dot_n * nx
    ut_y = uy - u_dot_n * ny
    ut_z = uz - u_dot_n * nz
    if q_wall is not None:
        # BFL: τ = ν · u_t / q  (q=0.5 → τ = 2ν·u_t, standard)
        inv_q = 1.0 / q_wall.clamp(min=1e-6)
        tau_x = nu * ut_x * inv_q
        tau_y = nu * ut_y * inv_q
        tau_z = nu * ut_z * inv_q
    else:
        # Standard half-way BB: τ = 2ν · u_t
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
