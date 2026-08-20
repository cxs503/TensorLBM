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
from dataclasses import dataclass

import torch

from .d3q19 import macroscopic3d
from .d3q27 import macroscopic27


def _macroscopic(f):
    """Auto-detect D3Q19 vs D3Q27 and return (rho, ux, uy, uz)."""
    if f.shape[0] == 27:
        return macroscopic27(f)
    return macroscopic3d(f)


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

        from .suboff_cad import SuboffConfig, suboff_radius_profile

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
    def from_naca(cls, solid, near, x_le, y_c, chord, m=0.04, p=0.40, t=0.12):
        """Analytical normal for NACA 4-digit airfoil (2D extruded).

        For a cambered airfoil (e.g. NACA 4412: m=0.04, p=0.40, t=0.12)
        the mean camber line y_camber(x) is NOT at y_c — it is shifted
        upward by the camber.  The upper/lower surface classification
        must use the camber line, not the chord centerline, otherwise
        lower-surface cells near the camber line get outward normals
        pointing UP instead of DOWN, corrupting the lift sign.

        Surface equations (body frame, x normalised 0–1):
          y_camber = m/p²·(2px − x²)            for x < p
          y_camber = m/(1−p)²·((1−2p) + 2px − x²)  for x ≥ p
          y_t      = 5t·(0.2969√x − 0.1260x − 0.3516x² + 0.2843x³ − 0.1015x⁴)
          upper:   y = y_camber + y_t
          lower:   y = y_camber − y_t

        The outward normal uses the FULL surface slope (camber + thickness
        derivative), and the sign (upper +1 / lower −1) is determined by
        comparing the cell y to the camber line, not the chord centerline.

        Parameters
        ----------
        m, p, t : float
            NACA 4-digit parameters (max camber, camber position, thickness).
            Defaults give NACA 4412.  Set m=0 for a symmetric airfoil.
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

        # --- Camber line y_camber(x) in lattice units ---
        # y_camber_lattice = y_c + camber(x) * chord
        camber = torch.where(
            xc < p,
            (m / (p ** 2)) * (2.0 * p * xc - xc ** 2),
            (m / ((1.0 - p) ** 2)) * ((1.0 - 2.0 * p) + 2.0 * p * xc - xc ** 2),
        )
        y_camber_lattice = y_c + camber * chord

        # --- Thickness derivative dy_t/dx ---
        dydx_t = 5.0 * t * (
            0.2969 / (2.0 * torch.sqrt(xc))
            - 0.1260
            - 0.7032 * xc
            + 0.8529 * xc ** 2
            - 0.4060 * xc ** 3
        )

        # --- Camber line derivative dy_camber/dx ---
        dydx_camber = torch.where(
            xc < p,
            (m / (p ** 2)) * (2.0 * p - 2.0 * xc),
            (m / ((1.0 - p) ** 2)) * (2.0 * p - 2.0 * xc),
        )

        # --- Upper/lower classification using camber line ---
        # Upper: y > y_camber, Lower: y < y_camber
        sign = torch.where(yy > y_camber_lattice, 1.0, -1.0)

        # --- Outward normal ---
        # Upper surface slope: dy/dx = dy_camber/dx + dy_t/dx
        # Lower surface slope: dy/dx = dy_camber/dx - dy_t/dx
        # Normal = (-dy/dx, sign, 0) / |n|
        dydx = dydx_camber + sign * dydx_t
        nx_n = -dydx * sign
        ny_n = sign.float()
        nz_n = torch.zeros_like(nx_n)
        norm = torch.sqrt(nx_n**2 + ny_n**2).clamp(min=1e-10)
        nx_n = nx_n / norm * near.float()
        ny_n = ny_n / norm * near.float()
        nz_n = nz_n / norm * near.float()
        return cls(near, nx_n, ny_n, nz_n)
    
    @classmethod
    def from_stl(cls, solid, near, vertices, faces, face_normals, origin, spacing,
                 dA_method="none"):
        """STL-derived surface normals for arbitrary geometry.

        Thin wrapper around :func:`tensorlbm.stl_geometry.SurfaceMesh_from_stl`
        that finds the nearest STL triangle for each near-wall cell and
        uses its face normal (flipped to point outward).

        Parameters
        ----------
        dA_method : {'none', 'stl_area', 'cos_theta'}, default 'none'
            Surface area element computation method.  See
            :func:`SurfaceMesh_from_stl` for details.
        """
        from .stl_geometry import SurfaceMesh_from_stl

        return SurfaceMesh_from_stl(
            solid, near, vertices, faces, face_normals, origin, spacing,
            dA_method=dA_method,
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


def suboff_smooth_q(solid, near, cx, cy, cz, length, radius, config=None):
    """q_smooth = true wall distance ``r_cell - R(x)`` for the SUBOFF hull.

    For each near-wall fluid cell, ``r_cell = sqrt((y-cy)^2 + (z-cz)^2)``
    is the radial distance from the hull axis and
    ``R(x) = suboff_radius_profile(xi) * radius`` is the local smooth-body
    radius in lattice units (xi = normalized axial coordinate, 0 at bow,
    1 at stern).  ``q_smooth`` is therefore the true distance from the cell
    centre to the *smooth* hull surface — the voxel staircase wall sits at
    a different distance.  Values are clamped to ``[0.05, 1.0]`` (same
    convention as the 3D-cylinder D20 7-formula study).

    Feed the result to :func:`drag_friction_integration` with
    ``formula='bfl'`` or ``'bfl_smooth'`` to get ``tau = nu * u_1 / q_smooth``,
    the BFL correction using the analytic body distance instead of the
    half-way bounce-back gap q=0.5.  On the 3D cylinder D20 (Re=40) this
    moved the total-drag error from -11.3% (standard) to -3.69%.

    Parameters
    ----------
    solid : torch.Tensor, bool, shape (nz, ny, nx)
        Solid mask (hull axis along x).
    near : torch.Tensor, bool, shape (nz, ny, nx)
        Near-wall mask (only these cells get a nonzero q).
    cx, cy, cz : float
        Hull centre coordinates (cells); the engine places the SUBOFF at
        ``(nx*0.25, ny*0.5, nz*0.5)``.
    length : float
        Hull length in lattice units (resolution L).
    radius : float
        Maximum hull radius in lattice units (R_max / dx).
    config : SuboffConfig or None
        Geometry configuration (defaults to real DARPA SUBOFF).

    Returns
    -------
    torch.Tensor
        ``q_smooth`` field, same shape as *solid*, zero outside *near*.
    """
    import numpy as np

    from .suboff_cad import SuboffConfig, suboff_radius_profile

    if config is None:
        config = SuboffConfig()
    nz, ny, nx = solid.shape
    device = solid.device
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    x_bow = cx - length / 2.0
    xi_np = ((xx - x_bow) / length).cpu().numpy()
    r_norm = suboff_radius_profile(xi_np, config)  # normalised [0, 1]
    r_lu = torch.tensor(r_norm * radius, device=device, dtype=torch.float32)
    r_cell = torch.sqrt((yy - cy) ** 2 + (zz - cz) ** 2)
    q = (r_cell - r_lu).clamp(0.05, 1.0)
    return q * near.to(device=device).float()


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
    # Bug 32: device sync
    _dev = field.device
    if nx_n.device != _dev:
        nx_n = nx_n.to(_dev); ny_n = ny_n.to(_dev); nz_n = nz_n.to(_dev)
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


@dataclass(frozen=True)
class BFLWallPressureDiagnostics:
    """Coverage of link-wise wall-pressure reconstruction."""

    boundary_cells: int
    requested_links: int
    usable_links: int
    fallback_cells: int
    minimum_active_q: float
    maximum_active_q: float


def _sample_at_offset_no_wrap(
    field: torch.Tensor,
    offset_zyx: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``field[index + offset]`` and an in-domain validity mask."""
    if field.ndim != 3:
        raise ValueError("field must have shape (nz,ny,nx)")
    out = torch.zeros_like(field)
    valid = torch.zeros(field.shape, dtype=torch.bool, device=field.device)
    target: list[slice] = []
    source: list[slice] = []
    for size, offset in zip(field.shape, offset_zyx, strict=True):
        if abs(offset) >= size:
            return out, valid
        if offset > 0:
            target.append(slice(0, size - offset))
            source.append(slice(offset, size))
        elif offset < 0:
            target.append(slice(-offset, size))
            source.append(slice(0, size + offset))
        else:
            target.append(slice(None))
            source.append(slice(None))
    target_index = tuple(target)
    source_index = tuple(source)
    out[target_index] = field[source_index]
    valid[target_index] = True
    return out, valid


def reconstruct_bfl_wall_pressure(
    pressure: torch.Tensor,
    mesh: SurfaceMesh,
    fluid_boundary_mask: torch.Tensor,
    q_field: torch.Tensor,
    *,
    solid: torch.Tensor,
) -> tuple[torch.Tensor, BFLWallPressureDiagnostics]:
    """Reconstruct pressure at BFL intersections using their actual ``q``.

    For a boundary link directed from the fluid node toward the solid, the
    next two samples are taken in the opposite (fluid) direction.  A
    quadratic through samples at link coordinates 0, 1 and 2 is evaluated at
    the wall coordinate ``-q``::

        p_w = (q+1)(q+2)/2 p_1 - q(q+2) p_2 + q(q+1)/2 p_3.

    Link estimates at the same boundary cell are averaged with their positive
    normal-alignment weights.  Invalid stencils never wrap across a physical
    domain face and fall back to the local fluid pressure, with coverage
    reported explicitly.
    """
    from .d3q19 import C

    if pressure.ndim != 3:
        raise ValueError("pressure must have shape (nz,ny,nx)")
    expected = (19, *pressure.shape)
    if fluid_boundary_mask.shape != expected:
        raise ValueError("fluid_boundary_mask must have shape (19,nz,ny,nx)")
    if q_field.shape != expected:
        raise ValueError("q_field must have shape (19,nz,ny,nx)")
    if solid.shape != pressure.shape or solid.dtype is not torch.bool:
        raise ValueError("solid must be bool with shape (nz,ny,nx)")
    if mesh.near.shape != pressure.shape:
        raise ValueError("surface mesh must share the pressure grid")

    device = pressure.device
    boundary_mask = fluid_boundary_mask.to(device=device, dtype=torch.bool)
    q = q_field.to(device=device, dtype=pressure.dtype)
    solid_device = solid.to(device=device)
    near = mesh.near.to(device=device, dtype=torch.bool)
    active = boundary_mask & near.unsqueeze(0)
    active_q = q[active]
    if active_q.numel() and (
        not bool(torch.isfinite(active_q).all())
        or bool(((active_q <= 0.0) | (active_q > 1.0)).any())
    ):
        raise ValueError("active BFL q values must be finite and lie in (0,1]")

    normals = tuple(
        component.to(device=device, dtype=pressure.dtype)
        for component in (mesh.nx_n, mesh.ny_n, mesh.nz_n)
    )
    weighted_pressure = torch.zeros_like(pressure)
    weight_sum = torch.zeros_like(pressure)
    usable_links = 0
    c = C.to(device=device)
    for direction in range(1, 19):
        link = active[direction]
        if not bool(link.any()):
            continue
        cx, cy, cz = (int(value) for value in c[direction].tolist())
        offset_one = (-cz, -cy, -cx)
        offset_two = (-2 * cz, -2 * cy, -2 * cx)
        p2, valid2 = _sample_at_offset_no_wrap(pressure, offset_one)
        p3, valid3 = _sample_at_offset_no_wrap(pressure, offset_two)
        solid2, _ = _sample_at_offset_no_wrap(solid_device, offset_one)
        solid3, _ = _sample_at_offset_no_wrap(solid_device, offset_two)
        usable = link & valid2 & valid3 & ~solid2 & ~solid3
        if not bool(usable.any()):
            continue
        q_link = q[direction]
        wall = (
            0.5 * (q_link + 1.0) * (q_link + 2.0) * pressure
            - q_link * (q_link + 2.0) * p2
            + 0.5 * q_link * (q_link + 1.0) * p3
        )
        link_length = math.sqrt(cx * cx + cy * cy + cz * cz)
        alignment = -(
            cx * normals[0] + cy * normals[1] + cz * normals[2]
        ) / link_length
        weight = torch.where(
            usable,
            alignment.clamp_min(0.0),
            torch.zeros_like(alignment),
        )
        weighted_pressure += weight * wall
        weight_sum += weight
        usable_links += int((weight > 0.0).sum().item())

    boundary_cells = near & active.any(dim=0)
    reconstructed = torch.where(
        weight_sum > 0.0,
        weighted_pressure / weight_sum.clamp_min(torch.finfo(pressure.dtype).tiny),
        pressure,
    )
    diagnostics = BFLWallPressureDiagnostics(
        boundary_cells=int(boundary_cells.sum().item()),
        requested_links=int(active.sum().item()),
        usable_links=usable_links,
        fallback_cells=int((boundary_cells & (weight_sum <= 0.0)).sum().item()),
        minimum_active_q=(float(active_q.min().item()) if active_q.numel() else math.nan),
        maximum_active_q=(float(active_q.max().item()) if active_q.numel() else math.nan),
    )
    return reconstructed, diagnostics


def integrate_bfl_projected_pressure(
    pressure: torch.Tensor,
    fluid_boundary_mask: torch.Tensor,
    q_field: torch.Tensor,
    *,
    solid: torch.Tensor,
    reconstruction: str = "quadratic",
) -> tuple[tuple[float, float, float], BFLWallPressureDiagnostics]:
    """Integrate pressure on axial BFL crossing faces.

    Each axial fluid-to-solid link represents one unit projected lattice face.
    Its body-force contribution is ``p_wall * c`` because the body outward
    normal is opposite the link direction.  Opposite face counts cancel a
    constant pressure exactly on every closed voxel body; unlike a calibrated
    nodal surface area, this is a discrete finite-volume identity.

    ``reconstruction`` controls the wall value on a link whose first fluid
    sample is a fractional distance ``q`` from the intersection.  ``local``
    uses that first value unchanged, ``linear`` extrapolates through the first
    two fluid samples, and ``quadratic`` (the historical default) uses three.
    The selectable orders are primarily useful for verification because
    high-order wall extrapolation can amplify an unresolved near-wall pressure
    mode even when all population values remain finite.
    """
    from .d3q19 import C

    if pressure.ndim != 3:
        raise ValueError("pressure must have shape (nz,ny,nx)")
    expected = (19, *pressure.shape)
    if fluid_boundary_mask.shape != expected or q_field.shape != expected:
        raise ValueError("BFL mask and q field must have shape (19,nz,ny,nx)")
    if solid.shape != pressure.shape or solid.dtype is not torch.bool:
        raise ValueError("solid must be bool with shape (nz,ny,nx)")
    if reconstruction not in {"local", "linear", "quadratic"}:
        raise ValueError(
            "reconstruction must be 'local', 'linear', or 'quadratic'",
        )
    device = pressure.device
    boundary = fluid_boundary_mask.to(device=device, dtype=torch.bool)
    q = q_field.to(device=device, dtype=pressure.dtype)
    solid_device = solid.to(device=device)
    c = C.to(device=device)
    force = torch.zeros(3, dtype=pressure.dtype, device=device)
    axial_active: list[torch.Tensor] = []
    axial_q_values: list[torch.Tensor] = []
    fallback_mask = torch.zeros_like(pressure, dtype=torch.bool)
    usable_links = 0
    for direction in range(1, 19):
        cx, cy, cz = (int(value) for value in c[direction].tolist())
        if abs(cx) + abs(cy) + abs(cz) != 1:
            continue
        link = boundary[direction]
        if not bool(link.any()):
            continue
        q_link = q[direction]
        active_q = q_link[link]
        if (
            not bool(torch.isfinite(active_q).all())
            or bool(((active_q <= 0.0) | (active_q > 1.0)).any())
        ):
            raise ValueError("active BFL q values must be finite and lie in (0,1]")
        offset_one = (-cz, -cy, -cx)
        offset_two = (-2 * cz, -2 * cy, -2 * cx)
        p2, valid2 = _sample_at_offset_no_wrap(pressure, offset_one)
        p3, valid3 = _sample_at_offset_no_wrap(pressure, offset_two)
        solid2, _ = _sample_at_offset_no_wrap(solid_device, offset_one)
        solid3, _ = _sample_at_offset_no_wrap(solid_device, offset_two)
        if reconstruction == "local":
            usable = link
            wall = pressure
        elif reconstruction == "linear":
            usable = link & valid2 & ~solid2
            wall = (q_link + 1.0) * pressure - q_link * p2
        else:
            usable = link & valid2 & valid3 & ~solid2 & ~solid3
            wall = (
                0.5 * (q_link + 1.0) * (q_link + 2.0) * pressure
                - q_link * (q_link + 2.0) * p2
                + 0.5 * q_link * (q_link + 1.0) * p3
            )
        wall = torch.where(usable, wall, pressure)
        link_pressure_sum = wall[link].sum()
        force += link_pressure_sum * torch.tensor(
            (cx, cy, cz), dtype=pressure.dtype, device=device,
        )
        axial_active.append(link)
        axial_q_values.append(active_q)
        fallback_mask |= link & ~usable
        usable_links += int(usable.sum().item())
    if axial_active:
        active_stack = torch.stack(axial_active)
        active_any = active_stack.any(dim=0)
        active_q = torch.cat(axial_q_values)
    else:
        active_stack = torch.zeros(
            (0, *pressure.shape), dtype=torch.bool, device=device,
        )
        active_any = torch.zeros_like(pressure, dtype=torch.bool)
        active_q = torch.empty(0, dtype=pressure.dtype, device=device)
    diagnostics = BFLWallPressureDiagnostics(
        boundary_cells=int(active_any.sum().item()),
        requested_links=int(active_stack.sum().item()),
        usable_links=usable_links,
        fallback_cells=int(fallback_mask.sum().item()),
        minimum_active_q=(float(active_q.min().item()) if active_q.numel() else math.nan),
        maximum_active_q=(float(active_q.max().item()) if active_q.numel() else math.nan),
    )
    return tuple(float(value.item()) for value in force), diagnostics


def drag_pressure_integration(f, mesh, dpS, extrap='none', p0_method='near_wall',
                              solid=None, p0_inlet_width=5,
                              fluid_boundary_mask=None, q_field=None):
    """Pressure drag: 3D force vector from pressure × normal × dA.

    F = -Σ (p_wall - p_0) · n · dA  (force on wall, negative of fluid force)
    Background pressure p_0 is subtracted to prevent spurious force from
    discrete surface non-closure (Σ n·dA ≠ 0 on staircase surface).

    Parameters
    ----------
    extrap : {'none', 'linear', 'quadratic', 'bfl_quadratic'}
        Wall-pressure extrapolation from near-wall cells along the dominant
        normal direction::

            'none'      → p_wall = p1                 (cell 1, current)
            'linear'    → p_wall = 2·p1 - p2          (1st-order extrap)
            'quadratic' → p_wall = 3·p1 - 3·p2 + p3   (2nd-order extrap)
            'bfl_quadratic' → link-wise quadratic reconstruction at each
                              Bouzidi intersection using its actual q value

        where p1 is the pressure at the near-wall cell, p2/p3 are pressures
        1/2 cells further into the fluid along the dominant normal axis.

    p0_method : {'near_wall', 'far_field', 'domain_avg', 'inlet'}
        Method for computing the background pressure p_0::

            'near_wall'  → average pressure at near-wall cells (original).
                           Can over-correct when wall function alters the
                           near-wall pressure field (negative Cd_p bug).
            'far_field'  → average pressure at fluid cells that are NOT
                           near-wall (bulk free-stream reference).  More
                           stable when wall functions change near-wall p.
            'domain_avg' → average pressure over ALL fluid cells.
            'inlet'      → average pressure at the first *p0_inlet_width*
                           x-planes (inlet/free-stream reference).

    solid : torch.Tensor or None
        Solid mask ``(nz, ny, nx)``.  Required for p0_method != 'near_wall'.

    p0_inlet_width : int
        Number of x-planes for the 'inlet' p0_method (default 5).

    Returns: (Cd_x, Cd_y, Cd_z) = (fx, fy, fz) / dpS
    """
    rho, _, _, _ = _macroscopic(f)
    p = (rho - 1.0) / 3.0
    # Subtract background pressure p_0 to prevent spurious force from
    # discrete surface non-closure (Σ n·dA ≠ 0 on staircase surface).
    mask_float = mesh.near.float()
    # Bug 32 fix: ensure mask on same device as p
    if mask_float.device != p.device:
        mask_float = mask_float.to(p.device)
    if solid is not None and solid.device != p.device:
        solid = solid.to(p.device)
    if p0_method == 'near_wall':
        n_p0 = mask_float.sum().clamp(min=1.0)
        p0 = (p * mask_float).sum() / n_p0
    elif p0_method == 'far_field':
        # Bulk fluid cells (fluid but NOT near-wall) — stable free-stream ref
        if solid is None:
            raise ValueError("solid mask required for p0_method='far_field'")
        far_mask = (~solid).float() * (1.0 - mask_float)
        n_p0 = far_mask.sum().clamp(min=1.0)
        p0 = (p * far_mask).sum() / n_p0
    elif p0_method == 'domain_avg':
        if solid is None:
            raise ValueError("solid mask required for p0_method='domain_avg'")
        fluid_mask = (~solid).float()
        n_p0 = fluid_mask.sum().clamp(min=1.0)
        p0 = (p * fluid_mask).sum() / n_p0
    elif p0_method == 'inlet':
        if solid is None:
            raise ValueError("solid mask required for p0_method='inlet'")
        inlet_mask = (~solid).float()
        inlet_mask[:, :, p0_inlet_width:] = 0.0
        n_p0 = inlet_mask.sum().clamp(min=1.0)
        p0 = (p * inlet_mask).sum() / n_p0
    else:
        raise ValueError(f"p0_method must be 'near_wall', 'far_field', "
                         f"'domain_avg', or 'inlet', got '{p0_method}'")
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
    elif extrap == 'bfl_quadratic':
        if solid is None or fluid_boundary_mask is None or q_field is None:
            raise ValueError(
                "solid, fluid_boundary_mask and q_field are required for "
                "extrap='bfl_quadratic'",
            )
        p_wall, _ = reconstruct_bfl_wall_pressure(
            p_corr, mesh, fluid_boundary_mask, q_field, solid=solid,
        )
    else:
        raise ValueError(
            "extrap must be 'none', 'linear', 'quadratic', or "
            f"'bfl_quadratic'; got '{extrap}'",
        )

    mask = mask_float * mesh.dA
    fpx = -(p_wall * mesh.nx_n * mask).sum()
    fpy = -(p_wall * mesh.ny_n * mask).sum()
    fpz = -(p_wall * mesh.nz_n * mask).sum()
    return float(fpx.item() / dpS), float(fpy.item() / dpS), float(fpz.item() / dpS)


def _wall_face_counts(solid):
    """Per-fluid-cell wall-face counts (nfx, nfy, nfz) of a voxel solid.

    ``nfx[i]`` counts how many of the cell's x-neighbours (i±1 in x) are
    solid, i.e. the number of solid-adjacent faces normal to x.  Shared by
    the 'faces' and 'mix50' friction formulas.
    """
    fluid = ~solid
    nfx = torch.zeros_like(solid, dtype=torch.float32)
    nfy = torch.zeros_like(solid, dtype=torch.float32)
    nfz = torch.zeros_like(solid, dtype=torch.float32)
    # x faces (solid at ±x)
    nfx[:, :, 1:-1] += (solid[:, :, 2:] & fluid[:, :, 1:-1]).float()
    nfx[:, :, 1:-1] += (solid[:, :, :-2] & fluid[:, :, 1:-1]).float()
    # y faces (solid at ±y)
    nfy[:, 1:-1, :] += (solid[:, 2:, :] & fluid[:, 1:-1, :]).float()
    nfy[:, 1:-1, :] += (solid[:, :-2, :] & fluid[:, 1:-1, :]).float()
    # z faces (solid at ±z)
    nfz[1:-1, :, :] += (solid[2:, :, :] & fluid[1:-1, :, :]).float()
    nfz[1:-1, :, :] += (solid[:-2, :, :] & fluid[1:-1, :, :]).float()
    return nfx, nfy, nfz


def drag_friction_integration(f, mesh, dpS, nu, q_wall=None, formula='standard',
                              solid=None):
    """Friction drag via 3D wall shear stress.

    Multiple friction formulas are supported via the *formula* parameter.
    All formulas use the tangential velocity u_t = u - (u·n)·n at near-wall
    cells.  For half-way bounce-back the wall is at distance q=0.5 from the
    near-wall fluid cell; u_1 is the tangential velocity at the near-wall
    cell (distance 0.5 from wall) and u_2 is at the second cell (distance
    1.5 from wall), obtained by shifting one cell along the dominant
    normal direction into the fluid.

    ==============  =================================================  ===========
    formula         expression (standard BB, Δn=0.5)                   exact for
    ==============  =================================================  ===========
    'standard'      τ = 2ν·u_1 = ν·u_1/0.5                             linear
    '2nd_order'     τ = ν·(3·u_1 − u_2)                                (task spec)
    'central'       τ = ν·u_2                                          (task spec)
    'lagrange'      τ = ν·(3·u_1 − u_2/3)                              linear+quad
    'bfl'           τ = ν·u_1/q  (requires q_wall)                     linear
    'bfl_smooth'    τ = ν·u_1/q_smooth  (alias of 'bfl'; q_wall is the
                    distance to the SMOOTH body surface, e.g. from
                    suboff_smooth_q, not the voxel half-gap)           linear
    'bfl_lagrange'  τ = ν·(u_1(q+1)/q − u_2·q/(q+1))  (requires q_wall)
                                                                       linear+quad
    'faces'         per-wall-face shear, dA=1 per face (requires solid)  linear
    'mix50'         Cd_f = 0.5·(standard + faces) (requires solid)      —
    ==============  =================================================  ===========

    The 'lagrange' formula is the exact second-order derivative for the
    non-uniform grid with sample points at distances 0 (wall, u=0),
    0.5 (u_1) and 1.5 (u_2) from the wall — it is exact for both linear
    and quadratic velocity profiles and is expected to converge best
    under grid refinement for smooth boundary layers.

    'bfl_lagrange' is the exact second-order Lagrange derivative on the
    non-uniform grid with the wall at 0, u_1 at distance q and u_2 at
    q+1 (q = actual fractional wall distance, e.g. from a Bouzidi
    intersection or the analytic body distance).  It reduces to 'lagrange'
    when q=0.5.

    'faces' integrates over the voxel staircase wall faces instead of
    near-wall cells: every fluid-solid face contributes its own shear
    (half-way BB: τ = 2ν·u_t with u_t tangential to the face), with dA=1
    per face.  This is the exact discrete friction of the voxelized body.
    Cells with two wall faces (staircase inner corners) then contribute
    both faces, which the cell-based 'standard' formula misses.  On a
    planar wall both give the same result.  The returned force is the
    sum of the face shear vectors; mesh.dA is not applied (faces are
    unit area).

    'mix50' evaluates 'standard' and 'faces' on the same field and returns
    their arithmetic mean.  The two bracket the smooth-body continuum
    friction: 'standard' (near-wall cell sum) is a lower bound that misses
    staircase inner-corner faces, while 'faces' (staircase-exact) is an
    upper bound because the voxel stair area exceeds the smooth wet area.
    The midpoint is the simplest principled interpolation between the two
    (SUBOFF Re=1000 L=96: Cd_tot = +0.25% vs Blasius 0.0420).

    Parameters
    ----------
    q_wall : torch.Tensor or None, shape (nz, ny, nx)
        Effective fractional wall distance at each near-wall cell.
        Used when formula='bfl' or formula='bfl_lagrange'.
    formula : str
        Friction formula: 'standard' (default), '2nd_order', 'central',
        'lagrange', 'bfl', 'bfl_lagrange', or 'faces'.
    solid : torch.Tensor or None, shape (nz, ny, nx), bool
        Solid mask.  Required for formula='faces'.

    Verified on Couette flow: Cf error = 0.00% (standard, q_wall=None).
    """
    rho, ux, uy, uz = _macroscopic(f)
    nx, ny, nz = mesh.nx_n, mesh.ny_n, mesh.nz_n
    u_dot_n = ux * nx + uy * ny + uz * nz
    ut_x = ux - u_dot_n * nx   # u_1 tangential (near-wall cell)
    ut_y = uy - u_dot_n * ny
    ut_z = uz - u_dot_n * nz

    if formula == 'standard':
        # τ = 2ν · u_1  (1st-order forward difference, Δn=0.5)
        tau_x = 2.0 * nu * ut_x
        tau_y = 2.0 * nu * ut_y
        tau_z = 2.0 * nu * ut_z
    elif formula in ('bfl', 'bfl_smooth'):
        # τ = ν · u_1 / q  (BFL corrected; q=0.5 → standard).
        # 'bfl_smooth' is an alias: q_wall is then the distance to the
        # SMOOTH body surface (e.g. suboff_smooth_q), not the voxel gap.
        if q_wall is None:
            raise ValueError(f"formula='{formula}' requires q_wall tensor")
        inv_q = 1.0 / q_wall.clamp(min=1e-6)
        tau_x = nu * ut_x * inv_q
        tau_y = nu * ut_y * inv_q
        tau_z = nu * ut_z * inv_q
    elif formula == 'bfl_lagrange':
        # τ = ν·[u_1·(q+1)/q − u_2·q/(q+1)]  [exact 2nd-order Lagrange
        # derivative at the wall for samples at 0 (wall, u=0), q (u_1),
        # q+1 (u_2) — see docstring; reduces to 'lagrange' at q=0.5]
        if q_wall is None:
            raise ValueError("formula='bfl_lagrange' requires q_wall tensor")
        ut2_x = _shift_along_normal_dominant(ut_x, mesh, steps=1)
        ut2_y = _shift_along_normal_dominant(ut_y, mesh, steps=1)
        ut2_z = _shift_along_normal_dominant(ut_z, mesh, steps=1)
        q_c = q_wall.clamp(min=1e-6)
        coeff1 = (q_c + 1.0) / q_c
        coeff2 = q_c / (q_c + 1.0)
        tau_x = nu * (coeff1 * ut_x - coeff2 * ut2_x)
        tau_y = nu * (coeff1 * ut_y - coeff2 * ut2_y)
        tau_z = nu * (coeff1 * ut_z - coeff2 * ut2_z)
    elif formula == 'faces':
        # Staircase-exact: integrate over fluid-solid wall faces.  For each
        # face the half-way BB shear is τ = 2ν·u_t with u_t tangential to
        # the (axis-aligned) face, dA = 1 per face.  Per cell the three
        # force components receive 2ν·u_x from every y/z-face, 2ν·u_y from
        # every x/z-face, and 2ν·u_z from every x/y-face.
        if solid is None:
            raise ValueError("formula='faces' requires solid mask")
        if solid.device != ux.device:
            solid = solid.to(ux.device)
        nfx, nfy, nfz = _wall_face_counts(solid)
        tau_x = 2.0 * nu * ux * (nfy + nfz)
        tau_y = 2.0 * nu * uy * (nfx + nfz)
        tau_z = 2.0 * nu * uz * (nfx + nfy)
        # unit face area; mesh.dA intentionally not applied
        ffx = tau_x.sum()
        ffy = tau_y.sum()
        ffz = tau_z.sum()
        return float(ffx.item() / dpS), float(ffy.item() / dpS), float(ffz.item() / dpS)
    elif formula == 'mix50':
        # Cd_f = 0.5·(standard + faces): midpoint interpolation between the
        # cell-based near-wall sum (lower bound: misses staircase
        # inner-corner faces) and the staircase-exact per-face sum (upper
        # bound: stair area exceeds the smooth wet area).  The smooth-body
        # continuum friction lies between the two; equal weighting is the
        # simple midpoint (SUBOFF Re=1000 L=96: +0.25% vs Blasius 0.0420).
        if solid is None:
            raise ValueError("formula='mix50' requires solid mask")
        if solid.device != ux.device:
            solid = solid.to(ux.device)
        mask = mesh.near.float() * mesh.dA
        std_sum = (
            2.0 * nu * (ut_x * mask).sum(),
            2.0 * nu * (ut_y * mask).sum(),
            2.0 * nu * (ut_z * mask).sum(),
        )
        nfx, nfy, nfz = _wall_face_counts(solid)
        face_sum = (
            (2.0 * nu * ux * (nfy + nfz)).sum(),
            (2.0 * nu * uy * (nfx + nfz)).sum(),
            (2.0 * nu * uz * (nfx + nfy)).sum(),
        )
        mix = tuple(0.5 * (std_sum[i] + face_sum[i]) for i in range(3))
        return (
            float(mix[0].item() / dpS),
            float(mix[1].item() / dpS),
            float(mix[2].item() / dpS),
        )
    elif formula in ('2nd_order', 'central', 'lagrange'):
        # Need u_2: tangential velocity at second cell from wall.
        # Shift velocity one cell along dominant normal into the fluid.
        ut2_x = _shift_along_normal_dominant(ut_x, mesh, steps=1)
        ut2_y = _shift_along_normal_dominant(ut_y, mesh, steps=1)
        ut2_z = _shift_along_normal_dominant(ut_z, mesh, steps=1)
        if formula == '2nd_order':
            # τ = ν·(3·u_1 − u_2)  [task-specified 2nd-order forward diff]
            tau_x = nu * (3.0 * ut_x - ut2_x)
            tau_y = nu * (3.0 * ut_y - ut2_y)
            tau_z = nu * (3.0 * ut_z - ut2_z)
        elif formula == 'central':
            # τ = ν·u_2  [task-specified central/forward diff, Δn=0.5]
            tau_x = nu * ut2_x
            tau_y = nu * ut2_y
            tau_z = nu * ut2_z
        else:  # 'lagrange'
            # τ = ν·(3·u_1 − u_2/3)  [exact 2nd-order for non-uniform grid:
            # wall at 0, u_1 at 0.5, u_2 at 1.5 — Lagrange interpolation]
            tau_x = nu * (3.0 * ut_x - ut2_x / 3.0)
            tau_y = nu * (3.0 * ut_y - ut2_y / 3.0)
            tau_z = nu * (3.0 * ut_z - ut2_z / 3.0)
    else:
        raise ValueError(
            f"formula must be 'standard', '2nd_order', 'central', "
            f"'lagrange', 'bfl', 'bfl_smooth', 'bfl_lagrange', 'faces', "
            f"or 'mix50'; got '{formula}'"
        )

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
    rho, _, _, _ = _macroscopic(f)
    p = (rho - 1.0) / 3.0
    return p / (0.5 * u_in**2)
