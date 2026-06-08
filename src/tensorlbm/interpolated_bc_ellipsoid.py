"""Ray-ellipsoid intersection for Bouzidi interpolated BC on ellipsoidal obstacles.

Adds :func:`compute_q_ellipsoid` to the interpolated BC module.
"""
from __future__ import annotations

import math

import torch

from .d3q19 import C as C3D


def compute_q_ellipsoid(
    nx: int,
    ny: int,
    nz: int,
    cx: float,
    cy: float,
    cz: float,
    a: float,
    b: float,
    alpha_deg: float = 0.0,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the BFL fractional-distance field *q* for a prolate spheroid.

    For every fluid node adjacent to the ellipsoid surface, computes the
    fractional distance *q ∈ (0, 1]* along each D3Q19 lattice link to the
    ellipsoid surface via ray-quadric intersection.

    The ellipsoid in body coordinates: (x_body/a)² + (y_body/b)² + (z_body/b)² = 1.
    Nose-up rotation by alpha_deg about z-axis is applied.

    Parameters
    ----------
    nx, ny, nz : int
        Grid dimensions.
    cx, cy, cz : float
        Ellipsoid centre.
    a : float
        Semi-major axis (streamwise).
    b : float
        Semi-minor axis (cross-stream).
    alpha_deg : float
        Angle of attack [degrees]. Nose-up = positive.
    device : torch.device

    Returns
    -------
    fluid_boundary_mask : torch.Tensor of bool, shape (19, nz, ny, nx)
        True where fluid node (i,j,k) has the ellipsoid boundary along direction d.
    q_field : torch.Tensor of float32, shape (19, nz, ny, nx)
        Fractional distance q for each (direction, fluid node) pair.
        Non-boundary entries are 0.5.
    """
    alpha = math.radians(alpha_deg)
    cos_a = math.cos(alpha)
    sin_a = math.sin(alpha)

    c = C3D.to(device)  # (19, 3)

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float64),
        torch.arange(ny, device=device, dtype=torch.float64),
        torch.arange(nx, device=device, dtype=torch.float64),
        indexing="ij",
    )

    fluid_boundary_mask = torch.zeros((19, nz, ny, nx), dtype=torch.bool, device=device)
    q_field = torch.full((19, nz, ny, nx), 0.5, dtype=torch.float32, device=device)

    for d in range(19):
        dcx = float(c[d, 0].item())
        dcy = float(c[d, 1].item())
        dcz = float(c[d, 2].item())
        if dcx == 0.0 and dcy == 0.0 and dcz == 0.0:
            continue

        # --- Neighbor point (fluid → solid direction) ---
        xn = xx + dcx
        yn = yy + dcy
        zn = zz + dcz

        # Transform both self and neighbor to body frame
        dx_s = xx - cx
        dy_s = yy - cy
        dz_s = zz - cz
        dx_n = xn - cx
        dy_n = yn - cy
        dz_n = zn - cz

        # Body-frame coordinates (nose-up rotation)
        xb_s = dx_s * cos_a - dy_s * sin_a
        yb_s = dx_s * sin_a + dy_s * cos_a
        zb_s = dz_s

        xb_n = dx_n * cos_a - dy_n * sin_a
        yb_n = dx_n * sin_a + dy_n * cos_a
        zb_n = dz_n

        # Ellipsoid inequality: (x/a)² + (y/b)² + (z/b)² ≤ 1
        r2_s = (xb_s / a)**2 + (yb_s / b)**2 + (zb_s / b)**2
        r2_n = (xb_n / a)**2 + (yb_n / b)**2 + (zb_n / b)**2

        self_is_fluid = r2_s > 1.0
        nb_is_solid = r2_n <= 1.0
        boundary = self_is_fluid & nb_is_solid

        if not boundary.any():
            continue

        # Direction vector in body frame
        dx_b = xb_n - xb_s  # = dcx·cos_a - dcy·sin_a in body frame
        dy_b = yb_n - yb_s  # = dcx·sin_a + dcy·cos_a
        dz_b = zb_n - zb_s  # = dcz

        # Ray-ellipsoid intersection: |(p0 + t*d)/[a,b,b]|² = 1
        # (x0+t*dx)²/a² + (y0+t*dy)²/b² + (z0+t*dz)²/b² = 1
        # A·t² + B·t + C = 0
        inv_a2 = 1.0 / (a * a)
        inv_b2 = 1.0 / (b * b)

        A_coef = dx_b**2 * inv_a2 + (dy_b**2 + dz_b**2) * inv_b2
        B_coef = 2.0 * (xb_s * dx_b * inv_a2 + (yb_s * dy_b + zb_s * dz_b) * inv_b2)
        C_coef = xb_s**2 * inv_a2 + (yb_s**2 + zb_s**2) * inv_b2 - 1.0

        discriminant = B_coef**2 - 4.0 * A_coef * C_coef
        safe_disc = torch.where(
            boundary & (discriminant >= 0.0),
            discriminant,
            torch.zeros_like(discriminant),
        )
        sqrt_disc = torch.sqrt(safe_disc)

        t1 = (-B_coef - sqrt_disc) / (2.0 * A_coef)
        t2 = (-B_coef + sqrt_disc) / (2.0 * A_coef)

        # Normalise: q = t / |d| in lattice units
        link_len = math.sqrt(dcx**2 + dcy**2 + dcz**2)
        q1 = t1 / link_len
        q2 = t2 / link_len

        valid1 = (t1 > 1e-10) & (q1 <= 1.0 + 1e-10)
        valid2 = (t2 > 1e-10) & (q2 <= 1.0 + 1e-10)

        q_val = torch.where(
            valid1 & valid2,
            torch.min(q1, q2),
            torch.where(valid1, q1, torch.where(valid2, q2, torch.full_like(q1, 0.5))),
        ).clamp(1e-6, 1.0).float()

        fluid_boundary_mask[d] = boundary
        q_field[d] = torch.where(boundary, q_val, q_field[d])

    return fluid_boundary_mask, q_field


__all__ = ["compute_q_ellipsoid"]
