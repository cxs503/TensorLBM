"""BFL spherical force method for curved boundaries in D3Q27 LBM.

Implements the Bouzidi-Firdaouss-Lallemand (BFL) interpolated bounce-back
boundary condition for curved surfaces (cylinder, sphere) with momentum
exchange force computation.

The BFL method places the wall at the exact geometric position rather than
at the half-way point between fluid and solid cells, achieving second-order
accuracy for curved boundaries.

BFL formulas (Bouzidi et al., 2001):
  For each boundary link with wall at distance ratio q ∈ (0,1):
    q < 0.5:  f_opp_new = 2q·f_i(x+c) + (1-2q)·f_i(x+2c)
    q ≥ 0.5:  f_opp_new = f_i(x+c)/(2q) + (2q-1)/(2q)·f_opp(x)

Momentum exchange with BFL:
  F = Σ (f_i_pre + f_opp_post) · c_i / q

Reference:
    Bouzidi, M., Firdaouss, M., & Lallemand, P. (2001).
    "Momentum transfer of a Boltzmann-lattice fluid with boundaries."
    Physics of Fluids, 13(11), 3452-3459.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from ..collision.d3q27 import C_D3Q27, OPPOSITE_D3Q27, W_D3Q27


@dataclass(frozen=True)
class SphericalForceResult:
    """Result of BFL spherical force computation.

    Attributes:
        force: (3,) force vector on the body (lattice units)
        drag: drag force (x-component)
        lift: lift force (y-component)
        side: side force (z-component)
        n_active_links: number of active boundary links
        q_min: minimum q-value
        q_max: maximum q-value
        q_mean: mean q-value
    """
    force: torch.Tensor
    drag: float
    lift: float
    side: float
    n_active_links: int
    q_min: float
    q_max: float
    q_mean: float


def compute_q_sphere(
    nx: int,
    ny: int,
    nz: int,
    cx: float,
    cy: float,
    cz: float,
    radius: float,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute BFL q-values for a sphere boundary.

    For each near-wall fluid cell, for each direction pointing toward solid,
    compute q = (distance from cell centre to wall) / |c_i|.

    Args:
        nx, ny, nz: Grid dimensions.
        cx, cy, cz: Sphere centre in grid coordinates.
        radius: Sphere radius in grid units.
        device: Torch device.
        dtype: Computation dtype (float64 recommended for accuracy).

    Returns:
        fluid_boundary_mask: (27, nz, ny, nx) bool
        q_field: (27, nz, ny, nx) float, per-direction fractional distance
    """
    c = C_D3Q27.to(device).to(dtype)  # (27, 3)

    # Create coordinate grids
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=dtype),
        torch.arange(ny, device=device, dtype=dtype),
        torch.arange(nx, device=device, dtype=dtype),
        indexing="ij",
    )

    fluid_boundary_mask = torch.zeros((27, nz, ny, nx), dtype=torch.bool, device=device)
    q_field = torch.full((27, nz, ny, nx), 0.5, dtype=torch.float32, device=device)

    R2 = radius ** 2

    for d in range(27):
        if d == 0:
            continue  # rest particle

        dx = float(c[d, 0].item())
        dy = float(c[d, 1].item())
        dz = float(c[d, 2].item())

        # Neighbour in direction d
        xx_nb = xx + dx
        yy_nb = yy + dy
        zz_nb = zz + dz

        # Distance squared from sphere centre
        dist_self = (xx - cx)**2 + (yy - cy)**2 + (zz - cz)**2
        dist_nb = (xx_nb - cx)**2 + (yy_nb - cy)**2 + (zz_nb - cz)**2

        # Current node is fluid, neighbour is solid
        self_is_fluid = dist_self > R2
        nb_is_solid = dist_nb <= R2
        boundary = self_is_fluid & nb_is_solid

        if not boundary.any():
            continue

        # Solve quadratic: |x + t*c - centre|^2 = R^2
        # (x-cx + t*dx)^2 + (y-cy + t*dy)^2 + (z-cz + t*dz)^2 = R^2
        ddx = xx - cx
        ddy = yy - cy
        ddz = zz - cz

        a_coef = dx**2 + dy**2 + dz**2  # |c|^2
        b_coef = 2.0 * (dx * ddx + dy * ddy + dz * ddz)
        c_coef = ddx**2 + ddy**2 + ddz**2 - R2

        discriminant = b_coef**2 - 4.0 * a_coef * c_coef
        safe_disc = torch.where(
            boundary & (discriminant >= 0.0),
            discriminant,
            torch.zeros_like(discriminant),
        )
        sqrt_disc = torch.sqrt(safe_disc)

        t1 = (-b_coef - sqrt_disc) / (2.0 * a_coef)
        t2 = (-b_coef + sqrt_disc) / (2.0 * a_coef)

        # q = t (fractional distance: 0=fluid cell, 1=solid cell)
        valid1 = (t1 > 1e-10) & (t1 <= 1.0 + 1e-10)
        valid2 = (t2 > 1e-10) & (t2 <= 1.0 + 1e-10)

        q_val = torch.where(
            valid1 & valid2,
            torch.min(t1, t2),
            torch.where(valid1, t1, torch.where(valid2, t2, torch.full_like(t1, 0.5))),
        ).clamp(1e-6, 1.0).float()

        fluid_boundary_mask[d] = boundary
        q_field[d] = torch.where(boundary, q_val, q_field[d])

    return fluid_boundary_mask, q_field


def compute_q_cylinder(
    nx: int,
    ny: int,
    nz: int,
    cx: float,
    cy: float,
    radius: float,
    device: torch.device,
    axis: str = "z",
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute BFL q-values for a 2D extruded cylinder.

    Args:
        nx, ny, nz: Grid dimensions.
        cx, cy: Cylinder centre in the cross-section plane.
        radius: Cylinder radius.
        device: Torch device.
        axis: Extrusion axis ('x', 'y', or 'z').
        dtype: Computation dtype.

    Returns:
        fluid_boundary_mask: (27, nz, ny, nx) bool
        q_field: (27, nz, ny, nx) float
    """
    c = C_D3Q27.to(device).to(dtype)

    if axis == "z":
        coord2, coord1 = torch.meshgrid(
            torch.arange(ny, device=device, dtype=dtype),
            torch.arange(nx, device=device, dtype=dtype),
            indexing="ij",
        )
        c1, c2 = cx, cy
        vi1, vi2, vai = 0, 1, 2
        bdim = 0
    elif axis == "y":
        coord2, coord1 = torch.meshgrid(
            torch.arange(nz, device=device, dtype=dtype),
            torch.arange(nx, device=device, dtype=dtype),
            indexing="ij",
        )
        c1, c2 = cx, nz / 2.0
        vi1, vi2, vai = 0, 2, 1
        bdim = 1
    elif axis == "x":
        coord2, coord1 = torch.meshgrid(
            torch.arange(nz, device=device, dtype=dtype),
            torch.arange(ny, device=device, dtype=dtype),
            indexing="ij",
        )
        c1, c2 = cy, nz / 2.0
        vi1, vi2, vai = 1, 2, 0
        bdim = 2
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")

    fluid_boundary_mask = torch.zeros((27, nz, ny, nx), dtype=torch.bool, device=device)
    q_field = torch.full((27, nz, ny, nx), 0.5, dtype=torch.float32, device=device)

    R2 = radius ** 2

    for d in range(27):
        if d == 0:
            continue

        dv1 = float(c[d, vi1].item())
        dv2 = float(c[d, vi2].item())
        dva = float(c[d, vai].item())

        # Pure axis direction: no intersection with extruded cylinder
        if dv1 == 0.0 and dv2 == 0.0:
            continue

        dist_self = (coord1 - c1)**2 + (coord2 - c2)**2
        dist_nb = (coord1 + dv1 - c1)**2 + (coord2 + dv2 - c2)**2

        self_is_fluid = dist_self > R2
        nb_is_solid = dist_nb <= R2
        boundary = self_is_fluid & nb_is_solid

        if not boundary.any():
            continue

        d1 = coord1 - c1
        d2 = coord2 - c2
        a_coef = dv1**2 + dv2**2
        if a_coef < 1e-10:
            continue
        b_coef = 2.0 * (dv1 * d1 + dv2 * d2)
        c_coef = d1**2 + d2**2 - R2

        discriminant = b_coef**2 - 4.0 * a_coef * c_coef
        safe_disc = torch.where(
            boundary & (discriminant >= 0.0),
            discriminant,
            torch.zeros_like(discriminant),
        )
        sqrt_disc = torch.sqrt(safe_disc)

        t1 = (-b_coef - sqrt_disc) / (2.0 * a_coef)
        t2 = (-b_coef + sqrt_disc) / (2.0 * a_coef)

        valid1 = (t1 > 1e-10) & (t1 <= 1.0 + 1e-10)
        valid2 = (t2 > 1e-10) & (t2 <= 1.0 + 1e-10)

        q_val = torch.where(
            valid1 & valid2,
            torch.min(t1, t2),
            torch.where(valid1, t1, torch.where(valid2, t2, torch.full_like(t1, 0.5))),
        ).clamp(1e-6, 1.0).float()

        boundary_3d = boundary.unsqueeze(bdim).expand(nz, ny, nx)
        q_val_3d = q_val.unsqueeze(bdim).expand(nz, ny, nx)

        fluid_boundary_mask[d] = boundary_3d
        q_field[d] = torch.where(boundary_3d, q_val_3d, q_field[d])

    return fluid_boundary_mask, q_field


def bfl_bounce_back(
    f: torch.Tensor,
    f_prev: torch.Tensor,
    fluid_boundary_mask: torch.Tensor,
    q_field: torch.Tensor,
    wall_velocity: Optional[tuple[float, float, float]] = None,
    rho_wall: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply BFL interpolated bounce-back for D3Q27.

    Args:
        f: Post-stream distribution (27, nz, ny, nx).
        f_prev: Pre-stream (post-collision) distribution (27, nz, ny, nx).
        fluid_boundary_mask: (27, nz, ny, nx) bool.
        q_field: (27, nz, ny, nx) float.
        wall_velocity: Optional (ux, uy, uz) for moving wall.
        rho_wall: Optional wall density for moving wall correction.

    Returns:
        Updated distribution tensor.
    """
    device = f.device
    dtype = f.dtype
    opp = OPPOSITE_D3Q27.to(device)

    # Promote q to population dtype for accuracy
    q = q_field.to(dtype=dtype)

    # Gather populations
    f_prev_opp = f_prev[opp]  # f_prev[opp[d]]

    # For q < 0.5: need f_prev[d] at x+c_d (upstream)
    c = C_D3Q27.to(device)
    f_prev_upstream = torch.stack([
        torch.roll(
            f_prev[d],
            shifts=(int(c[d, 2].item()), int(c[d, 1].item()), int(c[d, 0].item())),
            dims=(0, 1, 2),
        )
        for d in range(27)
    ])

    mask = fluid_boundary_mask
    mask_lin = (q < 0.5) & mask
    mask_quad = (~mask_lin) & mask

    # Linear regime: f_bc = 2q·f_prev[d](x) + (1-2q)·f_prev[d](x-c_d)
    f_bc_lin = 2.0 * q * f_prev + (1.0 - 2.0 * q) * f_prev_upstream

    # Quadratic regime: f_bc = f_prev[d]/(2q) + (2q-1)/(2q)·f_prev[opp]
    safe_q = torch.where(mask_quad, q, torch.ones_like(q))
    inv_2q = 1.0 / (2.0 * safe_q)
    f_bc_quad = f_prev * inv_2q + (2.0 * safe_q - 1.0) * inv_2q * f_prev_opp

    f_bc = torch.where(mask_lin, f_bc_lin, f_bc_quad)

    # Moving wall correction
    if wall_velocity is not None and rho_wall is not None:
        w = W_D3Q27.to(device=device, dtype=dtype)
        cs2 = 1.0 / 3.0
        c_float = c.to(dtype=dtype)
        c_dot_u = (c_float[:, 0] * wall_velocity[0]
                   + c_float[:, 1] * wall_velocity[1]
                   + c_float[:, 2] * wall_velocity[2])
        correction = 2.0 * rho_wall * w * c_dot_u / cs2  # (27,)
        correction_opp = correction[opp]  # opp-indexed
        nz, ny, nx = f.shape[1:]
        corr_3d = correction_opp.view(27, 1, 1, 1).expand(27, nz, ny, nx)
        f_bc = f_bc + corr_3d * mask.float()

    # Scatter: f_out[e] = f_bc[opp[e]] where mask[opp[e]] is True
    mask_for_e = mask[opp]
    f_bc_for_e = f_bc[opp]

    return torch.where(mask_for_e, f_bc_for_e, f)


def compute_bfl_force(
    f_pre: torch.Tensor,
    f_post: torch.Tensor,
    fluid_boundary_mask: torch.Tensor,
    q_field: torch.Tensor,
) -> SphericalForceResult:
    """Compute momentum exchange force with BFL q-correction.

    Force from each boundary link:
      F = (f_i_pre + f_opp_post) * c_i / q

    Args:
        f_pre: Pre-bounce-back distribution (27, nz, ny, nx).
        f_post: Post-bounce-back distribution (27, nz, ny, nx).
        fluid_boundary_mask: (27, nz, ny, nx) bool.
        q_field: (27, nz, ny, nx) float.

    Returns:
        SphericalForceResult with force components and statistics.
    """
    device = f_pre.device
    c = C_D3Q27.to(device).float()
    opp = OPPOSITE_D3Q27.to(device)

    force = torch.zeros(3, device=device, dtype=torch.float64)
    n_links = 0
    q_vals = []

    for d in range(1, 27):
        opp_d = int(opp[d].item())
        q_d = q_field[d]
        has_link = fluid_boundary_mask[d]

        if not has_link.any():
            continue

        q_clamped = q_d.clamp(min=0.01)
        q_mask = has_link.float()

        # Force = (f_pre[d] + f_post[opp_d]) * c_d / q
        fx = ((f_pre[d] + f_post[opp_d]) * float(c[d, 0].item()) / q_clamped * q_mask).sum()
        fy = ((f_pre[d] + f_post[opp_d]) * float(c[d, 1].item()) / q_clamped * q_mask).sum()
        fz = ((f_pre[d] + f_post[opp_d]) * float(c[d, 2].item()) / q_clamped * q_mask).sum()

        force[0] += fx
        force[1] += fy
        force[2] += fz
        n_links += int(has_link.sum().item())

        q_vals.append(q_d[has_link])

    if q_vals:
        all_q = torch.cat(q_vals)
        q_min = float(all_q.min().item())
        q_max = float(all_q.max().item())
        q_mean = float(all_q.mean().item())
    else:
        q_min = 0.0
        q_max = 0.0
        q_mean = 0.0

    return SphericalForceResult(
        force=force,
        drag=float(force[0].item()),
        lift=float(force[1].item()),
        side=float(force[2].item()),
        n_active_links=n_links,
        q_min=q_min,
        q_max=q_max,
        q_mean=q_mean,
    )


def bfl_step(
    f: torch.Tensor,
    f_prev: torch.Tensor,
    fluid_boundary_mask: torch.Tensor,
    q_field: torch.Tensor,
    wall_velocity: Optional[tuple[float, float, float]] = None,
    rho_wall: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, SphericalForceResult]:
    """Combined BFL bounce-back and force computation.

    Args:
        f: Post-stream distribution (27, nz, ny, nx).
        f_prev: Pre-stream distribution (27, nz, ny, nx).
        fluid_boundary_mask: (27, nz, ny, nx) bool.
        q_field: (27, nz, ny, nx) float.
        wall_velocity: Optional moving wall velocity.
        rho_wall: Optional wall density.

    Returns:
        (updated_f, force_result)
    """
    f_new = bfl_bounce_back(f, f_prev, fluid_boundary_mask, q_field, wall_velocity, rho_wall)
    force_result = compute_bfl_force(f, f_new, fluid_boundary_mask, q_field)
    return f_new, force_result
