"""Common Bouzidi–Firdaouss–Lallemand interpolated bounce-back module.

Provides a unified ``bouzidi_bounce_back_3d`` interface that works for both
**D3Q19** and **D3Q27** lattices by selecting the appropriate OPPOSITE
direction mapping.  The BFL interpolation formula (linear for *q* < 0.5,
quadratic for *q* ≥ 0.5) is lattice-agnostic; only the direction-index
mapping differs.

This module is a **post-streaming** boundary update that is collision-agnostic:
it can be freely combined with any collision operator (BGK, MRT, CG, …) or
turbulence model.  It handles curved surface boundaries (sphere, ellipsoid,
arbitrary STL) via the fractional-distance field *q*.

Reference
---------
Bouzidi, M., Firdaouss, M., & Lallemand, P. (2001).
"Momentum transfer of a Boltzmann-lattice fluid with boundaries."
*Physics of Fluids*, 13(11), 3452–3459.

Hot-path invariants
-------------------
* No GPU→CPU syncs (``.item()``, ``float(tensor)``) inside the BC path.
* The OPPOSITE array is looked up once at module load time.
"""
from __future__ import annotations

import math

import torch

from .d3q19 import OPPOSITE as _OPP19
from .d3q19 import C as _C19
from .d3q27 import OPPOSITE as _OPP27
from .d3q27 import C as _C27

__all__ = [
    "bouzidi_bounce_back_3d_common",
    "compute_q_sphere_27",
]

# Pre-computed OPPOSITE arrays as plain Python lists (no .item() in hot path)
_OPP19_LIST: list[int] = [int(x) for x in _OPP19.tolist()]
_OPP27_LIST: list[int] = [int(x) for x in _OPP27.tolist()]


# --------------------------------------------------------------------------- #
# Unified Bouzidi bounce-back for 3-D (D3Q19 / D3Q27)
# --------------------------------------------------------------------------- #

def bouzidi_bounce_back_3d_common(
    f: torch.Tensor,
    f_prev: torch.Tensor,
    fluid_nodes: torch.Tensor,
    q: torch.Tensor,
    direction: int,
    *,
    lattice: str = "D3Q19",
) -> torch.Tensor:
    """Apply the BFL interpolated bounce-back for one direction in 3-D.

    Works for both D3Q19 and D3Q27 by selecting the appropriate OPPOSITE
    direction mapping.  The interpolation formula is identical:

    - If *q* < 0.5: linear interpolation
      ``f_bc = 2*q*f_opp + (1 - 2*q)*f_prev[direction]``
    - If *q* ≥ 0.5: quadratic interpolation
      ``f_bc = f_opp / (2*q) + (2*q - 1) / (2*q) * f_prev[opp]``

    Args:
        f: Post-stream distribution tensor, shape ``(Q, nz, ny, nx)``
           where *Q* is 19 (D3Q19) or 27 (D3Q27).
        f_prev: Distribution tensor *before* the most recent stream step,
            same shape as *f*.
        fluid_nodes: Boolean mask of shape ``(nz, ny, nx)`` marking the
            fluid nodes adjacent to the solid boundary.
        q: Fractional distance tensor of shape ``(nz, ny, nx)`` with values
            in ``[0, 1]``.
        direction: Lattice direction index for which to apply the BC.
        lattice: ``"D3Q19"`` or ``"D3Q27"`` (case-insensitive).

    Returns:
        Updated distribution tensor with the interpolated populations set.

    Raises:
        ValueError: If the lattice is unsupported.
    """
    lattice_u = lattice.upper()
    if lattice_u == "D3Q19":
        opp_list = _OPP19_LIST
    elif lattice_u == "D3Q27":
        opp_list = _OPP27_LIST
    else:
        raise ValueError(
            f"lattice must be 'D3Q19' or 'D3Q27', got {lattice!r}"
        )

    opp = opp_list[direction]
    f_out = f.clone()

    q_cell = q[fluid_nodes]
    mask_lin = q_cell < 0.5
    mask_quad = ~mask_lin

    f_opp = f[opp][fluid_nodes]
    fp_opp = f_prev[opp][fluid_nodes]
    fp_d = f_prev[direction][fluid_nodes]

    # Linear interpolation (q < 0.5)
    f_bc_lin = 2.0 * q_cell * f_opp + (1.0 - 2.0 * q_cell) * fp_d

    # Quadratic interpolation (q >= 0.5)
    safe_q = torch.where(mask_quad, q_cell, torch.ones_like(q_cell))
    f_bc_quad = f_opp / (2.0 * safe_q) + (2.0 * safe_q - 1.0) / (2.0 * safe_q) * fp_opp

    f_bc = torch.where(mask_lin, f_bc_lin, f_bc_quad)

    target = f_out[direction].clone()
    target[fluid_nodes] = f_bc
    f_out[direction] = target

    return f_out


# --------------------------------------------------------------------------- #
# D3Q27 sphere q-field computation
# --------------------------------------------------------------------------- #

def compute_q_sphere_27(
    nx: int,
    ny: int,
    nz: int,
    cx: float,
    cy: float,
    cz: float,
    radius: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the BFL fractional-distance field *q* for a sphere (D3Q27).

    For every **fluid** node that is a direct D3Q27 lattice neighbour of the
    spherical boundary, this function analytically computes the fractional
    distance *q ∈ (0, 1]* from that fluid node to the point where the lattice
    link crosses the sphere surface, via ray-sphere intersection.

    Args:
        nx: Grid width (x).
        ny: Grid height (y).
        nz: Grid depth (z).
        cx: x-coordinate of the sphere centre.
        cy: y-coordinate of the sphere centre.
        cz: z-coordinate of the sphere centre.
        radius: Sphere radius in lattice units.
        device: Target PyTorch device.

    Returns:
        Tuple ``(fluid_boundary_mask, q_field)`` where
        - ``fluid_boundary_mask`` is a bool tensor of shape ``(27, nz, ny, nx)``.
        - ``q_field`` is a float32 tensor of shape ``(27, nz, ny, nx)``;
          non-boundary entries are 0.5.
    """
    c = _C27.to(device)  # (27, 3)

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float64),
        torch.arange(ny, device=device, dtype=torch.float64),
        torch.arange(nx, device=device, dtype=torch.float64),
        indexing="ij",
    )

    fluid_boundary_mask = torch.zeros((27, nz, ny, nx), dtype=torch.bool, device=device)
    q_field = torch.full((27, nz, ny, nx), 0.5, dtype=torch.float32, device=device)

    for d in range(27):
        dcx = float(c[d, 0].item())
        dcy = float(c[d, 1].item())
        dcz = float(c[d, 2].item())
        if dcx == 0.0 and dcy == 0.0 and dcz == 0.0:
            continue  # rest direction

        dist_nb = (xx + dcx - cx) ** 2 + (yy + dcy - cy) ** 2 + (zz + dcz - cz) ** 2
        nb_is_solid = dist_nb <= radius ** 2

        dist_self = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
        self_is_fluid = dist_self > radius ** 2

        boundary = self_is_fluid & nb_is_solid

        if not boundary.any():
            continue

        dx = xx - cx
        dy = yy - cy
        dz = zz - cz
        a_coef = dcx ** 2 + dcy ** 2 + dcz ** 2
        b_coef = 2.0 * (dcx * dx + dcy * dy + dcz * dz)
        c_coef = dx ** 2 + dy ** 2 + dz ** 2 - radius ** 2

        discriminant = b_coef ** 2 - 4.0 * a_coef * c_coef
        safe_disc = torch.where(
            boundary & (discriminant >= 0.0),
            discriminant,
            torch.zeros_like(discriminant),
        )
        sqrt_disc = torch.sqrt(safe_disc)

        t1 = (-b_coef - sqrt_disc) / (2.0 * a_coef)
        t2 = (-b_coef + sqrt_disc) / (2.0 * a_coef)

        link_len = math.sqrt(a_coef)
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
