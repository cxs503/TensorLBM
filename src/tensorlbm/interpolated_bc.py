"""Bouzidi–Firdaouss–Lallemand (2001) interpolated bounce-back boundary condition.

The standard (halfway) bounce-back treats the solid wall as lying exactly
halfway between the last fluid node and the first solid node, giving
first-order spatial accuracy. The BFL interpolated bounce-back determines
the fractional distance *q* from the fluid node to the actual curved wall
surface and uses linear (q < 0.5) or quadratic (q ≥ 0.5) interpolation,
raising the accuracy to second order.

Reference
---------
Bouzidi, M., Firdaouss, M., & Lallemand, P. (2001).
"Momentum transfer of a Boltzmann-lattice fluid with boundaries."
Physics of Fluids, 13(11), 3452–3459.
"""
from __future__ import annotations

import math

import torch

from .d2q9 import OPPOSITE, C


def bouzidi_bounce_back(
    f: torch.Tensor,
    f_prev: torch.Tensor,
    fluid_nodes: torch.Tensor,
    q: torch.Tensor,
    direction: int,
) -> torch.Tensor:
    """Apply the BFL interpolated bounce-back for one direction in 2-D.

    For each fluid node marked in *fluid_nodes* the incoming population in
    direction *direction* is reconstructed by interpolation:

    - If *q* < 0.5: linear interpolation uses the post-stream population at
      the fluid node and its upstream neighbour (``f_prev`` from the previous
      step).
    - If *q* ≥ 0.5: quadratic interpolation uses the fluid node population
      and the opposite-direction population from the previous step.

    Args:
        f: Post-stream distribution tensor, shape ``(9, ny, nx)``.
        f_prev: Distribution tensor *before* the most recent stream step,
            shape ``(9, ny, nx)``.
        fluid_nodes: Boolean mask of shape ``(ny, nx)`` marking the fluid
            nodes adjacent to the solid boundary.
        q: Fractional distance tensor of shape ``(ny, nx)`` with values in
            ``[0, 1]``. ``q = 0.5`` reproduces standard halfway bounce-back.
        direction: Lattice direction index (0–8) for which to apply the BC.
            The solid surface is reached by travelling in this direction from
            the fluid node.

    Returns:
        Updated distribution tensor with the interpolated populations set.
    """
    opp = int(OPPOSITE[direction].item())
    f_out = f.clone()

    q_cell = q[fluid_nodes]
    mask_lin = q_cell < 0.5
    mask_quad = ~mask_lin

    f[direction][fluid_nodes]
    f_opp = f[opp][fluid_nodes]
    fp_opp = f_prev[opp][fluid_nodes]

    fp_d = f_prev[direction][fluid_nodes]
    f_bc_lin = 2.0 * q_cell * f_opp + (1.0 - 2.0 * q_cell) * fp_d

    safe_q = torch.where(mask_quad, q_cell, torch.ones_like(q_cell))
    f_bc_quad = f_opp / (2.0 * safe_q) + (2.0 * safe_q - 1.0) / (2.0 * safe_q) * fp_opp

    f_bc = torch.where(mask_lin, f_bc_lin, f_bc_quad)

    target = f_out[direction].clone()
    target[fluid_nodes] = f_bc
    f_out[direction] = target

    return f_out


def compute_q_circle(
    nx: int,
    ny: int,
    cx: float,
    cy: float,
    radius: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the BFL fractional-distance field *q* for a circular obstacle.

    For every **fluid** node that is a direct lattice neighbour of the circular
    boundary, this function analytically computes the fractional distance
    *q ∈ (0, 1]* from that fluid node to the point where the lattice link
    crosses the circle surface.

    The intersection of the ray from fluid node **x** in direction **c** with
    the circle of centre (cx, cy) and radius *r* is found by solving the
    quadratic

    .. math::

        |\\mathbf{x} + t \\mathbf{c} - \\mathbf{x}_{centre}|^2 = r^2

    and taking the smallest positive root t*.  The fractional distance is
    ``q = t* / |c|`` (lattice links have unit length, so *q = t** for the
    face-centred and diagonal directions).  Nodes for which no intersection
    exists (pure fluid or pure solid) get ``q = 0.5`` (standard halfway BC).

    Args:
        nx: Grid width.
        ny: Grid height.
        cx: x-coordinate of the circle centre.
        cy: y-coordinate of the circle centre.
        radius: Circle radius in lattice units.
        device: Target PyTorch device.

    Returns:
        Tuple ``(fluid_boundary_mask, q_field)`` where

        - ``fluid_boundary_mask`` is a bool tensor of shape ``(9, ny, nx)``
          — ``True`` at ``[d, j, i]`` when fluid node ``(i, j)`` has the
          circle boundary along direction ``d``.
        - ``q_field`` is a float tensor of shape ``(9, ny, nx)`` with the
          fractional distance for each (direction, fluid node) pair;
          non-boundary entries are 0.5.
    """
    c = C.to(device)  # (9, 2)

    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float64),
        torch.arange(nx, device=device, dtype=torch.float64),
        indexing="ij",
    )  # both (ny, nx)

    fluid_boundary_mask = torch.zeros((9, ny, nx), dtype=torch.bool, device=device)
    q_field = torch.full((9, ny, nx), 0.5, dtype=torch.float32, device=device)

    for d in range(9):
        dcx = float(c[d, 0].item())
        dcy = float(c[d, 1].item())
        if dcx == 0.0 and dcy == 0.0:
            continue  # rest direction – no intersection

        # Neighbour in direction d
        # Is the neighbour a solid cell?
        dist_nb = (xx + dcx - cx) ** 2 + (yy + dcy - cy) ** 2
        nb_is_solid = dist_nb <= radius ** 2

        # Is the current node a fluid cell?
        dist_self = (xx - cx) ** 2 + (yy - cy) ** 2
        self_is_fluid = dist_self > radius ** 2

        boundary = self_is_fluid & nb_is_solid  # (ny, nx)

        if not boundary.any():
            continue

        # Solve quadratic: |x + t*c - centre|^2 = r^2
        # Let d_vec = x - centre, then:
        #   (t*c + d_vec)^2 = r^2
        #   |c|^2 t^2 + 2(c . d_vec) t + (|d_vec|^2 - r^2) = 0
        dx = xx - cx
        dy = yy - cy
        a_coef = dcx ** 2 + dcy ** 2  # |c|^2 (1.0 or 2.0)
        b_coef = 2.0 * (dcx * dx + dcy * dy)
        c_coef = dx ** 2 + dy ** 2 - radius ** 2

        discriminant = b_coef ** 2 - 4.0 * a_coef * c_coef
        # Only evaluate where boundary is True and discriminant >= 0
        safe_disc = torch.where(
            boundary & (discriminant >= 0.0),
            discriminant,
            torch.zeros_like(discriminant),
        )
        sqrt_disc = torch.sqrt(safe_disc)

        t1 = (-b_coef - sqrt_disc) / (2.0 * a_coef)
        t2 = (-b_coef + sqrt_disc) / (2.0 * a_coef)

        # Take smallest positive root; q = t / sqrt(a) to normalise to link length
        link_len = math.sqrt(a_coef)
        q1 = t1 / link_len
        q2 = t2 / link_len

        # Choose smallest q in (0, 1]
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


__all__ = ["bouzidi_bounce_back", "compute_q_circle"]
