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

import torch

from .d2q9 import OPPOSITE


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


__all__ = ["bouzidi_bounce_back"]
