#!/usr/bin/env python3
"""Vectorized BFL (Bouzidi-Firdaouss-Lallemand) interpolated bounce-back for D3Q19.

Fully vectorized version of bouzidi_bounce_back_d3q19 that avoids the
per-direction Python loop and boolean-mask advanced indexing.  Uses
torch.where with full-size tensors for SDAA efficiency.

Mathematics (identical to the reference implementation):
  For each direction d with a boundary link (fluid → solid):
    q = q_field[d]  (fractional distance, 0=fluid, 1=solid)
    fp_opp  = f_prev[opp_d]  (pre-stream, post-collision)
    fp_d    = f_prev[d]      (pre-stream, post-collision)
    fp_up   = f_prev[d](x-c_d)

    q < 0.5  (linear):    f_bc = 2q·fp_d + (1-2q)·fp_up
    q >= 0.5 (quadratic): f_bc = fp_d/(2q) + (2q-1)/(2q)·fp_opp

  The unknown population f[opp_d] at the fluid boundary cell is set to f_bc.

Vectorization key:
  opp is an involution (opp[opp[d]] = d), so for each output direction e:
    f_out[e] = where(mask[opp[e]], f_bc[opp[e]], f[e])
"""
from __future__ import annotations

import torch

from .bfl_common import bfl_bounce_back_common


def bouzidi_bounce_back_d3q19_vec(
    f: torch.Tensor,
    f_prev: torch.Tensor,
    fluid_boundary_mask: torch.Tensor,
    q_field: torch.Tensor,
    wall_correction: torch.Tensor | None = None,
) -> torch.Tensor:
    """Vectorized BFL interpolated bounce-back for ALL D3Q19 directions.

    Args:
        f: Post-stream distribution (19, nz, ny, nx)
        f_prev: Pre-stream distribution (19, nz, ny, nx)
        fluid_boundary_mask: (19, nz, ny, nx) bool
        q_field: (19, nz, ny, nx) float, per-direction fractional distance
        wall_correction: Optional (19, nz, ny, nx) float, moving-wall
            momentum correction added to f_bc.  For a wall moving with
            velocity **u_w**, the correction for the *unknown* population
            in direction ``opp_d`` is::

                corr[opp_d] = 2·ρ·w[opp_d]·(c[opp_d]·u_w)/cs²

            Since the scatter step sets ``f_out[e] = f_bc[opp[e]]``, the
            correction added to ``f_bc[d]`` is ``corr[opp[d]]``.  Pass a
            pre-computed ``(19, nz, ny, nx)`` tensor that already accounts
            for this opp-indexing (i.e. ``wall_correction[d]`` is the value
            to add to ``f_bc[d]``).

    Returns:
        Updated distribution tensor.
    """
    return bfl_bounce_back_common(
        f,
        f_prev,
        fluid_boundary_mask,
        q_field,
        lattice="D3Q19",
        wall_correction=wall_correction,
    )


__all__ = ["bouzidi_bounce_back_d3q19_vec"]
