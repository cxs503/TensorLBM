#!/usr/bin/env python3
"""Vectorized BFL (Bouzidi-Firdaouss-Lallemand) interpolated bounce-back for D3Q19.

Fully vectorized version of bouzidi_bounce_back_d3q19 that avoids the
per-direction Python loop and boolean-mask advanced indexing.  Uses
torch.where with full-size tensors for SDAA efficiency.

Mathematics (identical to the reference implementation):
  For each direction d with a boundary link (fluid → solid):
    q = q_field[d]  (fractional distance, 0=fluid, 1=solid)
    f_opp   = f[opp_d]       (post-stream, streamed from solid)
    fp_opp  = f_prev[opp_d]  (pre-stream, post-collision)
    fp_d    = f_prev[d]      (pre-stream, post-collision)

    q < 0.5  (linear):    f_bc = 2q·f_opp + (1-2q)·fp_d
    q >= 0.5 (quadratic): f_bc = f_opp/(2q) + (2q-1)/(2q)·fp_opp

  The unknown population f[opp_d] at the fluid boundary cell is set to f_bc.

Vectorization key:
  opp is an involution (opp[opp[d]] = d), so for each output direction e:
    f_out[e] = where(mask[opp[e]], f_bc[opp[e]], f[e])
"""
from __future__ import annotations

import torch

from .d3q19 import OPPOSITE


def bouzidi_bounce_back_d3q19_vec(
    f: torch.Tensor,
    f_prev: torch.Tensor,
    fluid_boundary_mask: torch.Tensor,
    q_field: torch.Tensor,
) -> torch.Tensor:
    """Vectorized BFL interpolated bounce-back for ALL D3Q19 directions.

    Args:
        f: Post-stream distribution (19, nz, ny, nx)
        f_prev: Pre-stream distribution (19, nz, ny, nx)
        fluid_boundary_mask: (19, nz, ny, nx) bool
        q_field: (19, nz, ny, nx) float, per-direction fractional distance

    Returns:
        Updated distribution tensor.
    """
    opp = OPPOSITE.to(f.device)  # (19,) int64

    # Gather per-direction quantities (all full-size, no advanced indexing)
    # f_opp_all[d]  = f[opp[d]]      — post-stream opposite
    # fp_opp_all[d] = f_prev[opp[d]] — pre-stream opposite
    # fp_d_all[d]   = f_prev[d]      — pre-stream same direction
    f_opp_all = f[opp]        # (19, nz, ny, nx)
    fp_opp_all = f_prev[opp]  # (19, nz, ny, nx)
    fp_d_all = f_prev         # (19, nz, ny, nx)

    q = q_field                               # (19, nz, ny, nx)
    mask = fluid_boundary_mask                 # (19, nz, ny, nx)

    mask_lin = (q < 0.5) & mask                # linear regime
    mask_quad = (~mask_lin) & mask             # quadratic regime

    # Linear: f_bc = 2q·f_opp + (1-2q)·fp_d
    f_bc_lin = 2.0 * q * f_opp_all + (1.0 - 2.0 * q) * fp_d_all

    # Quadratic: f_bc = f_opp/(2q) + (2q-1)/(2q)·fp_opp
    # safe_q >= 0.5 everywhere (1.0 for non-quadratic cells)
    safe_q = torch.where(mask_quad, q, torch.ones_like(q))
    inv_2q = 1.0 / (2.0 * safe_q)
    f_bc_quad = f_opp_all * inv_2q + (2.0 * safe_q - 1.0) * inv_2q * fp_opp_all

    f_bc = torch.where(mask_lin, f_bc_lin, f_bc_quad)  # (19, nz, ny, nx)

    # Scatter: for each output direction e, set f_out[e] = f_bc[opp[e]]
    # where mask[opp[e]] is True.
    # Since opp is an involution: d = opp[e], so mask_for_e = mask[opp].
    mask_for_e = mask[opp]     # (19, nz, ny, nx)
    f_bc_for_e = f_bc[opp]     # (19, nz, ny, nx)

    return torch.where(mask_for_e, f_bc_for_e, f)


__all__ = ["bouzidi_bounce_back_d3q19_vec"]
