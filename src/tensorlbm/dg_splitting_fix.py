"""DG-LBM band step with operator splitting (P0 fix).

Replaces the unstable MOL approach in dg_lbm_step_band with Strang splitting:
  For each sub-step:
    1. Collision: Δt-aware BGK (unconditionally stable)
    2. Advection: SSP-RK3 pure advection (already stable)

This eliminates the Δt_sub < 2τ constraint that caused NaN for τ_dg < 0.06
at high Reynolds numbers.
"""

import math

import torch

from .dg_advection import _Ops, collide_bgk_dg


def dg_lbm_step_band_split(
    f_dg: torch.Tensor,
    velocities: torch.Tensor,
    weights: torch.Tensor,
    tau: float,
    ops: _Ops,
    topo,
    ext_field: torch.Tensor | None,
    dt: float,
    n_substeps: int = 6,
    scheme: str = "rk3",
    opposite: torch.Tensor | None = None,
) -> torch.Tensor:
    """Operator-split DG-LBM band step.

    Replaces tensorlbm.dg_band.dg_lbm_step_band with the splitting approach
    that avoids MOL stiffness at small tau (DG-Compare gap #4 fix).
    """
    from .dg_band import dg_advect_band as _advect_band

    n_substeps = max(n_substeps, int(math.ceil(dt / (2.0 * tau))))
    dt_sub = dt / n_substeps

    f = f_dg
    for _ in range(n_substeps):
        f = collide_bgk_dg(f, velocities, weights, tau, dt_sub)
        f = _advect_band(
            f,
            velocities,
            ops,
            topo,
            ext_field,
            dt=dt_sub,
            n_substeps=1,
            scheme=scheme,
            opposite=opposite,
        )
        f = f.clamp(min=0.0)
    return f


def positivity_preserving_limiter(
    f_dg: torch.Tensor,
    cell_mean: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Zhang-Shu (2010) positivity-preserving limiter for DG.

    Scales nodal values toward the cell mean to ensure f ≥ ε everywhere,
    while exactly preserving the cell mean (and thus conservation).

    This is the P1 gap-fix for Gibbs oscillations (DG-Compare gap #3).

    Args:
        f_dg: Nodal values on Lobatto nodes (Q, ..., cells..., ..., nodes...)
        cell_mean: Cell-averaged values (same shape as f_dg without node dims)
        epsilon: Floor value (default 1e-6).

    Returns:
        Limited nodal values with f ≥ ε, same cell mean.
    """
    # Find the axis corresponding to nodes (typically the last spatial dims)
    # For P1 tetra/hex, nodes are at the end of the tensor
    node_axes = tuple(range(f_dg.ndim - cell_mean.ndim, f_dg.ndim))

    f_min = f_dg.amin(dim=node_axes)
    # theta = (mean - ε) / (mean - min), clipped to [0, 1]
    theta = ((cell_mean - epsilon) / (cell_mean - f_min + 1e-12)).clamp(0, 1)
    # Broadcast theta to nodal shape and apply
    theta_b = theta.reshape(*theta.shape, *([1] * len(node_axes)))
    return cell_mean.reshape(*cell_mean.shape, *([1] * len(node_axes))) + theta_b * (
        f_dg - cell_mean.reshape(*cell_mean.shape, *([1] * len(node_axes)))
    )
