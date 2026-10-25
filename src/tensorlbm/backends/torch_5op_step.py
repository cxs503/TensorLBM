"""Default 5-op PyTorch step implementation for the SUBOFF runner.

This is the original stream → force → BC → mass correction chain
extracted from :mod:`tensorlbm.suboff_cmk_kbc_runner` and lifted into
the backends step-impl layer.  Works on any PyTorch device (CPU or
CUDA) and is the default ``"torch_5op"`` step implementation.

This module is *behaviour-equivalent* to the original inline chain; it
exists only so that the SUBOFF runner can dispatch the per-step work
through ``tensorlbm.backends.get_step_impl()`` rather than reaching
into the kernel functions directly.
"""

from __future__ import annotations

import torch

from ..boundaries3d import far_field_bc_3d
from ..obstacles import compute_obstacle_forces_3d
from ..solver3d import correct_mass3d, stream3d

__all__ = ["step_torch_5op"]


def step_torch_5op(
    f: torch.Tensor,
    solid: torch.Tensor,
    u_in: float,
    step: int,
    mass_period: int,
    initial_mass: float,
    *,
    do_mass_correction: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One SUBOFF step on PyTorch (CPU or CUDA).

    Implements the canonical 5-op chain that has been in the SUBOFF
    runner since before the Triton integration:

    1.  ``stream3d``  — pull-stream via ``torch.roll``.
    2.  ``compute_obstacle_forces_3d`` — Ladd momentum-exchange.
    3.  ``far_field_bc_3d`` — 6-face Dirichlet BCs.
    4.  ``correct_mass3d`` — rescale to ``initial_mass`` (every
        ``mass_period`` steps).

    Args:
        f: Distribution ``(Q, nz, ny, nx)``, fp32, Q=19.
        solid: Bool/float obstacle mask ``(nz, ny, nx)``.
        u_in: Free-stream x-velocity (lattice units).
        step: 1-indexed step number (drives mass correction cadence).
        mass_period: Apply mass correction every ``mass_period`` steps.
        initial_mass: Target mass at t=0.
        do_mass_correction: If False, skip mass correction entirely.

    Returns:
        ``(f_new, fx, fy, fz)`` — post-step distribution (same shape
        as input) and three scalar force tensors (streamwise, lateral,
        vertical).
    """
    # 1. Stream.
    f = stream3d(f)

    # 2. Force reduction (post-stream, pre-bounce-back phase).
    fx_t, fy_t, fz_t = compute_obstacle_forces_3d(f, solid)

    # 3. 6-face far-field Dirichlet BC.
    f = far_field_bc_3d(f, u_in, obstacle_mask=solid)

    # 4. Mass correction every mass_period steps.
    if do_mass_correction and (step % mass_period == 0):
        f = correct_mass3d(f, initial_mass)

    return f, fx_t, fy_t, fz_t
