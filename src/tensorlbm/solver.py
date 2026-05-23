from __future__ import annotations

import functools
from typing import Any, cast

import torch

from .boundaries import (
    apply_simple_channel_boundaries,
    bounce_back_cells,
    cylinder_mask,
    make_channel_wall_mask,
)
from .d2q9 import C, equilibrium, macroscopic

# Cache for streaming index tensors keyed by (ny, nx, device_type, device_index)
_stream2d_cache: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

_M_D2Q9_DATA = [
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [-4.0, -1.0, -1.0, -1.0, -1.0, 2.0, 2.0, 2.0, 2.0],
    [4.0, -2.0, -2.0, -2.0, -2.0, 1.0, 1.0, 1.0, 1.0],
    [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0],
    [0.0, -2.0, 0.0, 2.0, 0.0, 1.0, -1.0, -1.0, 1.0],
    [0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0],
    [0.0, 0.0, -2.0, 0.0, 2.0, 1.0, 1.0, -1.0, -1.0],
    [0.0, 1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0],
]


def _invert_d2q9() -> list[list[float]]:
    import numpy as np

    matrix = np.array(_M_D2Q9_DATA, dtype=np.float64)
    return cast("list[list[float]]", np.linalg.inv(matrix).tolist())


_M_D2Q9_INV_DATA = _invert_d2q9()


@functools.cache
def _get_d2q9_mrt_matrices(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = torch.tensor(_M_D2Q9_DATA, dtype=torch.float32, device=device)
    matrix_inv = torch.tensor(_M_D2Q9_INV_DATA, dtype=torch.float32, device=device)
    return matrix, matrix_inv


def collide_bgk(f: torch.Tensor, tau: float) -> torch.Tensor:
    """Single-relaxation-time BGK collision step."""
    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)
    return f - (f - feq) / tau


def collide_mrt(
    f: torch.Tensor,
    tau: float,
    s_e: float = 1.64,
    s_eps: float = 1.54,
    s_q: float = 1.7,
) -> torch.Tensor:
    """Multi-relaxation-time (MRT) collision step for D2Q9.

    The physical shear viscosity is controlled by *tau* exactly as in BGK:
    ν = (τ − ½)/3. The extra relaxation rates *s_e*, *s_eps*, *s_q* damp
    the non-hydrodynamic moments and can be tuned independently to improve
    numerical stability at high Reynolds numbers.

    Moment ordering (rows of M):
        0: ρ  (conserved, s=0)
        1: e  (energy,          s=s_e)
        2: ε  (energy-square,   s=s_eps)
        3: jx (conserved, s=0)
        4: qx (heat-flux x,     s=s_q)
        5: jy (conserved, s=0)
        6: qy (heat-flux y,     s=s_q)
        7: pxx (stress,         s=1/tau)
        8: pxy (stress,         s=1/tau)

    Args:
        f: Distribution tensor of shape ``(9, ny, nx)``.
        tau: Relaxation time for shear stress (τ > ½).
        s_e: Relaxation rate for energy moment.
        s_eps: Relaxation rate for energy-square moment.
        s_q: Relaxation rate for heat-flux moments.

    Returns:
        Updated distribution tensor of the same shape.
    """
    device = f.device
    matrix, matrix_inv = _get_d2q9_mrt_matrices(device)

    s_nu = 1.0 / tau
    s_vec = torch.tensor(
        [0.0, s_e, s_eps, 0.0, s_q, 0.0, s_q, s_nu, s_nu],
        dtype=f.dtype,
        device=device,
    )

    ny, nx = f.shape[1], f.shape[2]
    f_flat = f.reshape(9, -1)
    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)
    feq_flat = feq.reshape(9, -1)

    moments = matrix @ f_flat
    moments_eq = matrix @ feq_flat
    moments_star = moments - s_vec.unsqueeze(1) * (moments - moments_eq)
    return (matrix_inv @ moments_star).reshape(9, ny, nx)


def stream(f: torch.Tensor) -> torch.Tensor:
    """Vectorised streaming by gathering from shifted source indices (periodic).

    Replaces the per-direction ``torch.roll`` loop with a single advanced-index
    gather, which is more GPU-friendly. Index tensors are cached per (shape,
    device) to avoid re-allocation on every call.
    """
    ny, nx = f.shape[1], f.shape[2]
    device = f.device
    c = C.to(device)

    cache_key = (ny, nx, device.type, device.index)
    if cache_key not in _stream2d_cache:
        y_src = (torch.arange(ny, device=device).unsqueeze(0) - c[:, 1].unsqueeze(1)) % ny
        x_src = (torch.arange(nx, device=device).unsqueeze(0) - c[:, 0].unsqueeze(1)) % nx
        q_idx = torch.arange(9, device=device).view(9, 1, 1).expand(9, ny, nx)
        y_idx = y_src.unsqueeze(2).expand(9, ny, nx)
        x_idx = x_src.unsqueeze(1).expand(9, ny, nx)
        _stream2d_cache[cache_key] = (q_idx, y_idx, x_idx)

    q_idx, y_idx, x_idx = _stream2d_cache[cache_key]
    return f[q_idx, y_idx, x_idx]


def correct_mass(f: torch.Tensor, target_mass: float) -> torch.Tensor:
    """Redistribute mass uniformly to correct global mass drift.

    Rescales the entire distribution tensor so that the sum of all
    populations equals *target_mass*. This corrects slow mass drift
    accumulated by inexact boundary conditions over many time steps.

    Args:
        f: Distribution tensor of shape ``(9, ny, nx)``.
        target_mass: Desired total mass (sum of all populations).

    Returns:
        Rescaled distribution tensor of the same shape.
    """
    current = f.sum()
    if current.abs() < 1e-30:
        return f
    return f * (target_mass / current)


__all__ = [
    "cylinder_mask",
    "make_channel_wall_mask",
    "bounce_back_cells",
    "apply_simple_channel_boundaries",
    "collide_bgk",
    "collide_mrt",
    "stream",
    "correct_mass",
]
