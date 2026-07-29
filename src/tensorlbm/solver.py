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

OPPOSITE_2D = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=torch.int64)

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


def collide_rlbm(f: torch.Tensor, tau: float) -> torch.Tensor:
    """Regularized BGK (RLBM) collision step for D2Q9.

    The non-equilibrium part of *f* is projected onto the second-order Hermite
    polynomial subspace before the BGK relaxation. This filters out the
    higher-order ghost (non-hydrodynamic) modes and significantly improves
    stability at low viscosity (τ → 0.5) without altering the recovered
    Navier–Stokes physics. See Latt & Chopard, *Math. Comput. Simul.* (2006).

    Reconstruction:

    .. math::
        \\Pi^{\\mathrm{neq}}_{\\alpha\\beta} =
            \\sum_i c_{i\\alpha} c_{i\\beta}\\,(f_i - f^{\\mathrm{eq}}_i)

    .. math::
        f^{\\mathrm{neq,reg}}_i =
            \\frac{w_i}{2 c_s^4}\\,
            (c_{i\\alpha} c_{i\\beta} - c_s^2 \\delta_{\\alpha\\beta})\\,
            \\Pi^{\\mathrm{neq}}_{\\alpha\\beta}

    Args:
        f:   Distribution tensor of shape ``(9, ny, nx)``.
        tau: Relaxation time (τ > 0.5). Kinematic viscosity ν = (τ − ½)/3.

    Returns:
        Updated distribution tensor of the same shape.
    """
    from .d2q9 import _c_on, _w_on  # noqa: PLC0415

    device = f.device
    c = _c_on(device).to(f.dtype)
    w = _w_on(device).to(f.dtype)

    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)
    fneq = f - feq

    cx = c[:, 0].view(9, 1, 1)
    cy = c[:, 1].view(9, 1, 1)

    # Second-order non-equilibrium moments Π_αβ = Σ_i c_iα c_iβ fneq_i
    pi_xx = (cx * cx * fneq).sum(dim=0)
    pi_yy = (cy * cy * fneq).sum(dim=0)
    pi_xy = (cx * cy * fneq).sum(dim=0)

    # Regularized non-equilibrium part using Hermite projection.
    # H_iαβ = c_iα c_iβ − c_s^2 δ_αβ; c_s^2 = 1/3, 1/(2 c_s^4) = 9/2
    cs2 = 1.0 / 3.0
    h_xx = cx * cx - cs2
    h_yy = cy * cy - cs2
    h_xy = cx * cy  # symmetric, contributes twice via αβ + βα
    w_view = w.view(9, 1, 1)
    fneq_reg = (9.0 / 2.0) * w_view * (h_xx * pi_xx + h_yy * pi_yy + 2.0 * h_xy * pi_xy)

    return feq + (1.0 - 1.0 / tau) * fneq_reg


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


def collide_trt(
    f: torch.Tensor,
    tau_plus: float,
    lambda_trt: float = 3.0 / 16.0,
) -> torch.Tensor:
    """Two-relaxation-time (TRT) collision step for D2Q9.

    The TRT model uses two independent relaxation rates:

    - *τ₊* (``tau_plus``) controls the symmetric part of the distribution
      and sets the kinematic viscosity: ν = (τ₊ − ½) / 3.
    - *τ₋* (anti-symmetric) is derived from the "magic" parameter Λ:
      τ₋ = ½ + Λ / (τ₊ − ½).  The magic number Λ = 3/16 eliminates wall
      placement errors in Poiseuille flow (Ginzburg 2008).

    Compared to BGK, TRT significantly improves accuracy for porous-media and
    wall-bounded flows at low viscosity by independently damping the
    anti-symmetric non-equilibrium moments.

    Reference
    ---------
    Ginzburg, I. (2008). Two-relaxation-time lattice Boltzmann scheme:
    About parametrization, velocity, pressure and mixed boundary conditions.
    *Commun. Comput. Phys.* 3(2), 427–478.

    Args:
        f:           Distribution tensor of shape ``(9, ny, nx)``.
        tau_plus:    Symmetric relaxation time (τ₊ > 0.5).
        lambda_trt:  Magic parameter Λ (default 3/16 eliminates Poiseuille
                     wall error).

    Returns:
        Updated distribution tensor of the same shape.
    """
    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)

    tau_minus = 0.5 + lambda_trt / (tau_plus - 0.5)

    opp = OPPOSITE_2D.to(f.device)
    f_plus = 0.5 * (f + f[opp])
    f_minus = 0.5 * (f - f[opp])
    feq_plus = 0.5 * (feq + feq[opp])
    feq_minus = 0.5 * (feq - feq[opp])

    return f - (f_plus - feq_plus) / tau_plus - (f_minus - feq_minus) / tau_minus


__all__ = [
    "cylinder_mask",
    "make_channel_wall_mask",
    "bounce_back_cells",
    "apply_simple_channel_boundaries",
    "collide_bgk",
    "collide_mrt",
    "collide_rlbm",
    "collide_trt",
    "stream",
    "correct_mass",
]
