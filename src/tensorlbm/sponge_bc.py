"""Sponge / absorbing-layer outlet boundary condition for LBM.

Acoustic reflections from non-physical outlet boundaries are a well-known
source of error in LBM simulations.  Commercial solvers (PowerFlow, XFlow)
use sponge-layer (absorbing zone) techniques to damp outgoing waves before
they reach the domain boundary and re-enter the interior.

This module implements two complementary damping strategies:

**Viscous sponge (Colonius & Lele, 2004)**
    The relaxation time τ in the sponge zone is progressively reduced
    (viscosity increased) towards the outlet, causing turbulent fluctuations
    to decay exponentially before reaching the boundary.  The modified
    collision step reads:

    .. math::

        f_i^{*} = f_i - \\frac{1}{\\tau + \\alpha(x)} (f_i - f_i^{eq})

    where the sponge profile is :math:`\\alpha(x) = A \\sigma^n(x)` with
    :math:`\\sigma = (x - x_0) / (x_1 - x_0) \\in [0, 1]`.

**Target-field sponge (Israeli & Orszag, 1981)**
    The distribution is relaxed towards a prescribed "target" field
    (typically the time-averaged mean flow) in the sponge zone:

    .. math::

        f_i^{*} = (1 - \\beta(x)) f_i + \\beta(x) f_i^{\\rm target}

    This method is more aggressive and better at absorbing vortical
    structures but requires a pre-computed target field.

Both 2-D (D2Q9) and 3-D (D3Q19, D3Q27) velocity sets are supported.

References
----------
* Israeli M. & Orszag S.A. (1981) J. Comput. Phys. 41 115.
* Colonius T. & Lele S.K. (2004) Prog. Aerosp. Sci. 40 345.
* Xu H. & Sagaut P. (2013) J. Comput. Phys. 232 435.
"""
from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Sponge profile
# ---------------------------------------------------------------------------

def sponge_profile(
    nx: int,
    x0: int,
    x1: int,
    amplitude: float = 0.5,
    exponent: float = 3.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Compute the 1-D sponge strength profile along x.

    Returns a 1-D tensor of length *nx*.  Inside ``[x0, x1]`` the profile
    rises from 0 to *amplitude* following a polynomial :math:`σ^n`.
    Outside this interval the value is 0.

    Args:
        nx: Number of grid cells in x.
        x0: Start of the sponge zone (x-index, inclusive).
        x1: End of the sponge zone (x-index, inclusive, typically nx-1).
        amplitude: Maximum sponge strength A.
        exponent: Polynomial exponent n (higher → sharper onset).
        device: Torch device.

    Returns:
        Sponge profile tensor of shape ``(nx,)``.
    """
    if device is None:
        device = torch.device("cpu")
    x = torch.arange(nx, device=device, dtype=torch.float32)
    sigma = torch.clamp((x - x0) / max(x1 - x0, 1), 0.0, 1.0)
    return amplitude * sigma**exponent


# ---------------------------------------------------------------------------
# Viscous sponge (2-D / 3-D agnostic)
# ---------------------------------------------------------------------------

def apply_viscous_sponge_2d(
    f: torch.Tensor,
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    tau0: float,
    sponge: torch.Tensor,
) -> torch.Tensor:
    """Apply viscous sponge relaxation to a D2Q9 distribution.

    In the sponge zone the effective relaxation time is reduced to
    ``tau_eff = tau0 / (1 + alpha)``, i.e., more viscous → stronger damping.

    Args:
        f: D2Q9 distribution, shape ``(9, ny, nx)``.
        rho: Density field, shape ``(ny, nx)``.
        ux: x-velocity, shape ``(ny, nx)``.
        uy: y-velocity, shape ``(ny, nx)``.
        tau0: Nominal relaxation time (τ = 0.5 + ν/cs²).
        sponge: 1-D sponge profile, shape ``(nx,)`` – broadcast over ny.

    Returns:
        Updated distribution tensor (same shape as *f*).
    """
    from .d2q9 import equilibrium

    alpha = sponge.view(1, -1)  # (1, nx)
    tau_eff = tau0 / (1.0 + alpha)   # (1, nx), reduced τ in sponge

    f_eq = equilibrium(rho, ux, uy)
    f_neq = f - f_eq
    # Only damp where alpha > 0
    damp = (alpha > 0).float()
    f_out = f - damp * f_neq / tau_eff.unsqueeze(0)
    return f_out


def apply_viscous_sponge_3d(
    f: torch.Tensor,
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    tau0: float,
    sponge: torch.Tensor,
    lattice: str = "D3Q19",
) -> torch.Tensor:
    """Apply viscous sponge relaxation to a 3-D LBM distribution.

    Args:
        f: Distribution tensor.
            - D3Q19: shape ``(19, nz, ny, nx)``
            - D3Q27: shape ``(27, nz, ny, nx)``
        rho: Density, shape ``(nz, ny, nx)``.
        ux, uy, uz: Velocity components, same shape as *rho*.
        tau0: Nominal relaxation time.
        sponge: 1-D sponge profile, shape ``(nx,)``.
        lattice: ``"D3Q19"`` (default) or ``"D3Q27"``.

    Returns:
        Updated distribution tensor.
    """
    if lattice == "D3Q27":
        from .d3q27 import equilibrium27 as equil
    else:
        from .d3q19 import equilibrium3d as equil

    alpha = sponge.view(1, 1, -1)   # (1, 1, nx)
    tau_eff = tau0 / (1.0 + alpha)  # (1, 1, nx)

    f_eq = equil(rho, ux, uy, uz)
    f_neq = f - f_eq
    damp = (alpha > 0).float()
    f_out = f - damp * f_neq / tau_eff.unsqueeze(0)
    return f_out


# ---------------------------------------------------------------------------
# Target-field sponge (2-D / 3-D)
# ---------------------------------------------------------------------------

def apply_target_sponge_2d(
    f: torch.Tensor,
    f_target: torch.Tensor,
    sponge: torch.Tensor,
) -> torch.Tensor:
    """Relax distribution towards a target field in the sponge zone (2-D).

    Args:
        f: Current D2Q9 distribution, shape ``(9, ny, nx)``.
        f_target: Target distribution, same shape.
        sponge: 1-D sponge profile, shape ``(nx,)`` — broadcast over directions and y.

    Returns:
        Updated distribution.
    """
    beta = sponge.view(1, 1, -1).clamp(0.0, 1.0)  # (1, 1, nx)
    return (1.0 - beta) * f + beta * f_target


def apply_target_sponge_3d(
    f: torch.Tensor,
    f_target: torch.Tensor,
    sponge: torch.Tensor,
) -> torch.Tensor:
    """Relax distribution towards a target field in the sponge zone (3-D).

    Args:
        f: Current distribution, shape ``(Q, nz, ny, nx)``.
        f_target: Target distribution, same shape.
        sponge: 1-D sponge profile, shape ``(nx,)`` — broadcast.

    Returns:
        Updated distribution.
    """
    beta = sponge.view(1, 1, 1, -1).clamp(0.0, 1.0)  # (1, 1, 1, nx)
    return (1.0 - beta) * f + beta * f_target


# ---------------------------------------------------------------------------
# Helper: build equilibrium target from mean flow
# ---------------------------------------------------------------------------

def build_mean_equilibrium_2d(
    rho_mean: torch.Tensor,
    ux_mean: torch.Tensor,
    uy_mean: torch.Tensor,
) -> torch.Tensor:
    """Build a D2Q9 equilibrium target from mean velocity/density fields.

    Suitable as the ``f_target`` argument for :func:`apply_target_sponge_2d`.

    Args:
        rho_mean: Mean density, shape ``(ny, nx)``.
        ux_mean: Mean x-velocity, same shape.
        uy_mean: Mean y-velocity, same shape.

    Returns:
        Equilibrium distribution ``f_eq``, shape ``(9, ny, nx)``.
    """
    from .d2q9 import equilibrium
    return equilibrium(rho_mean, ux_mean, uy_mean)


def build_mean_equilibrium_3d(
    rho_mean: torch.Tensor,
    ux_mean: torch.Tensor,
    uy_mean: torch.Tensor,
    uz_mean: torch.Tensor,
    lattice: str = "D3Q19",
) -> torch.Tensor:
    """Build a 3-D equilibrium target from mean velocity/density fields.

    Args:
        rho_mean: Mean density, shape ``(nz, ny, nx)``.
        ux_mean, uy_mean, uz_mean: Mean velocity components, same shape.
        lattice: ``"D3Q19"`` (default) or ``"D3Q27"``.

    Returns:
        Equilibrium distribution.
    """
    if lattice == "D3Q27":
        from .d3q27 import equilibrium27
        return equilibrium27(rho_mean, ux_mean, uy_mean, uz_mean)
    from .d3q19 import equilibrium3d
    return equilibrium3d(rho_mean, ux_mean, uy_mean, uz_mean)


__all__ = [
    "sponge_profile",
    "apply_viscous_sponge_2d",
    "apply_viscous_sponge_3d",
    "apply_target_sponge_2d",
    "apply_target_sponge_3d",
    "build_mean_equilibrium_2d",
    "build_mean_equilibrium_3d",
]
