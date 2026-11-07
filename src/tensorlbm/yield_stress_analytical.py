"""Analytical solutions for Bingham fluid channel flows.

Exact solutions for validation of the yield-stress LBM kernel:

1. **Bingham Couette flow**: shear-driven flow between parallel plates.
   The velocity profile is linear everywhere (uniform shear rate), so this
   validates the moving-wall boundary condition and the rheology in the
   yielded regime.  No plug region exists in simple Couette.

2. **Bingham Poiseuille flow**: pressure-driven (body-force driven) flow
   between parallel plates.  This has a central plug region where |τ| < τ_y
   and the fluid moves as a rigid body.  The plug half-width and plug
   velocity are known analytically.

3. **Newtonian limits**: as τ_y → 0, both solutions reduce to the
   classical Newtonian results.

All solutions are given in lattice units (ρ = 1, c_s² = 1/3).

References
----------
* Bird, R.B., Stewart, W.E. & Lightfoot, E.N. (2002) *Transport Phenomena*,
  2nd ed., Wiley, §8.3.
* Chhabra, R.P. & Richardson, J.F. (2008) *Non-Newtonian Flow and Applied
  Rheology*, 2nd ed., Butterworth-Heinemann, §5.2.
* Mitsoulis, E. (2004) *J. Non-Newtonian Fluid Mech.* 121, 1.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "bingham_couette_profile",
    "bingham_poiseuille_profile",
    "bingham_poiseuille_plug_half_width",
    "bingham_poiseuille_max_velocity",
    "newtonian_poiseuille_profile",
    "newtonian_couette_profile",
    "squeeze_film_lubrication_bounds",
]


def bingham_couette_profile(
    H: int,
    u_wall: float,
    tau_y: float,
    mu_0: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact Bingham Couette profile.

    For simple Couette flow (one wall moving at U_w, the other stationary),
    the shear rate is uniform: γ̇ = U_w / H.  The shear stress is
    τ = τ_y + μ₀·γ̇ everywhere (assuming |τ| > τ_y, i.e. the fluid is
    yielded throughout).  The velocity profile is linear:

        u(y) = U_w · (y − y_bottom) / H

    This is identical to the Newtonian Couette profile — the yield stress
    only affects the wall shear stress, not the velocity shape.

    Args:
        H: Channel height in lattice cells (fluid rows 1..H of ny=H+2 grid).
        u_wall: Wall velocity (top wall).
        tau_y: Yield stress.
        mu_0: Plastic viscosity.

    Returns:
        (u, y_phys): velocity profile and physical coordinates.
    """
    dtype = torch.float64
    y_phys = torch.arange(1, H + 1, dtype=dtype)
    u = u_wall * (y_phys - 0.5) / float(H)
    return u, y_phys


def bingham_poiseuille_plug_half_width(
    H: int,
    ax: float,
    tau_y: float,
    mu_0: float,
) -> float:
    """Plug half-width for Bingham Poiseuille flow.

    For body-force driven flow with acceleration ``ax``, the shear stress
    varies linearly from the centreline:

        τ(y) = ρ·ax·|y − y_c|

    The plug region is where |τ| ≤ τ_y, i.e.

        |y − y_c| ≤ τ_y / (ρ·ax) = y_p

    In lattice units (ρ = 1): y_p = τ_y / ax.

    The plug half-width in lattice cells is y_p, clipped to H/2.

    Args:
        H: Channel height (fluid rows).
        ax: Body force acceleration.
        tau_y: Yield stress.
        mu_0: Plastic viscosity.

    Returns:
        Plug half-width in lattice units.
    """
    if ax <= 0:
        return float(H) / 2.0  # no flow → entire channel is plug
    y_p = tau_y / ax
    return min(y_p, float(H) / 2.0)


def bingham_poiseuille_max_velocity(
    H: int,
    ax: float,
    tau_y: float,
    mu_0: float,
) -> float:
    """Maximum (plug) velocity for Bingham Poiseuille flow.

    The plug velocity is:

        u_max = (ax / (2·μ₀)) · [(H/2)² − y_p²] + (τ_y / μ₀) · [(H/2) − y_p]

    where y_p = τ_y / ax is the plug half-width.

    Args:
        H: Channel height.
        ax: Body force acceleration.
        tau_y: Yield stress.
        mu_0: Plastic viscosity.

    Returns:
        Maximum velocity (plug velocity).
    """
    Hh = float(H) / 2.0
    y_p = bingham_poiseuille_plug_half_width(H, ax, tau_y, mu_0)

    if y_p >= Hh:
        return 0.0  # entire channel is plug, no flow

    u_max = (ax / (2.0 * mu_0)) * (Hh * Hh - y_p * y_p) + (tau_y / mu_0) * (Hh - y_p)
    return u_max


def bingham_poiseuille_profile(
    H: int,
    ax: float,
    tau_y: float,
    mu_0: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact Bingham Poiseuille velocity profile.

    For body-force driven flow between parallel plates (half-height Hh = H/2,
    centreline at y_c = (H+1)/2):

    In the yielded region (|y − y_c| > y_p):

        u(y) = (ax / (2·μ₀)) · [(Hh)² − (y − y_c)²]
               − (τ_y / μ₀) · [|y − y_c| − y_p]

    In the plug region (|y − y_c| ≤ y_p):

        u(y) = u_max  (constant)

    where y_p = τ_y / ax is the plug half-width.

    Args:
        H: Channel height (fluid rows 1..H of ny=H+2 grid).
        ax: Body force acceleration (lattice units).
        tau_y: Yield stress (lattice units).
        mu_0: Plastic viscosity (lattice units).

    Returns:
        (u, y_phys): velocity profile and physical coordinates.
    """
    dtype = torch.float64
    Hh = float(H) / 2.0
    yc = float(H + 1) / 2.0  # centreline

    y_phys = torch.arange(1, H + 1, dtype=dtype)
    dy = torch.abs(y_phys - yc)

    if ax <= 0 or tau_y <= 0:
        # Newtonian limit or no force
        u = (ax / (2.0 * mu_0)) * (Hh * Hh - (y_phys - yc) ** 2)
        return u, y_phys

    y_p = tau_y / ax
    y_p = min(y_p, Hh)

    # Plug velocity
    u_max = (ax / (2.0 * mu_0)) * (Hh * Hh - y_p * y_p) + (tau_y / mu_0) * (Hh - y_p)

    # Full profile
    u = torch.where(
        dy <= y_p,
        torch.full_like(dy, u_max),
        (ax / (2.0 * mu_0)) * (Hh * Hh - dy * dy) - (tau_y / mu_0) * (dy - y_p),
    )

    # Ensure non-negative (numerical safety)
    u = torch.clamp(u, min=0.0)
    return u, y_phys


def newtonian_poiseuille_profile(
    H: int,
    ax: float,
    nu: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Newtonian Poiseuille profile (τ_y = 0 limit).

    u(y) = (ax / (2·ν)) · [(H/2)² − (y − y_c)²]

    Args:
        H: Channel height.
        ax: Body force acceleration.
        nu: Kinematic viscosity.

    Returns:
        (u, y_phys).
    """
    dtype = torch.float64
    Hh = float(H) / 2.0
    yc = float(H + 1) / 2.0
    y_phys = torch.arange(1, H + 1, dtype=dtype)
    u = (ax / (2.0 * nu)) * (Hh * Hh - (y_phys - yc) ** 2)
    return u, y_phys


def newtonian_couette_profile(
    H: int,
    u_wall: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Newtonian Couette profile (identical to Bingham Couette)."""
    dtype = torch.float64
    y_phys = torch.arange(1, H + 1, dtype=dtype)
    u = u_wall * (y_phys - 0.5) / float(H)
    return u, y_phys


def squeeze_film_lubrication_bounds(
    h: float,
    U: float,
    L: float,
    mu_0: float,
    tau_y: float,
) -> dict:
    """Squeeze-film lubrication theory bounds for Bingham fluid.

    For a thin gap of height h, length L, with relative velocity U:

    Newtonian lubrication force (Stefan-Reynolds):
        F_newton = μ₀ · U · L / h

    Bingham correction (upper bound, fully yielded):
        F_bingham_upper = F_newton · (1 + Bn / 3)

    where Bn = τ_y · h / (μ₀ · U) is the Bingham number.

    Lower bound (plug-dominated, approximate):
        F_bingham_lower = F_newton · max(1, Bn / 6)

    Args:
        h: Gap height.
        U: Relative velocity.
        L: Contact length.
        mu_0: Plastic viscosity.
        tau_y: Yield stress.

    Returns:
        Dictionary with newtonian, upper_bound, lower_bound, Bn.
    """
    if h <= 0 or U <= 0 or L <= 0:
        return {"newtonian": 0.0, "upper_bound": 0.0, "lower_bound": 0.0, "Bn": 0.0}

    Bn = tau_y * h / (mu_0 * U)
    F_newton = mu_0 * U * L / h
    F_upper = F_newton * (1.0 + Bn / 3.0)
    F_lower = F_newton * max(1.0, Bn / 6.0)

    return {
        "newtonian": F_newton,
        "upper_bound": F_upper,
        "lower_bound": F_lower,
        "Bn": Bn,
    }
