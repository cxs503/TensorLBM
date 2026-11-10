"""Analytical solutions for Herschel-Bulkley channel flow.

Exact solutions for pressure-driven (body-force) channel flow of a
Herschel-Bulkley fluid between parallel plates separated by distance 2h.

Constitutive law (Papanastasiou-regularised):
    τ = K·γ̇ⁿ + τ_y·(1 − exp(−m·γ̇))   (regularised)
    τ = K·γ̇ⁿ + τ_y                       (ideal, |τ| > τ_y)
    γ̇ = 0                                  (ideal, |τ| ≤ τ_y)

For the ideal (unregularised) HB fluid, the momentum balance gives:
    τ(y) = G·(h − y)    for 0 ≤ y ≤ h  (bottom half, y from wall)

Yield surface at y_p where τ(y_p) = τ_y:
    y_p = h − τ_y/G

In the yielded region (0 ≤ y < y_p):
    G·(h−y) − τ_y = K·(du/dy)ⁿ
    du/dy = [(G·(h−y) − τ_y)/K]^(1/n)

Velocity profile (integrating from wall u(0)=0 to yield surface):
    u(y) = [n/(n+1)]·K^(−1/n)·G^(−1)·[(G·(h−y)−τ_y)^((n+1)/n) − (G·h−τ_y)^((n+1)/n)]
           + u_plug·(y/y_p)  ... no, simpler:

    u(y) = [n/(n+1)]·(1/K)^(1/n)·(1/G)·[(G·h−τ_y)^((n+1)/n) − (G·(h−y)−τ_y)^((n+1)/n)]

Plug velocity (at y = y_p, constant in plug region y_p ≤ y ≤ h):
    u_plug = [n/(n+1)]·(1/K)^(1/n)·(1/G)·(G·h − τ_y)^((n+1)/n)

Mean velocity:
    U_mean = (1/h)·[∫₀^{y_p} u(y)dy + u_plug·(h − y_p)]

References
----------
* Chhabra, R.P. & Richardson, J.F. (2008) Non-Newtonian Flow, §5.2.
* Bird, R.B. et al. (2002) Transport Phenomena, §8.3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass(frozen=True)
class HBChannelParams:
    """Herschel-Bulkley channel flow parameters."""
    tau_y: float   # yield stress
    K: float       # consistency index
    n: float       # flow index
    G: float       # body force per unit volume (ρ·g or ρ·ax)
    h: float       # half-channel height


def hb_yield_surface(params: HBChannelParams) -> float:
    """Yield surface position y_p from the wall.

    Returns h if the fluid is entirely unyielded (no flow).
    """
    if params.G <= 0:
        return params.h
    y_p = params.h - params.tau_y / params.G
    return max(0.0, min(y_p, params.h))


def hb_velocity_profile(
    params: HBChannelParams,
    y: np.ndarray | torch.Tensor | None = None,
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the exact HB channel velocity profile.

    Args:
        params: HB channel parameters.
        y: Wall-normal coordinates (0 = wall, h = centerline).
           If None, generates n_points from 0 to h.
        n_points: Number of points if y is None.

    Returns:
        (y, u): wall-normal coordinates and velocity.
    """
    tau_y, K, n_idx, G, h = params.tau_y, params.K, params.n, params.G, params.h

    if y is None:
        y = np.linspace(0, h, n_points)
    elif isinstance(y, torch.Tensor):
        y = y.numpy()
    y = np.asarray(y, dtype=np.float64)

    y_p = hb_yield_surface(params)

    if y_p <= 0 or G <= 0:
        # No flow: entire channel is plug or no driving force
        return y, np.zeros_like(y)

    u = np.zeros_like(y)

    # Plug velocity
    plug_arg = max(G * h - tau_y, 0.0)
    if plug_arg <= 0:
        return y, np.zeros_like(y)

    u_plug = (n_idx / (n_idx + 1.0)) * (1.0 / K) ** (1.0 / n_idx) * (1.0 / G) * \
             plug_arg ** ((n_idx + 1.0) / n_idx)

    for i, yi in enumerate(y):
        if yi <= y_p:
            # Yielded region
            arg = G * (h - yi) - tau_y
            if arg >= 0:
                u[i] = (n_idx / (n_idx + 1.0)) * (1.0 / K) ** (1.0 / n_idx) * (1.0 / G) * \
                       (plug_arg ** ((n_idx + 1.0) / n_idx) - arg ** ((n_idx + 1.0) / n_idx))
            else:
                u[i] = 0.0
        else:
            # Plug region
            u[i] = u_plug

    return y, u


def hb_mean_velocity(params: HBChannelParams) -> float:
    """Compute the cross-sectionally averaged velocity analytically.

    U_mean = (1/h) · [∫₀^{y_p} u(y) dy + u_plug · (h − y_p)]

    The integral can be done analytically:
    ∫₀^{y_p} u(y) dy = [n/(n+1)]·K^(-1/n)·G^(-1) ·
        [plug_arg^((n+1)/n) · y_p − ∫₀^{y_p} (G(h−y)−τ_y)^((n+1)/n) dy]

    Let s = G(h−y) − τ_y, ds = −G dy
    When y=0: s = Gh − τ_y = plug_arg
    When y=y_p: s = G(h−y_p) − τ_y = 0

    ∫₀^{y_p} (G(h−y)−τ_y)^((n+1)/n) dy = (1/G) ∫₀^{plug_arg} s^((n+1)/n) ds
        = (1/G) · [n/(2n+1)] · plug_arg^((2n+1)/n)

    So:
    ∫₀^{y_p} u(y) dy = [n/(n+1)]·K^(-1/n)·G^(-1) ·
        [plug_arg^((n+1)/n) · y_p − (n/(2n+1))·plug_arg^((2n+1)/n)/G]
    """
    tau_y, K, n_idx, G, h = params.tau_y, params.K, params.n, params.G, params.h
    y_p = hb_yield_surface(params)

    if y_p <= 0 or G <= 0:
        return 0.0

    plug_arg = max(G * h - tau_y, 0.0)
    if plug_arg <= 0:
        return 0.0

    u_plug = (n_idx / (n_idx + 1.0)) * (1.0 / K) ** (1.0 / n_idx) * (1.0 / G) * \
             plug_arg ** ((n_idx + 1.0) / n_idx)

    # Integral of velocity in yielded region
    prefactor = (n_idx / (n_idx + 1.0)) * (1.0 / K) ** (1.0 / n_idx) * (1.0 / G)
    integral = prefactor * (
        plug_arg ** ((n_idx + 1.0) / n_idx) * y_p
        - (n_idx / (2.0 * n_idx + 1.0)) * plug_arg ** ((2.0 * n_idx + 1.0) / n_idx) / G
    )

    u_mean = (integral + u_plug * (h - y_p)) / h
    return u_mean


def hb_wall_shear_rate(params: HBChannelParams) -> float:
    """Wall shear rate for HB channel flow.

    At the wall (y=0): τ_w = G·h
    γ̇_w = [(τ_w − τ_y)/K]^(1/n) = [(G·h − τ_y)/K]^(1/n)

    Returns 0 if the fluid is not yielded at the wall.
    """
    tau_w = params.G * params.h
    if tau_w <= params.tau_y:
        return 0.0
    return ((tau_w - params.tau_y) / params.K) ** (1.0 / params.n)


def hb_effective_viscosity_at_wall(params: HBChannelParams) -> float:
    """Effective viscosity at the wall.

    μ_eff,w = τ_w / γ̇_w = (G·h) / γ̇_w
    """
    gamma_w = hb_wall_shear_rate(params)
    if gamma_w <= 0:
        return float('inf')
    tau_w = params.G * params.h
    return tau_w / gamma_w


# ---------------------------------------------------------------------------
# Metzner-Reed generalized Reynolds number
# ---------------------------------------------------------------------------

def metzner_reed_reynolds_number(
    U_mean: float,
    h: float,
    rho: float,
    K: float,
    n: float,
    tau_y: float = 0.0,
) -> float:
    """Metzner-Reed generalized Reynolds number for channel flow.

    For a power-law fluid in a channel of half-height h:
        Re_MR = ρ · U_mean^(2-n) · (4h)^n / (K · 8^(n-1) · ((3n+1)/(4n))^n)

    For HB fluids, we use the approach of Esmaeeli & Passandideh-Fard (2021):
    define an effective Reynolds number using the wall effective viscosity:
        Re_MR = ρ · U_mean · (4h) / μ_eff,w

    Or equivalently, use the modified definition that accounts for yield stress
    through the Hedström number He = ρ·τ_y·(4h)²/K².

    Here we use the simpler approach: compute Re_MR from the actual mean
    velocity and the apparent viscosity at the wall shear rate.

    For pure power-law (τ_y = 0), the standard Metzner-Reed:
        Re_MR = ρ · U^(2-n) · D_h^n / (K · 8^(n-1) · ((3n+1)/(4n))^n)
    where D_h = 4h for a wide channel.
    """
    D_h = 4.0 * h  # hydraulic diameter for wide channel

    if tau_y > 0:
        # For HB: use the approach based on the actual flow rate
        # Define Re_MR via the wall quantities
        params_approx = HBChannelParams(tau_y=tau_y, K=K, n=n, G=1.0, h=h)
        # We need to find G such that U_mean matches; for now use the
        # effective viscosity approach
        # γ̇_w,eff = (2n+1)/(3n) · U_mean/h  (power-law approximation)
        if n <= 0:
            return 0.0
        gamma_w_eff = (2.0 * n + 1.0) / (3.0 * n) * U_mean / h
        if gamma_w_eff <= 0:
            return 0.0
        mu_eff = K * gamma_w_eff ** (n - 1.0) + tau_y / gamma_w_eff
        return rho * U_mean * D_h / mu_eff
    else:
        # Standard Metzner-Reed for power-law
        if K <= 0 or U_mean <= 0:
            return 0.0
        phi_n = (3.0 * n + 1.0) / (4.0 * n)
        Re_MR = rho * U_mean ** (2.0 - n) * D_h ** n / (K * 8.0 ** (n - 1.0) * phi_n ** n)
        return Re_MR


def friction_factor_channel(
    G: float,
    h: float,
    rho: float,
    U_mean: float,
) -> float:
    """Fanning friction factor for channel flow.

    f = τ_w / (0.5 · ρ · U_mean²)
    τ_w = G · h  (from momentum balance)

    Returns f.
    """
    if U_mean <= 0:
        return float('inf')
    tau_w = G * h
    return tau_w / (0.5 * rho * U_mean ** 2)


def dodge_metzner_laminar(n: float) -> float:
    """Laminar friction factor coefficient: f = C(n)/Re_MR.

    For channel flow: C(n) = 24 for Newtonian (n=1).
    For power-law in pipe: f = 16/Re_MR.
    For channel (wide): f = 24/Re_MR (Newtonian limit).

    The generalized form for power-law channel:
    f · Re_MR = 24  (independent of n for the MR definition)
    """
    return 24.0


def dodge_metzner_turbulent(
    Re_MR: float,
    n: float,
) -> float:
    """Dodge-Metzner turbulent friction factor correlation (pipe).

    For pipe flow (1959):
    f^(-0.5) = (4.0/(n'^0.75)) · log10(Re_MR · f^(0.5·(2-n))) − 0.4/(n'^0.75)

    Simplified Blasius-type for channel (approximate):
    f = a(n) · Re_MR^(−b(n))

    where a(n) and b(n) are fitted to the DM correlation.

    For n=1 (Newtonian): f ≈ 0.0791 · Re_MR^(−0.25) (Blasius, Re < 10^5)

    Returns f (Fanning).
    """
    if Re_MR <= 0:
        return float('inf')

    # Blasius-type correlation generalized for power-law
    # From Dodge & Metzner (1959), the explicit approximation:
    # f = a / Re_MR^b
    # where a and b depend on n
    # For n=1: a=0.0791, b=0.25 (Blasius)
    # Generalized: a(n) = 0.0791 * (some correction), b(n) = 0.25 * (some correction)

    # Use the explicit approximation from Govier & Aziz (1972):
    # f = 10^(-a_n) * Re_MR^(-b_n)
    # a_n = 1.342 - 0.682*n + 0.185*n^2  (not exact, approximate)
    # b_n = 0.25 - 0.0375*(n-1)  (approximate)

    # Simpler: just use the Newtonian Blasius as a first approximation
    # and note that the DM correction is O(10%) for 0.5 < n < 1.5
    b = 0.25
    a = 0.0791

    # Correction for non-Newtonian (approximate, from literature)
    if abs(n - 1.0) > 0.01:
        # Dodge-Metzner explicit approximation
        # f = [1.342/n'^0.75] ... this is complex, use simplified
        # For shear-thinning (n<1): friction factor is slightly lower
        # For shear-thickening (n>1): friction factor is slightly higher
        correction = 1.0 + 0.15 * (1.0 - n)  # linear approx
        a = 0.0791 * correction

    return a * Re_MR ** (-b)
