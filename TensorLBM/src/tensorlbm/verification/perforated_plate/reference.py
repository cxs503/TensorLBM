"""Analytical reference solutions for perforated plate oscillatory flow.

Provides:
    - Idelchik quasi-steady discharge coefficient for sharp-edged orifice
    - Quasi-steady pressure drop formula
    - Stokes oscillating wall shear (for negative control)
"""

from __future__ import annotations

import math


def idelchik_cd_quasi_steady(
    beta: float,
    re: float = 1e5,
    edge_type: str = "sharp",
) -> float:
    """Quasi-steady discharge coefficient from Idelchik correlations.

    For a sharp-edged orifice in a pipe/channel:

        C_d = 1 / sqrt(ζ + 1)

    where ζ is the resistance coefficient.

    For sharp-edged orifice (Idelchik, Handbook of Hydraulic Resistance):
        ζ = (1 - β)² / β² + ζ_fric

    For high Re (turbulent), ζ_fric ≈ 0.5 for sharp edges.

    Simplified correlation for sharp-edged perforated plate:
        C_d ≈ 0.60 + 0.40 · β    (empirical fit for β ∈ [0.05, 0.40])

    Parameters
    ----------
    beta : float
        Open area ratio (0 < β < 1).
    re : float
        Reynolds number (for friction correction).
    edge_type : str
        'sharp' or 'rounded'.

    Returns
    -------
    C_d : float
        Discharge coefficient.
    """
    if beta <= 0 or beta >= 1:
        return float('nan')

    if edge_type == "sharp":
        # Empirical correlation for sharp-edged orifice
        # Valid for β ∈ [0.05, 0.40], Re > 1000
        # Based on Idelchik + experimental data
        c_d = 0.60 + 0.40 * beta
        # Clamp to physical range
        return max(0.4, min(0.9, c_d))
    elif edge_type == "rounded":
        # Rounded edges have higher C_d
        c_d = 0.80 + 0.15 * beta
        return max(0.6, min(0.98, c_d))
    else:
        raise ValueError(f"Unknown edge_type: {edge_type}")


def quasi_steady_pressure_drop(
    q: float,
    beta: float,
    area: float,
    rho: float = 1.0,
    c_d: float = 0.6,
) -> float:
    """Quasi-steady pressure drop across perforated plate.

    Δp = ½ ρ C_d⁻² (Q / (A β))² · sign(Q)

    Parameters
    ----------
    q : float
        Volume flow rate.
    beta : float
        Open area ratio.
    area : float
        Channel cross-section area.
    rho : float
        Fluid density.
    c_d : float
        Discharge coefficient.

    Returns
    -------
    delta_p : float
        Pressure drop.
    """
    if beta <= 0 or area <= 0 or c_d <= 0:
        return 0.0

    u_hole = q / (area * beta)
    delta_p = 0.5 * rho * (1.0 / c_d) ** 2 * u_hole * abs(u_hole)
    return delta_p


def stokes_oscillating_wall_shear(
    y: float,
    t: float,
    u0: float,
    omega: float,
    nu: float,
) -> float:
    """Stokes second problem: oscillating flat wall.

    u(y, t) = u₀ exp(-y/δ) cos(ωt - y/δ)

    where δ = sqrt(2ν/ω) is the Stokes layer thickness.

    Wall shear stress:
        τ_w = μ (∂u/∂y)|_{y=0} = -μ u₀ / δ [cos(ωt) + sin(ωt)]

    Parameters
    ----------
    y : float
        Wall-normal distance.
    t : float
        Time.
    u0 : float
        Wall oscillation amplitude.
    omega : float
        Angular frequency.
    nu : float
        Kinematic viscosity.

    Returns
    -------
    u : float
        Velocity at (y, t).
    """
    if nu <= 0 or omega <= 0:
        return 0.0

    delta = math.sqrt(2.0 * nu / omega)
    eta = y / delta

    u = u0 * math.exp(-eta) * math.cos(omega * t - eta)
    return u


def stokes_wall_shear_stress(
    t: float,
    u0: float,
    omega: float,
    nu: float,
    mu: float = 1.0,
) -> float:
    """Wall shear stress for Stokes second problem.

    τ_w(t) = -μ u₀ / δ [cos(ωt) + sin(ωt)]
           = -μ u₀ sqrt(ω/ν) cos(ωt - π/4)

    Parameters
    ----------
    t : float
        Time.
    u0 : float
        Wall oscillation amplitude.
    omega : float
        Angular frequency.
    nu : float
        Kinematic viscosity.
    mu : float
        Dynamic viscosity.

    Returns
    -------
    tau_w : float
        Wall shear stress.
    """
    if nu <= 0 or omega <= 0:
        return 0.0

    delta = math.sqrt(2.0 * nu / omega)
    tau_w = -mu * u0 / delta * (math.cos(omega * t) + math.sin(omega * t))
    return tau_w


def stokes_layer_thickness(
    omega: float,
    nu: float,
) -> float:
    """Stokes boundary layer thickness: δ = sqrt(2ν/ω)."""
    if nu <= 0 or omega <= 0:
        return float('inf')
    return math.sqrt(2.0 * nu / omega)
