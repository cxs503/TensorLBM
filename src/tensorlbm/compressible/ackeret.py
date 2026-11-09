"""Ackeret linearized supersonic theory for sonic boom verification.

Provides:
- Ackeret pressure coefficient: Cp = 2θ / sqrt(Ma² - 1)
- Pressure perturbation: Δp = ρ·U²·θ / sqrt(Ma² - 1)
- N-wave signature for wedge/diamond airfoils
- Comparison with exact oblique shock theory

References
----------
- Ackeret, J., "Luftkrafte auf Flugel, die mit grosserer als 
  Schallgeschwindigkeit bewegt werden", Z. Flugtech. Motorluft., 1925.
- Anderson, J.D., "Modern Compressible Flow", 3rd ed., McGraw-Hill, 2003.
- Whitham, G.B., "The flow pattern of a supersonic projectile", 
  Comm. Pure Appl. Math., 1952.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple


def ackeret_pressure_coefficient(Ma: float, theta_deg: float, gamma: float = 1.4) -> float:
    """Ackeret linearized theory pressure coefficient.
    
    Cp = 2·θ / sqrt(Ma² - 1)
    
    where θ is the flow deflection angle (positive = compression).
    
    Parameters
    ----------
    Ma : float
        Freestream Mach number (must be > 1).
    theta_deg : float
        Flow deflection angle in degrees.
    gamma : float
        Specific heat ratio.
    
    Returns
    -------
    Cp : float
        Pressure coefficient.
    """
    if Ma <= 1.0:
        raise ValueError(f"Ackeret theory requires Ma > 1, got Ma={Ma}")
    theta_rad = math.radians(theta_deg)
    beta = math.sqrt(Ma**2 - 1.0)
    Cp = 2.0 * theta_rad / beta
    return Cp


def ackeret_pressure_jump(Ma: float, theta_deg: float, 
                          rho_inf: float = 1.0, U_inf: float = 1.0) -> float:
    """Ackeret pressure jump across weak compression/expansion.
    
    Δp = ρ·U²·θ / sqrt(Ma² - 1)
    
    Parameters
    ----------
    Ma : float
        Freestream Mach number.
    theta_deg : float
        Deflection angle in degrees.
    rho_inf : float
        Freestream density.
    U_inf : float
        Freestream velocity.
    
    Returns
    -------
    dp : float
        Pressure jump.
    """
    if Ma <= 1.0:
        raise ValueError(f"Ackeret theory requires Ma > 1, got Ma={Ma}")
    theta_rad = math.radians(theta_deg)
    beta = math.sqrt(Ma**2 - 1.0)
    dp = rho_inf * U_inf**2 * theta_rad / beta
    return dp


def ackeret_lift_coefficient(Ma: float, theta_deg: float) -> float:
    """Ackeret lift coefficient for diamond airfoil at zero incidence.
    
    For a diamond airfoil with half-angle θ:
    C_L = 0 (symmetric at α=0)
    
    At angle of attack α:
    C_L = 4α / sqrt(Ma² - 1)
    
    Parameters
    ----------
    Ma : float
        Freestream Mach number.
    theta_deg : float
        Angle of attack in degrees (for diamond airfoil).
    
    Returns
    -------
    C_L : float
        Lift coefficient.
    """
    if Ma <= 1.0:
        raise ValueError(f"Ackeret theory requires Ma > 1, got Ma={Ma}")
    alpha_rad = math.radians(theta_deg)
    beta = math.sqrt(Ma**2 - 1.0)
    C_L = 4.0 * alpha_rad / beta
    return C_L


def ackeret_drag_coefficient(Ma: float, half_angle_deg: float) -> float:
    """Ackeret wave drag coefficient for diamond airfoil.
    
    C_D = 4θ² / sqrt(Ma² - 1)
    
    where θ is the half-angle of the diamond.
    
    Parameters
    ----------
    Ma : float
        Freestream Mach number.
    half_angle_deg : float
        Diamond half-angle in degrees.
    
    Returns
    -------
    C_D : float
        Wave drag coefficient.
    """
    if Ma <= 1.0:
        raise ValueError(f"Ackeret theory requires Ma > 1, got Ma={Ma}")
    theta_rad = math.radians(half_angle_deg)
    beta = math.sqrt(Ma**2 - 1.0)
    C_D = 4.0 * theta_rad**2 / beta
    return C_D


@dataclass
class NWaveSignature:
    """N-wave pressure signature result."""
    x_over_L: np.ndarray   # Normalized distance along signature
    dp_over_p_inf: np.ndarray  # Normalized pressure perturbation
    dp_max: float          # Peak overpressure (front shock)
    dp_min: float          # Peak underpressure (rear expansion)
    shock_separation: float  # Distance between front and rear shocks


def n_wave_signature(
    Ma: float,
    wedge_angle_deg: float,
    length_ratio: float = 1.0,
    n_points: int = 200,
) -> NWaveSignature:
    """Generate N-wave pressure signature for a wedge/diamond body.
    
    The N-wave consists of:
    - Front shock (compression) at x = 0
    - Linear expansion through the body
    - Rear shock (compression) at x = L
    
    For a diamond airfoil with half-angle θ and chord c:
    - Front shock: Δp/p∞ = γ·Ma²·2θ/(2·sqrt(Ma²-1))
    - Linear decrease through body length
    - Rear shock: same magnitude as front
    
    Parameters
    ----------
    Ma : float
        Freestream Mach number.
    wedge_angle_deg : float
        Wedge/diamond half-angle in degrees.
    length_ratio : float
        Body length / reference length.
    n_points : int
        Number of points in signature.
    
    Returns
    -------
    NWaveSignature
        N-wave pressure signature.
    """
    if Ma <= 1.0:
        raise ValueError(f"N-wave requires Ma > 1, got Ma={Ma}")
    
    theta_rad = math.radians(wedge_angle_deg)
    beta = math.sqrt(Ma**2 - 1.0)
    gamma = 1.4
    
    # Peak overpressure from Ackeret theory
    # Cp_front = 2θ/β, so Δp/p∞ = 0.5·γ·Ma²·Cp = γ·Ma²·θ/β
    dp_over_p_inf_front = gamma * Ma**2 * theta_rad / beta
    
    # N-wave shape: triangular (linear between shocks)
    x = np.linspace(-0.2, 1.2, n_points)
    dp = np.zeros(n_points)
    
    # Front shock at x=0
    # Linear expansion from x=0 to x=L (L=1)
    # Rear shock at x=L
    
    for i, xi in enumerate(x):
        if xi < 0:
            dp[i] = 0.0
        elif xi <= 0:  # Front shock (idealized as jump)
            dp[i] = dp_over_p_inf_front
        elif xi < length_ratio:
            # Linear decrease
            dp[i] = dp_over_p_inf_front * (1.0 - 2.0 * xi / length_ratio)
        elif xi <= length_ratio:
            dp[i] = -dp_over_p_inf_front
        else:
            dp[i] = 0.0
    
    # Smooth the shocks slightly (real N-wave has finite thickness)
    # Use a simple smoothing
    from scipy.ndimage import gaussian_filter1d
    try:
        dp = gaussian_filter1d(dp, sigma=1.0)
    except ImportError:
        pass  # No smoothing if scipy not available
    
    return NWaveSignature(
        x_over_L=x,
        dp_over_p_inf=dp,
        dp_max=float(dp.max()),
        dp_min=float(dp.min()),
        shock_separation=length_ratio,
    )


def compare_ackeret_vs_oblique_shock(
    Ma_range: np.ndarray = None,
    theta_deg: float = 10.0,
    gamma: float = 1.4,
) -> dict:
    """Compare Ackeret linearized theory with exact oblique shock.
    
    Parameters
    ----------
    Ma_range : np.ndarray
        Mach numbers to compare.
    theta_deg : float
        Deflection angle.
    gamma : float
        Specific heat ratio.
    
    Returns
    -------
    dict with comparison data.
    """
    if Ma_range is None:
        Ma_range = np.linspace(1.2, 4.0, 30)
    
    from tensorlbm.compressible.solver import oblique_shock_relations
    
    theta_rad = math.radians(theta_deg)
    
    Cp_ackeret = []
    Cp_exact = []
    Ma_valid = []
    
    for Ma in Ma_range:
        # Ackeret
        Cp_ack = ackeret_pressure_coefficient(Ma, theta_deg, gamma)
        Cp_ackeret.append(Cp_ack)
        
        # Exact oblique shock
        try:
            os_result = oblique_shock_relations(Ma, theta_rad, gamma)
            # Cp from pressure ratio: Cp = 2/(γ·Ma²) · (p2/p1 - 1)
            Cp_ex = 2.0 / (gamma * Ma**2) * (os_result.p_ratio - 1.0)
            Cp_exact.append(Cp_ex)
            Ma_valid.append(Ma)
        except (ValueError, ZeroDivisionError):
            Cp_exact.append(np.nan)
            Ma_valid.append(Ma)
    
    return {
        'Ma': np.array(Ma_valid),
        'Cp_ackeret': np.array(Cp_ackeret),
        'Cp_exact': np.array(Cp_exact),
        'theta_deg': theta_deg,
    }
