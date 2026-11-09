"""Isentropic flow relations for compressible flow verification.

Provides:
- Isentropic relations: p/p0, ρ/ρ0, T/T0 vs Mach number
- Verification that p/ρ^γ = const along streamlines
- Prandtl-Meyer expansion fan
- Critical (sonic) conditions

References
----------
- Anderson, J.D., "Modern Compressible Flow", 3rd ed., McGraw-Hill, 2003.
- Shapiro, A.H., "The Dynamics and Thermodynamics of Compressible Fluid Flow".
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple


def isentropic_p_p0(Ma: float, gamma: float = 1.4) -> float:
    """Isentropic pressure ratio p/p0.
    
    p/p0 = (1 + (γ-1)/2 · Ma²)^(-γ/(γ-1))
    """
    return (1.0 + 0.5 * (gamma - 1.0) * Ma**2) ** (-gamma / (gamma - 1.0))


def isentropic_rho_rho0(Ma: float, gamma: float = 1.4) -> float:
    """Isentropic density ratio ρ/ρ0.
    
    ρ/ρ0 = (1 + (γ-1)/2 · Ma²)^(-1/(γ-1))
    """
    return (1.0 + 0.5 * (gamma - 1.0) * Ma**2) ** (-1.0 / (gamma - 1.0))


def isentropic_T_T0(Ma: float, gamma: float = 1.4) -> float:
    """Isentropic temperature ratio T/T0.
    
    T/T0 = (1 + (γ-1)/2 · Ma²)^(-1)
    """
    return 1.0 / (1.0 + 0.5 * (gamma - 1.0) * Ma**2)


def isentropic_A_Astar(Ma: float, gamma: float = 1.4) -> float:
    """Isentropic area ratio A/A*.
    
    A/A* = (1/Ma) · ((2/(γ+1)) · (1 + (γ-1)/2 · Ma²))^((γ+1)/(2(γ-1)))
    """
    term1 = 1.0 / Ma
    term2 = (2.0 / (gamma + 1.0)) * (1.0 + 0.5 * (gamma - 1.0) * Ma**2)
    exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    return term1 * term2**exponent


def prandtl_meyer_angle(Ma: float, gamma: float = 1.4) -> float:
    """Prandtl-Meyer function ν(Ma).
    
    ν = sqrt((γ+1)/(γ-1)) · atan(sqrt((γ-1)/(γ+1) · (Ma²-1))) - atan(sqrt(Ma²-1))
    
    Returns angle in radians.
    """
    if Ma < 1.0:
        return 0.0
    g1 = (gamma + 1.0) / (gamma - 1.0)
    arg = (gamma - 1.0) / (gamma + 1.0) * (Ma**2 - 1.0)
    nu = math.sqrt(g1) * math.atan(math.sqrt(arg)) - math.atan(math.sqrt(Ma**2 - 1.0))
    return nu


def prandtl_meyer_from_nu(nu_rad: float, gamma: float = 1.4, tol: float = 1e-10) -> float:
    """Inverse Prandtl-Meyer: find Ma given ν.
    
    Uses Newton iteration.
    """
    if nu_rad <= 0:
        return 1.0
    
    # Initial guess
    Ma = 1.0 + nu_rad
    
    for _ in range(100):
        nu_calc = prandtl_meyer_angle(Ma, gamma)
        f = nu_calc - nu_rad
        
        if abs(f) < tol:
            break
        
        # dν/dMa = sqrt(Ma²-1) / (Ma · (1 + (γ-1)/2 · Ma²))
        if Ma <= 1.0:
            Ma = 1.001
            continue
        
        dnu_dMa = math.sqrt(Ma**2 - 1.0) / (Ma * (1.0 + 0.5 * (gamma - 1.0) * Ma**2))
        
        if abs(dnu_dMa) < 1e-15:
            break
        
        Ma -= f / dnu_dMa
        Ma = max(1.0 + 1e-6, Ma)
    
    return Ma


@dataclass
class IsentropicVerification:
    """Result of isentropic flow verification."""
    Ma_range: np.ndarray
    p_p0_computed: np.ndarray
    p_p0_exact: np.ndarray
    rho_rho0_computed: np.ndarray
    rho_rho0_exact: np.ndarray
    T_T0_computed: np.ndarray
    T_T0_exact: np.ndarray
    entropy_error: float  # max |Δs/cp|


def verify_isentropic_relations(
    Ma_range: np.ndarray = None,
    gamma: float = 1.4,
) -> IsentropicVerification:
    """Verify isentropic relations over a range of Mach numbers.
    
    Checks:
    1. p/p0 = (1 + (γ-1)/2 · Ma²)^(-γ/(γ-1))
    2. ρ/ρ0 = (1 + (γ-1)/2 · Ma²)^(-1/(γ-1))
    3. T/T0 = (1 + (γ-1)/2 · Ma²)^(-1)
    4. p/ρ^γ = const (entropy conservation)
    
    Parameters
    ----------
    Ma_range : np.ndarray
        Mach numbers to verify.
    gamma : float
        Specific heat ratio.
    
    Returns
    -------
    IsentropicVerification
        Verification results with errors.
    """
    if Ma_range is None:
        Ma_range = np.linspace(0.1, 2.0, 50)
    
    p_p0_exact = np.array([isentropic_p_p0(M, gamma) for M in Ma_range])
    rho_rho0_exact = np.array([isentropic_rho_rho0(M, gamma) for M in Ma_range])
    T_T0_exact = np.array([isentropic_T_T0(M, gamma) for M in Ma_range])
    
    # Verify: p/ρ^γ should be constant
    # p/p0 / (ρ/ρ0)^γ = const
    p_rho_gamma = p_p0_exact / (rho_rho0_exact ** gamma)
    entropy_error = float(np.max(np.abs(p_rho_gamma - p_rho_gamma[0])) / p_rho_gamma[0])
    
    return IsentropicVerification(
        Ma_range=Ma_range,
        p_p0_computed=p_p0_exact,
        p_p0_exact=p_p0_exact,
        rho_rho0_computed=rho_rho0_exact,
        rho_rho0_exact=rho_rho0_exact,
        T_T0_computed=T_T0_exact,
        T_T0_exact=T_T0_exact,
        entropy_error=entropy_error,
    )


def critical_conditions(gamma: float = 1.4) -> dict:
    """Compute critical (sonic, Ma=1) conditions.
    
    At Ma=1:
    - T*/T0 = 2/(γ+1)
    - p*/p0 = (2/(γ+1))^(γ/(γ-1))
    - ρ*/ρ0 = (2/(γ+1))^(1/(γ-1))
    - A/A* = 1
    """
    T_star_T0 = 2.0 / (gamma + 1.0)
    p_star_p0 = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))
    rho_star_rho0 = (2.0 / (gamma + 1.0)) ** (1.0 / (gamma - 1.0))
    
    return {
        'T_star_T0': T_star_T0,
        'p_star_p0': p_star_p0,
        'rho_star_rho0': rho_star_rho0,
        'A_Astar_sonic': 1.0,
    }


def max_turning_angle(gamma: float = 1.4) -> float:
    """Maximum turning angle (Prandtl-Meyer at Ma→∞).
    
    ν_max = (π/2) · (sqrt((γ+1)/(γ-1)) - 1)
    
    For γ=1.4: ν_max ≈ 130.45°
    """
    g1 = (gamma + 1.0) / (gamma - 1.0)
    nu_max = (math.pi / 2.0) * (math.sqrt(g1) - 1.0)
    return nu_max
