"""Analytical solutions for ASBL and Blasius boundary layers.

ASBL exact solution (Schlichting, Boundary-Layer Theory):
    u(y) = U∞ (1 - exp(-V y / ν))
    v(y) = -V  (constant suction)
    δ = ν / V  (boundary layer thickness, constant)
    C_f = 2V / U∞  (exact, independent of Re_x)
    τ_w = μ U∞ V / ν = ρ U∞ V

Blasius solution (flat plate, no suction):
    C_f,x = 0.664 / √Re_x  (local skin friction)
    C_f,L = 1.328 / √Re_L  (average skin friction over length L)
    δ ≈ 5.0 x / √Re_x  (99% thickness)
"""

import torch
import numpy as np
from typing import Union

ArrayLike = Union[torch.Tensor, np.ndarray, float]


def asbl_exact_velocity(
    y: ArrayLike,
    U_inf: float,
    V_suction: float,
    nu: float,
) -> ArrayLike:
    """ASBL exact velocity profile: u(y) = U∞ (1 - exp(-V y / ν)).

    Parameters
    ----------
    y : array-like, wall-normal coordinates
    U_inf : float, free-stream velocity
    V_suction : float, suction velocity (positive = into wall)
    nu : float, kinematic viscosity

    Returns
    -------
    u : array-like, streamwise velocity at each y
    """
    if isinstance(y, torch.Tensor):
        return U_inf * (1.0 - torch.exp(-V_suction * y / nu))
    else:
        y = np.asarray(y)
        return U_inf * (1.0 - np.exp(-V_suction * y / nu))


def asbl_skin_friction(U_inf: float, V_suction: float) -> float:
    """ASBL exact skin friction coefficient: C_f = 2V / U∞.

    This is a hard identity - independent of Re, x, or ν.

    Parameters
    ----------
    U_inf : float, free-stream velocity
    V_suction : float, suction velocity

    Returns
    -------
    C_f : float, skin friction coefficient
    """
    return 2.0 * V_suction / U_inf


def asbl_wall_shear_stress(
    U_inf: float,
    V_suction: float,
    nu: float,
    rho: float = 1.0,
) -> float:
    """ASBL wall shear stress: τ_w = ρ ν (∂u/∂y)|_0 = ρ U∞ V.

    Parameters
    ----------
    U_inf : float
    V_suction : float
    nu : float
    rho : float, density (default 1.0 for lattice units)

    Returns
    -------
    tau_w : float
    """
    return rho * U_inf * V_suction


def asbl_displacement_thickness(nu: float, V_suction: float) -> float:
    """ASBL displacement thickness: δ* = ν / V.

    Parameters
    ----------
    nu : float
    V_suction : float

    Returns
    -------
    delta_star : float
    """
    return nu / V_suction


def blasius_skin_friction(Re_x: float) -> float:
    """Blasius local skin friction: C_f = 0.664 / √Re_x.

    Parameters
    ----------
    Re_x : float, Reynolds number at position x

    Returns
    -------
    C_f : float
    """
    return 0.664 / np.sqrt(Re_x)


def blasius_similarity(eta: ArrayLike) -> ArrayLike:
    """Approximate Blasius velocity profile f'(η).

    Uses the Pohlhausen approximation:
        f'(η) ≈ tanh(a η) where a ≈ 0.332 for good accuracy.

    More accurate: use the series solution or tabulated values.

    Parameters
    ----------
    eta : array-like, similarity variable η = y √(U∞/(νx))

    Returns
    -------
    f_prime : array-like, u/U∞
    """
    if isinstance(eta, torch.Tensor):
        # Use a more accurate approximation
        # f'(η) ≈ 1 - exp(-a η - b η²) with a=0.332, b chosen for accuracy
        a = 0.33206
        b = 0.0564  # fitted for accuracy
        return 1.0 - torch.exp(-a * eta - b * eta**2)
    else:
        eta = np.asarray(eta)
        a = 0.33206
        b = 0.0564
        return 1.0 - np.exp(-a * eta - b * eta**2)


def blasius_velocity_profile(
    y: ArrayLike,
    x: float,
    U_inf: float,
    nu: float,
) -> ArrayLike:
    """Blasius velocity profile at position x.

    Parameters
    ----------
    y : array-like, wall-normal coordinates
    x : float, streamwise position
    U_inf : float, free-stream velocity
    nu : float, kinematic viscosity

    Returns
    -------
    u : array-like, streamwise velocity
    """
    eta = y * np.sqrt(U_inf / (nu * x))
    return U_inf * blasius_similarity(eta)
