"""Analytical solutions for circular Poiseuille flow.

For fully developed laminar flow in a circular pipe of radius R:
    u(r) = u_max * (1 - (r/R)^2)

where u_max is the centerline velocity, related to the pressure gradient by:
    u_max = (dp/dx) * R^2 / (4 * mu)

The volumetric flow rate is:
    Q = pi * R^2 * u_max / 2 = pi * R^4 * (dp/dx) / (8 * mu)

The wall shear stress is:
    tau_w = mu * (du/dr)|_R = 2 * mu * u_max / R

The friction factor is:
    f = 64 / Re_D
where Re_D = rho * u_mean * D / mu, D = 2R, u_mean = u_max / 2
"""

import numpy as np
from typing import Tuple


def poiseuille_velocity_profile(
    r: np.ndarray,
    r_max: float,
    u_max: float,
) -> np.ndarray:
    """Compute analytical Poiseuille velocity profile.

    Parameters
    ----------
    r : np.ndarray
        Radial positions
    r_max : float
        Pipe radius
    u_max : float
        Centerline (maximum) velocity

    Returns
    -------
    u : np.ndarray
        Velocity at each radial position
    """
    r = np.asarray(r, dtype=np.float64)
    u = np.zeros_like(r)
    mask = r <= r_max
    u[mask] = u_max * (1.0 - (r[mask] / r_max)**2)
    return u


def poiseuille_max_velocity(
    dp_dx: float,
    radius: float,
    mu: float,
) -> float:
    """Compute centerline velocity from pressure gradient.

    Parameters
    ----------
    dp_dx : float
        Pressure gradient (negative for flow in +x direction)
    radius : float
        Pipe radius
    mu : float
        Dynamic viscosity

    Returns
    -------
    u_max : float
        Centerline velocity
    """
    return -dp_dx * radius**2 / (4.0 * mu)


def poiseuille_flow_rate(
    u_max: float,
    radius: float,
) -> float:
    """Compute volumetric flow rate.

    Parameters
    ----------
    u_max : float
        Centerline velocity
    radius : float
        Pipe radius

    Returns
    -------
    Q : float
        Volumetric flow rate
    """
    return np.pi * radius**2 * u_max / 2.0


def poiseuille_wall_shear_stress(
    u_max: float,
    radius: float,
    mu: float,
) -> float:
    """Compute wall shear stress.

    Parameters
    ----------
    u_max : float
        Centerline velocity
    radius : float
        Pipe radius
    mu : float
        Dynamic viscosity

    Returns
    -------
    tau_w : float
        Wall shear stress magnitude
    """
    return 2.0 * mu * u_max / radius


def poiseuille_reynolds_number(
    u_max: float,
    diameter: float,
    nu: float,
) -> float:
    """Compute Reynolds number based on mean velocity and diameter.

    Parameters
    ----------
    u_max : float
        Centerline velocity
    diameter : float
        Pipe diameter
    nu : float
        Kinematic viscosity

    Returns
    -------
    Re : float
        Reynolds number
    """
    u_mean = u_max / 2.0
    return u_mean * diameter / nu
