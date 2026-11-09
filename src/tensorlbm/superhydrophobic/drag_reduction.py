"""Drag reduction quantification for superhydrophobic surfaces.

Computes the drag reduction rate:
    DR = (Cf_smooth - Cf_slip) / Cf_smooth

where Cf is the skin friction coefficient.

For a channel flow, the friction coefficient can be related to the
slip length. For Couette flow with slip length b_eff at one wall:
    Cf_slip / Cf_smooth = H / (H + b_eff)

For Poiseuille flow (pressure-driven) with symmetric slip:
    Cf_slip / Cf_smooth = 1 / (1 + 6*b_eff/H)  (approximate)

The exact relation depends on the flow configuration.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def friction_coefficient_channel(
    velocity_profile: np.ndarray,
    y_coords: np.ndarray,
    nu: float,
    H: float,
    rho: float = 1.0,
) -> float:
    """Compute skin friction coefficient from velocity profile.

    Cf = tau_w / (0.5 * rho * U_ref^2)

    where tau_w = mu * (du/dy)|_wall and U_ref is a reference velocity.

    For channel flow, we use the bulk velocity as reference:
        U_bulk = (1/H) * integral(u dy)

    Parameters
    ----------
    velocity_profile : array
        Streamwise velocity u(y).
    y_coords : array
        Wall-normal coordinates.
    nu : float
        Kinematic viscosity (lattice units: nu = cs^2 * (tau - 0.5) = (tau-0.5)/3).
    H : float
        Channel height.
    rho : float
        Fluid density.

    Returns
    -------
    Cf : float
        Skin friction coefficient.
    """
    # Wall shear stress from linear extrapolation near wall
    # Use points y=1,2,3 for gradient estimation
    n_fit = min(4, len(y_coords) // 4)
    y_fit = y_coords[1:n_fit+1]
    u_fit = velocity_profile[1:n_fit+1]

    if len(y_fit) >= 2:
        coeffs = np.polyfit(y_fit, u_fit, 1)
        du_dy_wall = coeffs[0]
    else:
        du_dy_wall = (velocity_profile[2] - velocity_profile[1]) / (y_coords[2] - y_coords[1])

    mu = rho * nu
    tau_w = mu * du_dy_wall

    # Bulk velocity
    U_bulk = np.trapezoid(velocity_profile, y_coords) / H

    if U_bulk < 1e-10:
        return 0.0

    Cf = tau_w / (0.5 * rho * U_bulk**2)
    return float(Cf)


def compute_drag_reduction(
    Cf_smooth: float,
    Cf_slip: float,
) -> dict:
    """Compute drag reduction rate.

    DR = (Cf_smooth - Cf_slip) / Cf_smooth

    Parameters
    ----------
    Cf_smooth : float
        Friction coefficient for smooth (no-slip) wall.
    Cf_slip : float
        Friction coefficient for superhydrophobic (slip) wall.

    Returns
    -------
    result : dict
        Contains:
        - 'DR': drag reduction rate (0 to 1)
        - 'DR_percent': drag reduction percentage
        - 'Cf_ratio': Cf_slip / Cf_smooth
    """
    if abs(Cf_smooth) < 1e-15:
        return {"DR": 0.0, "DR_percent": 0.0, "Cf_ratio": 1.0}

    DR = (Cf_smooth - Cf_slip) / Cf_smooth
    Cf_ratio = Cf_slip / Cf_smooth

    return {
        "DR": float(DR),
        "DR_percent": float(DR * 100),
        "Cf_ratio": float(Cf_ratio),
    }


def theoretical_drag_reduction_couette(b_eff: float, H: float) -> float:
    """Theoretical drag reduction for Couette flow with slip at one wall.

    For Couette flow, the wall shear stress is:
        tau_w = mu * U_lid / (H + b_eff)

    Compared to no-slip:
        tau_w_0 = mu * U_lid / H

    So: Cf_slip / Cf_smooth = H / (H + b_eff)
    And: DR = b_eff / (H + b_eff)

    Parameters
    ----------
    b_eff : float
        Effective slip length.
    H : float
        Channel height.

    Returns
    -------
    DR : float
        Theoretical drag reduction rate.
    """
    if H + b_eff < 1e-10:
        return 0.0
    return b_eff / (H + b_eff)


def theoretical_drag_reduction_poiseuille(b_eff: float, H: float) -> float:
    """Theoretical drag reduction for Poiseuille flow with symmetric slip.

    For pressure-driven channel flow with slip length b_eff at both walls:
        Q_slip / Q_no_slip = 1 + 6*b_eff/H  (for small b_eff/H)

    The friction factor ratio:
        Cf_slip / Cf_smooth = 1 / (1 + 6*b_eff/H)

    So: DR = 6*b_eff / (H + 6*b_eff)

    Parameters
    ----------
    b_eff : float
        Effective slip length.
    H : float
        Channel height.

    Returns
    -------
    DR : float
        Theoretical drag reduction rate.
    """
    denom = H + 6.0 * b_eff
    if abs(denom) < 1e-10:
        return 0.0
    return 6.0 * b_eff / denom
