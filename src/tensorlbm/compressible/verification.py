"""Verification utilities for compressible shock wave simulations."""

from __future__ import annotations

import math
import numpy as np

from .solver import normal_shock_relations


def compute_shock_profile_error(rho, u, p, M1, gamma=1.4):
    """Compute L2 errors of shock profile vs Rankine-Hugoniot exact solution.

    Parameters
    ----------
    rho, u, p : np.ndarray
        Numerical solution fields (1D).
    M1 : float
        Upstream Mach number.
    gamma : float
        Specific heat ratio.

    Returns
    -------
    dict with error metrics.
    """
    rh = normal_shock_relations(M1, gamma)

    rho1, p1 = 1.0, 1.0
    T1 = p1 / (rho1 * (gamma - 1.0))
    cs1 = math.sqrt(gamma * p1 / rho1)
    u1 = M1 * cs1

    rho2 = rho1 * rh.rho_ratio
    p2 = p1 * rh.p_ratio
    T2 = T1 * rh.T_ratio
    u2 = rh.M2 * math.sqrt(gamma * p2 / rho2)

    # Find shock position (midpoint of density transition)
    shock_idx = _find_shock_position(rho)

    n = len(rho)
    left_mask = np.arange(n) < shock_idx
    right_mask = np.arange(n) >= shock_idx

    # L2 errors
    if left_mask.any():
        rho_L2_left = np.sqrt(np.mean((rho[left_mask] - rho1) ** 2)) / rho1
        p_L2_left = np.sqrt(np.mean((p[left_mask] - p1) ** 2)) / p1
    else:
        rho_L2_left = p_L2_left = 0.0

    if right_mask.any():
        rho_L2_right = np.sqrt(np.mean((rho[right_mask] - rho2) ** 2)) / rho2
        p_L2_right = np.sqrt(np.mean((p[right_mask] - p2) ** 2)) / p2
    else:
        rho_L2_right = p_L2_right = 0.0

    n_left = left_mask.sum()
    n_right = right_mask.sum()
    n_total = n_left + n_right

    rho_L2 = np.sqrt((n_left * rho_L2_left ** 2 + n_right * rho_L2_right ** 2) / n_total)
    p_L2 = np.sqrt((n_left * p_L2_left ** 2 + n_right * p_L2_right ** 2) / n_total)

    return {
        "rho_L2": float(rho_L2),
        "p_L2": float(p_L2),
        "shock_idx": int(shock_idx),
    }


def extract_shock_position(rho):
    """Extract shock position as midpoint of density transition."""
    return int(_find_shock_position(rho))


def _find_shock_position(rho_1d):
    """Find shock position as midpoint of density transition."""
    rho_min = rho_1d.min()
    rho_max = rho_1d.max()
    rho_mid = 0.5 * (rho_min + rho_max)
    crossings = np.where(np.diff(np.sign(rho_1d - rho_mid)))[0]
    if len(crossings) == 0:
        return len(rho_1d) // 2
    return int(crossings[0])
