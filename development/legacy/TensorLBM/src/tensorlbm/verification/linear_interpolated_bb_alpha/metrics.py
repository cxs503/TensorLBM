"""Error metrics for linear-interpolated BB alpha scheme validation.

Provides standardized error computation functions for comparing
LBM simulation results against analytical solutions.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np


@dataclasses.dataclass
class ConvergenceResult:
    """Result of a single convergence study case."""
    grid_size: int
    radius: float
    use_interpolated: bool
    steps_to_converge: int
    l2_error: float
    linf_error: float
    max_velocity_error: float
    flow_rate_error: float
    u_max_sim: float
    u_max_analytical: float
    Q_sim: float
    Q_analytical: float

    @property
    def method_name(self) -> str:
        return "interpolated-bb-alpha" if self.use_interpolated else "standard-bb"


def compute_l2_error(
    u_sim: np.ndarray,
    u_analytical: np.ndarray,
    r: np.ndarray,
    radius: float,
    normalize: bool = True,
) -> float:
    """Compute L2 norm of velocity error.

    Parameters
    ----------
    u_sim : np.ndarray
        Simulated velocity profile.
    u_analytical : np.ndarray
        Analytical velocity profile.
    r : np.ndarray
        Radial positions.
    radius : float
        Pipe radius.
    normalize : bool
        If True, normalize by max analytical velocity.

    Returns
    -------
    l2_error : float
        L2 error norm.
    """
    mask = r <= radius
    if not mask.any():
        return float('nan')

    u_s = u_sim[mask]
    u_a = u_analytical[mask]

    l2 = float(np.sqrt(np.mean((u_s - u_a)**2)))

    if normalize:
        u_max = float(np.max(u_a))
        if u_max > 0:
            l2 /= u_max

    return l2


def compute_linfinity_error(
    u_sim: np.ndarray,
    u_analytical: np.ndarray,
    r: np.ndarray,
    radius: float,
    normalize: bool = True,
) -> float:
    """Compute L-infinity (max) norm of velocity error.

    Parameters
    ----------
    u_sim : np.ndarray
        Simulated velocity profile.
    u_analytical : np.ndarray
        Analytical velocity profile.
    r : np.ndarray
        Radial positions.
    radius : float
        Pipe radius.
    normalize : bool
        If True, normalize by max analytical velocity.

    Returns
    -------
    linf_error : float
        L-infinity error norm.
    """
    mask = r <= radius
    if not mask.any():
        return float('nan')

    u_s = u_sim[mask]
    u_a = u_analytical[mask]

    linf = float(np.max(np.abs(u_s - u_a)))

    if normalize:
        u_max = float(np.max(u_a))
        if u_max > 0:
            linf /= u_max

    return linf


def compute_flow_rate_error(
    Q_sim: float,
    Q_analytical: float,
) -> float:
    """Compute relative error in volumetric flow rate.

    Parameters
    ----------
    Q_sim : float
        Simulated flow rate.
    Q_analytical : float
        Analytical flow rate.

    Returns
    -------
    error : float
        Relative error |Q_sim - Q_analytical| / Q_analytical.
    """
    if Q_analytical == 0:
        return float('nan')
    return abs(Q_sim - Q_analytical) / abs(Q_analytical)


def compute_max_velocity_error(
    u_max_sim: float,
    u_max_analytical: float,
) -> float:
    """Compute relative error in maximum (centerline) velocity.

    Parameters
    ----------
    u_max_sim : float
        Simulated max velocity.
    u_max_analytical : float
        Analytical max velocity.

    Returns
    -------
    error : float
        Relative error |u_max_sim - u_max_analytical| / u_max_analytical.
    """
    if u_max_analytical == 0:
        return float('nan')
    return abs(u_max_sim - u_max_analytical) / abs(u_max_analytical)


def compute_convergence_rate(
    errors: list,
    grid_sizes: list,
) -> float:
    """Compute observed convergence rate from error series.

    For second-order convergence: error ~ O(h^2), so
    rate = log(e1/e2) / log(h1/h2) should be ~2.

    Parameters
    ----------
    errors : list of float
        Error values for each grid size.
    grid_sizes : list of int
        Corresponding grid sizes.

    Returns
    -------
    rate : float
        Observed convergence rate.
    """
    if len(errors) < 2 or len(grid_sizes) < 2:
        return float('nan')

    # Use last two points for rate estimate
    e1, e2 = errors[-2], errors[-1]
    h1, h2 = 1.0 / grid_sizes[-2], 1.0 / grid_sizes[-1]

    if e1 <= 0 or e2 <= 0:
        return float('nan')

    return float(np.log(e1 / e2) / np.log(h1 / h2))
