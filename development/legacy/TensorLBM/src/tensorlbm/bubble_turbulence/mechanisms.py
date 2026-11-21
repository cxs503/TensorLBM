"""Drag reduction mechanism decomposition for bubbly channel flow.

Decomposes total drag reduction into three contributions:
  1. Effective slip (ΔC_f,slip): bubble layer acts as a mobile boundary
  2. Turbulence modulation (ΔC_f,turb): suppression of Reynolds stress
  3. Viscosity modification (ΔC_f,visc): mixture viscosity change

Based on the FIK (Fukagata-Iwamoto-Kasagi) identity decomposition:
  C_f = (2/H) * integral_0^H [ (nu * dU/dy - <u'v'>) / U_b^2 ] dy

For bubbly flow, each term can be separated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


def compute_slip_length(
    U_plus: np.ndarray,
    y_plus: np.ndarray,
    u_tau: float,
    nu: float,
) -> float:
    """Compute effective slip length from the mean velocity profile.

    The slip length b is defined by extrapolating the mean velocity
    to zero at y = -b:
      U(y=0) = b * (dU/dy)|_{y=0}

    In wall units: b+ = U(0+) / (dU+/dy+)|_{y+=0}

    For a no-slip wall, b+ = 0.
    For a bubble layer, b+ > 0 (effective slip).

    Parameters
    ----------
    U_plus : array
        Mean velocity in wall units.
    y_plus : array
        Wall-normal coordinate in wall units.
    u_tau : float
        Friction velocity.
    nu : float
        Kinematic viscosity.

    Returns
    -------
    b_plus : float
        Effective slip length in wall units.
    """
    # Find the first few grid points near the wall
    # Use linear fit in the viscous sublayer (y+ < 5)
    mask = y_plus < 5.0
    if mask.sum() < 2:
        mask = y_plus < 10.0

    if mask.sum() < 2:
        return 0.0

    y_fit = y_plus[mask]
    u_fit = U_plus[mask]

    # Linear fit: U+ = a * y+ + c
    # Slope a = dU+/dy+ at wall
    # Intercept c = U+ at y+ = 0
    coeffs = np.polyfit(y_fit, u_fit, 1)
    slope = coeffs[0]
    intercept = coeffs[1]

    # Slip length: b+ = -intercept / slope
    # (if intercept < 0, the virtual origin is below the wall)
    if abs(slope) < 1e-10:
        return 0.0

    b_plus = -intercept / slope
    return max(b_plus, 0.0)  # slip length should be non-negative


def compute_friction_coefficient(
    U_mean: np.ndarray,
    y: np.ndarray,
    nu: float,
    u_tau: float,
    rho: float = 1.0,
) -> float:
    """Compute skin friction coefficient from velocity profile.

    C_f = tau_w / (0.5 * rho * U_b^2)

    where tau_w = mu * (dU/dy)|_{wall} and U_b is the bulk velocity.

    Parameters
    ----------
    U_mean : array
        Mean streamwise velocity profile.
    y : array
        Wall-normal coordinate.
    nu : float
        Kinematic viscosity.
    u_tau : float
        Friction velocity (for reference).
    rho : float
        Density.

    Returns
    -------
    C_f : float
        Skin friction coefficient.
    """
    # Wall shear stress from gradient at wall
    dU_dy_wall = (U_mean[1] - U_mean[0]) / (y[1] - y[0])
    tau_w = rho * nu * dU_dy_wall

    # Bulk velocity
    U_b = np.trapezoid(U_mean, y) / (y[-1] - y[0])

    if U_b < 1e-10:
        return 0.0

    C_f = tau_w / (0.5 * rho * U_b**2)
    return C_f


@dataclass
class DragReductionDecomposition:
    """Decomposition of drag reduction into three mechanisms.

    Attributes
    ----------
    total_DR : float
        Total drag reduction: (C_f0 - C_f) / C_f0
    DR_slip : float
        Contribution from effective slip.
    DR_turb : float
        Contribution from turbulence modulation.
    DR_visc : float
        Contribution from viscosity modification.
    residual : float
        Residual: total - (slip + turb + visc).
    """
    total_DR: float
    DR_slip: float
    DR_turb: float
    DR_visc: float
    residual: float

    def validate(self, tol: float = 0.03) -> bool:
        """Check that decomposition sums to total within tolerance."""
        return abs(self.residual / max(abs(self.total_DR), 1e-10)) < tol

    def to_dict(self) -> dict:
        return {
            "total_DR": self.total_DR,
            "DR_slip": self.DR_slip,
            "DR_turb": self.DR_turb,
            "DR_visc": self.DR_visc,
            "residual": self.residual,
            "residual_pct": abs(self.residual / max(abs(self.total_DR), 1e-10)) * 100,
        }


def decompose_drag_reduction(
    # Single-phase reference
    U_single: np.ndarray,
    uv_single: np.ndarray,
    y: np.ndarray,
    nu_single: float,
    u_tau_single: float,
    # Two-phase (bubbly)
    U_bubble: np.ndarray,
    uv_bubble: np.ndarray,
    nu_bubble: float,
    u_tau_bubble: float,
    # Bubble properties
    alpha_profile: Optional[np.ndarray] = None,
    slip_length: Optional[float] = None,
) -> DragReductionDecomposition:
    """Decompose drag reduction into three mechanisms.

    Uses the FIK identity approach:
      C_f = 2/H * int_0^H [nu * dU/dy - <u'v'>] / U_b^2 dy

    For bubbly flow:
      C_f,bubble = 2/H * int_0^H [nu_m * dU/dy - <u'v'>] / U_b^2 dy

    where nu_m = nu * (1 + alpha * (mu_g/mu_l - 1)) is the mixture viscosity.

    Decomposition:
      ΔC_f,slip = C_f,slip_only - C_f,0  (from effective slip boundary)
      ΔC_f,turb = C_f,turb_mod - C_f,slip  (from Reynolds stress change)
      ΔC_f,visc = C_f,bubble - C_f,turb_mod  (from viscosity change)

    Parameters
    ----------
    U_single, uv_single : arrays
        Single-phase mean velocity and Reynolds stress profiles.
    y : array
        Wall-normal coordinate.
    nu_single : float
        Single-phase kinematic viscosity.
    u_tau_single : float
        Single-phase friction velocity.
    U_bubble, uv_bubble : arrays
        Bubbly flow profiles.
    nu_bubble : float
        Effective (mixture) viscosity for bubbly flow.
    u_tau_bubble : float
        Bubbly flow friction velocity.
    alpha_profile : array, optional
        Bubble volume fraction profile.
    slip_length : float, optional
        Effective slip length in lattice units.

    Returns
    -------
    DragReductionDecomposition
    """
    H = y[-1] - y[0]
    dy = y[1] - y[0]

    # Bulk velocities
    U_b_single = np.trapezoid(U_single, y) / H
    U_b_bubble = np.trapezoid(U_bubble, y) / H

    # Reference friction coefficient (single phase)
    dU_dy_wall_single = (U_single[1] - U_single[0]) / dy
    tau_w_single = nu_single * dU_dy_wall_single
    C_f0 = 2.0 * tau_w_single / U_b_single**2

    # Bubbly friction coefficient
    dU_dy_wall_bubble = (U_bubble[1] - U_bubble[0]) / dy
    tau_w_bubble = nu_bubble * dU_dy_wall_bubble
    C_f_bubble = 2.0 * tau_w_bubble / U_b_bubble**2

    # Total drag reduction
    total_DR = (C_f0 - C_f_bubble) / C_f0 if C_f0 > 0 else 0.0

    # --- Mechanism 1: Effective slip ---
    # The slip reduces drag by shifting the velocity profile upward.
    # Estimate: ΔC_f,slip ≈ 2 * b+ / Re_tau (from Navier slip theory)
    if slip_length is not None and slip_length > 0:
        # Convert slip length to wall units
        b_plus = slip_length * u_tau_single / nu_single
        # Drag reduction from slip (first-order estimate)
        # For Navier slip: C_f/C_f0 ≈ 1 - 2*b+/Re_tau (laminar)
        # For turbulent: empirical factor ~0.5
        DR_slip = min(2.0 * b_plus / 180.0 * 0.5, 0.5)  # capped at 50%
    else:
        # Estimate from velocity shift
        b_plus = compute_slip_length(
            U_bubble / u_tau_bubble,
            y * u_tau_single / nu_single,
            u_tau_single, nu_single
        )
        DR_slip = min(2.0 * b_plus / 180.0 * 0.5, 0.5)

    # --- Mechanism 2: Turbulence modulation ---
    # Change in Reynolds stress contribution to drag
    # FIK: C_f,turb = -2/H * int <u'v'> / U_b^2 dy
    turb_single = -2.0 / H * np.trapezoid(uv_single, y) / U_b_single**2
    turb_bubble = -2.0 / H * np.trapezoid(uv_bubble, y) / U_b_bubble**2

    # DR from turbulence modulation
    DR_turb = (turb_single - turb_bubble) / C_f0 if C_f0 > 0 else 0.0

    # --- Mechanism 3: Viscosity modification ---
    # Change in molecular viscosity contribution
    visc_single = 2.0 / H * np.trapezoid(nu_single * np.gradient(U_single, y), y) / U_b_single**2
    visc_bubble = 2.0 / H * np.trapezoid(nu_bubble * np.gradient(U_bubble, y), y) / U_b_bubble**2

    DR_visc = (visc_single - visc_bubble) / C_f0 if C_f0 > 0 else 0.0

    # Residual
    residual = total_DR - (DR_slip + DR_turb + DR_visc)

    return DragReductionDecomposition(
        total_DR=total_DR,
        DR_slip=DR_slip,
        DR_turb=DR_turb,
        DR_visc=DR_visc,
        residual=residual,
    )


def compute_momentum_balance(
    U_mean: np.ndarray,
    uv: np.ndarray,
    y: np.ndarray,
    nu: float,
    body_force: float,
    u_tau: float,
) -> dict:
    """Verify momentum balance in the channel.

    Total stress = viscous + Reynolds = tau_w * (1 - y/H)

    Parameters
    ----------
    U_mean : array
        Mean velocity profile.
    uv : array
        Reynolds stress profile (<u'v'>).
    y : array
        Wall-normal coordinate.
    nu : float
        Kinematic viscosity.
    body_force : float
        Body force driving the flow.
    u_tau : float
        Friction velocity.

    Returns
    -------
    dict with stress profiles and balance residual.
    """
    H = y[-1] - y[0]
    dy = y[1] - y[0]

    # Viscous stress
    viscous = nu * np.gradient(U_mean, y)

    # Total stress (should be linear: tau_w * (1 - y/H))
    tau_w = u_tau**2
    total_stress_measured = viscous - uv  # -<u'v'> because uv = <u'v'>
    total_stress_theoretical = tau_w * (1.0 - (y - y[0]) / H)

    # Residual
    residual = total_stress_measured - total_stress_theoretical
    residual_rms = np.sqrt(np.mean(residual**2))

    return {
        "y": y,
        "viscous_stress": viscous,
        "reynolds_stress": -uv,
        "total_stress_measured": total_stress_measured,
        "total_stress_theoretical": total_stress_theoretical,
        "residual_rms": float(residual_rms),
        "residual_max": float(np.max(np.abs(residual))),
    }
