"""Sonic boom N-wave simulation using compressible LBM.

Implements:
- Supersonic wedge flow simulation
- N-wave pressure signature extraction
- Ackeret theory comparison

References
----------
- Ackeret, J. (1925). "Air forces on airfoils moving faster than sound"
- Bailey, J.R. (1986). Sonic boom N-wave experimental data
- Whitham, G.B. (1952). "The flow pattern of a supersonic projectile"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .naca0012 import (
    GAMMA, CS2,
    _d2q9_equilibrium_density,
    _d2q9_equilibrium_temperature,
    _macroscopic_ddf,
    _stream_ddf,
    _apply_bounce_back,
    _apply_riemann_farfield,
    _compute_artificial_viscosity,
)


# ============================================================================
# Supersonic wedge geometry
# ============================================================================

def create_wedge_mask(
    nx: int, ny: int,
    wedge_angle_deg: float,
    wedge_length_ratio: float = 0.5,
    wedge_y_center: float = 0.5
) -> torch.Tensor:
    """Create wall mask for supersonic wedge.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions.
    wedge_angle_deg : float
        Wedge half-angle in degrees.
    wedge_length_ratio : float
        Wedge length as fraction of domain.
    wedge_y_center : float
        Y-position of wedge center (fraction of domain height).

    Returns
    -------
    wall_mask : torch.Tensor
        Boolean mask (True = wall).
    """
    wall_mask = torch.zeros(ny, nx, dtype=torch.bool)

    wedge_angle_rad = math.radians(wedge_angle_deg)
    wedge_length = int(nx * wedge_length_ratio)
    y_center = ny * wedge_y_center  # float center

    tan_theta = math.tan(wedge_angle_rad)

    for i in range(wedge_length):
        # Upper and lower wedge boundaries at this x
        y_upper = y_center + i * tan_theta
        y_lower = y_center - i * tan_theta

        # Mark all cells between upper and lower boundaries
        for j in range(ny):
            if y_lower <= j <= y_upper:
                wall_mask[j, i] = True

    return wall_mask


# ============================================================================
# Sonic boom N-wave simulation
# ============================================================================

@dataclass
class SonicBoomResult:
    """Results from sonic boom simulation."""
    Ma: float
    wedge_angle_deg: float
    dp_sim: float  # simulated overpressure
    dp_ackeret: float  # Ackeret theory overpressure
    dp_error_pct: float  # error percentage
    n_wave_profile: Optional[np.ndarray]  # pressure signature
    pressure_field: Optional[np.ndarray]  # full pressure field
    shock_angle_sim: float  # simulated shock angle
    shock_angle_theory: float  # theoretical shock angle (Mach angle)


def simulate_sonic_boom(
    Ma: float = 2.0,
    wedge_angle_deg: float = 10.0,
    nx: int = 400,
    ny: int = 200,
    n_steps: int = 8000,
    omega_f: float = 0.8,
    omega_g: float = 0.8,
    c_h: float = 0.2,
    report_interval: int = 500,
) -> SonicBoomResult:
    """Simulate sonic boom N-wave from supersonic wedge.

    Parameters
    ----------
    Ma : float
        Freestream Mach number.
    wedge_angle_deg : float
        Wedge half-angle in degrees.
    nx, ny : int
        Grid resolution.
    n_steps : int
        Number of time steps.
    omega_f, omega_g : float
        Relaxation parameters.
    c_h : float
        Artificial viscosity coefficient.
    report_interval : int
        Reporting interval.

    Returns
    -------
    SonicBoomResult with simulation results.
    """
    device = torch.device("cpu")

    # Freestream conditions
    rho_inf = 1.0
    T_inf = 1.0
    cs_lattice = math.sqrt(CS2)
    u_inf = Ma * cs_lattice

    # Create wedge geometry
    wall_mask = create_wedge_mask(nx, ny, wedge_angle_deg)

    # Initialize
    rho = torch.full((ny, nx), rho_inf, device=device)
    ux = torch.full((ny, nx), u_inf, device=device)
    uy = torch.zeros(ny, nx, device=device)
    T = torch.full((ny, nx), T_inf, device=device)

    # Zero velocity inside wedge
    ux = torch.where(wall_mask, torch.tensor(0.0, device=device), ux)
    uy = torch.where(wall_mask, torch.tensor(0.0, device=device), uy)

    # Initialize distributions
    f = _d2q9_equilibrium_density(rho, ux, uy)
    g = _d2q9_equilibrium_temperature(T, ux, uy)
    f, g = _apply_bounce_back(f, g, wall_mask)

    # Pressure history at probe location (downstream of wedge)
    probe_x = int(0.7 * nx)
    probe_y = int(0.5 * ny)
    pressure_history = []

    # Simulation loop
    for step in range(n_steps):
        # Collision
        rho, ux, uy, T = _macroscopic_ddf(f, g)

        # Artificial viscosity
        mu_art = _compute_artificial_viscosity(ux, uy, rho, c_h)
        omega_eff_f = omega_f / (1.0 + mu_art)
        omega_eff_g = omega_g / (1.0 + mu_art)

        f_eq = _d2q9_equilibrium_density(rho, ux, uy)
        g_eq = _d2q9_equilibrium_temperature(T, ux, uy)

        f = f - omega_eff_f.unsqueeze(0) * (f - f_eq)
        g = g - omega_eff_g.unsqueeze(0) * (g - g_eq)

        # Streaming
        f, g = _stream_ddf(f, g)

        # Boundary conditions
        f, g = _apply_bounce_back(f, g, wall_mask)
        f, g = _apply_riemann_farfield(
            f, g, rho_inf, u_inf, 0.0, T_inf,
            nx, ny
        )

        # Record pressure at probe
        if (step + 1) % report_interval == 0:
            rho, ux, uy, T = _macroscopic_ddf(f, g)
            p_probe = (rho[probe_y, probe_x] * T[probe_y, probe_x]).item()
            pressure_history.append(p_probe)

    # Final state
    rho, ux, uy, T = _macroscopic_ddf(f, g)
    pressure_field = (rho * T).cpu().numpy()

    # Extract N-wave signature
    # Find shock position and measure overpressure
    n_wave_profile = _extract_n_wave(pressure_field, probe_y, nx)

    # Compute overpressure
    p_inf = rho_inf * T_inf
    dp_sim = max(pressure_history) - p_inf if pressure_history else 0.0

    # Ackeret theory
    dp_ackeret = _ackeret_overpressure(Ma, wedge_angle_deg, rho_inf, u_inf)

    # Shock angle
    mach_angle = math.asin(1.0 / Ma)
    shock_angle_sim = _estimate_shock_angle(pressure_field, wedge_angle_deg)

    return SonicBoomResult(
        Ma=Ma,
        wedge_angle_deg=wedge_angle_deg,
        dp_sim=dp_sim,
        dp_ackeret=dp_ackeret,
        dp_error_pct=abs(dp_sim - dp_ackeret) / dp_ackeret * 100 if dp_ackeret > 0 else 0.0,
        n_wave_profile=n_wave_profile,
        pressure_field=pressure_field,
        shock_angle_sim=shock_angle_sim,
        shock_angle_theory=mach_angle,
    )


def _extract_n_wave(pressure_field: np.ndarray, probe_y: int, nx: int) -> np.ndarray:
    """Extract N-wave pressure signature along horizontal line."""
    return pressure_field[probe_y, :]


def _ackeret_overpressure(Ma: float, theta_deg: float, rho: float, U: float) -> float:
    """Ackeret theory for sonic boom overpressure.

    Δp = ρ·U²·θ / √(Ma² - 1)
    """
    theta_rad = math.radians(theta_deg)
    return rho * U**2 * theta_rad / math.sqrt(Ma**2 - 1)


def _estimate_shock_angle(pressure_field: np.ndarray, wedge_angle_deg: float) -> float:
    """Estimate shock angle from pressure field gradient.

    Uses maximum pressure gradient to locate shock.
    """
    ny, nx = pressure_field.shape

    # Find shock position at mid-height
    y_mid = ny // 2
    dp_dx = np.diff(pressure_field[y_mid, :])
    shock_idx = np.argmax(dp_dx)

    # Estimate shock angle from geometry
    # For a wedge, shock angle β satisfies: tan(β) = y_shock / x_shock
    # This is a simplified estimate
    x_shock = shock_idx / nx
    y_shock = 0.5  # mid-height

    # Simple estimate: shock angle ≈ Mach angle for weak shocks
    # For stronger shocks, use oblique shock relations
    return math.atan(y_shock / max(x_shock, 0.01))


# ============================================================================
# Oblique shock relations for comparison
# ============================================================================

def oblique_shock_angle(Ma: float, theta_deg: float, gamma: float = GAMMA) -> float:
    """Compute oblique shock angle using θ-β-M relation (weak shock solution).

    Parameters
    ----------
    Ma : float
        Freestream Mach number.
    theta_deg : float
        Flow deflection angle (wedge angle) in degrees.
    gamma : float
        Specific heat ratio.

    Returns
    -------
    beta : float
        Shock angle in radians.
    """
    theta_rad = math.radians(theta_deg)

    # Check if deflection angle is possible
    # Maximum deflection angle for given Ma
    sin_mu = 1.0 / Ma
    mu = math.asin(sin_mu)  # Mach angle

    # Use iterative method with better initial guess
    # Start slightly above Mach angle
    beta = mu + 0.05

    for _ in range(200):
        sin_b = math.sin(beta)
        cos_b = math.cos(beta)

        if sin_b <= sin_mu:
            beta = mu + 0.05
            continue

        Ma_n_sq = Ma**2 * sin_b**2

        # θ-β-M relation: tan(θ) = 2·cot(β)·(Ma²·sin²(β) - 1) / (Ma²·(γ + cos(2β)) + 2)
        cot_b = cos_b / sin_b
        num = 2.0 * cot_b * (Ma_n_sq - 1.0)
        den = Ma**2 * (gamma + math.cos(2.0 * beta)) + 2.0

        if den <= 1e-10:
            beta += 0.01
            continue

        theta_calc = math.atan(num / den)
        f = theta_calc - theta_rad

        if abs(f) < 1e-10:
            break

        # Numerical derivative
        db = 1e-8
        beta_d = beta + db
        sin_bd = math.sin(beta_d)
        cos_bd = math.cos(beta_d)
        if sin_bd <= sin_mu:
            db = -db
            beta_d = beta + db
            sin_bd = math.sin(beta_d)
            cos_bd = math.cos(beta_d)

        Ma_n_sq_d = Ma**2 * sin_bd**2
        cot_bd = cos_bd / sin_bd
        num_d = 2.0 * cot_bd * (Ma_n_sq_d - 1.0)
        den_d = Ma**2 * (gamma + math.cos(2.0 * beta_d)) + 2.0
        theta_d = math.atan(num_d / den_d) if den_d > 1e-10 else theta_calc
        df = (theta_d - theta_calc) / db

        if abs(df) < 1e-15:
            beta += 0.01
            continue

        beta_new = beta - f / df

        # Ensure we stay in weak shock regime (beta close to Mach angle)
        if beta_new < mu + 0.001:
            beta_new = mu + 0.001
        if beta_new > math.pi / 2 - 0.001:
            beta_new = math.pi / 2 - 0.001

        beta = beta_new

    return beta


def oblique_shock_pressure_ratio(Ma: float, theta_deg: float, gamma: float = GAMMA) -> float:
    """Compute pressure ratio across oblique shock."""
    beta = oblique_shock_angle(Ma, theta_deg, gamma)
    Ma_n = Ma * math.sin(beta)

    # Normal shock relation for normal component
    p_ratio = 1 + 2 * gamma / (gamma + 1) * (Ma_n**2 - 1)
    return p_ratio
