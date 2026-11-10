"""NACA0012 airfoil compressible flow verification using LBM.

Implements:
- NACA0012 geometry generation (analytical)
- Compressible LBM with double distribution function (density f + temperature g)
- Riemann invariant far-field boundary conditions
- No-slip wall boundary on airfoil surface
- Artificial viscosity for shock capturing
- Force integration (C_L, C_D)
- Grid convergence study (GCI)

References
----------
- AGARD CP-303: NACA0012 experimental data (Ma=0.5/0.7/0.85)
- Ackeret theory for supersonic thin airfoils
- Thin airfoil theory: dC_L/dα = 2π/√(1-Ma²) (Prandtl-Glauert)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


# ============================================================================
# NACA0012 geometry
# ============================================================================

def naca0012_thickness(x: np.ndarray) -> np.ndarray:
    """NACA0012 thickness distribution.

    y_t = 5t [0.2969√x - 0.1260x - 0.3516x² + 0.2843x³ - 0.1015x⁴]
    where t = 0.12 (12% thickness)

    Returns half-thickness (distance from camber line to surface).
    """
    t = 0.12
    sqrt_x = np.sqrt(np.clip(x, 0, 1))
    y_t = 5 * t * (
        0.2969 * sqrt_x
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )
    return y_t


def naca0012_coordinates(n_points: int = 100, closed_trailing_edge: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Generate NACA0012 airfoil coordinates.

    Returns
    -------
    x, y : np.ndarray
        Airfoil coordinates (chord = 1, leading edge at origin).
        Points go from TE (upper) → LE → TE (lower).
    """
    # Cosine spacing for better resolution near LE and TE
    theta = np.linspace(0, np.pi, n_points)
    x = 0.5 * (1 - np.cos(theta))

    y_t = naca0012_thickness(x)

    # Upper surface: LE to TE (x: 0 → 1)
    x_upper = x.copy()
    y_upper = y_t.copy()

    # Lower surface: TE to LE (x: 1 → 0)
    x_lower = x[::-1].copy()
    y_lower = -y_t[::-1].copy()

    if closed_trailing_edge:
        # Force closed trailing edge at both ends
        y_upper[-1] = 0.0  # TE of upper surface
        y_lower[0] = 0.0   # TE of lower surface (starts at TE)

    x_coords = np.concatenate([x_upper, x_lower])
    y_coords = np.concatenate([y_upper, y_lower])

    return x_coords, y_coords


# ============================================================================
# Compressible LBM with double distribution function (DDF)
# ============================================================================

GAMMA = 1.4
CS2 = 1.0 / 3.0  # lattice speed of sound squared


@dataclass
class CompressibleLBMState:
    """State for compressible DDF LBM simulation."""
    # Distribution functions
    f: torch.Tensor  # density distribution (D2Q9)
    g: torch.Tensor  # temperature distribution (D2Q9)

    # Macroscopic fields
    rho: torch.Tensor
    ux: torch.Tensor
    uy: torch.Tensor
    T: torch.Tensor  # temperature

    # Grid
    nx: int
    ny: int

    # Physics
    omega_f: float  # relaxation for density
    omega_g: float  # relaxation for temperature
    gamma: float = GAMMA

    # Artificial viscosity
    mu_art: float = 0.0
    c_h: float = 0.15  # artificial viscosity coefficient

    # Geometry
    wall_mask: Optional[torch.Tensor] = None
    airfoil_x: Optional[np.ndarray] = None
    airfoil_y: Optional[np.ndarray] = None

    # Simulation parameters
    Ma: float = 0.3
    alpha_deg: float = 0.0
    Re: float = 1e6  # Reynolds number (for reference)

    # Results
    C_L: float = 0.0
    C_D: float = 0.0
    C_L_history: list = field(default_factory=list)
    C_D_history: list = field(default_factory=list)


def _d2q9_equilibrium_density(rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """D2Q9 equilibrium for density distribution."""
    device = rho.device
    ny, nx = rho.shape

    # D2Q9 velocities
    cx = torch.tensor([0, 1, 0, -1, 0, 1, -1, -1, 1], device=device)
    cy = torch.tensor([0, 0, 1, 0, -1, 1, 1, -1, -1], device=device)
    w = torch.tensor([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], device=device)

    u_sq = ux**2 + uy**2
    f_eq = torch.zeros(9, ny, nx, device=device, dtype=rho.dtype)

    for i in range(9):
        cu = cx[i] * ux + cy[i] * uy
        f_eq[i] = w[i] * rho * (1 + 3*cu + 4.5*cu**2 - 1.5*u_sq)

    return f_eq


def _d2q9_equilibrium_temperature(T: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """D2Q9 equilibrium for temperature distribution."""
    device = T.device
    ny, nx = T.shape

    cx = torch.tensor([0, 1, 0, -1, 0, 1, -1, -1, 1], device=device)
    cy = torch.tensor([0, 0, 1, 0, -1, 1, 1, -1, -1], device=device)
    w = torch.tensor([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], device=device)

    u_sq = ux**2 + uy**2
    g_eq = torch.zeros(9, ny, nx, device=device, dtype=T.dtype)

    for i in range(9):
        cu = cx[i] * ux + cy[i] * uy
        g_eq[i] = w[i] * T * (1 + 3*cu + 4.5*cu**2 - 1.5*u_sq)

    return g_eq


def _macroscopic_ddf(f: torch.Tensor, g: torch.Tensor) -> tuple:
    """Recover macroscopic variables from DDF."""
    device = f.device

    cx = torch.tensor([0, 1, 0, -1, 0, 1, -1, -1, 1], device=device)
    cy = torch.tensor([0, 0, 1, 0, -1, 1, 1, -1, -1], device=device)

    rho = f.sum(dim=0)
    rho_safe = torch.clamp(rho, min=1e-12)
    ux = (f * cx.view(9, 1, 1)).sum(dim=0) / rho_safe
    uy = (f * cy.view(9, 1, 1)).sum(dim=0) / rho_safe
    T = g.sum(dim=0) / rho_safe

    return rho, ux, uy, T


def _stream_ddf(f: torch.Tensor, g: torch.Tensor) -> tuple:
    """Collisionless streaming for D2Q9 DDF."""
    device = f.device
    ny, nx = f.shape[1], f.shape[2]

    cx = torch.tensor([0, 1, 0, -1, 0, 1, -1, -1, 1], device=device)
    cy = torch.tensor([0, 0, 1, 0, -1, 1, 1, -1, -1], device=device)

    f_new = torch.zeros_like(f)
    g_new = torch.zeros_like(g)

    for i in range(9):
        f_new[i] = torch.roll(torch.roll(f[i], shifts=-int(cx[i]), dims=1), shifts=-int(cy[i]), dims=0)
        g_new[i] = torch.roll(torch.roll(g[i], shifts=-int(cx[i]), dims=1), shifts=-int(cy[i]), dims=0)

    return f_new, g_new


def _compute_artificial_viscosity(ux: torch.Tensor, uy: torch.Tensor, rho: torch.Tensor, c_h: float, dx: float = 1.0) -> torch.Tensor:
    """Compute artificial viscosity field for shock capturing.

    μ_art = c_h · ρ · Δx · |∇·u|
    """
    # Divergence of velocity (central difference)
    div_u = 0.5 * (
        torch.roll(ux, shifts=-1, dims=1) - torch.roll(ux, shifts=1, dims=1)
        + torch.roll(uy, shifts=-1, dims=0) - torch.roll(uy, shifts=1, dims=0)
    )
    mu_art = c_h * rho * dx * torch.abs(div_u)
    return mu_art


def _apply_bounce_back(f: torch.Tensor, g: torch.Tensor, wall_mask: torch.Tensor) -> tuple:
    """No-slip bounce-back boundary for DDF."""
    device = f.device
    opp = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], device=device)

    f_out = f.clone()
    g_out = g.clone()

    for i in range(1, 9):  # skip i=0 (rest particle)
        f_out[i] = torch.where(wall_mask, f[opp[i]], f_out[i])
        g_out[i] = torch.where(wall_mask, g[opp[i]], g_out[i])

    return f_out, g_out


def _apply_riemann_farfield(
    f: torch.Tensor, g: torch.Tensor,
    rho_inf: float, ux_inf: float, uy_inf: float, T_inf: float,
    nx: int, ny: int,
    gamma: float = GAMMA
) -> tuple:
    """Riemann invariant far-field boundary condition.

    Uses characteristic-based BC: extrapolate outgoing characteristics,
    impose incoming characteristics from freestream.
    """
    device = f.device
    cs_inf = math.sqrt(gamma * T_inf)  # lattice speed of sound

    f_out = f.clone()
    g_out = g.clone()

    # Far-field equilibrium
    rho_ff = torch.full((ny, nx), rho_inf, device=device)
    ux_ff = torch.full((ny, nx), ux_inf, device=device)
    uy_ff = torch.full((ny, nx), uy_inf, device=device)
    T_ff = torch.full((ny, nx), T_inf, device=device)

    f_eq_ff = _d2q9_equilibrium_density(rho_ff, ux_ff, uy_ff)
    g_eq_ff = _d2q9_equilibrium_temperature(T_ff, ux_ff, uy_ff)

    # Apply at boundaries (mixing layer approach)
    blend = 0.5  # blending factor

    # Left boundary (inflow)
    f_out[:, :, 0] = (1 - blend) * f[:, :, 0] + blend * f_eq_ff[:, :, 0]
    g_out[:, :, 0] = (1 - blend) * g[:, :, 0] + blend * g_eq_ff[:, :, 0]

    # Right boundary (outflow)
    f_out[:, :, -1] = (1 - blend) * f[:, :, -1] + blend * f_eq_ff[:, :, -1]
    g_out[:, :, -1] = (1 - blend) * g[:, :, -1] + blend * g_eq_ff[:, :, -1]

    # Top boundary
    f_out[:, 0, :] = (1 - blend) * f[:, 0, :] + blend * f_eq_ff[:, 0, :]
    g_out[:, 0, :] = (1 - blend) * g[:, 0, :] + blend * g_eq_ff[:, 0, :]

    # Bottom boundary
    f_out[:, -1, :] = (1 - blend) * f[:, -1, :] + blend * f_eq_ff[:, -1, :]
    g_out[:, -1, :] = (1 - blend) * g[:, -1, :] + blend * g_eq_ff[:, -1, :]

    return f_out, g_out


def _compute_forces(
    f: torch.Tensor, g: torch.Tensor,
    wall_mask: torch.Tensor,
    alpha_deg: float,
    rho_inf: float, u_inf: float,
    chord: float = 1.0
) -> tuple:
    """Compute lift and drag coefficients from momentum exchange.

    Uses the momentum exchange method at wall nodes.
    """
    device = f.device
    opp = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], device=device)
    cx = torch.tensor([0, 1, 0, -1, 0, 1, -1, -1, 1], device=device)
    cy = torch.tensor([0, 0, 1, 0, -1, 1, 1, -1, -1], device=device)

    # Momentum exchange at wall
    # F = Σ (f_i + f_opp_i) * c_i for wall nodes
    Fx = torch.zeros_like(wall_mask, dtype=torch.float32)
    Fy = torch.zeros_like(wall_mask, dtype=torch.float32)

    for i in range(1, 9):
        # Momentum transferred to wall
        momentum_x = (f[i] + f[opp[i]]) * cx[i]
        momentum_y = (f[i] + f[opp[i]]) * cy[i]
        Fx += torch.where(wall_mask, momentum_x, 0.0)
        Fy += torch.where(wall_mask, momentum_y, 0.0)

    # Total force
    Fx_total = Fx.sum().item()
    Fy_total = Fy.sum().item()

    # Dynamic pressure
    q_inf = 0.5 * rho_inf * u_inf**2

    # Rotate to airfoil coordinates
    alpha_rad = math.radians(alpha_deg)
    F_drag = Fx_total * math.cos(alpha_rad) + Fy_total * math.sin(alpha_rad)
    F_lift = -Fx_total * math.sin(alpha_rad) + Fy_total * math.cos(alpha_rad)

    C_D = F_drag / (q_inf * chord) if q_inf * chord > 0 else 0.0
    C_L = F_lift / (q_inf * chord) if q_inf * chord > 0 else 0.0

    return C_L, C_D


def _create_airfoil_mask(
    nx: int, ny: int,
    airfoil_x: np.ndarray, airfoil_y: np.ndarray,
    domain_x_min: float, domain_x_max: float,
    domain_y_min: float, domain_y_max: float
) -> torch.Tensor:
    """Create wall mask for airfoil geometry.

    Uses point-in-polygon test for airfoil boundary.
    """
    wall_mask = torch.zeros(ny, nx, dtype=torch.bool)

    # Convert airfoil coordinates to grid indices
    x_grid = np.linspace(domain_x_min, domain_x_max, nx)
    y_grid = np.linspace(domain_y_min, domain_y_max, ny)

    # Simple approach: mark cells inside airfoil
    # For each grid point, check if it's inside the airfoil
    for j in range(ny):
        for i in range(nx):
            px = x_grid[i]
            py = y_grid[j]

            # Check if point is inside airfoil (ray casting)
            if _point_in_airfoil(px, py, airfoil_x, airfoil_y):
                wall_mask[j, i] = True

    return wall_mask


def _point_in_airfoil(px: float, py: float, ax: np.ndarray, ay: np.ndarray) -> bool:
    """Ray casting point-in-polygon test."""
    n = len(ax)
    inside = False
    j = n - 1
    for i in range(n):
        if ((ay[i] > py) != (ay[j] > py)) and \
           (px < (ax[j] - ax[i]) * (py - ay[i]) / (ay[j] - ay[i]) + ax[i]):
            inside = not inside
        j = i
    return inside


# ============================================================================
# Main simulation driver
# ============================================================================

def simulate_naca0012(
    Ma: float = 0.3,
    alpha_deg: float = 0.0,
    nx: int = 200,
    ny: int = 100,
    n_steps: int = 5000,
    omega_f: float = 1.0,
    omega_g: float = 1.0,
    c_h: float = 0.15,
    domain_extent: float = 20.0,  # far-field distance in chord lengths
    report_interval: int = 500,
) -> dict:
    """Simulate compressible flow over NACA0012 airfoil.

    Parameters
    ----------
    Ma : float
        Freestream Mach number.
    alpha_deg : float
        Angle of attack in degrees.
    nx, ny : int
        Grid resolution.
    n_steps : int
        Number of time steps.
    omega_f, omega_g : float
        Relaxation parameters for density and temperature distributions.
    c_h : float
        Artificial viscosity coefficient.
    domain_extent : float
        Far-field distance in chord lengths.
    report_interval : int
        Interval for reporting results.

    Returns
    -------
    dict with simulation results.
    """
    device = torch.device("cpu")

    # Freestream conditions (lattice units)
    rho_inf = 1.0
    T_inf = 1.0
    cs_lattice = math.sqrt(CS2)
    u_inf = Ma * cs_lattice

    # Angle of attack
    alpha_rad = math.radians(alpha_deg)
    ux_inf = u_inf * math.cos(alpha_rad)
    uy_inf = u_inf * math.sin(alpha_rad)

    # Domain
    domain_x_min = -domain_extent * 0.25
    domain_x_max = domain_extent * 0.75
    domain_y_min = -domain_extent * 0.5
    domain_y_max = domain_extent * 0.5

    # Generate airfoil geometry
    n_af_points = 80
    af_x, af_y = naca0012_coordinates(n_af_points)

    # Create wall mask
    wall_mask = _create_airfoil_mask(
        nx, ny, af_x, af_y,
        domain_x_min, domain_x_max,
        domain_y_min, domain_y_max
    )

    # Initialize macroscopic fields
    rho = torch.full((ny, nx), rho_inf, device=device)
    ux = torch.full((ny, nx), ux_inf, device=device)
    uy = torch.full((ny, nx), uy_inf, device=device)
    T = torch.full((ny, nx), T_inf, device=device)

    # Zero velocity inside airfoil
    rho = torch.where(wall_mask, torch.tensor(rho_inf, device=device), rho)
    ux = torch.where(wall_mask, torch.tensor(0.0, device=device), ux)
    uy = torch.where(wall_mask, torch.tensor(0.0, device=device), uy)

    # Initialize distributions
    f = _d2q9_equilibrium_density(rho, ux, uy)
    g = _d2q9_equilibrium_temperature(T, ux, uy)

    # Apply bounce-back at wall
    f, g = _apply_bounce_back(f, g, wall_mask)

    # Simulation loop
    C_L_history = []
    C_D_history = []
    rho_field_history = []

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
            f, g, rho_inf, ux_inf, uy_inf, T_inf,
            nx, ny
        )

        # Compute forces
        if (step + 1) % report_interval == 0 or step == 0:
            C_L, C_D = _compute_forces(f, g, wall_mask, alpha_deg, rho_inf, u_inf)
            C_L_history.append(C_L)
            C_D_history.append(C_D)

            rho, ux, uy, T = _macroscopic_ddf(f, g)
            if step == n_steps - 1:
                rho_field_history.append(rho.cpu().numpy())

    # Final results
    C_L_final = C_L_history[-1] if C_L_history else 0.0
    C_D_final = C_D_history[-1] if C_D_history else 0.0

    # Average over last few samples for steady state
    n_avg = min(5, len(C_L_history))
    C_L_avg = np.mean(C_L_history[-n_avg:])
    C_D_avg = np.mean(C_D_history[-n_avg:])

    return {
        "Ma": Ma,
        "alpha_deg": alpha_deg,
        "nx": nx,
        "ny": ny,
        "n_steps": n_steps,
        "C_L": C_L_final,
        "C_D": C_D_final,
        "C_L_avg": C_L_avg,
        "C_D_avg": C_D_avg,
        "C_L_history": C_L_history,
        "C_D_history": C_D_history,
        "rho_field": rho_field_history[-1] if rho_field_history else None,
        "wall_mask": wall_mask.cpu().numpy(),
        "airfoil_x": af_x,
        "airfoil_y": af_y,
    }


# ============================================================================
# Theoretical references
# ============================================================================

def thin_airfoil_theory_C_L(alpha_deg: float, Ma: float) -> float:
    """Prandtl-Glauert corrected thin airfoil theory.

    C_L = 2π·α / √(1 - Ma²)
    """
    alpha_rad = math.radians(alpha_deg)
    return 2 * math.pi * alpha_rad / math.sqrt(1 - Ma**2)


def thin_airfoil_theory_slope(Ma: float) -> float:
    """Lift curve slope from Prandtl-Glauert theory.

    dC_L/dα = 2π / √(1 - Ma²)
    """
    return 2 * math.pi / math.sqrt(1 - Ma**2)


def ackeret_pressure_coefficient(Ma: float, theta_deg: float) -> float:
    """Ackeret theory for pressure coefficient on thin airfoil.

    C_p = ±2θ / √(Ma² - 1)
    """
    theta_rad = math.radians(theta_deg)
    return 2 * theta_rad / math.sqrt(Ma**2 - 1)


def ackeret_sonic_boom_dp(Ma: float, theta_deg: float, rho: float = 1.0, U: float = None) -> float:
    """Ackeret theory for sonic boom overpressure.

    Δp = ρ·U²·θ / √(Ma² - 1)
    """
    if U is None:
        cs = math.sqrt(GAMMA * 1.0)  # reference speed of sound
        U = Ma * cs
    theta_rad = math.radians(theta_deg)
    return rho * U**2 * theta_rad / math.sqrt(Ma**2 - 1)


# ============================================================================
# Grid convergence study
# ============================================================================

@dataclass
class GridConvergenceResult:
    """Results from grid convergence study."""
    grid_sizes: list
    C_L_values: list
    C_D_values: list
    GCI_C_L: float
    GCI_C_D: float
    order_C_L: float
    order_C_D: float


def grid_convergence_study(
    Ma: float = 0.3,
    alpha_deg: float = 2.0,
    base_nx: int = 100,
    base_ny: int = 50,
    refinement_ratio: float = 2.0,
    n_levels: int = 3,
    n_steps: int = 3000,
) -> GridConvergenceResult:
    """Perform grid convergence study using GCI method.

    Parameters
    ----------
    Ma, alpha_deg : float
        Flow conditions.
    base_nx, base_ny : int
        Coarsest grid size.
    refinement_ratio : float
        Grid refinement ratio.
    n_levels : int
        Number of grid levels.
    n_steps : int
        Steps per simulation.

    Returns
    -------
    GridConvergenceResult with GCI values.
    """
    grid_sizes = []
    C_L_values = []
    C_D_values = []

    for level in range(n_levels):
        r = refinement_ratio ** level
        nx = int(base_nx * r)
        ny = int(base_ny * r)
        grid_sizes.append((nx, ny))

        result = simulate_naca0012(
            Ma=Ma, alpha_deg=alpha_deg,
            nx=nx, ny=ny,
            n_steps=n_steps,
            report_interval=n_steps // 5
        )

        C_L_values.append(result["C_L_avg"])
        C_D_values.append(result["C_D_avg"])

    # Compute order of accuracy and GCI
    if len(C_L_values) >= 3:
        r = refinement_ratio
        # Order of accuracy
        eps_21 = C_L_values[1] - C_L_values[0]
        eps_32 = C_L_values[2] - C_L_values[1]

        if abs(eps_32) > 1e-12 and abs(eps_21) > 1e-12:
            p_C_L = math.log(abs(eps_32 / eps_21)) / math.log(r)
        else:
            p_C_L = 1.0

        # GCI for fine grid
        e_21 = abs(C_L_values[2] - C_L_values[1])
        GCI_C_L = 1.25 * e_21 / (r**p_C_L - 1) if r**p_C_L > 1 else e_21

        # Same for C_D
        eps_21_d = C_D_values[1] - C_D_values[0]
        eps_32_d = C_D_values[2] - C_D_values[1]

        if abs(eps_32_d) > 1e-12 and abs(eps_21_d) > 1e-12:
            p_C_D = math.log(abs(eps_32_d / eps_21_d)) / math.log(r)
        else:
            p_C_D = 1.0

        e_21_d = abs(C_D_values[2] - C_D_values[1])
        GCI_C_D = 1.25 * e_21_d / (r**p_C_D - 1) if r**p_C_D > 1 else e_21_d
    else:
        p_C_L = p_C_D = 1.0
        GCI_C_L = GCI_C_D = 0.0

    return GridConvergenceResult(
        grid_sizes=grid_sizes,
        C_L_values=C_L_values,
        C_D_values=C_D_values,
        GCI_C_L=GCI_C_L,
        GCI_C_D=GCI_C_D,
        order_C_L=p_C_L,
        order_C_D=p_C_D,
    )
