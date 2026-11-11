"""Tip vortex cavitation inception: Lamb-Oseen pressure anchor × LBM resolution × McCormick scaling.

Mission P284W3-N05: validate the pressure-based cavitation inception criterion
for tip vortices using:

(a) Lamb-Oseen / Rankine vortex core pressure drop analytical anchor
(b) LBM-based vortex resolution (D2Q9 BGK/MRT)
(c) McCormick (1962) empirical inception scaling

Physics
-------
Tip vortex cavitation occurs when the minimum pressure in the vortex core
drops below the vapour pressure. The inception cavitation number is:

    σ_i = (p_∞ - p_v) / (0.5 ρ V²)

evaluated at the condition p_min = p_v.

For a Lamb-Oseen vortex with circulation Γ and core radius r_c:
    u_θ(r) = Γ/(2πr) × (1 - exp(-r²/r_c²))

The pressure field follows from radial equilibrium dp/dr = ρ u_θ²/r:
    Δp = p(∞) - p(0) = ρΓ²/(4π²r_c²) × ln(2)   [Lamb-Oseen]
    Δp = ρΓ²/(4π²r_c²)                           [Rankine]

The Lamb-Oseen result includes the factor ln(2) ≈ 0.693 from the smooth
vorticity distribution, while the Rankine vortex (sharp core boundary)
gives the upper bound.

McCormick (1962) scaling for tip vortex cavitation inception:
    σ_i ∝ (Γ/(V c))^{2/3} × (Re)^{-1/6}

where Γ is the tip vortex circulation, V is the inflow velocity, c is the
chord, and Re is the chord-based Reynolds number.

References
----------
Lamb, H. (1932) Hydrodynamics, 6th ed., Cambridge Univ. Press.
Oseen, C. W. (1911) "Über Wirbelbewegung in einer reibenden Flüssigkeit",
    Arkiv för Matematik, Astronomi och Fysik, 7(11):1-34.
McCormick, B. W. (1962) "On Cavitation Inception from a Tip Vortex",
    J. Ship Research, 6(1-2):16-26.
Saffman, P. G. (1992) Vortex Dynamics, Cambridge Univ. Press.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

__all__ = [
    # Analytical vortex models
    "lamb_oseen_velocity",
    "lamb_oseen_pressure",
    "rankine_velocity",
    "rankine_pressure",
    "lamb_oseen_core_pressure_drop",
    "rankine_core_pressure_drop",
    # LBM vortex simulation
    "LambOseenLBMConfig",
    "init_lamb_oseen_lbm",
    "run_lamb_oseen_lbm",
    "compute_pressure_from_velocity",
    # Tip vortex model
    "TipVortexConfig",
    "tip_vortex_trajectory",
    "tip_vortex_circulation_decay",
    # Cavitation inception
    "cavitation_inception_number",
    "mccormick_inception_scaling",
    "inception_sigma_from_pmin",
]


# =============================================================================
# Analytical vortex models
# =============================================================================


def lamb_oseen_velocity(
    r: np.ndarray,
    gamma: float,
    r_c: float,
) -> np.ndarray:
    """Lamb-Oseen vortex tangential velocity profile.

    u_θ(r) = Γ/(2πr) × (1 - exp(-r²/r_c²))

    Args:
        r: Radial distance array (must be > 0 to avoid singularity).
        gamma: Circulation Γ (positive = counterclockwise).
        r_c: Vortex core radius.

    Returns:
        Tangential velocity u_θ at each r.

    Notes:
        At r → 0: u_θ → Γ r / (2π r_c²) (solid-body rotation limit).
        At r → ∞: u_θ → Γ / (2πr) (potential vortex limit).
    """
    r = np.asarray(r, dtype=np.float64)
    # Handle r=0 gracefully (limit is 0)
    u_theta = np.zeros_like(r)
    mask = r > 1e-15
    r_safe = r[mask]
    u_theta[mask] = (gamma / (2.0 * np.pi * r_safe)) * (
        1.0 - np.exp(-(r_safe**2) / r_c**2)
    )
    return u_theta


def lamb_oseen_pressure(
    r: np.ndarray,
    gamma: float,
    r_c: float,
    rho: float = 1.0,
    p_inf: float = 0.0,
) -> np.ndarray:
    """Lamb-Oseen vortex pressure field from radial equilibrium.

    dp/dr = ρ u_θ²/r, integrated from r to ∞:
    p(r) = p_∞ - ρ ∫_r^∞ u_θ²(r')/r' dr'

    The integral has a closed form:
    ∫_r^∞ u_θ²/r' dr' = Γ²/(4π²) × [ln(2)/r_c² - E₁(r²/r_c²)/(2r_c²)
                        + (1-exp(-r²/r_c²))²/(2r²) + ...]

    For numerical accuracy, we compute the integral directly.

    Args:
        r: Radial distance array.
        gamma: Circulation Γ.
        r_c: Vortex core radius.
        rho: Fluid density.
        p_inf: Pressure at infinity.

    Returns:
        Pressure p(r) at each radial location.
    """
    r = np.asarray(r, dtype=np.float64)
    p = np.full_like(r, p_inf)

    for i, ri in np.ndenumerate(r):
        if ri < 1e-15:
            # p(0) = p_inf - Δp_total
            dp_total = lamb_oseen_core_pressure_drop(gamma, r_c, rho)
            p[i] = p_inf - dp_total
        else:
            # Numerical integration from ri to R_max
            r_max = 20.0 * r_c  # sufficient for convergence
            n_pts = 10000
            r_int = np.linspace(ri, r_max, n_pts)
            u_th = lamb_oseen_velocity(r_int, gamma, r_c)
            integrand = u_th**2 / r_int
            dp = rho * np.trapezoid(integrand, r_int)
            p[i] = p_inf - dp

    return p


def lamb_oseen_core_pressure_drop(
    gamma: float,
    r_c: float,
    rho: float = 1.0,
) -> float:
    """Analytical pressure drop at the center of a Lamb-Oseen vortex.

    Δp = p(∞) - p(0) = ρ Γ² ln(2) / (4 π² r_c²)

    Derivation:
    Δp = ρ ∫₀^∞ u_θ²/r dr = ρ Γ²/(4π²) ∫₀^∞ (1-exp(-r²/r_c²))²/r³ dr

    Substituting x = r²/r_c²:
    = ρ Γ²/(8π² r_c²) ∫₀^∞ (1-exp(-x))²/x² dx

    The integral ∫₀^∞ (1-e^{-x})²/x² dx = 2 ln 2 (Frullani integral).

    Therefore: Δp = ρ Γ² ln(2) / (4 π² r_c²)

    Args:
        gamma: Circulation Γ.
        r_c: Vortex core radius.
        rho: Fluid density.

    Returns:
        Pressure drop Δp = p(∞) - p(0) (positive value).
    """
    return rho * gamma**2 * math.log(2.0) / (4.0 * math.pi**2 * r_c**2)


def rankine_velocity(
    r: np.ndarray,
    gamma: float,
    r_c: float,
) -> np.ndarray:
    """Rankine combined vortex tangential velocity.

    u_θ(r) = Γ r / (2π r_c²)    for r ≤ r_c  (solid-body rotation)
    u_θ(r) = Γ / (2π r)         for r > r_c   (potential vortex)

    Args:
        r: Radial distance array.
        gamma: Circulation Γ.
        r_c: Vortex core radius.

    Returns:
        Tangential velocity u_θ at each r.
    """
    r = np.asarray(r, dtype=np.float64)
    u_theta = np.where(
        r <= r_c,
        gamma * r / (2.0 * np.pi * r_c**2),
        gamma / (2.0 * np.pi * np.maximum(r, 1e-15)),
    )
    return u_theta


def rankine_pressure(
    r: np.ndarray,
    gamma: float,
    r_c: float,
    rho: float = 1.0,
    p_inf: float = 0.0,
) -> np.ndarray:
    """Rankine vortex pressure field.

    Inside core (r ≤ r_c): p(r) = p(0) + ρ Γ² r² / (8 π² r_c⁴)
    Outside (r > r_c):     p(r) = p_∞ - ρ Γ² / (8 π² r²)

    Args:
        r: Radial distance array.
        gamma: Circulation Γ.
        r_c: Vortex core radius.
        rho: Fluid density.
        p_inf: Pressure at infinity.

    Returns:
        Pressure p(r) at each radial location.
    """
    r = np.asarray(r, dtype=np.float64)
    dp_total = rankine_core_pressure_drop(gamma, r_c, rho)
    p0 = p_inf - dp_total

    p = np.where(
        r <= r_c,
        p0 + rho * gamma**2 * r**2 / (8.0 * np.pi**2 * r_c**4),
        p_inf - rho * gamma**2 / (8.0 * np.pi**2 * np.maximum(r, 1e-15) ** 2),
    )
    return p


def rankine_core_pressure_drop(
    gamma: float,
    r_c: float,
    rho: float = 1.0,
) -> float:
    """Analytical pressure drop at center of Rankine vortex.

    Δp = p(∞) - p(0) = ρ Γ² / (4 π² r_c²)

    This is the sum of:
    - Inside contribution: ρ Γ² / (8 π² r_c²)
    - Outside contribution: ρ Γ² / (8 π² r_c²)

    Args:
        gamma: Circulation Γ.
        r_c: Vortex core radius.
        rho: Fluid density.

    Returns:
        Pressure drop Δp = p(∞) - p(0) (positive value).
    """
    return rho * gamma**2 / (4.0 * math.pi**2 * r_c**2)


# =============================================================================
# LBM-based Lamb-Oseen vortex simulation
# =============================================================================


@dataclass
class LambOseenLBMConfig:
    """Configuration for LBM simulation of Lamb-Oseen vortex.

    The vortex is initialized as an initial condition and allowed to
    evolve under the Navier-Stokes equations (via LBM). The Lamb-Oseen
    vortex is the self-similar solution for a decaying point vortex,
    so the core radius grows as r_c(t) = sqrt(r_c0² + 4νt).
    """

    nx: int = 128  # Grid size (square domain)
    ny: int = 128
    gamma: float = 0.1  # Circulation (lattice units)
    r_c0: float = 8.0  # Initial core radius (lattice units)
    tau: float = 0.6  # Relaxation time (ν = (τ - 0.5) × c_s² × dt)
    n_steps: int = 1000  # Number of time steps
    output_interval: int = 100  # Steps between outputs
    rho0: float = 1.0  # Reference density
    use_mrt: bool = False  # Use MRT collision (more stable)


def init_lamb_oseen_lbm(config: LambOseenLBMConfig) -> dict:
    """Initialize LBM fields for Lamb-Oseen vortex.

    Sets up the equilibrium distribution functions corresponding to the
    Lamb-Oseen velocity field on a periodic domain.

    Args:
        config: LBM configuration.

    Returns:
        Dictionary with keys:
            'f': distribution functions, shape (9, ny, nx)
            'ux', 'uy': velocity fields
            'rho': density field
            'x', 'y': coordinate arrays
    """
    from tensorlbm.d2q9 import C, W_EXACT, equilibrium

    nx, ny = config.nx, config.ny
    gamma = config.gamma
    r_c = config.r_c0
    rho0 = config.rho0

    # Coordinate arrays (cell-centered)
    x = np.arange(nx) - nx / 2.0
    y = np.arange(ny) - ny / 2.0
    xg, yg = np.meshgrid(x, y)

    # Radial distance from center
    r = np.sqrt(xg**2 + yg**2)

    # Lamb-Oseen velocity field
    u_theta = lamb_oseen_velocity(r, gamma, r_c)

    # Convert to Cartesian components
    # u_x = -u_θ × sin(θ) = -u_θ × y/r
    # u_y = u_θ × cos(θ) = u_θ × x/r
    mask = r > 1e-10
    ux = np.zeros((ny, nx))
    uy = np.zeros((ny, nx))
    ux[mask] = -u_theta[mask] * yg[mask] / r[mask]
    uy[mask] = u_theta[mask] * xg[mask] / r[mask]

    # Initialize density (uniform for incompressible vortex)
    rho = np.full((ny, nx), rho0)

    # Compute equilibrium distributions
    ux_t = torch.tensor(ux, dtype=torch.float64)
    uy_t = torch.tensor(uy, dtype=torch.float64)
    rho_t = torch.tensor(rho, dtype=torch.float64)

    f = equilibrium(rho_t, ux_t, uy_t).numpy()

    return {
        "f": f,
        "ux": ux,
        "uy": uy,
        "rho": rho,
        "x": x,
        "y": y,
        "config": config,
    }


def compute_pressure_from_velocity(
    ux: np.ndarray,
    uy: np.ndarray,
    rho: float = 1.0,
    p_ref: float = 0.0,
) -> np.ndarray:
    """Compute pressure field from velocity using radial equilibrium.

    For a vortex flow, the pressure satisfies:
    ∇p = ρ (u·∇)u

    For axisymmetric vortex: dp/dr = ρ u_θ²/r

    This function computes the pressure by integrating the momentum
    equation from a reference point (typically the domain boundary).

    Args:
        ux: x-velocity field, shape (ny, nx).
        uy: y-velocity field, shape (ny, nx).
        rho: Fluid density.
        p_ref: Reference pressure at domain boundary.

    Returns:
        Pressure field, shape (ny, nx).
    """
    ny, nx = ux.shape
    x = np.arange(nx) - nx / 2.0
    y = np.arange(ny) - ny / 2.0
    xg, yg = np.meshgrid(x, y)
    r = np.sqrt(xg**2 + yg**2)

    # Compute tangential velocity
    mask = r > 1e-10
    u_theta = np.zeros((ny, nx))
    # u_θ = (-ux sin θ + uy cos θ) = (-ux y/r + uy x/r)
    u_theta[mask] = (-ux[mask] * yg[mask] / r[mask] + uy[mask] * xg[mask] / r[mask])

    # Integrate dp/dr = ρ u_θ²/r radially
    # For each angular direction, integrate from r_max to r
    p = np.full((ny, nx), p_ref)

    # Use azimuthal average for stability
    r_bins = np.linspace(0, max(nx, ny) / 2, 100)
    u_theta_avg = np.zeros(len(r_bins) - 1)
    for ib in range(len(r_bins) - 1):
        annulus = (r >= r_bins[ib]) & (r < r_bins[ib + 1]) & mask
        if annulus.any():
            u_theta_avg[ib] = np.abs(u_theta[annulus]).mean()

    # Integrate from outside in
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    dp_dr = rho * u_theta_avg**2 / np.maximum(r_centers, 1e-10)
    n_bins_actual = len(r_bins)
    p_avg = np.zeros(n_bins_actual - 1)
    p_avg[-1] = p_ref
    for ib in range(n_bins_actual - 3, -1, -1):
        dr = r_bins[ib + 1] - r_bins[ib]
        p_avg[ib] = p_avg[ib + 1] + dp_dr[ib] * dr

    # Map back to 2D
    for iy in range(ny):
        for ix in range(nx):
            ri = r[iy, ix]
            if ri < r_bins[0]:
                p[iy, ix] = p_avg[0]
            elif ri >= r_bins[-1]:
                p[iy, ix] = p_ref
            else:
                # Linear interpolation
                ib = np.searchsorted(r_bins, ri) - 1
                ib = max(0, min(ib, len(p_avg) - 1))
                p[iy, ix] = p_avg[ib]

    return p


def run_lamb_oseen_lbm(
    config: LambOseenLBMConfig,
    verbose: bool = False,
) -> dict:
    """Run LBM simulation of Lamb-Oseen vortex evolution.

    The vortex decays viscously: r_c(t) = sqrt(r_c0² + 4νt).
    We track the core pressure minimum and compare with the analytical
    Lamb-Oseen solution at the evolved core radius.

    Args:
        config: LBM configuration.
        verbose: Print progress.

    Returns:
        Dictionary with simulation results including:
            'p_min_history': time series of minimum pressure
            'r_c_history': time series of measured core radius
            'final_ux', 'final_uy': final velocity fields
            'final_rho': final density field
            'analytical_dp': analytical pressure drops for comparison
    """
    from tensorlbm.d2q9 import C, OPPOSITE, W_EXACT

    nx, ny = config.nx, config.ny
    tau = config.tau
    n_steps = config.n_steps

    # Kinematic viscosity: ν = (τ - 0.5) / 3 (in lattice units with c_s² = 1/3)
    nu = (tau - 0.5) / 3.0

    # Initialize
    state = init_lamb_oseen_lbm(config)
    f = state["f"].copy()

    # Exact weights for double precision
    w = np.array(W_EXACT, dtype=np.float64)
    c = C.numpy()

    # Tracking arrays
    p_min_history = []
    r_c_history = []
    t_history = []
    analytical_dp_history = []

    # Sound speed squared
    cs2 = 1.0 / 3.0

    for step in range(n_steps + 1):
        # Compute macroscopic fields
        rho = np.sum(f, axis=0)
        ux = np.sum(f * c[:, 0:1].reshape(9, 1, 1), axis=0) / rho
        uy = np.sum(f * c[:, 1:2].reshape(9, 1, 1), axis=0) / rho

        # Compute pressure from velocity field (radial equilibrium)
        p = compute_pressure_from_velocity(ux, uy, config.rho0, p_ref=cs2 * config.rho0)
        p_min = p.min()

        # Analytical prediction at current time
        # Lamb-Oseen self-similar: r_c(t) = sqrt(r_c0² + 4νt)
        r_c_analytical = math.sqrt(config.r_c0**2 + 4.0 * nu * step)
        dp_analytical = lamb_oseen_core_pressure_drop(
            config.gamma, r_c_analytical, config.rho0
        )

        # Measure core radius from velocity profile
        r_arr = np.sqrt(state["x"] ** 2 + state["y"] ** 2)
        u_theta_meas = np.sqrt(ux**2 + uy**2)
        r_bins = np.linspace(0, min(nx, ny) / 2, 50)
        u_avg = np.zeros(len(r_bins) - 1)
        for ib in range(len(r_bins) - 1):
            mask_bin = (r_arr >= r_bins[ib]) & (r_arr < r_bins[ib + 1])
            if mask_bin.any():
                u_avg[ib] = u_theta_meas[mask_bin].mean()
        if u_avg.max() > 1e-10:
            r_max_idx = np.argmax(u_avg)
            r_c_meas = 0.5 * (r_bins[r_max_idx] + r_bins[r_max_idx + 1])
        else:
            r_c_meas = config.r_c0

        p_min_history.append(p_min)
        r_c_history.append(r_c_meas)
        t_history.append(step)
        analytical_dp_history.append(dp_analytical)

        if verbose and step % config.output_interval == 0:
            p_inf = cs2 * config.rho0
            dp_measured = p_inf - p_min
            print(
                f"Step {step}: p_min={p_min:.6f}, Δp_meas={dp_measured:.6f}, "
                f"Δp_anal={dp_analytical:.6f}, r_c_meas={r_c_meas:.2f}, "
                f"r_c_anal={r_c_analytical:.2f}"
            )

        if step == n_steps:
            break

        # Collision (BGK)
        feq = np.zeros_like(f)
        for k in range(9):
            cu = c[k, 0] * ux + c[k, 1] * uy
            usq = ux**2 + uy**2
            feq[k] = w[k] * rho * (
                1.0 + cu / cs2 + 0.5 * cu**2 / cs2**2 - 0.5 * usq / cs2
            )

        f = f - (f - feq) / tau

        # Streaming (periodic boundaries)
        f_new = np.zeros_like(f)
        for k in range(9):
            f_new[k] = np.roll(f[k], shift=(c[k, 1], c[k, 0]), axis=(0, 1))
        f = f_new

    # Final fields
    rho_final = np.sum(f, axis=0)
    ux_final = np.sum(f * c[:, 0:1].reshape(9, 1, 1), axis=0) / rho_final
    uy_final = np.sum(f * c[:, 1:2].reshape(9, 1, 1), axis=0) / rho_final

    return {
        "p_min_history": np.array(p_min_history),
        "r_c_history": np.array(r_c_history),
        "t_history": np.array(t_history),
        "analytical_dp_history": np.array(analytical_dp_history),
        "final_ux": ux_final,
        "final_uy": uy_final,
        "final_rho": rho_final,
        "nu": nu,
        "config": config,
    }


# =============================================================================
# Tip vortex model
# =============================================================================


@dataclass
class TipVortexConfig:
    """Configuration for tip vortex cavitation analysis.

    Models the tip vortex shed from a finite-span hydrofoil and
    computes the cavitation inception number.
    """

    # Hydrofoil parameters
    chord: float = 0.1  # chord length [m]
    span: float = 0.2  # span length [m]
    velocity: float = 10.0  # inflow velocity [m/s]
    alpha_deg: float = 5.0  # angle of attack [degrees]

    # Fluid properties
    rho: float = 998.0  # water density [kg/m³]
    nu: float = 1.0e-6  # kinematic viscosity [m²/s]
    p_v: float = 2330.0  # vapour pressure [Pa] (water at 20°C)

    # Vortex parameters
    r_c_over_c: float = 0.05  # core radius / chord (typical: 0.02-0.1)
    gamma_over_Vc: float = None  # circulation parameter (computed from CL if None)
    CL: float = None  # lift coefficient (computed from thin airfoil theory if None)


def tip_vortex_circulation_decay(
    x: np.ndarray,
    gamma_0: float,
    b: float,
    nu: float,
) -> np.ndarray:
    """Tip vortex circulation decay with downstream distance.

    Based on the viscous diffusion model:
    Γ(x) = Γ₀ × (1 - A_v × sqrt(ν x / (V b²)))

    where A_v ≈ 0.6 is an empirical constant, x is downstream distance,
    V is inflow velocity, and b is the span.

    For small x, the decay is approximately linear in sqrt(x).

    Args:
        x: Downstream distance array [m].
        gamma_0: Initial circulation at trailing edge [m²/s].
        b: Hydrofoil span [m].
        nu: Kinematic viscosity [m²/s].

    Returns:
        Circulation Γ(x) at each downstream location.
    """
    x = np.asarray(x, dtype=np.float64)
    A_v = 0.6  # empirical constant (McCormick 1962)
    # Decay factor
    decay = np.sqrt(nu * np.maximum(x, 0) / (1.0**2 * b**2))
    gamma = gamma_0 * np.maximum(1.0 - A_v * decay, 0.1)
    return gamma


def tip_vortex_trajectory(
    x: np.ndarray,
    gamma_0: float,
    V: float,
    r_c0: float,
    nu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute tip vortex core radius and trajectory.

    The core radius grows by viscous diffusion:
    r_c(x) = sqrt(r_c0² + 4 ν x / V)

    The vortex convects downstream at approximately the inflow velocity
    and rolls up toward the tip.

    Args:
        x: Downstream distance array [m].
        gamma_0: Initial circulation [m²/s].
        V: Inflow velocity [m/s].
        r_c0: Initial core radius [m].
        nu: Kinematic viscosity [m²/s].

    Returns:
        Tuple of (r_c(x), y_vortex(x)):
            r_c: Core radius at each x [m]
            y_vortex: Lateral position of vortex core [m]
    """
    x = np.asarray(x, dtype=np.float64)
    # Core radius growth (Lamb-Oseen self-similar)
    r_c = np.sqrt(r_c0**2 + 4.0 * nu * x / V)

    # Lateral trajectory: vortex rolls up from trailing edge
    # Simplified model: y ≈ b/2 × (1 - exp(-x/(0.1c)))
    # This represents the roll-up toward the free tip
    # For a semi-span b/2, the tip vortex starts at the tip and stays there
    y_vortex = np.full_like(x, 0.0)  # stays at tip location (y=0 reference)

    return r_c, y_vortex


# =============================================================================
# Cavitation inception
# =============================================================================


def cavitation_inception_number(
    gamma: float,
    r_c: float,
    V: float,
    rho: float,
    p_inf: float,
    p_v: float,
    vortex_model: str = "lamb_oseen",
) -> float:
    """Compute cavitation inception number for a tip vortex.

    The inception cavitation number σ_i is defined as:
    σ_i = (p_∞ - p_v) / (0.5 ρ V²)

    at the condition where p_min = p_v in the vortex core.

    For a Lamb-Oseen vortex:
    p_min = p_∞ - Δp = p_∞ - ρ Γ² ln(2) / (4 π² r_c²)

    Setting p_min = p_v:
    p_∞ - p_v = ρ Γ² ln(2) / (4 π² r_c²)

    Therefore: σ_i = ρ Γ² ln(2) / (4 π² r_c² × 0.5 ρ V²)
                   = Γ² ln(2) / (2 π² r_c² V²)

    Args:
        gamma: Vortex circulation Γ [m²/s].
        r_c: Vortex core radius [m].
        V: Reference velocity [m/s].
        rho: Fluid density [kg/m³].
        p_inf: Freestream pressure [Pa].
        p_v: Vapour pressure [Pa].
        vortex_model: 'lamb_oseen' or 'rankine'.

    Returns:
        Inception cavitation number σ_i.
    """
    if vortex_model == "lamb_oseen":
        dp = lamb_oseen_core_pressure_drop(gamma, r_c, rho)
    elif vortex_model == "rankine":
        dp = rankine_core_pressure_drop(gamma, r_c, rho)
    else:
        raise ValueError(f"Unknown vortex model: {vortex_model}")

    q_dyn = 0.5 * rho * V**2
    sigma_i = dp / q_dyn
    return sigma_i


def inception_sigma_from_pmin(
    p_min: float,
    p_inf: float,
    p_v: float,
    rho: float,
    V: float,
) -> float:
    """Compute inception sigma from measured minimum pressure.

    At inception, p_min = p_v, so:
    σ_i = (p_∞ - p_v) / (0.5 ρ V²)

    But if p_min > p_v (no cavitation yet), we can compute the
    critical σ at which cavitation would occur:
    σ_crit = (p_∞ - p_min) / (0.5 ρ V²)

    Args:
        p_min: Minimum pressure in vortex core [Pa].
        p_inf: Freestream pressure [Pa].
        p_v: Vapour pressure [Pa].
        rho: Fluid density [kg/m³].
        V: Reference velocity [m/s].

    Returns:
        Critical cavitation number σ_crit.
    """
    q_dyn = 0.5 * rho * V**2
    sigma_crit = (p_inf - p_min) / q_dyn
    return sigma_crit


def mccormick_inception_scaling(
    gamma: float,
    V: float,
    c: float,
    Re: float,
    r_c_over_c: float = 0.05,
) -> float:
    """McCormick (1962) empirical cavitation inception scaling.

    McCormick found that the inception cavitation number scales as:
    σ_i = K × (Γ/(V c))^{2/3} × Re^{-1/6}

    where K ≈ 3.5 is an empirical constant (depends on hydrofoil geometry),
    Γ is the tip vortex circulation, V is inflow velocity, c is chord,
    and Re = V c / ν is the Reynolds number.

    The exponent 2/3 on circulation and -1/6 on Reynolds number come from
    the balance between vortex strength and viscous core diffusion.

    Args:
        gamma: Tip vortex circulation Γ [m²/s].
        V: Inflow velocity [m/s].
        c: Chord length [m].
        Re: Reynolds number V c / ν.
        r_c_over_c: Core radius to chord ratio (for reference).

    Returns:
        Inception cavitation number σ_i from McCormick scaling.
    """
    K = 3.5  # empirical constant (McCormick 1962, elliptic planform)
    gamma_star = gamma / (V * c)  # dimensionless circulation
    sigma_i = K * gamma_star ** (2.0 / 3.0) * Re ** (-1.0 / 6.0)
    return sigma_i
