"""Diagnostics for SC/D2Q9 two-phase LBM.

This module provides:
1. Parasitic flow SNR analysis — comparing spurious velocity against
   capillary pressure to ensure the signal dominates the noise.
2. Surface tension measurement from the Laplace law.
3. Three-radius self-consistency gate — verifying that Δp vs 1/R is
   linear with R² > 0.95 and slope = σ within 5%.

The parasitic (spurious) currents are a well-known artefact of lattice
Boltzmann two-phase models.  They arise from the imbalance between the
discrete pressure tensor and the capillary force at the interface.  The
key diagnostic is the SNR:

    SNR = Δp_capillary / p_dynamic_spurious
        = (σ/κ) / (ρ cs max|u_spurious|)

where κ = 1/R is the interface curvature.  For the simulation to be
physically meaningful, we require SNR > 3 (the capillary pressure jump
must be at least 3× larger than the dynamic pressure from spurious flow).

References
----------
Wagner & Pohl (2004) Europhys. Lett. 65, 145 (spurious currents in SC).
Biferale et al. (2020) Phys. Rev. Lett. 125, 024503 (spurious flow reduction).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from .sc_core import SCConfig, SCState, init_droplet, run_simulation
from .pressure import momentum_flux_pressure, pressure_jump_droplet


@dataclass
class SNRResult:
    """Result of parasitic flow SNR analysis."""
    max_u_spurious: float  # Maximum spurious velocity magnitude
    sigma: float  # Surface tension (from Laplace fit or input)
    R_droplet: float  # Droplet radius
    kappa: float  # Curvature 1/R
    dp_capillary: float  # Capillary pressure jump σ/R (2D)
    p_dynamic: float  # Dynamic pressure from spurious flow ρ cs max|u|
    snr: float  # Signal-to-noise ratio
    pass_snr: bool  # SNR > 3?


@dataclass
class LaplaceResult:
    """Result of Laplace three-radius self-consistency test."""
    radii: List[float]  # Input radii [R1, R2, R3]
    radii_eq: List[float]  # Measured equilibrium radii
    dp_values: List[float]  # Measured pressure jumps [Δp1, Δp2, Δp3]
    sigma_per_radius: List[float]  # σ_i = Δp_i * R_i for each radius
    sigma_fit: float  # Surface tension from linear fit
    R_squared: float  # R² of linear fit Δp vs 1/R
    slope_error_pct: float  # |slope - σ_input| / σ_input * 100
    pass_laplace: bool  # R² > 0.95 and slope error < 5%?
    pass_sigma: bool  # All per-radius σ within 5% of fit?


def parasitic_flow_snr(
    state: SCState,
    center: tuple[float, float],
    R_droplet: float,
    sigma: float | None = None,
    interface_width: float = 3.0,
) -> SNRResult:
    """Compute parasitic flow signal-to-noise ratio.

    The SNR is defined as:
        SNR = Δp_capillary / p_dynamic_spurious

    where:
    - Δp_capillary = σ / R (2D Laplace law)
    - p_dynamic_spurious = ρ_l * cs * max|u_spurious|

    The spurious velocity is measured in the liquid phase near the interface.

    Args:
        state: SC state with a static droplet.
        center: Droplet center (cx, cy).
        R_droplet: Droplet radius.
        sigma: Surface tension. If None, estimated from coexistence.
        interface_width: Interface width for sampling.

    Returns:
        SNRResult with all diagnostics.
    """
    device = torch.device(state.config.device)
    dtype = state.config.dtype

    # Measure spurious velocity (max |u| in liquid near interface)
    rho = state.rho
    ux = state.ux
    uy = state.uy

    # Liquid mask: ρ > (ρ_l + ρ_v) / 2 ≈ 1.06 for standard coexistence
    rho_mid = 1.06  # Approximate mid-density for psi_exp with G=5
    liquid_mask = rho > rho_mid

    # Near-interface region: within 2*interface_width of the interface
    ny, nx = rho.shape
    y = torch.arange(ny, device=device, dtype=dtype)
    x = torch.arange(nx, device=device, dtype=dtype)
    Y, X = torch.meshgrid(y, x, indexing='ij')

    dx = X - center[0]
    dy = Y - center[1]
    dx = dx - nx * torch.round(dx / nx)
    dy = dy - ny * torch.round(dy / ny)
    r = torch.sqrt(dx**2 + dy**2)

    # Interface band: R - 3w < r < R + 3w, in liquid
    interface_band = (r > R_droplet - 3*interface_width) & (r < R_droplet + 3*interface_width)
    sample_mask = liquid_mask & interface_band

    if sample_mask.sum() > 0:
        u_mag = torch.sqrt(ux**2 + uy**2)
        max_u = u_mag[sample_mask].max().item()
    else:
        # Fallback: use global max in liquid
        u_mag = torch.sqrt(ux**2 + uy**2)
        max_u = u_mag[liquid_mask].max().item() if liquid_mask.sum() > 0 else 0.0

    # Estimate surface tension if not provided
    if sigma is None:
        # For psi_exp with G=5, σ ≈ 0.056 (from literature/calibration)
        sigma = 0.056

    # Capillary pressure (2D: Δp = σ/R)
    kappa = 1.0 / R_droplet
    dp_capillary = sigma * kappa

    # Dynamic pressure from spurious flow
    cs = math.sqrt(1.0 / 3.0)
    rho_l = 1.96  # Liquid density for standard coexistence
    p_dynamic = rho_l * cs * max_u

    # SNR
    snr = dp_capillary / p_dynamic if p_dynamic > 0 else float('inf')

    return SNRResult(
        max_u_spurious=max_u,
        sigma=sigma,
        R_droplet=R_droplet,
        kappa=kappa,
        dp_capillary=dp_capillary,
        p_dynamic=p_dynamic,
        snr=snr,
        pass_snr=snr > 3.0,
    )


def measure_surface_tension_laplace(
    config: SCConfig,
    R: float,
    n_steps: int = 10000,
    warmup_steps: int = 2000,
    sample_interval: int = 100,
) -> dict:
    """Measure surface tension from a single droplet using Laplace law.

    Runs a static droplet simulation and measures:
    - Equilibrium radius R_eq (from density threshold)
    - Pressure jump Δp (from momentum flux)
    - Surface tension σ = Δp * R_eq (2D)

    Args:
        config: Simulation configuration.
        R: Initial droplet radius.
        n_steps: Total simulation steps.
        warmup_steps: Steps to discard before measurement.
        sample_interval: Diagnostic sampling interval.

    Returns:
        Dictionary with measured quantities.
    """
    # Initialize droplet
    state = init_droplet(config, R=R)

    # Run warmup
    state = run_simulation(state, warmup_steps, sample_interval=sample_interval)

    # Run measurement phase
    state = run_simulation(state, n_steps - warmup_steps, sample_interval=sample_interval)

    # Measure equilibrium radius from density field
    rho = state.rho
    rho_mid = 1.06  # Mid-density for coexistence
    liquid_mask = rho > rho_mid
    # Equivalent radius from area: A = π R² → R = sqrt(A/π)
    area = liquid_mask.sum().item()
    R_eq = math.sqrt(area / math.pi)

    # Measure pressure jump
    # Estimate center from liquid mass center
    device = torch.device(config.device)
    dtype = config.dtype
    ny, nx = rho.shape
    y = torch.arange(ny, device=device, dtype=dtype)
    x = torch.arange(nx, device=device, dtype=dtype)
    Y, X = torch.meshgrid(y, x, indexing='ij')

    # Periodic center of mass
    mass = liquid_mask.float().sum()
    if mass > 0:
        # Use circular mean for periodic coordinates
        theta_x = 2 * math.pi * X / nx
        theta_y = 2 * math.pi * Y / ny
        cx = torch.atan2(
            (liquid_mask.float() * torch.sin(theta_x)).sum(),
            (liquid_mask.float() * torch.cos(theta_x)).sum()
        ) * nx / (2 * math.pi)
        cy = torch.atan2(
            (liquid_mask.float() * torch.sin(theta_y)).sum(),
            (liquid_mask.float() * torch.cos(theta_y)).sum()
        ) * ny / (2 * math.pi)
        cx = cx.item() % nx
        cy = cy.item() % ny
    else:
        cx, cy = nx / 2, ny / 2

    center = (cx, cy)

    # Pressure jump from momentum flux
    p_jump = pressure_jump_droplet(state, center, R_eq)

    # The measured dp includes both the bulk coexistence pressure difference
    # AND the Laplace pressure jump due to curvature.
    # dp_total = dp_bulk + dp_Laplace
    # dp_Laplace = σ/R (2D)
    # 
    # To extract σ, we need to subtract dp_bulk.
    # dp_bulk = p_liquid_bulk - p_vapour_bulk (from EOS)
    # For SC with psi_exp and G=5: dp_bulk ≈ 1.197
    # 
    # Alternatively, we can measure dp for multiple radii and fit:
    # dp(R) = dp_bulk + σ/R
    # The slope of dp vs 1/R gives σ.
    # 
    # For a single droplet, we estimate dp_bulk from the coexistence densities.
    from tensorlbm.sc_d2q9_interface import psi_exp
    rho_l = state.rho[liquid_mask].mean().item() if liquid_mask.sum() > 0 else 1.96
    rho_v = state.rho[~liquid_mask & (state.rho < 0.5)].mean().item() if (~liquid_mask & (state.rho < 0.5)).sum() > 0 else 0.16
    psi_l = psi_exp(torch.tensor([rho_l])).item()
    psi_v = psi_exp(torch.tensor([rho_v])).item()
    cs2 = 1.0 / 3.0
    dp_bulk = (rho_l - rho_v) * cs2 + (state.config.G / 2.0) * cs2 * (psi_l**2 - psi_v**2)
    
    dp_laplace = p_jump['dp'] - dp_bulk
    dp = dp_laplace

    # Surface tension (2D: σ = Δp_Laplace * R)
    sigma_measured = dp * R_eq if dp > 0 else 0.0

    # Max spurious velocity
    u_mag = torch.sqrt(state.ux**2 + state.uy**2)
    max_u = u_mag[liquid_mask].max().item() if liquid_mask.sum() > 0 else 0.0

    return {
        'R_init': R,
        'R_eq': R_eq,
        'center': center,
        'dp': dp,
        'p_inside': p_jump['p_inside'],
        'p_outside': p_jump['p_outside'],
        'sigma_measured': sigma_measured,
        'max_u': max_u,
        'mass': mass.item(),
        'step': state.step,
    }


def laplace_three_radius_gate(
    config: SCConfig,
    radii: List[float] = None,
    n_steps: int = 15000,
    warmup_steps: int = 3000,
    sample_interval: int = 200,
    sigma_input: float = 0.056,
) -> LaplaceResult:
    """Laplace three-radius self-consistency gate.

    Runs static droplet simulations at three radii and verifies:
    1. Δp vs 1/R is linear with R² > 0.95
    2. Slope = σ within 5% of input surface tension
    3. Per-radius σ_i = Δp_i * R_i are consistent

    This is the machine-verifiable criterion for interface tension
    measurability in the SC/D2Q9 model.

    Args:
        config: Base simulation configuration (nx, ny will be adjusted per R).
        radii: List of three radii. Default [20, 40, 80].
        n_steps: Steps per droplet simulation.
        warmup_steps: Warmup steps per simulation.
        sample_interval: Diagnostic interval.
        sigma_input: Expected surface tension for comparison.

    Returns:
        LaplaceResult with all diagnostics and pass/fail.
    """
    if radii is None:
        radii = [20.0, 40.0, 80.0]

    if len(radii) != 3:
        raise ValueError("Laplace gate requires exactly 3 radii")

    results = []
    for R in radii:
        # Adjust domain size: L = 4*R (to avoid periodic image interaction)
        L = int(4 * R)
        config_R = SCConfig(
            nx=L, ny=L,
            tau=config.tau,
            G=config.G,
            rho0=config.rho0,
            psi_fn=config.psi_fn,
            gradient_scheme=config.gradient_scheme,
            device=config.device,
            dtype=config.dtype,
        )
        res = measure_surface_tension_laplace(
            config_R, R, n_steps, warmup_steps, sample_interval
        )
        results.append(res)

    # Extract data for linear fit
    R_eq_list = [r['R_eq'] for r in results]
    dp_list = [r['dp'] for r in results]
    inv_R_list = [1.0 / R for R in R_eq_list]

    # Linear fit: Δp = σ * (1/R) + b
    # For 2D: Δp = σ/R, so intercept should be ~0
    inv_R_arr = np.array(inv_R_list)
    dp_arr = np.array(dp_list)

    # Least squares fit
    A = np.vstack([inv_R_arr, np.ones(len(inv_R_arr))]).T
    fit_result = np.linalg.lstsq(A, dp_arr, rcond=None)
    coeffs = fit_result[0]
    sigma_fit = coeffs[0]  # slope = σ
    intercept = coeffs[1]

    # R² calculation
    dp_pred = sigma_fit * inv_R_arr + intercept
    ss_res = np.sum((dp_arr - dp_pred)**2)
    ss_tot = np.sum((dp_arr - np.mean(dp_arr))**2)
    R_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Per-radius surface tension
    sigma_per_radius = [dp * R for dp, R in zip(dp_list, R_eq_list)]

    # Check consistency
    slope_error_pct = abs(sigma_fit - sigma_input) / sigma_input * 100.0
    max_sigma_dev = max(abs(s - sigma_fit) / sigma_fit * 100.0 for s in sigma_per_radius)

    pass_laplace = R_squared > 0.95 and slope_error_pct < 5.0
    pass_sigma = max_sigma_dev < 5.0

    return LaplaceResult(
        radii=radii,
        radii_eq=R_eq_list,
        dp_values=dp_list,
        sigma_per_radius=sigma_per_radius,
        sigma_fit=sigma_fit,
        R_squared=R_squared,
        slope_error_pct=slope_error_pct,
        pass_laplace=pass_laplace,
        pass_sigma=pass_sigma,
    )
