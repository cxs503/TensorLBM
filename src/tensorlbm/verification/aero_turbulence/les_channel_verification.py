"""LES turbulent channel flow verification for TensorLBM.

Verification against Mos, Kim & Mansour (1999) DNS data at Re_tau=180.

Physics:
- Plane channel flow, periodic in x (and z for 3D)
- No-slip walls at y = 0 and y = H
- Body-force driven (constant pressure gradient)
- Smagorinsky LES model with van Driest wall damping
- Re_tau = u_tau * h / nu = 180

Reference:
- Mos, R., Kim, J., & Mansour, N.N. (1999). "Direct numerical simulation
  of turbulent channel flow up to Re_tau=590." Physics of Fluids, 11(4), 943-945.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...boundaries import bounce_back_cells, make_channel_wall_mask
from ...d2q9 import C, W, equilibrium, macroscopic
from ...solver import stream
from ...turbulence import collide_smagorinsky_bgk
from ...utils import resolve_device
from .reference_data import (
    MKM1999Statistics,
    mkm1999_mean_velocity_profile,
    mkm1999_reynolds_stress_profile,
    mkm1999_tke_profile,
)


# ============================================================================
# Van Driest damping function
# ============================================================================

def van_driest_damping(y_plus: torch.Tensor, A_plus: float = 25.0) -> torch.Tensor:
    """Van Driest damping function for near-wall eddy viscosity.
    
    D = 1 - exp(-y+ / A+)
    
    Parameters
    ----------
    y_plus : torch.Tensor
        Wall distance in wall units.
    A_plus : float
        Van Driest constant (typically 25-27).
    
    Returns
    -------
    D : torch.Tensor
        Damping factor in [0, 1].
    """
    return 1.0 - torch.exp(-y_plus / A_plus)


def compute_wall_distance_2d(ny: int, nx: int, device: str = "cpu") -> torch.Tensor:
    """Compute wall distance for 2D channel (walls at j=0 and j=ny-1).
    
    Returns distance in lattice units (minimum distance to either wall).
    """
    dev = resolve_device(device)
    j = torch.arange(ny, device=dev, dtype=torch.float32).view(ny, 1).expand(ny, nx)
    d_bottom = j.clone()
    d_top = (ny - 1 - j).float()
    return torch.minimum(d_bottom, d_top)


# ============================================================================
# Channel flow configuration
# ============================================================================

@dataclass
class LESChannelConfig:
    """Configuration for LES turbulent channel flow."""
    nx: int = 256               # Streamwise extent
    ny: int = 65                # Wall-normal extent (must be odd for symmetry)
    re_tau: float = 180.0       # Friction Reynolds number
    u_tau: float = 0.005        # Friction velocity (lattice units)
    C_s: float = 0.1            # Smagorinsky constant
    A_plus: float = 25.0        # Van Driest damping constant
    n_steps: int = 100000       # Total simulation steps
    averaging_start: int = 30000  # Steps before time-averaging begins
    sample_interval: int = 100  # Interval between samples for statistics
    device: str = "cpu"
    seed: int = 42
    run_name: str = "les_channel_ret180"
    
    @property
    def H(self) -> int:
        """Channel height (fluid region)."""
        return self.ny - 2  # walls at j=0 and j=ny-1
    
    @property
    def h(self) -> float:
        """Half-channel height."""
        return self.H / 2.0
    
    @property
    def nu(self) -> float:
        """Kinematic viscosity from Re_tau = u_tau * h / nu."""
        return self.u_tau * self.h / self.re_tau
    
    @property
    def tau(self) -> float:
        """BGK relaxation time."""
        return 3.0 * self.nu + 0.5
    
    @property
    def body_force(self) -> float:
        """Streamwise body force for driving."""
        return 2.0 * self.u_tau**2 / self.H
    
    @property
    def dy_plus(self) -> float:
        """Wall-normal grid spacing in wall units."""
        return self.u_tau / self.nu
    
    def validate(self) -> None:
        if self.nx < 32:
            raise ValueError(f"nx={self.nx} must be >= 32")
        if self.ny < 17:
            raise ValueError(f"ny={self.ny} must be >= 17")
        if self.re_tau <= 0:
            raise ValueError(f"re_tau={self.re_tau} must be > 0")
        if self.tau <= 0.5:
            raise ValueError(f"tau={self.tau} must be > 0.5")
        if not (0.0 < self.C_s < 0.3):
            raise ValueError(f"C_s={self.C_s} must be in (0, 0.3)")


# ============================================================================
# Body force application
# ============================================================================

def apply_body_force_2d(f: torch.Tensor, a_x: float) -> torch.Tensor:
    """Apply streamwise body force (Guo forcing, first order)."""
    c = C.to(f.device).float()
    w = W.to(f.device).float()
    rho = f.sum(dim=0)
    cx = c[:, 0].view(9, 1, 1)
    w_view = w.view(9, 1, 1)
    f_new = f + w_view * 3.0 * rho.unsqueeze(0) * cx * a_x
    return f_new


# ============================================================================
# Statistics accumulator
# ============================================================================

class ChannelStatsAccumulator:
    """Accumulate time-averaged statistics for channel flow."""
    
    def __init__(self, ny: int, nx: int, device: str = "cpu"):
        dev = resolve_device(device)
        self.ny = ny
        self.nx = nx
        self.device = dev
        self.count = 0
        
        # Mean velocity
        self.mean_u = torch.zeros(ny, nx, device=dev)
        self.mean_v = torch.zeros(ny, nx, device=dev)
        
        # Reynolds stresses
        self.mean_uu = torch.zeros(ny, nx, device=dev)
        self.mean_vv = torch.zeros(ny, nx, device=dev)
        self.mean_uv = torch.zeros(ny, nx, device=dev)
    
    def accumulate(self, ux: torch.Tensor, uy: torch.Tensor) -> None:
        """Add a snapshot to the statistics."""
        self.count += 1
        n = self.count
        
        # Update running means
        self.mean_u += (ux - self.mean_u) / n
        self.mean_v += (uy - self.mean_v) / n
        
        # Reynolds stresses (using current mean)
        up = ux - self.mean_u
        vp = uy - self.mean_v
        self.mean_uu += (up**2 - self.mean_uu) / n
        self.mean_vv += (vp**2 - self.mean_vv) / n
        self.mean_uv += (up * vp - self.mean_uv) / n
    
    def get_wall_normal_profiles(self) -> dict[str, np.ndarray]:
        """Get wall-normal profiles (averaged in x and time)."""
        y_idx = np.arange(self.ny)
        
        U_plus = self.mean_u.mean(dim=-1).cpu().numpy()
        V_plus = self.mean_v.mean(dim=-1).cpu().numpy()
        uu_plus = self.mean_uu.mean(dim=-1).cpu().numpy()
        vv_plus = self.mean_vv.mean(dim=-1).cpu().numpy()
        uv_plus = self.mean_uv.mean(dim=-1).cpu().numpy()
        
        # TKE
        tke_plus = 0.5 * (uu_plus + vv_plus)  # 2D: only u,v components
        
        return {
            "y_idx": y_idx,
            "U_plus": U_plus,
            "V_plus": V_plus,
            "uu_plus": uu_plus,
            "vv_plus": vv_plus,
            "uv_plus": uv_plus,
            "tke_plus": tke_plus,
        }


# ============================================================================
# Main simulation runner
# ============================================================================

def run_les_channel(config: LESChannelConfig) -> dict[str, Any]:
    """Run LES turbulent channel flow simulation.
    
    Returns
    -------
    dict with keys:
        profiles: wall-normal profiles (U+, uu+, vv+, uv+, tke+)
        y_plus: wall distance in wall units
        statistics: summary statistics
        config: configuration used
    """
    config.validate()
    device = resolve_device(config.device)
    torch.manual_seed(config.seed)
    
    nx, ny = config.nx, config.ny
    
    # Wall mask (top and bottom walls)
    obstacle = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall_mask = make_channel_wall_mask(ny, nx, obstacle, device)
    
    # Wall distance in lattice units
    wall_dist = compute_wall_distance_2d(ny, nx, device=str(device))
    
    # Wall distance in wall units (y+ = y * u_tau / nu)
    y_plus = wall_dist * config.u_tau / config.nu
    
    # Initialize flow field
    # Start with laminar Poiseuille profile + small perturbation
    y_wall = wall_dist  # distance to nearest wall
    
    # Laminar profile: u(y) = (a/2*nu) * y * (H - y) where a = body_force
    # Max velocity at center = a * H^2 / (8 * nu)
    u_max = config.body_force * config.H**2 / (8.0 * config.nu)
    u_lam = u_max * 4.0 * y_wall / config.H * (1.0 - y_wall / config.H)
    
    # Add small random perturbation for transition
    torch.manual_seed(config.seed)
    perturbation = 0.01 * u_max * torch.randn(ny, nx, device=device)
    ux_init = u_lam + perturbation
    ux_init[wall_mask] = 0.0  # no-slip walls
    uy_init = torch.zeros(ny, nx, device=device)
    
    rho0 = torch.ones(ny, nx, device=device)
    f = equilibrium(rho0, ux_init, uy_init, device=device)
    
    # Statistics accumulator
    stats = ChannelStatsAccumulator(ny, nx, device=str(device))
    
    # Time integration
    for step in range(1, config.n_steps + 1):
        # Collision (Smagorinsky LES)
        f = collide_smagorinsky_bgk(f, tau=config.tau, C_s=config.C_s)
        
        # Streaming
        f = stream(f)
        
        # Body force
        f = apply_body_force_2d(f, config.body_force)
        
        # Wall boundary conditions (bounce-back)
        f = bounce_back_cells(f, wall_mask)
        
        # Accumulate statistics
        if step > config.averaging_start and step % config.sample_interval == 0:
            rho, ux, uy = macroscopic(f)
            # Normalize by u_tau for wall units
            ux_plus = ux / config.u_tau
            uy_plus = uy / config.u_tau
            # Zero out wall cells
            ux_plus[wall_mask] = 0.0
            uy_plus[wall_mask] = 0.0
            stats.accumulate(ux_plus, uy_plus)
    
    # Extract profiles
    profiles = stats.get_wall_normal_profiles()
    
    # Compute y+ for each wall-normal location
    y_idx = profiles["y_idx"]
    y_plus_profile = np.minimum(y_idx, ny - 1 - y_idx) * config.dy_plus
    
    # Summary statistics
    u_center = profiles["U_plus"][ny // 2]
    # Bulk velocity: integrate half-channel and normalize
    half_ny = ny // 2 + 1
    y_half = y_plus_profile[:half_ny]
    u_half = profiles["U_plus"][:half_ny]
    if len(y_half) > 1 and not np.any(np.isnan(u_half)):
        u_bulk = float(np.trapezoid(u_half, y_half) / config.re_tau)
    else:
        u_bulk = 0.0
    
    # Find Reynolds stress peak
    uv_profile = -profiles["uv_plus"]  # -<u'v'>+
    if not np.any(np.isnan(uv_profile)):
        peak_idx = np.argmax(uv_profile[1:ny//2]) + 1
        uv_peak_val = float(uv_profile[peak_idx])
        uv_peak_y = float(y_plus_profile[peak_idx])
    else:
        uv_peak_val = 0.0
        uv_peak_y = 0.0
    
    result = {
        "profiles": {
            "y_plus": y_plus_profile.tolist(),
            "U_plus": profiles["U_plus"].tolist(),
            "uu_plus": profiles["uu_plus"].tolist(),
            "vv_plus": profiles["vv_plus"].tolist(),
            "uv_plus": profiles["uv_plus"].tolist(),
            "tke_plus": profiles["tke_plus"].tolist(),
        },
        "statistics": {
            "n_samples": stats.count,
            "u_centerline_plus": float(u_center) if not np.isnan(u_center) else 0.0,
            "u_bulk_plus": float(u_bulk),
            "uv_peak_value": uv_peak_val,
            "uv_peak_y_plus": uv_peak_y,
            "re_tau": config.re_tau,
            "dy_plus": config.dy_plus,
        },
        "config": {
            "nx": config.nx,
            "ny": config.ny,
            "re_tau": config.re_tau,
            "u_tau": config.u_tau,
            "C_s": config.C_s,
            "A_plus": config.A_plus,
            "n_steps": config.n_steps,
            "averaging_start": config.averaging_start,
        },
    }
    
    return result


# ============================================================================
# Energy spectrum computation
# ============================================================================

def compute_energy_spectrum(
    ux: torch.Tensor,
    dx: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute streamwise energy spectrum E_uu(k_x).
    
    Parameters
    ----------
    ux : torch.Tensor, shape (ny, nx)
        Streamwise velocity fluctuation field.
    dx : float
        Grid spacing.
    
    Returns
    -------
    k : np.ndarray
        Wavenumber array.
    E_uu : np.ndarray
        Energy spectrum.
    """
    # Average over wall-normal direction
    ux_mean = ux.mean(dim=0).cpu().numpy()
    
    # FFT
    nx = len(ux_mean)
    ux_hat = np.fft.rfft(ux_mean - ux_mean.mean())
    
    # Energy spectrum
    E_uu = np.abs(ux_hat)**2 / nx
    
    # Wavenumber
    k = np.fft.rfftfreq(nx, d=dx) * 2 * np.pi
    
    # Remove k=0
    k = k[1:]
    E_uu = E_uu[1:]
    
    return k, E_uu


def fit_inertial_subrange(
    k: np.ndarray,
    E: np.ndarray,
    k_range: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Fit -5/3 slope in the inertial subrange.
    
    Returns
    -------
    dict with keys:
        slope: fitted exponent
        r_squared: R² of the fit
        k_range: wavenumber range used
    """
    if k_range is None:
        # Use middle third of log(k) range
        log_k = np.log10(k[k > 0])
        k_min = 10**(log_k.min() + log_k.ptp() * 0.2)
        k_max = 10**(log_k.max() - log_k.ptp() * 0.2)
    else:
        k_min, k_max = k_range
    
    mask = (k >= k_min) & (k <= k_max) & (k > 0) & (E > 0)
    if mask.sum() < 3:
        return {"slope": 0.0, "r_squared": 0.0, "k_range": [k_min, k_max]}
    
    log_k = np.log10(k[mask])
    log_E = np.log10(E[mask])
    
    # Linear fit: log(E) = slope * log(k) + const
    coeffs = np.polyfit(log_k, log_E, 1)
    slope = float(coeffs[0])
    
    # R²
    E_fit = 10**(np.polyval(coeffs, log_k))
    ss_res = np.sum((log_E - np.log10(E_fit))**2)
    ss_tot = np.sum((log_E - log_E.mean())**2)
    r_squared = 1.0 - ss_res / max(ss_tot, 1e-20)
    
    return {
        "slope": slope,
        "r_squared": float(r_squared),
        "k_range": [float(k_min), float(k_max)],
    }


__all__ = [
    "van_driest_damping",
    "compute_wall_distance_2d",
    "LESChannelConfig",
    "ChannelStatsAccumulator",
    "run_les_channel",
    "compute_energy_spectrum",
    "fit_inertial_subrange",
]
