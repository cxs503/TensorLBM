"""NACA0012 airfoil verification for TensorLBM.

Verification against Abbott & von Doenhoff (1945) experimental data.
Standard LBM (D2Q9) is valid for Ma < 0.3 (weakly compressible regime).
For Ma = 0.3, we are at the upper limit of standard LBM validity.
For Ma > 0.3, compressibility errors become significant and results
should be interpreted with caution (honest limitation disclosure).

Physics:
- NACA 0012 symmetric airfoil (12% thickness)
- Chord-based Reynolds number Re_c
- Mach number Ma = U_inf / c_s (c_s = 1/sqrt(3) in lattice units)
- Angles of attack: 0° to 15°

Reference:
- Abbott, I.H. & von Doenhoff, A.E. (1945). "Theory of Wing Sections"
  Dover Publications, 1959 reprint. NACA0012 data at Re ≈ 3-9×10^6.
- Thin airfoil theory: dCl/dα = 2π per radian (incompressible, inviscid)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...boundaries import (
    apply_simple_channel_boundaries,
    bounce_back_cells,
    compute_obstacle_forces,
    make_channel_wall_mask,
)
from ...d2q9 import C, W, equilibrium, macroscopic
from ...solver import collide_mrt, correct_mass, stream
from ...utils import resolve_device
from .reference_data import abbott_cd_interp, abbott_cl_interp, thin_airfoil_cl


# ============================================================================
# NACA 4-digit geometry
# ============================================================================

def naca0012_thickness(xc: np.ndarray) -> np.ndarray:
    """NACA 0012 thickness distribution (symmetric, m=p=0, t=0.12).
    
    yt = 5*t*(0.2969*sqrt(x) - 0.1260*x - 0.3516*x^2 + 0.2843*x^3 - 0.1015*x^4)
    """
    t = 0.12
    yt = 5.0 * t * (
        0.2969 * np.sqrt(np.clip(xc, 0, None))
        - 0.1260 * xc
        - 0.3516 * xc**2
        + 0.2843 * xc**3
        - 0.1015 * xc**4
    )
    return yt


def build_naca0012_mask(
    nx: int, ny: int, chord: float, alpha_deg: float,
    cx: float | None = None, cy: float | None = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Build solid mask for NACA0012 airfoil at given angle of attack.
    
    Parameters
    ----------
    nx, ny : int
        Domain size in lattice units.
    chord : float
        Chord length in lattice units.
    alpha_deg : float
        Angle of attack in degrees.
    cx, cy : float, optional
        Center position (quarter-chord). Defaults to domain center.
    device : str
        Torch device.
    
    Returns
    -------
    mask : torch.Tensor of bool, shape (ny, nx)
        True where solid (airfoil interior).
    """
    if cx is None:
        cx = nx / 3.0
    if cy is None:
        cy = ny / 2.0
    
    alpha_rad = math.radians(alpha_deg)
    cos_a = math.cos(alpha_rad)
    sin_a = math.sin(alpha_rad)
    
    # Chordwise points (high resolution for smooth geometry)
    n_pts = max(300, int(chord * 8))
    xc = np.linspace(0.0, 1.0, n_pts)
    yt = naca0012_thickness(xc)
    
    # Upper and lower surfaces (symmetric: y_upper = yt, y_lower = -yt)
    x_upper = (xc - 0.25) * chord * cos_a + yt * chord * sin_a + cx
    y_upper = -(xc - 0.25) * chord * sin_a + yt * chord * cos_a + cy
    x_lower = (xc - 0.25) * chord * cos_a - yt * chord * sin_a + cx
    y_lower = -(xc - 0.25) * chord * sin_a - yt * chord * cos_a + cy
    
    # Build polygon mask using scanline fill
    poly_x = np.concatenate([x_upper, x_lower[::-1]])
    poly_y = np.concatenate([y_upper, y_lower[::-1]])
    
    mask = torch.zeros(ny, nx, dtype=torch.bool, device=device)
    
    x_min = max(0, int(poly_x.min()) - 1)
    x_max = min(nx - 1, int(poly_x.max()) + 1)
    y_min = max(0, int(poly_y.min()) - 1)
    y_max = min(ny - 1, int(poly_y.max()) + 1)
    
    for j in range(y_min, y_max + 1):
        intersections = []
        for i in range(len(poly_x) - 1):
            x1, y1 = poly_x[i], poly_y[i]
            x2, y2 = poly_x[i + 1], poly_y[i + 1]
            if (y1 <= j and y2 > j) or (y2 <= j and y1 > j):
                x_int = x1 + (j - y1) * (x2 - x1) / (y2 - y1 + 1e-10)
                intersections.append(x_int)
        intersections.sort()
        for k in range(0, len(intersections) - 1, 2):
            i_start = max(int(intersections[k]), x_min)
            i_end = min(int(intersections[k + 1]) + 1, x_max + 1)
            if i_start < i_end:
                mask[j, i_start:i_end] = True
    
    return mask


# ============================================================================
# Airfoil simulation
# ============================================================================

@dataclass
class AirfoilSimConfig:
    """Configuration for NACA0012 airfoil simulation."""
    chord: float = 100.0          # Chord in lattice units
    alpha_deg: float = 5.0        # Angle of attack
    u_in: float = 0.05            # Inlet velocity (Ma = u_in * sqrt(3))
    re: float = 500.0             # Chord-based Reynolds number
    nx: int = 400                 # Domain width
    ny: int = 200                 # Domain height
    n_steps: int = 20000          # Total simulation steps
    warmup_steps: int = 5000      # Steps before averaging
    use_mrt: bool = True          # Use MRT collision (more stable)
    device: str = "cpu"
    seed: int = 42
    
    @property
    def ma(self) -> float:
        """Mach number (lattice speed of sound c_s = 1/sqrt(3))."""
        return self.u_in * math.sqrt(3.0)
    
    @property
    def nu(self) -> float:
        """Kinematic viscosity from Re = U*c/nu."""
        return self.u_in * self.chord / self.re
    
    @property
    def tau(self) -> float:
        """BGK relaxation time."""
        return 3.0 * self.nu + 0.5
    
    @property
    def blockage_ratio(self) -> float:
        """Approximate blockage ratio (airfoil max thickness / domain height)."""
        return 0.12 * self.chord / self.ny


def run_airfoil_sim(config: AirfoilSimConfig) -> dict[str, Any]:
    """Run NACA0012 airfoil simulation and return Cl, Cd.
    
    Returns
    -------
    dict with keys:
        cl_sim, cd_sim: simulated lift and drag coefficients
        ma: Mach number
        re: Reynolds number
        alpha_deg: angle of attack
        blockage_ratio: domain blockage ratio
        n_samples: number of averaging samples
    """
    device = resolve_device(config.device)
    torch.manual_seed(config.seed)
    
    # Build geometry
    mask = build_naca0012_mask(
        config.nx, config.ny, config.chord, config.alpha_deg,
        device=device,
    )
    wall_mask = make_channel_wall_mask(config.ny, config.nx, mask, device)
    
    # Initialize flow field (uniform flow)
    rho0 = torch.ones(config.ny, config.nx, device=device)
    ux0 = torch.full_like(rho0, config.u_in)
    uy0 = torch.zeros_like(rho0)
    ux0[mask] = 0.0
    uy0[mask] = 0.0
    f = equilibrium(rho0, ux0, uy0, device=device)
    
    # Dynamic pressure for coefficient normalization
    q_inf = 0.5 * config.u_in**2 * config.chord  # per unit span
    initial_mass = float(rho0.sum().item())
    
    # Time-averaging
    cl_samples = []
    cd_samples = []
    
    for step in range(1, config.n_steps + 1):
        # Collision
        if config.use_mrt and config.tau < 0.8:
            f = collide_mrt(f, tau=config.tau)
        else:
            from ...solver import collide_bgk
            f = collide_bgk(f, tau=config.tau)
        
        # Streaming
        f = stream(f)
        
        # Boundary conditions
        f = apply_simple_channel_boundaries(
            f, u_in=config.u_in, wall_mask=wall_mask,
            obstacle_mask=torch.zeros_like(mask),
        )
        f = bounce_back_cells(f, mask)
        
        # Mass conservation correction
        f = correct_mass(f, initial_mass)
        
        # Force computation (momentum exchange)
        if step > config.warmup_steps:
            fx, fy = compute_obstacle_forces(f, mask)
            cd = float(fx.item()) / q_inf
            cl = float(fy.item()) / q_inf
            cl_samples.append(cl)
            cd_samples.append(cd)
    
    # Compute mean and uncertainty
    if len(cl_samples) > 0:
        cl_mean = float(np.mean(cl_samples))
        cd_mean = float(np.mean(cd_samples))
        cl_std = float(np.std(cl_samples))
        cd_std = float(np.std(cd_samples))
    else:
        cl_mean = cd_mean = cl_std = cd_std = 0.0
    
    return {
        "cl_sim": cl_mean,
        "cd_sim": cd_mean,
        "cl_std": cl_std,
        "cd_std": cd_std,
        "ma": config.ma,
        "re": config.re,
        "alpha_deg": config.alpha_deg,
        "blockage_ratio": config.blockage_ratio,
        "n_samples": len(cl_samples),
        "tau": config.tau,
    }


# ============================================================================
# Polar sweep
# ============================================================================

def run_polar_sweep(
    alpha_list: list[float],
    chord: float = 100.0,
    u_in: float = 0.05,
    re: float = 500.0,
    nx: int = 400,
    ny: int = 200,
    n_steps: int = 20000,
    warmup_steps: int = 5000,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run NACA0012 polar sweep over multiple angles of attack.
    
    Returns
    -------
    dict with keys:
        alpha_list: list of angles
        cl_sim, cd_sim: simulated coefficients
        cl_ref, cd_ref: Abbott-von Doenhoff reference
        cl_err_pct, cd_err_pct: percentage errors
        thin_airfoil_cl: thin airfoil theory values
    """
    results = {
        "alpha_list": alpha_list,
        "cl_sim": [],
        "cd_sim": [],
        "cl_ref": [],
        "cd_ref": [],
        "thin_airfoil_cl": [],
        "cl_err_pct": [],
        "cd_err_pct": [],
        "ma": [],
        "re": re,
    }
    
    for alpha in alpha_list:
        config = AirfoilSimConfig(
            chord=chord,
            alpha_deg=alpha,
            u_in=u_in,
            re=re,
            nx=nx,
            ny=ny,
            n_steps=n_steps,
            warmup_steps=warmup_steps,
            device=device,
        )
        
        sim = run_airfoil_sim(config)
        cl_ref = abbott_cl_interp(alpha)
        cd_ref = abbott_cd_interp(alpha)
        
        # Error computation
        cl_err = abs(sim["cl_sim"] - cl_ref) / max(abs(cl_ref), 0.01) * 100.0
        cd_err = abs(sim["cd_sim"] - cd_ref) / max(abs(cd_ref), 0.001) * 100.0
        
        results["cl_sim"].append(sim["cl_sim"])
        results["cd_sim"].append(sim["cd_sim"])
        results["cl_ref"].append(cl_ref)
        results["cd_ref"].append(cd_ref)
        results["thin_airfoil_cl"].append(thin_airfoil_cl(alpha))
        results["cl_err_pct"].append(cl_err)
        results["cd_err_pct"].append(cd_err)
        results["ma"].append(sim["ma"])
    
    return results


# ============================================================================
# Lift slope analysis
# ============================================================================

def compute_lift_slope(
    alpha_list: list[float],
    cl_list: list[float],
    linear_range: tuple[float, float] = (-4.0, 8.0),
) -> dict[str, float]:
    """Compute lift curve slope dCl/dα from simulation data.
    
    Parameters
    ----------
    alpha_list : list of float
        Angles of attack [deg].
    cl_list : list of float
        Corresponding lift coefficients.
    linear_range : tuple
        (alpha_min, alpha_max) for linear fit [deg].
    
    Returns
    -------
    dict with keys:
        slope_per_rad: dCl/dα [per radian]
        slope_per_deg: dCl/dα [per degree]
        thin_airfoil_slope: 2π per radian
        deviation_pct: percentage deviation from thin airfoil theory
    """
    alpha_arr = np.array(alpha_list)
    cl_arr = np.array(cl_list)
    
    # Select linear range
    mask = (alpha_arr >= linear_range[0]) & (alpha_arr <= linear_range[1])
    if mask.sum() < 2:
        return {
            "slope_per_rad": 0.0,
            "slope_per_deg": 0.0,
            "thin_airfoil_slope": 2.0 * math.pi,
            "deviation_pct": 100.0,
        }
    
    alpha_lin = alpha_arr[mask]
    cl_lin = cl_arr[mask]
    
    # Linear fit: Cl = slope * alpha + intercept
    coeffs = np.polyfit(np.radians(alpha_lin), cl_lin, 1)
    slope_per_rad = float(coeffs[0])
    slope_per_deg = slope_per_rad / math.radians(1.0)
    
    thin_slope = 2.0 * math.pi
    deviation_pct = abs(slope_per_rad - thin_slope) / thin_slope * 100.0
    
    return {
        "slope_per_rad": slope_per_rad,
        "slope_per_deg": slope_per_deg,
        "thin_airfoil_slope": thin_slope,
        "deviation_pct": deviation_pct,
    }


__all__ = [
    "naca0012_thickness",
    "build_naca0012_mask",
    "AirfoilSimConfig",
    "run_airfoil_sim",
    "run_polar_sweep",
    "compute_lift_slope",
]
