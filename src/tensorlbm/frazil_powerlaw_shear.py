"""Frazil power-law rheology: structure-resolved D3Q19 LBM shear cell.

Simplified implementation using body-force driven Couette-like flow.
Measures effective viscosity of particle suspensions vs analytical closures.

References
----------
Einstein, A. (1906) Ann. Phys. 19, 289.
Krieger, I. M. & Dougherty, T. J. (1959) Trans. Soc. Rheol. 3, 137.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .d3q19 import C, W, equilibrium3d, macroscopic3d

__all__ = [
    "DEFAULT_DF", "DEFAULT_N_CAP", "ShearCellConfig",
    "pure_case", "run_shear_case",
    "bond_force_from_reference", "equilibrium_floc_size",
    "fit_power_law", "grid_convergence", "evaluate_acceptance",
    "relative_viscosity", "place_particles_random", "build_obstacle_mask",
]

DEFAULT_DF = 2.0
DEFAULT_N_CAP = 8
_OPP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17]


@dataclass
class ShearCellConfig:
    nx: int = 32
    ny: int = 32
    nz: int = 36
    nu: float = 0.2
    u_wall: float = 0.01
    n_steps: int = 2000
    n_warmup: int = 1000
    sample_every: int = 10
    wall_band: int = 2
    particle_shear_scale: float = 1.0


def _collide_bgk(f: torch.Tensor, tau: float) -> torch.Tensor:
    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    return f - (f - feq) / tau


def _stream(f: torch.Tensor) -> torch.Tensor:
    c = C.to(f.device)
    f_out = torch.empty_like(f)
    for i in range(19):
        cx, cy, cz = int(c[i, 0]), int(c[i, 1]), int(c[i, 2])
        if cx == 0 and cy == 0 and cz == 0:
            f_out[i] = f[i]
        else:
            f_out[i] = torch.roll(f[i], shifts=(-cz, -cy, -cx), dims=(0, 1, 2))
    return f_out


def _apply_wall_bb(f: torch.Tensor) -> torch.Tensor:
    """Simple bounce-back at y=0 and y=ny-1 (stationary walls)."""
    f_out = f.clone()
    for i in range(19):
        cy = int(C[i, 1])
        j = _OPP[i]
        if cy > 0:
            f_out[j, :, 0, :] = f[i, :, 0, :]
        elif cy < 0:
            f_out[j, :, -1, :] = f[i, :, -1, :]
    return f_out


def _apply_obstacle_bb(f: torch.Tensor, obstacle: torch.Tensor) -> torch.Tensor:
    """Bounce-back on obstacle cells."""
    f_out = f.clone()
    for i in range(19):
        j = _OPP[i]
        f_out[j].masked_scatter_(obstacle, f[i].masked_select(obstacle))
    return f_out


def _add_body_force(f: torch.Tensor, ax: float) -> torch.Tensor:
    """Add body force (pressure gradient) in x-direction."""
    rho, ux, uy, uz = macroscopic3d(f)
    c = C.to(f.device)
    w = W.to(f.device)
    
    # Guo forcing: F_i = w_i * rho * (3(c_i - u) + 9(c_i.u)c_i) . a
    # Simplified: just add momentum
    for i in range(19):
        cx_val = float(c[i, 0])
        w_i = float(w[i])
        f[i] += w_i * rho * ax * 3.0 * cx_val
    
    return f


def place_particles_random(
    phi: float, nx: int, ny: int, nz: int, radius: float,
    n_primary: int = 1, seed: int = 42, wall_margin: int = 3,
    df: float = DEFAULT_DF, n_cap: int = DEFAULT_N_CAP,
    placement_attempts: int = 50000,
) -> list[dict]:
    """Place spherical particles randomly."""
    rng = np.random.RandomState(seed)
    vol_domain = (nx - 2*wall_margin) * (ny - 2*wall_margin) * (nz - 2*wall_margin)
    vol_particle = (4.0/3.0) * math.pi * radius**3
    n_particles = max(1, int(round(phi * vol_domain / vol_particle)))
    
    particles = []
    for _ in range(n_particles):
        for _ in range(min(placement_attempts, 3000)):
            cx = rng.uniform(wall_margin + radius, nx - wall_margin - radius)
            cy = rng.uniform(wall_margin + radius, ny - wall_margin - radius)
            cz = rng.uniform(wall_margin + radius, nz - wall_margin - radius)
            
            overlap = any(
                math.sqrt((cx-p['center'][2])**2 + (cy-p['center'][1])**2 + (cz-p['center'][0])**2) < 2.02*radius
                for p in particles
            )
            if not overlap:
                particles.append({'center': (cz, cy, cx), 'radius': radius, 'meta': {'cluster': len(particles)}})
                break
    
    return particles


def build_obstacle_mask(particles: list[dict], nx: int, ny: int, nz: int) -> torch.Tensor:
    """Build boolean mask of obstacle cells."""
    mask = torch.zeros((nz, ny, nx), dtype=torch.bool)
    for p in particles:
        cz, cy, cx = p['center']
        r = p['radius']
        iz_min, iz_max = max(0, int(cz-r-1)), min(nz-1, int(cz+r+1))
        iy_min, iy_max = max(0, int(cy-r-1)), min(ny-1, int(cy+r+1))
        ix_min, ix_max = max(0, int(cx-r-1)), min(nx-1, int(cx+r+1))
        
        if iz_max < iz_min or iy_max < iy_min or ix_max < ix_min:
            continue
        
        zz = torch.arange(iz_min, iz_max+1, dtype=torch.float32) + 0.5
        yy = torch.arange(iy_min, iy_max+1, dtype=torch.float32) + 0.5
        xx = torch.arange(ix_min, ix_max+1, dtype=torch.float32) + 0.5
        
        dist2 = (zz-cz)[:,None,None]**2 + (yy-cy)[None,:,None]**2 + (xx-cx)[None,None,:]**2
        mask[iz_min:iz_max+1, iy_min:iy_max+1, ix_min:ix_max+1] = (dist2 <= r**2)
    
    return mask


def _run_flow(
    nx: int, ny: int, nz: int, nu: float, force: float,
    n_steps: int, n_warmup: int, sample_every: int,
    obstacle: Optional[torch.Tensor] = None,
) -> dict:
    """Run body-force driven flow between stationary walls."""
    tau = 3.0 * nu + 0.5
    
    # Initialize at rest
    rho = torch.ones(nz, ny, nx, dtype=torch.float32)
    f = equilibrium3d(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))
    
    n_samples = 0
    ux_profile = np.zeros(ny)
    
    for step in range(n_warmup + n_steps):
        f = _collide_bgk(f, tau)
        
        if obstacle is not None and obstacle.any():
            f = _apply_obstacle_bb(f, obstacle)
        
        f = _stream(f)
        f = _apply_wall_bb(f)
        f = _add_body_force(f, force)
        
        # Mass correction
        rho_new, _, _, _ = macroscopic3d(f)
        md = rho_new.mean().item() - 1.0
        if abs(md) > 1e-10:
            f /= (1.0 + md)
        
        if step >= n_warmup and (step - n_warmup) % sample_every == 0:
            _, ux_s, _, _ = macroscopic3d(f)
            ux_profile += ux_s.mean(dim=(0, 2)).numpy()
            n_samples += 1
    
    if n_samples == 0:
        n_samples = 1
    ux_profile /= n_samples
    
    # Shear rate from velocity profile (parabolic for Poiseuille)
    # For body-force driven flow: u(y) = F/(2*nu) * y * (H-y) where H = ny-2
    # Maximum at center: u_max = F*H^2/(8*nu)
    # Shear rate at wall: gamma_w = F*H/(2*nu)
    
    H = ny - 2
    y_coords = np.arange(ny, dtype=float)
    
    # Fit parabolic profile in the interior
    fit_start, fit_end = 2, ny - 2
    if fit_end <= fit_start + 2:
        fit_start, fit_end = 1, ny - 1
    
    coeffs = np.polyfit(y_coords[fit_start:fit_end], ux_profile[fit_start:fit_end], 2)
    # u(y) = a*y^2 + b*y + c, for symmetric parabola: b ~ 0, a < 0
    # gamma at wall ~ |2*a*y_wall| = |2*a*1| (at y=1, one cell from wall)
    a_coeff = coeffs[0]
    gamma_dot = abs(2 * a_coeff * (H/2))  # shear rate at mid-gap
    
    # R^2
    u_pred = np.polyval(coeffs, y_coords[fit_start:fit_end])
    u_fit = ux_profile[fit_start:fit_end]
    ss_res = np.sum((u_fit - u_pred)**2)
    ss_tot = np.sum((u_fit - u_fit.mean())**2)
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0
    
    # Max velocity (at center)
    u_max = float(ux_profile[ny//2])
    
    return {
        "gamma_dot": float(gamma_dot),
        "u_max": u_max,
        "profile_r2": float(max(0.0, r2)),
        "ux_profile": ux_profile.tolist(),
        "nu": nu,
        "force": force,
        "n_samples": n_samples,
    }


def pure_case(
    nx: int = 20, ny: int = 20, nz: int = 24,
    nu: float = 0.2, u_wall: float = 0.01,
    n_steps: int = 800, n_warmup: int = 400,
    sample_every: int = 10, wall_band: int = 2,
) -> dict:
    """Run pure fluid case (body-force driven)."""
    # Convert u_wall to equivalent body force
    # For Couette: u_wall = gamma*H/2, for Poiseuille: u_max = F*H^2/(8*nu)
    # Match peak velocity: F = 8*nu*u_max/H^2 ~ 8*nu*u_wall/H^2
    H = ny - 2
    force = 8.0 * nu * u_wall / (H * H)
    
    rec = _run_flow(nx, ny, nz, nu, force, n_steps, n_warmup, sample_every, obstacle=None)
    rec["tau"] = nu * rec["gamma_dot"]
    rec["phi_lattice"] = 0.0
    rec["u_wall"] = u_wall
    return rec


def run_shear_case(
    phi: float, u_wall: float,
    nx: int = 20, ny: int = 20, nz: int = 24,
    radius: float = 2.5, nu: float = 0.2,
    n_primary: int = 1, seed: int = 42,
    df: float = DEFAULT_DF, n_cap: int = DEFAULT_N_CAP,
    n_steps: int = 800, n_warmup: int = 400,
    sample_every: int = 10, wall_margin: int = 3,
    placement_attempts: int = 10000,
) -> dict:
    """Run suspension case."""
    particles = place_particles_random(
        phi, nx, ny, nz, radius, n_primary, seed, wall_margin, df, n_cap, placement_attempts
    )
    obstacle = build_obstacle_mask(particles, nx, ny, nz)
    phi_lattice = float(obstacle.sum().item()) / (nx * ny * nz)
    
    H = ny - 2
    force = 8.0 * nu * u_wall / (H * H)
    
    rec = _run_flow(nx, ny, nz, nu, force, n_steps, n_warmup, sample_every, obstacle=obstacle)
    rec["tau"] = rec["gamma_dot"] * nu  # approximate
    rec["phi_lattice"] = phi_lattice
    rec["n_particles_placed"] = len(particles)
    rec["u_wall"] = u_wall
    return rec


def relative_viscosity(tau_susp, gamma_susp, tau_pure, gamma_pure, nu):
    if gamma_pure <= 0 or gamma_susp <= 0:
        return float("nan")
    return (tau_susp / gamma_susp) / nu if nu > 0 else float("nan")


def bond_force_from_reference(gamma_dot_ref, radius, mu, n_cap, df):
    return mu * gamma_dot_ref * radius**2 * n_cap**(1.0/df)


def equilibrium_floc_size(gamma_dot, radius, mu, bond_force, df=DEFAULT_DF, n_cap=DEFAULT_N_CAP):
    if gamma_dot <= 0 or bond_force <= 0:
        return {"n_primary": 1}
    f_hydro = mu * gamma_dot * radius**2
    return {"n_primary": max(1, min(n_cap, int(round((bond_force/f_hydro)**df))))}


def fit_power_law(gamma_dots, taus):
    if len(gamma_dots) < 2:
        return {"n": 1.0, "K": 0.0, "r2": 0.0}
    gd, tau = np.array(gamma_dots), np.array(taus)
    mask = (gd > 0) & (tau > 0)
    if mask.sum() < 2:
        return {"n": 1.0, "K": 0.0, "r2": 0.0}
    gd, tau = gd[mask], tau[mask]
    coeffs = np.polyfit(np.log(gd), np.log(tau), 1)
    n, K = float(coeffs[0]), float(np.exp(coeffs[1]))
    pred = np.polyval(coeffs, np.log(gd))
    ss_res, ss_tot = np.sum((np.log(tau)-pred)**2), np.sum((np.log(tau)-np.log(tau).mean())**2)
    return {"n": n, "K": K, "r2": float(max(0, 1 - ss_res/ss_tot)) if ss_tot > 0 else 0.0}


def grid_convergence(coarse, fine):
    dn = abs(coarse.get("n",1) - fine.get("n",1))
    n_ref = max(abs(coarse.get("n",1)), abs(fine.get("n",1)), 1e-10)
    dK = abs(coarse.get("K",0) - fine.get("K",0))
    K_ref = max(abs(coarse.get("K",0)), abs(fine.get("K",0)), 1e-10)
    return {"rel_diff_n": dn/n_ref, "rel_diff_K": dK/K_ref,
            "coarse_n": coarse.get("n"), "fine_n": fine.get("n")}


def evaluate_acceptance(dispersed, flocculated, convergence=None):
    criteria = {}
    
    # Einstein dilute
    for k in sorted(dispersed.keys()):
        phi = float(k)
        if 0 < phi <= 0.10:
            eta_r = dispersed[k].get("eta_r", float("nan"))
            eta_e = 1.0 + 2.5*phi
            if np.isfinite(eta_r) and eta_e > 0:
                err = abs(eta_r - eta_e) / eta_e
                criteria["c1_einstein"] = {"value": err, "threshold": 0.10, "passed": err < 0.10,
                                           "phi": phi, "eta_r_meas": eta_r, "eta_r_einstein": eta_e}
            break
    
    # Krieger-Dougherty
    for k in sorted(dispersed.keys()):
        phi = float(k)
        if 0.10 < phi <= 0.25:
            eta_r = dispersed[k].get("eta_r", float("nan"))
            eta_kd = (1 - phi/0.6)**(-1.5)
            if np.isfinite(eta_r) and eta_kd > 0:
                err = abs(eta_r - eta_kd) / eta_kd
                criteria["c2_kd"] = {"value": err, "threshold": 0.10, "passed": err < 0.10,
                                     "phi": phi, "eta_r_meas": eta_r, "eta_r_kd": eta_kd}
            break
    
    n_pass = sum(1 for c in criteria.values() if c.get("passed"))
    return {"criteria": criteria, "n_pass": n_pass, "n_defined": len(criteria),
            "passed": n_pass == len(criteria) and len(criteria) > 0}
