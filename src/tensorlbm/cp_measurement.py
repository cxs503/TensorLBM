"""Surface pressure coefficient (Cp) measurement for LBM benchmarks.

Cp = (p - p_inf) / (0.5 * rho * U_inf²)
   = 2*(rho - 1.0) / (3 * u_in²)  (lattice units, rho_0=1.0)

Measures Cp at near-wall fluid cells (first off-wall cell).
Outputs: Cp_min, Cp_max, Cp_mean, Cp_rms, and sampled distribution.
"""
from __future__ import annotations
import torch
import math
from typing import Optional


def compute_cp_field(f, u_in, rho_ref=1.0):
    """Compute Cp field from distribution function.
    
    Args:
        f: D3Q27 distribution [27, nz, ny, nx]
        u_in: inflow velocity (lattice units)
        rho_ref: reference density (default 1.0)
    
    Returns:
        Cp field [nz, ny, nx]
    """
    rho = f.sum(0)  # density
    cp = 2.0 * (rho - rho_ref) / (3.0 * u_in**2)
    return cp


def compute_cp_surface(f, solid, u_in, rho_ref=1.0):
    """Compute Cp at near-wall fluid cells (surface pressure).
    
    Args:
        f: distribution [27, nz, ny, nx]
        solid: boolean mask [nz, ny, nx]
        u_in: inflow velocity
        rho_ref: reference density
    
    Returns:
        cp_surface: Cp values at near-wall cells [nz, ny, nx] (NaN elsewhere)
        near_wall: boolean mask of near-wall fluid cells
    """
    fluid = ~solid
    # Near-wall: fluid cells adjacent to solid
    near_wall = torch.zeros_like(solid)
    for ax, sgn in [(2,1),(2,-1),(1,1),(1,-1),(0,1),(0,-1)]:
        near_wall |= (fluid & torch.roll(solid, sgn, dims=ax))
    
    cp = compute_cp_field(f, u_in, rho_ref)
    cp_surface = torch.where(near_wall, cp, torch.full_like(cp, float('nan')))
    
    return cp_surface, near_wall


def cp_statistics(cp_surface, near_wall):
    """Compute Cp statistics on the surface.
    
    Returns: dict with Cp_min, Cp_max, Cp_mean, Cp_rms, n_points
    """
    cp_vals = cp_surface[near_wall]
    if cp_vals.numel() == 0:
        return {'Cp_min': 0, 'Cp_max': 0, 'Cp_mean': 0, 'Cp_rms': 0, 'n_points': 0}
    
    return {
        'Cp_min': float(cp_vals.min()),
        'Cp_max': float(cp_vals.max()),
        'Cp_mean': float(cp_vals.mean()),
        'Cp_rms': float(cp_vals.std()),
        'n_points': int(cp_vals.numel()),
    }


def cp_profile_along_x(f, solid, u_in, rho_ref=1.0):
    """Cp profile along x-axis (for airfoils, ships, buildings).
    
    Returns: (x_positions, cp_upper, cp_lower) sampled at each x station.
    For 2D-like geometries (thin z), cp_upper = max Cp at each x, cp_lower = min.
    """
    cp = compute_cp_field(f, u_in, rho_ref)
    fluid = ~solid
    near_wall = torch.zeros_like(solid)
    for ax, sgn in [(2,1),(2,-1),(1,1),(1,-1),(0,1),(0,-1)]:
        near_wall |= (fluid & torch.roll(solid, sgn, dims=ax))
    
    nz, ny, nx = solid.shape
    x_vals = []
    cp_upper = []
    cp_lower = []
    
    for ix in range(nx):
        col = near_wall[:, :, ix]
        if col.any():
            cp_col = cp[:, :, ix]
            cp_vals = cp_col[col]
            x_vals.append(ix)
            cp_upper.append(float(cp_vals.max()))
            cp_lower.append(float(cp_vals.min()))
    
    return x_vals, cp_upper, cp_lower


def cp_profile_angular(f, solid, u_in, cx, cy, rho_ref=1.0, n_angles=36):
    """Cp profile vs angle (for cylinder, sphere).
    
    Returns: (angles_deg, cp_values) sampled at n_angles around the body.
    """
    cp = compute_cp_field(f, u_in, rho_ref)
    fluid = ~solid
    near_wall = torch.zeros_like(solid)
    for ax, sgn in [(2,1),(2,-1),(1,1),(1,-1),(0,1),(0,-1)]:
        near_wall |= (fluid & torch.roll(solid, sgn, dims=ax))
    
    nz, ny, nx = solid.shape
    zz, yy, xx = torch.meshgrid(torch.arange(nz), torch.arange(ny), torch.arange(nx), indexing='ij')
    
    # Ensure cx, cy are on the same device as the tensors
    device = solid.device
    cx_t = torch.tensor(float(cx), device=device)
    cy_t = torch.tensor(float(cy), device=device)
    
    angles = []
    cp_vals = []
    
    for i in range(n_angles):
        theta = 2 * math.pi * i / n_angles
        dx = math.cos(theta)
        dy = math.sin(theta)
        mask_angle = near_wall & (torch.abs((yy.float()-cy_t)*dx - (xx.float()-cx_t)*dy) < 3.0)
        dot = (xx.float()-cx_t)*dx + (yy.float()-cy_t)*dy
        mask_angle &= (dot > 0)
        
        if mask_angle.any():
            cp_angle = cp[mask_angle]
            angles.append(math.degrees(theta))
            cp_vals.append(float(cp_angle.mean()))
    
    return angles, cp_vals


def print_cp_report(f, solid, u_in, geometry_type='generic', cx=None, cy=None, rho_ref=1.0):
    """Print Cp report for a benchmark.
    
    Args:
        f: distribution
        solid: mask
        u_in: inflow velocity
        geometry_type: 'airfoil', 'cylinder', 'sphere', 'building', 'generic'
        cx, cy: center for angular profiles
        rho_ref: reference density
    """
    cp_surface, near_wall = compute_cp_surface(f, solid, u_in, rho_ref)
    stats = cp_statistics(cp_surface, near_wall)
    
    print(f"  Cp statistics:", flush=True)
    print(f"    Cp_min={stats['Cp_min']:.4f}  Cp_max={stats['Cp_max']:.4f}", flush=True)
    print(f"    Cp_mean={stats['Cp_mean']:.4f}  Cp_rms={stats['Cp_rms']:.4f}", flush=True)
    print(f"    n_points={stats['n_points']}", flush=True)
    
    if geometry_type in ('airfoil', 'ship', 'building', 'generic'):
        x_vals, cp_upper, cp_lower = cp_profile_along_x(f, solid, u_in, rho_ref)
        if x_vals:
            # Sample at 10 stations
            n = len(x_vals)
            step = max(n//10, 1)
            print(f"  Cp profile (x, upper, lower):", flush=True)
            for i in range(0, n, step):
                print(f"    x={x_vals[i]:4d}  Cp_up={cp_upper[i]:.3f}  Cp_lo={cp_lower[i]:.3f}", flush=True)
    
    if geometry_type in ('cylinder', 'sphere') and cx is not None and cy is not None:
        angles, cp_vals = cp_profile_angular(f, solid, u_in, cx, cy, rho_ref)
        if angles:
            print(f"  Cp vs angle:", flush=True)
            for a, c in zip(angles, cp_vals):
                print(f"    θ={a:5.1f}°  Cp={c:.3f}", flush=True)
    
    return stats
