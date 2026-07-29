"""BFL (Bouzidi-Firdaouss-Lallemand) interpolated bounce-back for curved boundaries.

Standard half-way bounce-back places the wall at the midpoint between
fluid and solid cells. For curved surfaces (cylinder, sphere), this creates
a staircase approximation. BFL interpolates to the exact wall position.

Reference:
  Bouzidi, Firdaouss & Lallemand (2001) "Momentum transfer of a 
  Boltzmann-lattice fluid on a curved boundary", Phys. Fluids 13:3452.

BFL formulas:
  For each boundary link with wall at distance ratio q ∈ (0,1):
    q < 0.5:  f_opp_new = 2q·f_i(x+c) + (1-2q)·f_i(x+2c)
    q ≥ 0.5:  f_opp_new = f_i(x+c)/(2q) + (2q-1)/(2q)·f_opp(x)

Momentum exchange with BFL:
  F = (f_i_pre + f_opp_post) · c_i / q
  (the 1/q factor accounts for fractional wall position)
"""
import torch
import math
from typing import Optional, Tuple


def compute_q_values_cylinder(
    solid: torch.Tensor,
    cx: float,
    cy: float,
    R: float,
    c: torch.Tensor,  # (19, 3) velocity set
) -> torch.Tensor:
    """Compute q-values for a cylinder boundary.
    
    For each near-wall fluid cell, for each direction pointing toward solid,
    compute q = (distance from cell centre to wall) / |c_i|.
    
    Args:
        solid: (nz, ny, nx) bool mask.
        cx, cy: cylinder centre in grid coordinates.
        R: cylinder radius in grid units.
        c: (19, 3) D3Q19 velocity vectors.
    
    Returns:
        q: (19, nz, ny, nx) float, q-value for each direction at each cell.
           0 means no boundary link in this direction.
    """
    nz, ny, nx = solid.shape
    fluid = ~solid
    
    # Create coordinate grids
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=solid.device, dtype=torch.float32),
        torch.arange(ny, device=solid.device, dtype=torch.float32),
        torch.arange(nx, device=solid.device, dtype=torch.float32),
        indexing='ij'
    )
    
    # Distance from cylinder centre (in 2D: y-x plane)
    dist = torch.sqrt((xx - cx)**2 + (yy - cy)**2)  # (nz, ny, nx)
    
    q = torch.zeros(19, nz, ny, nx, device=solid.device, dtype=torch.float32)
    
    for i in range(18):  # skip rest particle (18)
        ci = c[i]  # (3,)
        # Neighbor cell in direction c_i
        nx_i = xx + ci[2]  # x-component
        ny_i = yy + ci[1]  # y-component
        nz_i = zz + ci[0]  # z-component
        
        # Check if neighbor is within bounds and solid
        valid = (nx_i >= 0) & (nx_i < nx) & (ny_i >= 0) & (ny_i < ny) & (nz_i >= 0) & (nz_i < nz)
        nx_i_clamp = nx_i.clamp(0, nx-1).long()
        ny_i_clamp = ny_i.clamp(0, ny-1).long()
        nz_i_clamp = nz_i.clamp(0, nz-1).long()
        
        neighbor_solid = solid[nz_i_clamp, ny_i_clamp, nx_i_clamp] & valid & fluid
        
        if neighbor_solid.any():
            # Distance from current cell to wall along direction c_i
            # Wall is at radius R from centre
            # Parametric: point = (x, y) + t * (ci_x, ci_y), t ∈ [0, 1]
            # (x + t*ci_x - cx)^2 + (y + t*ci_y - cy)^2 = R^2
            # Solve for t: a*t^2 + b*t + c = 0
            a = ci[2]**2 + ci[1]**2  # dx^2 + dy^2 (skip z for 2D)
            b = 2 * (xx * ci[2] + yy * ci[1] - cx * ci[2] - cy * ci[1])
            cc = (xx - cx)**2 + (yy - cy)**2 - R**2
            
            discriminant = b**2 - 4*a*cc
            valid_quad = (discriminant > 0) & neighbor_solid
            
            if valid_quad.any():
                t = (-b - torch.sqrt(discriminant.clamp(min=0))) / (2 * a)
                # q = t (fraction along the link, should be in (0, 1))
                q_i = t.clamp(0.01, 0.99)
                q[i] = torch.where(valid_quad, q_i, torch.zeros_like(q_i))
    
    return q


def bfl_bounce_back(
    f: torch.Tensor,
    solid: torch.Tensor,
    near: torch.Tensor,
    q: torch.Tensor,
    c: torch.Tensor,  # (19, 3)
) -> torch.Tensor:
    """Apply BFL interpolated bounce-back.
    
    For each boundary link with q-value:
      q < 0.5:  f_opp_new = 2q·f_i(x+c) + (1-2q)·f_i(x+2c)
      q ≥ 0.5:  f_opp_new = f_i(x+c)/(2q) + (2q-1)/(2q)·f_opp(x)
    """
    nz, ny, nx = solid.shape
    device = f.device
    
    for i in range(18):
        if i == 18:
            continue
        opp = (i + 9) % 18
        if opp == i:
            continue
        
        ci = c[i]  # (3,)
        di, dj, dk = int(ci[2].item()), int(ci[1].item()), int(ci[0].item())
        
        # q-values for this direction
        q_i = q[i]  # (nz, ny, nx)
        has_link = (q_i > 0.01) & near
        
        if not has_link.any():
            continue
        
        # Get f at neighbor (x+c_i) and (x+2c_i)
        # x+c_i
        f_1 = torch.zeros_like(f[i])
        f_1[:, :, 1:-1] = f[i, :, :, 1+di:1+di+nx-2] if di != 0 else f[i, :, :, 1:-1]
        # Simplified: use roll but mask out non-link cells
        f_1 = torch.roll(f[i], (-dk, -dj, -di), dims=(0, 1, 2))
        
        # x+2c_i
        f_2 = torch.roll(f[i], (-2*dk, -2*dj, -2*di), dims=(0, 1, 2))
        
        # BFL interpolation
        mask = has_link.float()
        q_clamped = q_i.clamp(0.01, 0.99)
        
        # Case 1: q < 0.5
        case1 = (q_i < 0.5) & has_link
        if case1.any():
            f_new_1 = 2 * q_clamped * f_1 + (1 - 2 * q_clamped) * f_2
            f[opp] = torch.where(case1, f_new_1, f[opp])
        
        # Case 2: q >= 0.5
        case2 = (q_i >= 0.5) & has_link
        if case2.any():
            f_new_2 = f_1 / (2 * q_clamped) + (2 * q_clamped - 1) / (2 * q_clamped) * f[opp]
            f[opp] = torch.where(case2, f_new_2, f[opp])
    
    return f


def bfl_momentum_exchange_drag(
    f_pre: torch.Tensor,  # pre-bounce-back distribution
    f_post: torch.Tensor,  # post-bounce-back distribution
    solid: torch.Tensor,
    near: torch.Tensor,
    q: torch.Tensor,
    c: torch.Tensor,  # (19, 3)
    dpS: float,
) -> float:
    """BFL momentum exchange drag with q-correction.
    
    Force from each boundary link:
      F = (f_i_pre + f_opp_post) * c_i / q
    
    The 1/q factor accounts for the fractional wall position.
    """
    device = f_pre.device
    cx_k = c[:, 0].view(19, 1, 1, 1).to(device).float()
    
    dfric = torch.zeros(1, device=device)
    
    for i in range(18):
        if i == 18:
            continue
        opp = (i + 9) % 18
        if opp == i:
            continue
        
        q_i = q[i]
        has_link = (q_i > 0.01) & near
        
        if has_link.any():
            q_clamped = q_i.clamp(min=0.01)
            # Force = (f_i_pre + f_opp_post) * c_i_x / q
            force = (f_pre[i] + f_post[opp]) * cx_k[i] / q_clamped
            dfric += (force * has_link.float()).sum()
    
    return float(dfric.item() / dpS)


# ── Test ──
if __name__ == '__main__':
    print("BFL module compiled OK.")
    print("Key: q-value computation + interpolated bounce-back + q-corrected drag")
