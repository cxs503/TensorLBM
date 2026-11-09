"""LBM explicit microstructure simulation for superhydrophobic surfaces.

Simulates a 3D channel flow (D3Q19) with a patterned bottom wall consisting
of alternating no-slip (solid ridge) and free-slip (gas-liquid interface)
stripes. The effective slip length is extracted by fitting the velocity
profile to the analytical Couette solution with Navier slip.

Method
------
1. Set up a 3D channel with periodic x (streamwise), bounded y (wall-normal),
   periodic z (spanwise).
2. Bottom wall (y=0): alternating stripes of bounce-back (no-slip) and
   specular reflection (free-slip) along the x-direction.
3. Top wall (y=ny-1): moving lid (Couette).
4. Run to steady state, extract u_x(y) profile averaged over x,z.
5. Fit the near-wall velocity to extract b_eff.

For Couette flow (top wall moving at U_lid, no pressure gradient):
    u(y) = U_lid * (y + b_eff) / (H + 2*b_eff)
where H = ny - 1 is the channel height.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional

from tensorlbm.d3q19 import C, W, OPPOSITE, equilibrium3d, macroscopic3d
from tensorlbm.boundary.boundaries3d import bounce_back_cells_3d, free_slip_cells_3d


@dataclass
class MicrostructureConfig:
    """Configuration for explicit microstructure LBM simulation."""
    nx: int = 64          # streamwise cells
    ny: int = 33          # wall-normal cells (channel height = ny-1)
    nz: int = 32          # spanwise cells
    a_lu: int = 8         # free-slip stripe width (lattice units)
    b_lu: int = 8         # no-slip stripe width (lattice units)
    tau: float = 0.6      # relaxation time
    u_lid: float = 0.05   # lid velocity (Couette)
    n_steps: int = 5000   # simulation steps
    output_interval: int = 500
    device: str = "cpu"


def create_stripe_mask(
    nx: int,
    nz: int,
    a_lu: int,
    b_lu: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create masks for alternating stripe pattern along x-direction.

    The pattern repeats with period (a_lu + b_lu):
    - x mod period < a_lu: free-slip (gas-liquid interface)
    - x mod period >= a_lu: no-slip (solid ridge)

    Returns
    -------
    free_slip_mask : (nz, nx) bool tensor
    no_slip_mask : (nz, nx) bool tensor
    """
    period = a_lu + b_lu
    x = torch.arange(nx, device=device)
    x_mod = x % period

    is_free = x_mod < a_lu
    is_noslip = ~is_free

    free_slip_mask = is_free.unsqueeze(0).expand(nz, -1).clone()
    no_slip_mask = is_noslip.unsqueeze(0).expand(nz, -1).clone()

    return free_slip_mask, no_slip_mask


def stream3d(f: torch.Tensor) -> torch.Tensor:
    """Standard pull streaming for D3Q19 with periodic x,z and bounded y."""
    c_dev = C.to(f.device)
    f_new = torch.empty_like(f)
    for q in range(19):
        cx, cy, cz = int(c_dev[q, 0]), int(c_dev[q, 1]), int(c_dev[q, 2])
        shifted = f[q]
        if cx != 0:
            shifted = torch.roll(shifted, shifts=cx, dims=-1)
        if cz != 0:
            shifted = torch.roll(shifted, shifts=cz, dims=-3)
        if cy != 0:
            shifted = torch.roll(shifted, shifts=cy, dims=-2)
        f_new[q] = shifted
    return f_new


def apply_couette_bc(
    f: torch.Tensor,
    bottom_free_mask: torch.Tensor,
    bottom_noslip_mask: torch.Tensor,
    u_lid: float,
    ny: int,
) -> torch.Tensor:
    """Apply Couette flow boundary conditions.

    Bottom wall (y=0): patterned - free-slip or bounce-back per stripe.
    Top wall (y=ny-1): moving lid with velocity u_lid (bounce-back + correction).

    Note: f has shape (19, nz, ny, nx).
    bottom_free_mask has shape (nz, nx).
    We need to create 3D masks of shape (nz, ny, nx).
    """
    device = f.device
    opp_dev = OPPOSITE.to(device)
    nz = f.shape[1]
    nx = f.shape[3]

    # --- Bottom wall (y=0) ---
    # Create 3D mask: shape (nz, ny, nx), True only at y=0 where stripe says so
    bottom_3d_free = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    bottom_3d_free[:, 0, :] = bottom_free_mask  # y=0 plane

    bottom_3d_noslip = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    bottom_3d_noslip[:, 0, :] = bottom_noslip_mask

    # Free-slip at bottom (specular reflection in y)
    f = free_slip_cells_3d(f, bottom_3d_free, axis=1)

    # No-slip at bottom (bounce-back)
    f = bounce_back_cells_3d(f, bottom_3d_noslip)

    # --- Top wall (y=ny-1) - moving lid ---
    top_3d = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    top_3d[:, ny - 1, :] = True

    # Bounce-back at top
    f_bb_top = f[opp_dev]
    f = torch.where(top_3d.unsqueeze(0), f_bb_top, f)

    # Moving wall correction for top lid
    if abs(u_lid) > 1e-12:
        c_dev = C.to(device).float()
        w_dev = W.to(device).float()
        cu = c_dev[:, 0] * u_lid
        correction = 2.0 * w_dev * 1.0 * 3.0 * cu
        f = f + correction.view(19, 1, 1, 1) * top_3d.unsqueeze(0).float()

    return f


def run_explicit_microstructure(
    config: Optional[MicrostructureConfig] = None,
    verbose: bool = False,
) -> dict:
    """Run LBM simulation with explicit microstructure pattern.

    Returns
    -------
    result : dict
        Contains velocity profile, slip length, etc.
    """
    if config is None:
        config = MicrostructureConfig()

    device = torch.device(config.device)
    nx, ny, nz = config.nx, config.ny, config.nz

    # Create stripe masks
    free_mask, noslip_mask = create_stripe_mask(
        nx, nz, config.a_lu, config.b_lu, device
    )

    # Initialize equilibrium at rest
    rho = torch.ones((nz, ny, nx), dtype=torch.float32, device=device)
    ux = torch.zeros_like(rho)
    uy = torch.zeros_like(rho)
    uz = torch.zeros_like(rho)
    f = equilibrium3d(rho, ux, uy, uz)

    # Time stepping
    for step in range(config.n_steps):
        # Collision (BGK)
        rho, ux, uy, uz = macroscopic3d(f)
        feq = equilibrium3d(rho, ux, uy, uz)
        f = f - (f - feq) / config.tau

        # Streaming
        f = stream3d(f)

        # Boundary conditions
        f = apply_couette_bc(f, free_mask, noslip_mask, config.u_lid, ny)

        if verbose and (step + 1) % config.output_interval == 0:
            rho, ux, uy, uz = macroscopic3d(f)
            u_max = ux.max().item()
            print(f"Step {step+1}: u_max = {u_max:.6f}")

    # Extract velocity profile
    rho, ux, uy, uz = macroscopic3d(f)
    u_profile = ux.mean(dim=(0, 2)).cpu().numpy()
    y_coords = np.arange(ny)

    # Extract effective slip length
    b_eff, u_slip = extract_slip_length_from_profile(
        u_profile, y_coords, config.u_lid, ny
    )

    phi = config.a_lu / (config.a_lu + config.b_lu)

    return {
        "velocity_profile": u_profile,
        "y_coords": y_coords,
        "b_eff_measured": b_eff,
        "u_slip": u_slip,
        "config": config,
        "period": config.a_lu + config.b_lu,
        "phi": phi,
        "a_lu": config.a_lu,
        "b_lu": config.b_lu,
    }


def extract_slip_length_from_profile(
    u_profile: np.ndarray,
    y_coords: np.ndarray,
    u_lid: float,
    ny: int,
) -> tuple[float, float]:
    """Extract effective slip length from velocity profile.

    For Couette flow with slip length b_eff:
        u(y) = u_lid * (y + b_eff) / (H + 2*b_eff)

    At bottom wall: u_slip = u_lid * b_eff / (H + 2*b_eff)
    => b_eff = u_slip * H / (u_lid - 2*u_slip)
    """
    H = ny - 1

    n_fit = min(4, ny // 4)
    y_fit = y_coords[1:n_fit+1].astype(float)
    u_fit = u_profile[1:n_fit+1]

    if len(y_fit) >= 2:
        coeffs = np.polyfit(y_fit, u_fit, 1)
        u_slip = coeffs[1]  # intercept = u at y=0
    else:
        u_slip = u_profile[1]

    denom = u_lid - 2.0 * u_slip
    if abs(denom) < 1e-10:
        b_eff = float("inf")
    else:
        b_eff = u_slip * H / denom

    b_eff = max(b_eff, 0.0)
    return b_eff, u_slip
