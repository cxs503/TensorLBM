"""Navier slip BC equivalent path for superhydrophobic surfaces.

Instead of resolving the microstructure explicitly, this path applies
the Philip(1972) effective slip length as a macroscopic Navier slip BC
and verifies that the resulting flow field matches the explicit
microstructure simulation.

Method
------
1. Compute b_eff from Philip(1972) for the given stripe geometry.
2. Run a smooth-wall channel flow with Navier slip BC (b_eff) at the bottom.
3. Compare velocity profiles and slip velocities with the explicit simulation.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional

from tensorlbm.d3q19 import C, W, OPPOSITE, equilibrium3d, macroscopic3d
from tensorlbm.boundary.navier_slip_bc import navier_slip_cells_3d, slip_length_to_beta
from .philip1972 import philip_slip_length_lattice


@dataclass
class NavierSlipConfig:
    """Configuration for Navier slip equivalent simulation."""
    nx: int = 64
    ny: int = 33
    nz: int = 32
    a_lu: int = 8         # free-slip stripe width
    b_lu: int = 8         # no-slip stripe width
    tau: float = 0.6
    u_lid: float = 0.05
    n_steps: int = 5000
    device: str = "cpu"


def stream3d_periodic_xz(f: torch.Tensor) -> torch.Tensor:
    """Stream with periodic x and z, bounded y."""
    f_new = torch.empty_like(f)
    c_dev = C.to(f.device)
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


def run_navier_slip_equivalent(
    config: Optional[NavierSlipConfig] = None,
    verbose: bool = False,
) -> dict:
    """Run channel flow with Navier slip BC using Philip(1972) slip length.

    Returns
    -------
    result : dict
        Contains:
        - 'velocity_profile': u_x(y) averaged over x,z
        - 'y_coords': y positions
        - 'b_eff_applied': Philip slip length used
        - 'u_slip': measured slip velocity at bottom wall
        - 'b_eff_measured': extracted from velocity profile
    """
    if config is None:
        config = NavierSlipConfig()

    device = torch.device(config.device)
    nx, ny, nz = config.nx, config.ny, config.nz

    # Compute Philip(1972) slip length
    b_eff_philip = philip_slip_length_lattice(config.a_lu, config.b_lu, direction="parallel")

    # Initialize
    rho = torch.ones((nz, ny, nx), dtype=torch.float32, device=device)
    ux = torch.zeros_like(rho)
    uy = torch.zeros_like(rho)
    uz = torch.zeros_like(rho)
    f = equilibrium3d(rho, ux, uy, uz)

    # Wall masks
    bottom_mask = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    bottom_mask[:, 0, :] = True
    top_mask = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    top_mask[:, ny - 1, :] = True

    opp_dev = OPPOSITE.to(device)

    for step in range(config.n_steps):
        # Collision
        rho, ux, uy, uz = macroscopic3d(f)
        feq = equilibrium3d(rho, ux, uy, uz)
        f = f - (f - feq) / config.tau

        # Streaming
        f = stream3d_periodic_xz(f)

        # Bottom wall: Navier slip with Philip's b_eff
        f = navier_slip_cells_3d(f, bottom_mask, axis=1, Ls=b_eff_philip)

        # Top wall: moving lid (no-slip bounce-back + correction)
        f_bb_top = f[opp_dev]
        f = torch.where(top_mask.unsqueeze(0), f_bb_top, f)
        if abs(config.u_lid) > 1e-12:
            c_dev = C.to(device).float()
            w_dev = W.to(device).float()
            cu = c_dev[:, 0] * config.u_lid
            correction = 2.0 * w_dev * 1.0 * 3.0 * cu
            f = f + correction.view(19, 1, 1, 1) * top_mask.unsqueeze(0).float()

        if verbose and (step + 1) % 500 == 0:
            rho, ux, uy, uz = macroscopic3d(f)
            print(f"Step {step+1}: u_max = {ux.max().item():.6f}")

    # Extract profile
    rho, ux, uy, uz = macroscopic3d(f)
    u_profile = ux.mean(dim=(0, 2)).cpu().numpy()
    y_coords = np.arange(ny)

    # Extract slip velocity
    H = ny - 1
    n_fit = min(4, ny // 4)
    y_fit = y_coords[1:n_fit+1].astype(float)
    u_fit = u_profile[1:n_fit+1]
    coeffs = np.polyfit(y_fit, u_fit, 1)
    u_slip = coeffs[1]  # intercept

    # Measured b_eff from Couette formula
    denom = config.u_lid - 2.0 * u_slip
    if abs(denom) < 1e-10:
        b_eff_measured = float("inf")
    else:
        b_eff_measured = max(u_slip * H / denom, 0.0)

    return {
        "velocity_profile": u_profile,
        "y_coords": y_coords,
        "b_eff_applied": b_eff_philip,
        "b_eff_measured": b_eff_measured,
        "u_slip": u_slip,
        "config": config,
    }
