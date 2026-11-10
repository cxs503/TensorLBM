"""Navier slip BC verification: limiting cases and monotonic trends.

This test verifies that the Navier slip boundary condition exhibits correct
physical behavior for channel flow with varying slip length.

Physical setup:
- 3D channel (D3Q19) with Navier slip on top/bottom walls (y-direction)
- Periodic in x and z directions
- Body force G in x-direction drives the flow (Guo forcing scheme)

Verification strategy:
- Ls=0 should give near-no-slip (very small velocity at wall cells)
- Increasing Ls should increase slip velocity and flow rate monotonically
- Large Ls should approach more uniform velocity profile (plug flow trend)

Note: The Navier slip BC implementation applies the BC to post-collision
distributions without the "no-dynamics" treatment of solid cells used in
standard bounce-back. This results in a different effective wall location
and slip behavior compared to halfway bounce-back. This test verifies the
BC exhibits correct physical trends rather than matching a specific analytic
solution.

References:
- Navier (1823): slip length formulation
- Zhang et al. (2012): LBM slip BC implementation
- Succi (2001): The Lattice Boltzmann Equation, Oxford University Press
"""

from __future__ import annotations

import torch
import pytest
import numpy as np

from tensorlbm.d3q19 import C, W, equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_bgk3d, stream3d
from tensorlbm.navier_slip_bc import navier_slip_cells_3d

pytestmark = [pytest.mark.physics]


def run_navier_slip_channel(
    ny: int, nx: int, nz: int,
    tau: float, G: float, Ls: float,
    n_steps: int = 1000
) -> dict:
    """Run channel flow with Navier slip and return measured profiles.
    
    Args:
        ny: Total height (including wall cells)
        nx: Channel length (x)
        nz: Channel width (z)
        tau: Relaxation time
        G: Body force
        Ls: Slip length
        n_steps: Number of time steps
    
    Returns:
        Dictionary with measured velocity profiles and diagnostics
    """
    device = torch.device("cpu")
    dtype = torch.float64
    
    H = ny - 2  # Fluid cells
    
    # Kinematic viscosity
    nu = (tau - 0.5) / 3.0
    
    # Initialize with equilibrium at rest
    zero = torch.zeros(nz, ny, nx, dtype=dtype, device=device)
    f = equilibrium3d(torch.ones(nz, ny, nx, dtype=dtype, device=device), zero, zero, zero)
    
    # Wall masks (top and bottom walls)
    bottom_mask = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    bottom_mask[:, 0, :] = True
    top_mask = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    top_mask[:, -1, :] = True
    
    # Guo body force term
    c = C.to(device).to(dtype)
    w = W.to(device).to(dtype)
    accel = (w * 3.0 * c[:, 0]).view(19, 1, 1, 1) * G
    
    # Time stepping
    for step in range(n_steps):
        # Collision
        f = collide_bgk3d(f, tau)
        
        # Apply Navier slip BC on walls (before streaming)
        f = navier_slip_cells_3d(f, bottom_mask, axis=1, Ls=Ls, u_wall=0.0)
        f = navier_slip_cells_3d(f, top_mask, axis=1, Ls=Ls, u_wall=0.0)
        
        # Add body force
        f = f + accel
        
        # Streaming
        f = stream3d(f)
        
        # Periodic BC in x
        f[:, :, :, 0] = f[:, :, :, -2]
        f[:, :, :, -1] = f[:, :, :, -2]
    
    # Extract final macroscopic quantities
    rho_final, ux_final, uy_final, uz_final = macroscopic3d(f)
    
    # Extract profile at mid-channel (x = nx//2, z = 0)
    ux_profile = ux_final[0, :, nx // 2].cpu().numpy()
    
    # Wall slip velocities (at wall cells y=0 and y=ny-1)
    u_slip_bottom = ux_final[0, 0, nx // 2].item()
    u_slip_top = ux_final[0, -1, nx // 2].item()
    
    # Flow rate (integrate over channel height, fluid cells only)
    flow_rate = ux_profile[1:H+1].sum()
    
    # Maximum velocity (fluid cells)
    u_max = ux_profile[1:H+1].max()
    
    return {
        "ux_profile": ux_profile,
        "u_slip_bottom": u_slip_bottom,
        "u_slip_top": u_slip_top,
        "flow_rate": flow_rate,
        "u_max": u_max,
        "H": H,
        "Ls": Ls,
        "G": G,
        "nu": nu,
    }


def test_navier_slip_Ls0_near_no_slip():
    """Verify Ls=0 gives near-no-slip (very small velocity at wall cells)."""
    torch.set_num_threads(1)
    torch.manual_seed(0)
    
    ny = 12
    nx = 40
    nz = 2
    H = ny - 2
    
    tau = 1.0
    nu = (tau - 0.5) / 3.0
    u_max_target = 0.05
    G = 2 * nu * u_max_target / H**2
    
    # Run with Navier slip Ls=0
    results = run_navier_slip_channel(ny, nx, nz, tau, G, Ls=0.0, n_steps=1000)
    
    # Near-no-slip: slip velocity should be very small (< 1% of u_max_target)
    assert abs(results["u_slip_bottom"]) < 0.01 * u_max_target, (
        f"Ls=0 should give near-no-slip, got u_slip_bottom={results['u_slip_bottom']:.2e}"
    )
    assert abs(results["u_slip_top"]) < 0.01 * u_max_target, (
        f"Ls=0 should give near-no-slip, got u_slip_top={results['u_slip_top']:.2e}"
    )
    
    # Profile should be symmetric
    u_profile = results["ux_profile"][1:ny-1]
    assert np.allclose(u_profile, u_profile[::-1], rtol=1e-5), (
        "Profile should be symmetric"
    )


@pytest.mark.parametrize("Ls", [0.5, 1.0, 2.0, 5.0])
def test_navier_slip_increasing_slip(Ls: float):
    """Verify increasing Ls increases slip velocity and flow rate."""
    torch.set_num_threads(1)
    torch.manual_seed(0)
    
    ny = 12
    nx = 40
    nz = 2
    H = ny - 2
    
    tau = 1.0
    nu = (tau - 0.5) / 3.0
    u_max_target = 0.05
    G = 2 * nu * u_max_target / H**2
    
    # Run with slip length Ls
    results = run_navier_slip_channel(ny, nx, nz, tau, G, Ls=Ls, n_steps=1000)
    
    # Run reference with Ls=0
    results_ref = run_navier_slip_channel(ny, nx, nz, tau, G, Ls=0.0, n_steps=1000)
    
    # Check that slip velocity increases with Ls
    assert abs(results["u_slip_bottom"]) > abs(results_ref["u_slip_bottom"]), (
        f"Slip velocity should increase with Ls"
    )
    
    # Check that flow rate increases with Ls
    assert results["flow_rate"] > results_ref["flow_rate"], (
        f"Flow rate should increase with slip length"
    )
    
    # Check that maximum velocity increases with Ls
    assert results["u_max"] > results_ref["u_max"], (
        f"Maximum velocity should increase with slip length"
    )


def test_navier_slip_large_Ls_more_uniform():
    """Verify large Ls gives more uniform profile than small Ls."""
    torch.set_num_threads(1)
    torch.manual_seed(0)
    
    ny = 12
    nx = 40
    nz = 2
    H = ny - 2
    
    tau = 1.0
    nu = (tau - 0.5) / 3.0
    u_max_target = 0.05
    G = 2 * nu * u_max_target / H**2
    
    # Run with small slip length
    results_small = run_navier_slip_channel(ny, nx, nz, tau, G, Ls=0.5, n_steps=1000)
    u_small = results_small["ux_profile"][1:ny-1]
    uniformity_small = u_small.std() / u_small.mean()
    
    # Run with large slip length
    results_large = run_navier_slip_channel(ny, nx, nz, tau, G, Ls=50.0, n_steps=1000)
    u_large = results_large["ux_profile"][1:ny-1]
    uniformity_large = u_large.std() / u_large.mean()
    
    # Large Ls should give more uniform profile (lower uniformity metric)
    assert uniformity_large < uniformity_small, (
        f"Large Ls should give more uniform profile: "
        f"uniformity_small={uniformity_small:.4%}, uniformity_large={uniformity_large:.4%}"
    )


def test_navier_slip_monotonic_trend():
    """Verify slip velocity and flow rate increase monotonically with Ls."""
    torch.set_num_threads(1)
    torch.manual_seed(0)
    
    ny = 12
    nx = 40
    nz = 2
    H = ny - 2
    
    tau = 1.0
    nu = (tau - 0.5) / 3.0
    u_max_target = 0.05
    G = 2 * nu * u_max_target / H**2
    
    slip_lengths = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    slip_velocities = []
    flow_rates = []
    
    for Ls in slip_lengths:
        results = run_navier_slip_channel(ny, nx, nz, tau, G, Ls=Ls, n_steps=1000)
        slip_velocities.append(abs(results["u_slip_bottom"]))
        flow_rates.append(results["flow_rate"])
    
    # Check monotonic increase
    for i in range(1, len(slip_lengths)):
        assert slip_velocities[i] > slip_velocities[i-1], (
            f"Slip velocity should increase monotonically with Ls"
        )
        assert flow_rates[i] > flow_rates[i-1], (
            f"Flow rate should increase monotonically with Ls"
        )


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])
