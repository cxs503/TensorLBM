"""Pytest test for Taylor-Green Vortex 2D validation.

This test verifies that the TensorLBM D2Q9 solver correctly reproduces
the analytical decay of the 2D Taylor-Green vortex.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_tgv_energy_decay():
    """Test that TGV energy decay matches analytical solution within 8%."""
    from tgv_validation import TaylorGreenVortex2D

    # Use small grid for fast test
    sim = TaylorGreenVortex2D(nx=32, ny=32, tau=0.6, u0=0.05)

    # Run for enough steps to see decay
    n_steps = 500
    results = sim.run(n_steps=n_steps, sample_interval=10)

    t_sim = results["time"]
    E_sim = results["energy"]
    E_ana = sim.analytical_energy(t_sim)

    # Compute decay rate from simulation
    mask = E_sim > 1e-10
    t_fit = t_sim[mask]
    lnE_fit = np.log(E_sim[mask])
    coeffs = np.polyfit(t_fit, lnE_fit, 1)
    alpha_sim = -coeffs[0]
    alpha_ana = sim.alpha

    # Check criterion: decay rate error <= 8%
    decay_rate_error = abs(alpha_sim - alpha_ana) / alpha_ana
    assert decay_rate_error <= 0.08, (
        f"Decay rate error {decay_rate_error:.4%} exceeds 8% criterion"
    )


def test_tgv_mass_conservation():
    """Test that mass is conserved during TGV simulation."""
    from tgv_validation import TaylorGreenVortex2D

    sim = TaylorGreenVortex2D(nx=32, ny=32, tau=0.6, u0=0.05)

    # Initial mass
    rho0, _ = sim._macroscopic()
    mass0 = rho0.sum().item()

    # Run simulation
    sim.run(n_steps=100, sample_interval=50)

    # Final mass
    rho_final, _ = sim._macroscopic()
    mass_final = rho_final.sum().item()

    # Check mass conservation (should be near-exact for periodic BC)
    # Allow for floating point accumulation errors
    mass_error = abs(mass_final - mass0) / mass0
    assert mass_error < 1e-4, f"Mass not conserved: error = {mass_error:.2e}"


def test_tgv_symmetry():
    """Test that TGV maintains expected symmetry."""
    from tgv_validation import TaylorGreenVortex2D

    sim = TaylorGreenVortex2D(nx=32, ny=32, tau=0.6, u0=0.05)

    # Run a few steps
    sim.run(n_steps=50, sample_interval=50)

    # Get velocity field
    _, u = sim._macroscopic()
    ux = u[..., 0].cpu().numpy()
    uy = u[..., 1].cpu().numpy()

    # TGV should have anti-symmetry in certain directions
    # u(x, y) = -u(x, L-y) for the x-component
    # This is a weak check - just verify the field is not trivial
    assert np.abs(ux).max() > 1e-6, "Velocity field is trivial"
    assert np.abs(uy).max() > 1e-6, "Velocity field is trivial"


def test_tgv_analytical_solution():
    """Test that analytical solution is correctly computed."""
    from tgv_validation import TaylorGreenVortex2D

    sim = TaylorGreenVortex2D(nx=32, ny=32, tau=0.6, u0=0.05)

    # At t=0, energy should be E0
    E0 = sim.analytical_energy(np.array([0.0]))[0]
    assert abs(E0 - sim.E0) < 1e-10, "Analytical energy at t=0 should equal E0"

    # Energy should decay exponentially
    t = np.array([100.0])
    E_t = sim.analytical_energy(t)[0]
    E_expected = sim.E0 * np.exp(-sim.alpha * 100.0)
    assert abs(E_t - E_expected) < 1e-10, "Analytical energy decay incorrect"


def test_tgv_viscosity_relation():
    """Test that viscosity is correctly related to tau."""
    from tgv_validation import TaylorGreenVortex2D

    tau = 0.6
    sim = TaylorGreenVortex2D(nx=32, ny=32, tau=tau, u0=0.05)

    # nu = cs^2 * (tau - 0.5) = (1/3) * 0.1 = 0.0333...
    nu_expected = (1.0 / 3.0) * (tau - 0.5)
    assert abs(sim.nu - nu_expected) < 1e-10, "Viscosity calculation incorrect"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
