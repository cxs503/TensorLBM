"""Tests for two-component SC solver with EDM forcing.

These tests validate the two-component Shan-Chen LBM implementation
for layered Poiseuille flow.
"""

import pytest
import numpy as np
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from tensorlbm.verification.layered_poiseuille.two_component_sc import (
    TwoComponentSCSolver, TwoComponentSCConfig, equilibrium, macroscopic,
    psi_exp
)
from tensorlbm.verification.layered_poiseuille.analytic import solve_layered


class TestTwoComponentSCBasic:
    """Basic functionality tests."""
    
    def test_solver_creation(self):
        """Test that solver can be created."""
        cfg = TwoComponentSCConfig(
            ny=16, nx=4,
            tau_a=0.8, tau_b=1.2,
            G=2.0, g_x=1e-5, g_y=0.0,
            steps=100,
        )
        solver = TwoComponentSCSolver(cfg)
        assert solver.f_a.shape == (9, 16, 4)
        assert solver.f_b.shape == (9, 16, 4)
    
    def test_step_advances_time(self):
        """Test that step() advances the simulation."""
        cfg = TwoComponentSCConfig(
            ny=16, nx=4,
            tau_a=0.8, tau_b=1.2,
            G=2.0, g_x=1e-5, g_y=0.0,
            steps=10,
        )
        solver = TwoComponentSCSolver(cfg)
        assert solver.step_count == 0
        solver.step()
        assert solver.step_count == 1
        solver.step()
        assert solver.step_count == 2
    
    def test_velocity_develops(self):
        """Test that velocity develops under body force."""
        cfg = TwoComponentSCConfig(
            ny=16, nx=4,
            tau_a=0.8, tau_b=1.2,
            G=2.0, g_x=1e-4, g_y=0.0,  # Stronger force for faster development
            steps=100,
        )
        solver = TwoComponentSCSolver(cfg)
        
        # Initial velocity
        _, ux_init, _ = macroscopic(solver.f_a)
        ux_max_init = ux_init.max().item()
        
        # Run simulation
        for _ in range(50):
            solver.step()
        
        # Final velocity
        macro = solver.get_macroscopic()
        ux_max_final = macro["ux_total"].max().item()
        
        # Velocity should increase
        assert ux_max_final > ux_max_init, "Velocity did not develop"


class TestEquilibrium:
    """Test equilibrium distribution."""
    
    def test_equilibrium_sums_to_rho(self):
        """Test that equilibrium sums to density."""
        rho = torch.ones(4, 4) * 2.0
        ux = torch.zeros(4, 4)
        uy = torch.zeros(4, 4)
        
        feq = equilibrium(rho, ux, uy)
        rho_recovered = feq.sum(dim=0)
        
        assert torch.allclose(rho, rho_recovered, atol=1e-6)
    
    def test_equilibrium_momentum(self):
        """Test that equilibrium gives correct momentum."""
        rho = torch.ones(4, 4) * 2.0
        ux = torch.ones(4, 4) * 0.1
        uy = torch.ones(4, 4) * 0.05
        
        feq = equilibrium(rho, ux, uy)
        rho_rec, ux_rec, uy_rec = macroscopic(feq)
        
        assert torch.allclose(rho, rho_rec, atol=1e-6)
        assert torch.allclose(ux, ux_rec, atol=1e-6)
        assert torch.allclose(uy, uy_rec, atol=1e-6)


class TestPseudopotential:
    """Test pseudopotential function."""
    
    def test_psi_exp_monotonic(self):
        """Test that exponential pseudopotential is monotonic."""
        rho = torch.linspace(0.1, 5.0, 50)
        psi = psi_exp(rho)
        
        # Should be monotonically increasing
        assert torch.all(psi[1:] >= psi[:-1])
    
    def test_psi_exp_bounds(self):
        """Test pseudopotential bounds."""
        rho = torch.tensor([0.0, 1.0, 10.0])
        psi = psi_exp(rho, rho0=1.0)
        
        # psi(0) = 0, psi(inf) -> rho0
        assert psi[0] == pytest.approx(0.0, abs=1e-6)
        assert psi[2] < 1.0  # Should approach rho0 but not exceed


class TestAnalyticalComparison:
    """Test comparison with analytical solution."""
    
    def test_analytical_solution_exists(self):
        """Test that analytical solution can be computed."""
        sol = solve_layered(
            height=30.0,
            interfaces=[15.0],
            nus=[0.1, 0.2],
            g=1e-5
        )
        
        y = np.linspace(1.0, 29.0, 50)  # Avoid exact wall positions
        u = sol.u(y)
        
        # Should be positive in interior
        assert np.all(u > 0)
    
    def test_analytical_continuity(self):
        """Test that analytical solution is approximately continuous at interface."""
        sol = solve_layered(
            height=30.0,
            interfaces=[15.0],
            nus=[0.1, 0.2],
            g=1e-5
        )
        
        # Evaluate just below and above interface
        y_below = np.array([14.5])
        y_above = np.array([15.5])
        
        u_below = sol.u(y_below)[0]
        u_above = sol.u(y_above)[0]
        
        # Should be approximately continuous (within 10% due to viscosity jump)
        rel_diff = abs(u_below - u_above) / max(u_below, u_above)
        assert rel_diff < 0.1, f"Solution not continuous: {rel_diff:.2%}"


class TestForcingSchemes:
    """Test different forcing schemes."""
    
    @pytest.mark.parametrize("forcing", ["edm", "guo", "velocity_shift"])
    def test_forcing_scheme_runs(self, forcing):
        """Test that each forcing scheme can run."""
        cfg = TwoComponentSCConfig(
            ny=16, nx=4,
            tau_a=0.8, tau_b=1.2,
            G=2.0, g_x=1e-5, g_y=0.0,
            steps=10,
            forcing=forcing,
        )
        solver = TwoComponentSCSolver(cfg)
        
        # Should run without error
        for _ in range(5):
            solver.step()
        
        # Should produce finite values
        macro = solver.get_macroscopic()
        assert np.isfinite(macro["ux_total"].max().item())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
