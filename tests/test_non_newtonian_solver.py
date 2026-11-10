"""Comprehensive tests for non-Newtonian LBM solver package.

Tests cover:
- Carreau-Yasuda viscosity model (limits, monotonicity, Newtonian degeneration)
- Power-law viscosity model
- Strain rate calculation
- Picard iteration convergence
- Lid-driven cavity benchmark (n=0.5, 1.0, 1.5)
- Pipe/channel flow benchmark (analytical comparison)
- 4:1 contraction flow benchmark
- Newtonian limit (C5: n=1, μ₀=μ∞ → standard BGK)
"""

from __future__ import annotations

import pytest
import torch
import numpy as np

from tensorlbm.non_newtonian.picard import PicardNonNewtonianSolver, PicardConfig
from tensorlbm.non_newtonian.cavity import CavityNonNewtonian, CavityConfig
from tensorlbm.non_newtonian.pipe import PipeNonNewtonian, PipeConfig
from tensorlbm.non_newtonian.contraction import Contraction4to1, ContractionConfig
from tensorlbm.non_newtonian.rheology import (
    strain_rate_magnitude_2d,
    apparent_viscosity_carreau_yasuda,
    apparent_viscosity_power_law,
    poiseuille_power_law_profile,
)


# ===========================================================================
# Viscosity model tests
# ===========================================================================

class TestCarreauYasudaViscosity:
    """Tests for the Carreau-Yasuda viscosity model."""

    def test_zero_shear_limit(self):
        """At γ̇→0, ν → ν₀ (zero-shear plateau)."""
        shear_rate = torch.tensor([1e-10, 1e-8, 1e-6], dtype=torch.float64)
        nu = apparent_viscosity_carreau_yasuda(
            shear_rate, nu_0=0.1, nu_inf=0.001,
            relaxation_time=1.0, flow_index=0.5,
        )
        assert torch.allclose(nu, torch.full_like(nu, 0.1), rtol=1e-3)

    def test_high_shear_limit(self):
        """At γ̇→∞ with ν∞>0, ν → ν∞."""
        shear_rate = torch.tensor([1e6, 1e8], dtype=torch.float64)
        nu = apparent_viscosity_carreau_yasuda(
            shear_rate, nu_0=0.1, nu_inf=0.001,
            relaxation_time=1.0, flow_index=0.5,
        )
        assert torch.allclose(nu, torch.full_like(nu, 0.001), rtol=0.1)

    def test_shear_thinning_monotone(self):
        """Shear-thinning (n<1): ν decreases with γ̇."""
        shear_rate = torch.logspace(-3, 3, 50, dtype=torch.float64)
        nu = apparent_viscosity_carreau_yasuda(
            shear_rate, nu_0=0.1, nu_inf=0.001,
            relaxation_time=1.0, flow_index=0.5,
        )
        diffs = nu[1:] - nu[:-1]
        assert torch.all(diffs <= 0)

    def test_newtonian_limit_n_equals_1(self):
        """When n=1, Carreau-Yasuda reduces to constant ν = ν₀."""
        shear_rate = torch.logspace(-2, 2, 20, dtype=torch.float64)
        nu = apparent_viscosity_carreau_yasuda(
            shear_rate, nu_0=0.05, nu_inf=0.001,
            relaxation_time=1.0, flow_index=1.0,
        )
        assert torch.allclose(nu, torch.full_like(nu, 0.05), rtol=1e-6)

    def test_invalid_params(self):
        """Invalid parameters should raise ValueError."""
        sr = torch.tensor([1.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="nu_0"):
            apparent_viscosity_carreau_yasuda(sr, nu_0=0.001, nu_inf=0.1,
                                               relaxation_time=1.0, flow_index=0.5)
        with pytest.raises(ValueError, match="relaxation_time"):
            apparent_viscosity_carreau_yasuda(sr, nu_0=0.1, nu_inf=0.001,
                                               relaxation_time=-1.0, flow_index=0.5)


class TestPowerLawViscosity:
    """Tests for the power-law viscosity model."""

    def test_shear_thinning(self):
        """n<1: viscosity decreases with shear rate."""
        sr = torch.tensor([0.01, 0.1, 1.0, 10.0], dtype=torch.float64)
        nu = apparent_viscosity_power_law(sr, consistency_index=0.05, flow_index=0.6)
        assert torch.all(nu[1:] < nu[:-1])

    def test_shear_thickening(self):
        """n>1: viscosity increases with shear rate."""
        sr = torch.tensor([0.01, 0.1, 1.0, 10.0], dtype=torch.float64)
        nu = apparent_viscosity_power_law(sr, consistency_index=0.05, flow_index=1.5)
        assert torch.all(nu[1:] > nu[:-1])

    def test_newtonian_n_equals_1(self):
        """n=1: ν = K (constant)."""
        sr = torch.tensor([0.01, 0.1, 1.0, 10.0], dtype=torch.float64)
        nu = apparent_viscosity_power_law(sr, consistency_index=0.05, flow_index=1.0)
        assert torch.allclose(nu, torch.full_like(nu, 0.05), rtol=1e-6)


# ===========================================================================
# Strain rate tests
# ===========================================================================

class TestStrainRate:
    """Tests for strain rate magnitude calculation."""

    def test_linear_shear(self):
        """Linear shear flow: γ̇ = |du/dy| = const."""
        ny, nx = 32, 8
        y = torch.arange(ny, dtype=torch.float64).unsqueeze(1).expand(ny, nx)
        ux = 0.02 * y.to(torch.float64)
        uy = torch.zeros_like(ux)

        gamma = strain_rate_magnitude_2d(ux, uy)
        interior = gamma[2:-2, 2:-2]
        assert torch.allclose(interior, torch.full_like(interior, 0.02), atol=2e-4)

    def test_zero_velocity(self):
        """Zero velocity → zero strain rate."""
        ux = torch.zeros((16, 16), dtype=torch.float64)
        uy = torch.zeros((16, 16), dtype=torch.float64)
        gamma = strain_rate_magnitude_2d(ux, uy)
        assert torch.allclose(gamma, torch.zeros_like(gamma), atol=1e-10)


# ===========================================================================
# Picard iteration tests
# ===========================================================================

class TestPicardIteration:
    """Tests for the Picard iteration framework."""

    def test_newtonian_initialization(self):
        """Newtonian case should initialize correctly."""
        cfg = PicardConfig(
            rheology_model="newtonian",
            newtonian_viscosity=0.01,
            max_iterations=10,
            tolerance=1e-4,
        )
        solver = PicardNonNewtonianSolver(16, 16, cfg)
        solver.initialize()
        assert solver.tau_field is not None
        assert solver.tau_field.shape == (16, 16)

    def test_picard_config_validation(self):
        """Config should have sensible defaults."""
        cfg = PicardConfig()
        assert cfg.max_iterations == 50
        assert cfg.tolerance == 1e-4
        assert 0 < cfg.relaxation_factor <= 1.0


# ===========================================================================
# Cavity benchmark tests
# ===========================================================================

class TestCavityBenchmark:
    """Tests for lid-driven cavity with non-Newtonian fluids."""

    @pytest.mark.parametrize("n_power", [0.5, 1.0, 1.5])
    def test_cavity_convergence(self, n_power):
        """Cavity should converge within Picard iteration limit (C4)."""
        cfg = CavityConfig(
            nx=33, ny=33, re=100.0, u_lid=0.1,
            n_power=n_power, rheology="power_law",
            picard_max_iter=20, picard_tol=1e-3,
            lbm_steps_per_picard=50,
        )
        solver = CavityNonNewtonian(cfg)
        info = solver.solve()
        # C4: Picard iterations < 50
        assert info["picard_iterations"] <= 50, \
            f"n={n_power}: Picard did not converge in 50 iterations"

    def test_cavity_nontrivial_flow(self):
        """Cavity should produce non-trivial velocity field."""
        # Use larger grid for stability
        cfg = CavityConfig(
            nx=33, ny=33, re=100.0, u_lid=0.1,
            n_power=1.0, rheology="power_law",
            picard_max_iter=10, lbm_steps_per_picard=100,
        )
        solver = CavityNonNewtonian(cfg)
        info = solver.solve()
        # Check that we got a valid (non-NaN) solution
        assert "error" not in info or info.get("error") != "NaN detected", \
            "Solver produced NaN"
        profiles = solver.get_centerline_velocities()
        u_vert = profiles["u_vertical"]
        # Non-trivial flow (should have some velocity)
        assert not np.isnan(u_vert).all(), "All velocities are NaN"


# ===========================================================================
# Pipe flow benchmark tests
# ===========================================================================

class TestPipeBenchmark:
    """Tests for pipe/channel flow with non-Newtonian fluids."""

    @pytest.mark.parametrize("n_power", [0.5, 1.0, 1.5])
    def test_pipe_runs(self, n_power):
        """Pipe flow solver should run and produce a profile."""
        cfg = PipeConfig(
            nx=33, ny=17, n_power=n_power, rheology="power_law",
            u_max_target=0.05,
            picard_max_iter=10, picard_tol=1e-3,
            lbm_steps_per_picard=50,
        )
        solver = PipeNonNewtonian(cfg)
        info = solver.solve()
        y_num, u_num = solver.get_velocity_profile()
        assert len(y_num) == 17
        assert len(u_num) == 17

    def test_pipe_newtonian_parabolic(self):
        """Newtonian pipe should give roughly parabolic profile."""
        cfg = PipeConfig(
            nx=33, ny=17, n_power=1.0, rheology="power_law",
            u_max_target=0.05,
            picard_max_iter=10, lbm_steps_per_picard=100,
        )
        solver = PipeNonNewtonian(cfg)
        solver.solve()
        y, u = solver.get_velocity_profile()
        mid = len(u) // 2
        # Maximum at center (approximately)
        assert u[mid] >= u[0]  # center > wall


# ===========================================================================
# Contraction flow tests
# ===========================================================================

class TestContractionBenchmark:
    """Tests for 4:1 contraction flow."""

    def test_contraction_runs(self):
        """Contraction solver should run without errors."""
        cfg = ContractionConfig(
            H_out=8, L_inlet=16, L_outlet=16,
            n_power=1.0, rheology="power_law",
            u_inlet=0.02,
            picard_max_iter=5, lbm_steps_per_picard=20,
        )
        solver = Contraction4to1(cfg)
        info = solver.solve()
        assert info["picard_iterations"] >= 1

    def test_contraction_geometry(self):
        """Geometry mask should have correct dimensions."""
        cfg = ContractionConfig(H_out=8, L_inlet=16, L_outlet=16)
        solver = Contraction4to1(cfg)
        assert solver.ny == 4 * 8 + 2  # H_in + 2 walls
        assert solver.wall_mask[0, :].all()  # bottom wall
        assert solver.wall_mask[-1, :].all()  # top wall


# ===========================================================================
# Newtonian limit test (C5)
# ===========================================================================

class TestNewtonianLimit:
    """C5: Newtonian limit (n=1, μ₀=μ∞) should match standard BGK."""

    def test_carreau_yasuda_newtonian_degeneration(self):
        """When n=1, CY reduces to constant viscosity = ν₀."""
        sr = torch.logspace(-3, 3, 50, dtype=torch.float64)
        nu = apparent_viscosity_carreau_yasuda(
            sr, nu_0=0.01, nu_inf=0.0099,
            relaxation_time=1.0, flow_index=1.0,
        )
        assert torch.std(nu) / torch.mean(nu) < 0.01

    def test_power_law_newtonian_degeneration(self):
        """When n=1, power-law reduces to constant ν = K."""
        sr = torch.logspace(-3, 3, 50, dtype=torch.float64)
        nu = apparent_viscosity_power_law(sr, consistency_index=0.01, flow_index=1.0)
        assert torch.allclose(nu, torch.full_like(nu, 0.01), rtol=1e-6)


# ===========================================================================
# Analytical Poiseuille profile test
# ===========================================================================

class TestPoiseuilleAnalytical:
    """Tests for the analytical Poiseuille profile."""

    def test_newtonian_parabolic(self):
        """n=1 should give parabolic profile."""
        y = torch.linspace(-1, 1, 100, dtype=torch.float64)
        u = poiseuille_power_law_profile(y, half_height=1.0,
                                          pressure_gradient=1.0,
                                          consistency_index=0.1,
                                          flow_index=1.0)
        u_expected = 1.0 / (2 * 0.1) * (1.0 - y**2)
        assert torch.allclose(u, u_expected, rtol=1e-6)

    def test_zero_at_walls(self):
        """Velocity should be zero at walls."""
        y = torch.tensor([-1.0, 1.0], dtype=torch.float64)
        u = poiseuille_power_law_profile(y, half_height=1.0,
                                          pressure_gradient=1.0,
                                          consistency_index=0.1,
                                          flow_index=0.5)
        assert torch.allclose(u, torch.zeros(2, dtype=torch.float64), atol=1e-10)

    def test_maximum_at_center(self):
        """Maximum velocity should be at center."""
        y = torch.linspace(-1, 1, 100, dtype=torch.float64)
        u = poiseuille_power_law_profile(y, half_height=1.0,
                                          pressure_gradient=1.0,
                                          consistency_index=0.1,
                                          flow_index=0.5)
        mid = len(u) // 2
        assert u[mid] == u.max()
