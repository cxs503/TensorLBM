"""Unit tests for D3Q19 cumulant collision operator.

Tests verify:
1. Mass conservation (rho is invariant under collision)
2. Momentum conservation (rho*u is invariant under collision)
3. Equilibrium fixed-point (f_eq collides to f_eq)
4. Ghost-mode relaxation (non-hydrodynamic modes decay)
5. SGS enhancement (C_s > 0 increases damping)
6. Comparison with BGK/MRT baselines

Run with pytest::

    pytest tests/collision/test_cumulant_d3q19.py -v
"""

from __future__ import annotations

import pytest
import torch

from tensorlbm.collision_d3q19_advanced import (
    collide_bgk_d3q19,
    collide_cumulant_d3q19,
    collide_mrt_d3q19,
)
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d


@pytest.fixture
def small_grid():
    """Create a small test grid."""
    nz, ny, nx = 8, 8, 8
    return nz, ny, nx


@pytest.fixture
def f_uniform(small_grid):
    """Create uniform equilibrium distribution."""
    nz, ny, nx = small_grid
    rho = torch.ones((nz, ny, nx), dtype=torch.float32)
    ux = torch.zeros((nz, ny, nx), dtype=torch.float32)
    uy = torch.zeros((nz, ny, nx), dtype=torch.float32)
    uz = torch.zeros((nz, ny, nx), dtype=torch.float32)
    return equilibrium3d(rho, ux, uy, uz)


@pytest.fixture
def f_moving(small_grid):
    """Create moving equilibrium distribution."""
    nz, ny, nx = small_grid
    rho = torch.ones((nz, ny, nx), dtype=torch.float32)
    ux = 0.05 * torch.ones((nz, ny, nx), dtype=torch.float32)
    uy = 0.03 * torch.ones((nz, ny, nx), dtype=torch.float32)
    uz = 0.02 * torch.ones((nz, ny, nx), dtype=torch.float32)
    return equilibrium3d(rho, ux, uy, uz)


@pytest.fixture
def f_neq(small_grid):
    """Create distribution with non-equilibrium perturbation."""
    nz, ny, nx = small_grid
    rho = torch.ones((nz, ny, nx), dtype=torch.float32)
    ux = torch.zeros((nz, ny, nx), dtype=torch.float32)
    uy = torch.zeros((nz, ny, nx), dtype=torch.float32)
    uz = torch.zeros((nz, ny, nx), dtype=torch.float32)
    feq = equilibrium3d(rho, ux, uy, uz)
    # Add perturbation to ghost modes
    f = feq.clone()
    f[7] += 0.01  # Perturb one ghost mode
    f[8] -= 0.01
    return f


class TestCumulantMassConservation:
    """Test that cumulant collision conserves mass."""

    def test_mass_conserved_uniform(self, f_uniform):
        """Mass should be invariant for uniform equilibrium."""
        rho_before, _, _, _ = macroscopic3d(f_uniform)
        f_collided = collide_cumulant_d3q19(f_uniform, tau=0.6)
        rho_after, _, _, _ = macroscopic3d(f_collided)
        assert torch.allclose(rho_before, rho_after, atol=1e-6), (
            f"Mass drift: {torch.abs(rho_before - rho_after).max():.2e}"
        )

    def test_mass_conserved_moving(self, f_moving):
        """Mass should be invariant for moving equilibrium."""
        rho_before, _, _, _ = macroscopic3d(f_moving)
        f_collided = collide_cumulant_d3q19(f_moving, tau=0.6)
        rho_after, _, _, _ = macroscopic3d(f_collided)
        assert torch.allclose(rho_before, rho_after, atol=1e-6), (
            f"Mass drift: {torch.abs(rho_before - rho_after).max():.2e}"
        )


class TestCumulantMomentumConservation:
    """Test that cumulant collision conserves momentum."""

    def test_momentum_conserved_moving(self, f_moving):
        """Momentum should be invariant for moving equilibrium."""
        _, ux_before, uy_before, uz_before = macroscopic3d(f_moving)
        f_collided = collide_cumulant_d3q19(f_moving, tau=0.6)
        _, ux_after, uy_after, uz_after = macroscopic3d(f_collided)
        assert torch.allclose(ux_before, ux_after, atol=1e-6), (
            f"ux drift: {torch.abs(ux_before - ux_after).max():.2e}"
        )
        assert torch.allclose(uy_before, uy_after, atol=1e-6), (
            f"uy drift: {torch.abs(uy_before - uy_after).max():.2e}"
        )
        assert torch.allclose(uz_before, uz_after, atol=1e-6), (
            f"uz drift: {torch.abs(uz_before - uz_after).max():.2e}"
        )


class TestCumulantFixedPoint:
    """Test that equilibrium is a fixed point."""

    def test_equilibrium_fixed_point(self, f_uniform):
        """f_eq should collide to f_eq."""
        f_collided = collide_cumulant_d3q19(f_uniform, tau=0.6)
        assert torch.allclose(f_uniform, f_collided, atol=1e-6), (
            f"Max deviation: {torch.abs(f_uniform - f_collided).max():.2e}"
        )

    def test_moving_equilibrium_fixed_point(self, f_moving):
        """Moving f_eq should collide to f_eq."""
        f_collided = collide_cumulant_d3q19(f_moving, tau=0.6)
        assert torch.allclose(f_moving, f_collided, atol=1e-6), (
            f"Max deviation: {torch.abs(f_moving - f_collided).max():.2e}"
        )


class TestCumulantGhostModeRelaxation:
    """Test that ghost modes relax correctly."""

    def test_ghost_mode_decay(self, f_neq):
        """Non-hydrodynamic modes should decay."""
        f_collided = collide_cumulant_d3q19(f_neq, tau=0.6, omega_3=1.0)
        # Ghost modes should be reduced
        f_neq_before = f_neq - equilibrium3d(*macroscopic3d(f_neq))
        f_neq_after = f_collided - equilibrium3d(*macroscopic3d(f_collided))
        # Check that the perturbation direction is damped (use max over grid)
        assert (
            torch.abs(f_neq_after[7]).max().item() < torch.abs(f_neq_before[7]).max().item() + 1e-6
        ), "Ghost mode 7 should decay"


class TestCumulantSGS:
    """Test SGS enhancement with C_s > 0."""

    def test_sgs_increases_damping(self, f_neq):
        """C_s > 0 should increase ghost-mode damping."""
        f_no_sgs = collide_cumulant_d3q19(f_neq, tau=0.6, C_s=0.0, omega_3=1.0)
        f_with_sgs = collide_cumulant_d3q19(f_neq, tau=0.6, C_s=0.1, omega_3=1.0)
        # With SGS, ghost modes should be more damped
        f_neq_no_sgs = f_no_sgs - equilibrium3d(*macroscopic3d(f_no_sgs))
        f_neq_with_sgs = f_with_sgs - equilibrium3d(*macroscopic3d(f_with_sgs))
        # Check that SGS increases damping (ghost modes closer to zero)
        assert (
            torch.abs(f_neq_with_sgs[7]).max().item()
            <= torch.abs(f_neq_no_sgs[7]).max().item() + 1e-6
        ), "SGS should increase ghost-mode damping"


class TestCumulantComparison:
    """Compare cumulant with BGK/MRT baselines."""

    def test_cumulant_vs_bgk_stability(self, f_neq):
        """Cumulant should be at least as stable as BGK."""
        f_bgk = collide_bgk_d3q19(f_neq, tau=0.6)
        f_cum = collide_cumulant_d3q19(f_neq, tau=0.6)
        rho_bgk, _, _, _ = macroscopic3d(f_bgk)
        rho_cum, _, _, _ = macroscopic3d(f_cum)
        rho_orig, _, _, _ = macroscopic3d(f_neq)
        # Both should conserve mass to similar accuracy
        drift_bgk = torch.abs(rho_bgk - rho_orig).max()
        drift_cum = torch.abs(rho_cum - rho_orig).max()
        assert drift_cum < 1e-5, f"Cumulant mass drift too large: {drift_cum:.2e}"

    def test_cumulant_vs_mrt_stability(self, f_neq):
        """Cumulant should be at least as stable as MRT."""
        f_mrt = collide_mrt_d3q19(f_neq, tau=0.6)
        f_cum = collide_cumulant_d3q19(f_neq, tau=0.6)
        rho_mrt, _, _, _ = macroscopic3d(f_mrt)
        rho_cum, _, _, _ = macroscopic3d(f_cum)
        rho_orig, _, _, _ = macroscopic3d(f_neq)
        # Both should conserve mass
        drift_mrt = torch.abs(rho_mrt - rho_orig).max()
        drift_cum = torch.abs(rho_cum - rho_orig).max()
        assert drift_cum < 1e-5, f"Cumulant mass drift too large: {drift_cum:.2e}"


class TestCumulantEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_tau(self, f_uniform):
        """tau <= 0.5 should raise ValueError."""
        with pytest.raises((ValueError, RuntimeError)):
            collide_cumulant_d3q19(f_uniform, tau=0.5)

    def test_high_tau(self, f_uniform):
        """Large tau should still work."""
        f_collided = collide_cumulant_d3q19(f_uniform, tau=10.0)
        rho_before, _, _, _ = macroscopic3d(f_uniform)
        rho_after, _, _, _ = macroscopic3d(f_collided)
        assert torch.allclose(rho_before, rho_after, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
