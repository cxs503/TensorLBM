"""Tests for H2-FRAZIL-RHEOLOGY: frazil power-law suspension rheology.

Tests the rheology closures (Einstein, Krieger-Dougherty) and the
LBM shear cell infrastructure.
"""

import math
import numpy as np
import pytest
import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorlbm.frazil_powerlaw_mature import (
    einstein_relative_viscosity,
    einstein_batchelor_green,
    krieger_dougherty,
    hydrodynamic_volume_fraction,
    physical_suspension,
    voxelise,
    RADIUS_FRACTION,
)
from tensorlbm.frazil_powerlaw_shear import (
    fit_power_law,
    place_particles_random,
    build_obstacle_mask,
    pure_case,
    DEFAULT_DF,
    DEFAULT_N_CAP,
)


# ---------------------------------------------------------------------------
# Rheology closures (analytical, should be exact)
# ---------------------------------------------------------------------------
class TestRheologyClosures:
    """Test the analytical rheology closures."""
    
    def test_einstein_phi_zero(self):
        """Einstein at phi=0 should give eta_r = 1."""
        assert einstein_relative_viscosity(0.0) == 1.0
    
    def test_einstein_dilute(self):
        """Einstein: eta_r = 1 + 2.5*phi."""
        for phi in [0.01, 0.02, 0.05]:
            expected = 1.0 + 2.5 * phi
            assert abs(einstein_relative_viscosity(phi) - expected) < 1e-10
    
    def test_einstein_at_005(self):
        """Einstein at phi=0.05: eta_r = 1.125."""
        assert abs(einstein_relative_viscosity(0.05) - 1.125) < 1e-10
    
    def test_ebg_reduces_to_einstein_at_low_phi(self):
        """EBG should be close to Einstein at low phi."""
        phi = 0.01
        e = einstein_relative_viscosity(phi)
        ebg = einstein_batchelor_green(phi)
        # EBG = 1 + 2.5*phi + 6.2*phi^2, at phi=0.01 the phi^2 term is 6.2e-4
        assert abs(ebg - e) < 0.01
    
    def test_kd_phi_zero(self):
        """KD at phi=0 should give eta_r = 1."""
        assert krieger_dougherty(0.0) == 1.0
    
    def test_kd_at_020(self):
        """KD at phi=0.20: eta_r = (1-0.2/0.6)^(-1.5) = 1.837."""
        expected = (1.0 - 0.20/0.60)**(-1.5)
        assert abs(krieger_dougherty(0.20) - expected) < 1e-10
    
    def test_kd_at_030(self):
        """KD at phi=0.30: eta_r = (1-0.3/0.6)^(-1.5) = 2.828."""
        expected = (1.0 - 0.30/0.60)**(-1.5)
        assert abs(krieger_dougherty(0.30) - expected) < 1e-10
    
    def test_kd_diverges_at_phi_max(self):
        """KD should diverge at phi_max."""
        assert krieger_dougherty(0.60) == float("inf")
    
    def test_kd_monotone_increasing(self):
        """KD should be monotonically increasing."""
        phis = np.linspace(0, 0.55, 20)
        etas = [krieger_dougherty(p) for p in phis]
        for i in range(1, len(etas)):
            assert etas[i] > etas[i-1]
    
    def test_hydrodynamic_volume_fraction_single_sphere(self):
        """For single spheres, phi_h = phi_lattice."""
        assert hydrodynamic_volume_fraction(0.1, n_primary=1) == 0.1
    
    def test_hydrodynamic_volume_fraction_floc(self):
        """For fractal aggregates, phi_h > phi_lattice."""
        phi_h = hydrodynamic_volume_fraction(0.1, n_primary=8, df=2.0)
        assert phi_h > 0.1


# ---------------------------------------------------------------------------
# Power law fitting
# ---------------------------------------------------------------------------
class TestPowerLawFit:
    """Test the power-law fitting function."""
    
    def test_perfect_power_law(self):
        """Fit should recover exact n and K for perfect power-law data."""
        K_true, n_true = 0.5, 0.8
        gamma = [0.001, 0.01, 0.1]
        tau = [K_true * g**n_true for g in gamma]
        fit = fit_power_law(gamma, tau)
        assert abs(fit["n"] - n_true) < 1e-6
        assert abs(fit["K"] - K_true) < 1e-6
        assert fit["r2"] > 0.999
    
    def test_newtonian_limit(self):
        """Newtonian fluid: n=1, K=nu."""
        nu = 0.2
        gamma = [0.001, 0.01, 0.1]
        tau = [nu * g for g in gamma]
        fit = fit_power_law(gamma, tau)
        assert abs(fit["n"] - 1.0) < 1e-6
        assert abs(fit["K"] - nu) < 1e-6
    
    def test_too_few_points(self):
        """Should handle < 2 points gracefully."""
        fit = fit_power_law([0.01], [0.002])
        assert fit["n"] == 1.0
        assert fit["K"] == 0.0


# ---------------------------------------------------------------------------
# Particle placement
# ---------------------------------------------------------------------------
class TestParticlePlacement:
    """Test particle placement and voxelization."""
    
    def test_place_zero_phi(self):
        """phi=0 should place at least 1 particle (minimum)."""
        particles = place_particles_random(0.0, 20, 20, 24, 2.5)
        assert len(particles) >= 1
    
    def test_place_small_phi(self):
        """Small phi should place a few particles."""
        particles = place_particles_random(0.05, 20, 20, 24, 2.5)
        assert len(particles) >= 1
    
    def test_no_overlap(self):
        """Placed particles should not overlap."""
        particles = place_particles_random(0.10, 20, 20, 24, 2.5, seed=42)
        for i, p1 in enumerate(particles):
            for j, p2 in enumerate(particles):
                if i >= j:
                    continue
                dist = math.sqrt(sum((a-b)**2 for a, b in zip(p1['center'], p2['center'])))
                assert dist >= 2.0 * p1['radius'] * 0.99  # small tolerance
    
    def test_obstacle_mask_shape(self):
        """Obstacle mask should have correct shape."""
        particles = place_particles_random(0.05, 20, 20, 24, 2.5)
        mask = build_obstacle_mask(particles, 20, 20, 24)
        assert mask.shape == (24, 20, 20)
        assert mask.dtype == torch.bool
    
    def test_obstacle_mask_nonzero(self):
        """Obstacle mask should have some True cells."""
        particles = place_particles_random(0.10, 20, 20, 24, 2.5)
        mask = build_obstacle_mask(particles, 20, 20, 24)
        assert mask.sum() > 0


# ---------------------------------------------------------------------------
# Physical suspension
# ---------------------------------------------------------------------------
class TestPhysicalSuspension:
    """Test physical suspension creation."""
    
    def test_create_dispersed(self):
        """Create a dispersed suspension."""
        susp = physical_suspension(0.05, family="dispersed", seed=42)
        assert susp.family == "dispersed"
        assert len(susp.particles) >= 1
    
    def test_create_flocculated(self):
        """Create a flocculated suspension."""
        susp = physical_suspension(0.10, family="flocculated", n_primary=4, seed=42)
        assert susp.family == "flocculated"
        assert len(susp.particles) >= 1
    
    def test_voxelise(self):
        """Voxelise a suspension."""
        susp = physical_suspension(0.05, family="dispersed", seed=42)
        particles, info = voxelise(susp, n_xy=20, n_z=24)
        assert info['n_particles'] >= 1
        assert info['phi_lattice'] >= 0


# ---------------------------------------------------------------------------
# LBM solver (smoke test)
# ---------------------------------------------------------------------------
class TestLBMSolver:
    """Smoke tests for the LBM shear cell solver."""
    
    def test_pure_case_runs(self):
        """Pure case should run without error."""
        rec = pure_case(nx=16, ny=16, nz=20, nu=0.2, u_wall=0.005,
                       n_steps=100, n_warmup=50, sample_every=10)
        assert "gamma_dot" in rec
        assert "profile_r2" in rec
        assert rec["profile_r2"] > 0.5  # should be reasonably linear
    
    def test_pure_case_phi_zero(self):
        """Pure case should have phi_lattice = 0."""
        rec = pure_case(nx=16, ny=16, nz=20, nu=0.2, u_wall=0.005,
                       n_steps=100, n_warmup=50, sample_every=10)
        assert rec["phi_lattice"] == 0.0
