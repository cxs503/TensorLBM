"""Tests for Schäfer-Turek 2D-2 cylinder benchmark validation suite.

These tests verify the code functionality and methodology.
Full quantitative validation requires extended computation time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorlbm.boundaries import (
    bounce_back_cells,
    compute_obstacle_forces,
    cylinder_mask,
    make_channel_wall_mask,
)
from tensorlbm.d2q9 import equilibrium
from tensorlbm.solver import collide_bgk, stream


class TestSchaeferTurekGeometry:
    """Test geometry setup for Schäfer-Turek benchmark."""

    def test_cylinder_mask_creation(self):
        """Verify cylinder mask is created correctly."""
        nx, ny = 100, 50
        cx, cy = 33, 25
        radius = 6.0
        
        mask = cylinder_mask(nx, ny, cx, cy, radius, device="cpu")
        
        assert mask.shape == (ny, nx)
        assert mask.dtype == torch.bool
        assert mask.sum() > 0
        # Check that center is inside mask
        assert mask[cy, cx].item() is True

    def test_channel_wall_mask(self):
        """Verify channel wall mask excludes cylinder."""
        nx, ny = 100, 50
        cx, cy = 33, 25
        radius = 6.0
        
        obstacle = cylinder_mask(nx, ny, cx, cy, radius, device="cpu")
        wall_mask = make_channel_wall_mask(ny, nx, obstacle, device="cpu")
        
        # Top and bottom rows should be walls (except where cylinder intersects)
        assert wall_mask[0, :].sum() > 0
        assert wall_mask[-1, :].sum() > 0
        # Walls should not overlap with obstacle
        assert (wall_mask & obstacle).sum() == 0

    def test_geometry_ratios(self):
        """Verify Schäfer-Turek geometry ratios."""
        D_lu = 20.0
        H_lu = 4.1 * D_lu  # H/D = 4.1
        ny = int(round(H_lu)) + 1
        
        # Check H/D ratio
        actual_ratio = (ny - 1) / D_lu
        assert abs(actual_ratio - 4.1) < 0.1


class TestSchaeferTurekPhysics:
    """Test physics parameters for Re=100."""

    def test_reynolds_number_setup(self):
        """Verify Re=100 parameters."""
        D_lu = 20.0
        u_mean = 0.05
        Re = 100.0
        
        nu = u_mean * D_lu / Re
        tau = 3.0 * nu + 0.5
        
        # Check tau is in stable range
        assert tau > 0.5
        assert tau < 2.0
        
        # Verify Re
        Re_check = u_mean * D_lu / nu
        assert abs(Re_check - Re) < 1e-6

    def test_mach_number(self):
        """Verify low Mach number condition."""
        u_mean = 0.05
        c_s = 1.0 / 3.0**0.5  # Speed of sound in lattice units
        Ma = u_mean / c_s
        
        assert Ma < 0.1, f"Ma={Ma} too high for incompressible flow"


class TestSchaeferTurekSolver:
    """Test solver functionality."""

    def test_short_simulation(self):
        """Run a very short simulation to verify code works."""
        nx, ny = 60, 30
        radius = 4.0
        u_in = 0.06
        Re = 100.0
        nu = u_in * 2.0 * radius / Re
        tau = 3.0 * nu + 0.5
        
        dev = torch.device("cpu")
        mask = cylinder_mask(nx, ny, nx // 3, ny // 2, radius, device=dev)
        wall_mask = make_channel_wall_mask(ny, nx, mask, device=dev)
        
        rho0 = torch.ones(ny, nx, device=dev)
        ux0 = torch.full_like(rho0, u_in)
        f = equilibrium(rho0, ux0, torch.zeros_like(rho0), device=dev)
        
        # Run 50 steps
        for step in range(50):
            f = collide_bgk(f, tau=tau)
            f = stream(f)
            fx, fy = compute_obstacle_forces(f, mask)
            combined = wall_mask | mask
            f = bounce_back_cells(f, combined)
        
        # Check no NaN
        assert not torch.isnan(f).any(), "NaN detected in distribution function"
        assert not torch.isnan(fx), "NaN detected in force"

    def test_force_computation(self):
        """Verify force computation returns finite values."""
        nx, ny = 60, 30
        radius = 4.0
        u_in = 0.06
        Re = 100.0
        nu = u_in * 2.0 * radius / Re
        tau = 3.0 * nu + 0.5
        
        dev = torch.device("cpu")
        mask = cylinder_mask(nx, ny, nx // 3, ny // 2, radius, device=dev)
        
        rho0 = torch.ones(ny, nx, device=dev)
        ux0 = torch.full_like(rho0, u_in)
        f = equilibrium(rho0, ux0, torch.zeros_like(rho0), device=dev)
        
        # Run a few steps
        for _ in range(10):
            f = collide_bgk(f, tau=tau)
            f = stream(f)
        
        fx, fy = compute_obstacle_forces(f, mask)
        
        assert torch.isfinite(fx), "Drag force is not finite"
        assert torch.isfinite(fy), "Lift force is not finite"


class TestSchaeferTurekPostProcessing:
    """Test post-processing and analysis functions."""

    def test_strouhal_calculation(self):
        """Verify Strouhal number calculation from synthetic data."""
        # Create synthetic periodic signal
        n_points = 1000
        dt = 5
        t = np.arange(n_points) * dt
        
        # Signal with known frequency
        D_lu = 20.0
        u_mean = 0.05
        St_true = 0.165
        f_true = St_true * u_mean / D_lu
        period = 1.0 / f_true
        
        cl = 0.1 * np.sin(2 * np.pi * t / period)
        
        # FFT
        cl_centered = cl - np.mean(cl)
        fft_vals = np.fft.rfft(cl_centered)
        fft_mag = np.abs(fft_vals)
        fft_freq = np.fft.rfftfreq(n_points, d=dt)
        
        # Find peak
        peak_idx = np.argmax(fft_mag[1:]) + 1
        f_peak = fft_freq[peak_idx]
        St_calc = f_peak * D_lu / u_mean
        
        # Should be close to true value
        assert abs(St_calc - St_true) < 0.01

    def test_drag_coefficient_normalization(self):
        """Verify drag coefficient normalization."""
        # Synthetic force
        fx = 0.01  # lattice units
        D_lu = 20.0
        u_mean = 0.05
        rho = 1.0
        
        A_ref = D_lu * 1.0  # per unit depth
        q = 0.5 * rho * u_mean**2
        
        cd = fx / (q * A_ref)
        
        # Should be positive and reasonable
        assert cd > 0
        assert cd < 100  # Sanity check


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
