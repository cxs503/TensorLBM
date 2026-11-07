"""Tests for compressible shock wave verification suite."""

import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tensorlbm.compressible_shock import (
    GAMMA,
    CS2,
    DoubleMachReflectionResult,
    ForwardFacingStepResult,
    NormalShockResult,
    apply_bounce_back,
    apply_inlet,
    apply_outlet,
    bgk_collision,
    mach2_from_mach1,
    macroscopic,
    p_ratio,
    rh_ratio,
    run_verification,
    simulate_double_mach_reflection,
    simulate_forward_facing_step,
    simulate_normal_shock,
    stream_step,
    t_ratio,
)


# ---------------------------------------------------------------------------
# Unit tests for Rankine-Hugoniot relations
# ---------------------------------------------------------------------------
class TestRankineHugoniot:
    def test_rh_ratio_mach1(self):
        """At Mach 1, density ratio should be 1."""
        assert abs(rh_ratio(1.0) - 1.0) < 1e-10

    def test_rh_ratio_mach2(self):
        """Mach 2 density ratio."""
        expected = (GAMMA + 1) * 4 / ((GAMMA - 1) * 4 + 2)
        assert abs(rh_ratio(2.0) - expected) < 1e-10

    def test_p_ratio_mach1(self):
        """At Mach 1, pressure ratio should be 1."""
        assert abs(p_ratio(1.0) - 1.0) < 1e-10

    def test_t_ratio_mach1(self):
        """At Mach 1, temperature ratio should be 1."""
        assert abs(t_ratio(1.0) - 1.0) < 1e-10

    def test_mach2_from_mach1(self):
        """Downstream Mach number after normal shock."""
        # For Mach 2, downstream should be subsonic
        m2 = mach2_from_mach1(2.0)
        assert m2 < 1.0
        assert abs(m2 - 0.5774) < 1e-4  # approximate value


# ---------------------------------------------------------------------------
# Unit tests for D2Q9 BGK solver
# ---------------------------------------------------------------------------
class TestBGKSolver:
    def test_equilibrium_conservation(self):
        """Equilibrium should conserve mass, momentum."""
        rho = torch.ones(10, 10)
        ux = torch.ones(10, 10) * 0.1
        uy = torch.ones(10, 10) * 0.1
        f_eq = tensorlbm.d2q9.equilibrium(rho, ux, uy)
        rho_rec, ux_rec, uy_rec = tensorlbm.d2q9.macroscopic(f_eq)
        assert torch.allclose(rho, rho_rec, atol=1e-6)
        assert torch.allclose(ux, ux_rec, atol=1e-6)
        assert torch.allclose(uy, uy_rec, atol=1e-6)

    def test_stream_step_conservation(self):
        """Streaming should conserve mass."""
        f = torch.rand(9, 10, 10)
        f_streamed = stream_step(f)
        rho_orig = f.sum(dim=0)
        rho_new = f_streamed.sum(dim=0)
        assert torch.allclose(rho_orig, rho_new, atol=1e-10)

    def test_bounce_back_no_slip(self):
        """Bounce-back should give zero velocity at wall."""
        ny, nx = 10, 10
        f = torch.rand(9, ny, nx)
        wall_mask = torch.zeros(ny, nx)
        wall_mask[0, :] = 1.0
        f_bb = apply_bounce_back(f, wall_mask)
        rho, ux, uy = macroscopic(f_bb)
        assert torch.allclose(ux[0, :], torch.zeros_like(ux[0, :]), atol=1e-6)
        assert torch.allclose(uy[0, :], torch.zeros_like(uy[0, :]), atol=1e-6)


# ---------------------------------------------------------------------------
# Integration tests for normal shock
# ---------------------------------------------------------------------------
class TestNormalShock:
    @pytest.mark.slow
    def test_normal_shock_mach2(self):
        """Normal shock at Mach 2 should match R-H relations within 1%."""
        res = simulate_normal_shock(2.0, nx=200, ny=50, omega=1.5, n_steps=5000)
        assert res.rho_err < 1.0, f"rho_err={res.rho_err:.2f}% > 1%"
        assert res.p_err < 1.0, f"p_err={res.p_err:.2f}% > 1%"
        assert res.t_err < 1.0, f"t_err={res.t_err:.2f}% > 1%"

    @pytest.mark.slow
    def test_normal_shock_mach5(self):
        """Normal shock at Mach 5 should match R-H relations within 1%."""
        res = simulate_normal_shock(5.0, nx=200, ny=50, omega=1.5, n_steps=5000)
        assert res.rho_err < 1.0, f"rho_err={res.rho_err:.2f}% > 1%"
        assert res.p_err < 1.0, f"p_err={res.p_err:.2f}% > 1%"
        assert res.t_err < 1.0, f"t_err={res.t_err:.2f}% > 1%"

    @pytest.mark.slow
    def test_normal_shock_mach10(self):
        """Normal shock at Mach 10 should match R-H relations within 1%."""
        res = simulate_normal_shock(10.0, nx=200, ny=50, omega=1.5, n_steps=5000)
        assert res.rho_err < 1.0, f"rho_err={res.rho_err:.2f}% > 1%"
        assert res.p_err < 1.0, f"p_err={res.p_err:.2f}% > 1%"
        assert res.t_err < 1.0, f"t_err={res.t_err:.2f}% > 1%"


# ---------------------------------------------------------------------------
# Integration tests for double Mach reflection
# ---------------------------------------------------------------------------
class TestDoubleMachReflection:
    @pytest.mark.slow
    def test_dmr_density_field_shape(self):
        """Double Mach reflection should produce correct density field shape."""
        res = simulate_double_mach_reflection(nx=200, ny=100, n_steps=2000)
        assert res.density_field.shape == (100, 200)
        assert res.density_field.min() > 0  # density should be positive


# ---------------------------------------------------------------------------
# Integration tests for forward-facing step
# ---------------------------------------------------------------------------
class TestForwardFacingStep:
    @pytest.mark.slow
    def test_ffs_density_field_shape(self):
        """Forward-facing step should produce correct density field shape."""
        res = simulate_forward_facing_step(nx=200, ny=50, n_steps=2000)
        assert res.density_field.shape == (50, 200)
        assert res.density_field.min() > 0  # density should be positive


# ---------------------------------------------------------------------------
# Full verification run
# ---------------------------------------------------------------------------
class TestFullVerification:
    @pytest.mark.slow
    def test_run_verification(self):
        """Run full verification and check results structure."""
        results = run_verification()
        assert "normal_shock_M2" in results
        assert "normal_shock_M3" in results
        assert "normal_shock_M5" in results
        assert "normal_shock_M8" in results
        assert "normal_shock_M10" in results
        assert "double_mach_reflection" in results
        assert "forward_facing_step" in results
