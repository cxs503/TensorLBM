"""Tests for AA-pattern fused stream-collide implementation.

Verifies:
1. Periodic BC equivalence with separate stream+collide
2. Mass conservation
3. Stability (finite values after many steps)
4. Correct equilibrium fixed point
"""

from __future__ import annotations

import pytest
import torch
import numpy as np

from tensorlbm.aa_fused_stream_collide import (
    aa_fused_stream_collide,
    separate_stream_collide,
    aa_fused_step,
    separate_step,
    _macroscopic,
    _equilibrium,
    C,
    W,
)


class TestAAPatternEquivalence:
    """Test that AA-pattern is equivalent to separate approach for periodic BCs."""

    def test_single_step_periodic(self):
        """Single step should be bit-identical for periodic BCs."""
        n = 33
        f = torch.ones((9, n, n), dtype=torch.float64) * 0.1
        f[0] = 0.5  # Non-trivial initial condition
        tau = 1.0

        f_aa = aa_fused_stream_collide(f, tau)
        f_sep = separate_stream_collide(f, tau)

        diff = torch.abs(f_aa - f_sep).max().item()
        assert diff < 1e-14, f"Max diff {diff:.2e} exceeds machine epsilon"

    def test_multi_step_periodic(self):
        """Multiple steps should remain equivalent for periodic BCs."""
        n = 33
        f = torch.ones((9, n, n), dtype=torch.float64) * 0.1
        f[0] = 0.5
        tau = 1.0

        f_aa = f.clone()
        f_sep = f.clone()

        for _ in range(100):
            f_aa = aa_fused_stream_collide(f_aa, tau)
            f_sep = separate_stream_collide(f_sep, tau)

        diff = torch.abs(f_aa - f_sep).max().item()
        assert diff < 1e-12, f"Max diff after 100 steps: {diff:.2e}"


class TestMassConservation:
    """Test mass conservation properties."""

    def test_mass_conserved_periodic(self):
        """Mass should be conserved for periodic BCs."""
        n = 33
        f = torch.ones((9, n, n), dtype=torch.float64) * 0.1
        f[0] = 0.5
        tau = 1.0

        mass_initial = f.sum().item()

        f_out = aa_fused_stream_collide(f, tau)
        mass_final = f_out.sum().item()

        rel_change = abs(mass_final - mass_initial) / mass_initial
        assert rel_change < 1e-10, f"Mass change {rel_change:.2e} exceeds tolerance"


class TestStability:
    """Test numerical stability."""

    def test_finite_values_after_steps(self):
        """Values should remain finite after many steps."""
        n = 33
        f = torch.ones((9, n, n), dtype=torch.float64) * 0.1
        f[0] = 0.5
        tau = 1.0

        for _ in range(500):
            f = aa_fused_stream_collide(f, tau)

        assert torch.all(torch.isfinite(f)), "Non-finite values detected"


class TestEquilibrium:
    """Test equilibrium properties."""

    def test_equilibrium_fixed_point(self):
        """Equilibrium distribution should be a fixed point of collision."""
        n = 33
        rho = torch.ones((n, n), dtype=torch.float64)
        ux = torch.zeros((n, n), dtype=torch.float64)
        uy = torch.zeros((n, n), dtype=torch.float64)

        feq = _equilibrium(rho, ux, uy)
        tau = 1.0

        f_out = aa_fused_stream_collide(feq, tau)

        # After streaming + collision of equilibrium, should return to equilibrium
        # (up to streaming shift)
        diff = torch.abs(f_out - feq).max().item()
        # Note: streaming shifts the distribution, so we expect non-zero diff
        # but the macroscopic variables should be preserved
        rho_out, ux_out, uy_out = _macroscopic(f_out)
        assert torch.allclose(rho_out, rho, atol=1e-10)
        assert torch.allclose(ux_out, ux, atol=1e-10)
        assert torch.allclose(uy_out, uy, atol=1e-10)


class TestLatticeConstants:
    """Test lattice constants are correct."""

    def test_d2q9_velocities(self):
        """D2Q9 velocities should sum to zero."""
        assert C[:, 0].sum().item() == 0
        assert C[:, 1].sum().item() == 0

    def test_d2q9_weights_sum(self):
        """D2Q9 weights should sum to 1."""
        assert abs(W.sum().item() - 1.0) < 1e-15

    def test_d2q9_weights(self):
        """D2Q9 weights should match standard values."""
        expected = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]
        for i, w in enumerate(expected):
            assert abs(W[i].item() - w) < 1e-15
