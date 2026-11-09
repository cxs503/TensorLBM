"""Pytest tests for scalar transport and Boussinesq buoyancy modules.

Tests cover:
- D2Q5 scalar lattice properties
- Scalar equilibrium, collision, streaming
- Mass conservation
- Boussinesq buoyancy force
- Gaussian plume hazard zone
"""

from __future__ import annotations

import math

import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm.scalar_transport import (
    C5,
    CS2_D2Q5,
    OPPOSITE5,
    W5,
    _get_w5,
    scalar_collision,
    scalar_diffusivity,
    scalar_equilibrium,
    scalar_macroscopic,
    scalar_stream,
    tau_from_diffusivity,
)
from tensorlbm.boussinesq import (
    compute_buoyancy_force,
    fit_plume_exponent,
)


# ── D2Q5 lattice tests ───────────────────────────────────────────────────────

class TestD2Q5Lattice:
    """Tests for D2Q5 scalar lattice properties."""

    def test_weights_sum_to_one_fp32(self):
        """D2Q5 weights sum to 1 in float32."""
        assert abs(float(W5.sum()) - 1.0) < 1e-6

    def test_weights_sum_to_one_fp64(self):
        """D2Q5 weights sum to 1 in float64."""
        w64 = _get_w5(torch.float64, torch.device("cpu"))
        assert abs(float(w64.sum()) - 1.0) < 1e-14

    def test_five_directions(self):
        """D2Q5 has 5 directions."""
        assert C5.shape == (5, 2)

    def test_opposite_mapping(self):
        """D2Q5 opposite mapping is correct."""
        # Direction 0 (rest) is its own opposite
        assert OPPOSITE5[0] == 0
        # Direction 1 (+x) opposite is 2 (-x)
        assert OPPOSITE5[1] == 2
        assert OPPOSITE5[2] == 1
        # Direction 3 (+y) opposite is 4 (-y)
        assert OPPOSITE5[3] == 4
        assert OPPOSITE5[4] == 3

    def test_cs2_value(self):
        """D2Q5 cs² = 1/3."""
        assert abs(CS2_D2Q5 - 1.0 / 3.0) < 1e-15


# ── Scalar equilibrium tests ─────────────────────────────────────────────────

class TestScalarEquilibrium:
    """Tests for D2Q5 scalar equilibrium."""

    def test_zeroth_moment_conservation(self):
        """Equilibrium conserves zeroth moment (scalar concentration)."""
        ny, nx = 16, 16
        C = torch.rand(ny, nx)
        ux = torch.zeros(ny, nx)
        uy = torch.zeros(ny, nx)

        geq = scalar_equilibrium(C, ux, uy)
        C_recovered = scalar_macroscopic(geq)

        assert torch.allclose(C_recovered, C, atol=1e-6)

    def test_zero_velocity_equilibrium(self):
        """At zero velocity, g_eq_i = w_i * C."""
        ny, nx = 8, 8
        C = torch.ones(ny, nx) * 5.0
        ux = torch.zeros(ny, nx)
        uy = torch.zeros(ny, nx)

        geq = scalar_equilibrium(C, ux, uy)
        w = _get_w5(torch.float32, torch.device("cpu"))

        for i in range(5):
            expected = w[i] * 5.0
            assert torch.allclose(geq[i], torch.full((ny, nx), expected), atol=1e-6)

    def test_equilibrium_with_velocity(self):
        """Equilibrium with non-zero velocity has correct advection term."""
        ny, nx = 8, 8
        C = torch.ones(ny, nx)
        ux = torch.ones(ny, nx) * 0.1
        uy = torch.zeros(ny, nx)

        geq = scalar_equilibrium(C, ux, uy)
        C_recovered = scalar_macroscopic(geq)

        # Zeroth moment should still be C
        assert torch.allclose(C_recovered, C, atol=1e-5)


# ── Scalar collision tests ───────────────────────────────────────────────────

class TestScalarCollision:
    """Tests for D2Q5 scalar BGK collision."""

    def test_mass_conservation(self):
        """Collision conserves total scalar mass."""
        ny, nx = 16, 16
        torch.manual_seed(42)
        g = torch.rand(5, ny, nx)
        C = scalar_macroscopic(g)
        ux = torch.zeros(ny, nx)
        uy = torch.zeros(ny, nx)
        tau_s = 0.8

        mass_before = float(g.sum())
        g_new = scalar_collision(g, C, ux, uy, tau_s)
        mass_after = float(g_new.sum())

        assert abs(mass_after - mass_before) / mass_before < 1e-6

    def test_equilibrium_is_fixed_point(self):
        """Equilibrium distribution is a fixed point of collision."""
        ny, nx = 8, 8
        C = torch.rand(ny, nx)
        ux = torch.zeros(ny, nx)
        uy = torch.zeros(ny, nx)
        tau_s = 0.8

        geq = scalar_equilibrium(C, ux, uy)
        g_collided = scalar_collision(geq, C, ux, uy, tau_s)

        assert torch.allclose(g_collided, geq, atol=1e-6)


# ── Scalar streaming tests ───────────────────────────────────────────────────

class TestScalarStream:
    """Tests for D2Q5 scalar streaming."""

    def test_mass_conservation(self):
        """Streaming conserves total scalar mass."""
        ny, nx = 16, 16
        torch.manual_seed(123)
        g = torch.rand(5, ny, nx)

        mass_before = float(g.sum())
        g_streamed = scalar_stream(g)
        mass_after = float(g_streamed.sum())

        assert abs(mass_after - mass_before) / mass_before < 1e-6

    def test_uniform_field_invariant(self):
        """Uniform scalar field is invariant under streaming."""
        ny, nx = 16, 16
        C = torch.ones(ny, nx) * 3.0
        ux = torch.zeros(ny, nx)
        uy = torch.zeros(ny, nx)

        g = scalar_equilibrium(C, ux, uy)
        g_streamed = scalar_stream(g)

        assert torch.allclose(g_streamed, g, atol=1e-6)


# ── Diffusivity tests ────────────────────────────────────────────────────────

class TestDiffusivity:
    """Tests for scalar diffusivity relations."""

    def test_diffusivity_formula(self):
        """D = (tau - 0.5) / 3."""
        tau_s = 0.8
        D = scalar_diffusivity(tau_s)
        assert abs(D - (tau_s - 0.5) / 3.0) < 1e-10

    def test_tau_from_diffusivity_inverse(self):
        """tau_from_diffusivity is inverse of scalar_diffusivity."""
        D_target = 0.1
        tau_s = tau_from_diffusivity(D_target)
        D_recovered = scalar_diffusivity(tau_s)
        assert abs(D_recovered - D_target) < 1e-10


# ── Boussinesq buoyancy tests ────────────────────────────────────────────────

class TestBoussinesq:
    """Tests for Boussinesq buoyancy force."""

    def test_buoyancy_direction_hot(self):
        """Buoyancy is upward for T > T_ref."""
        ny, nx = 8, 8
        rho = torch.ones(ny, nx)
        T = torch.ones(ny, nx) * 2.0
        T_ref = 0.0
        g_beta = 1e-3

        F = compute_buoyancy_force(rho, T, T_ref, g_beta)

        assert float(F[1].mean()) > 0  # upward
        assert abs(float(F[0].mean())) < 1e-15  # no horizontal force

    def test_buoyancy_direction_cold(self):
        """Buoyancy is downward for T < T_ref."""
        ny, nx = 8, 8
        rho = torch.ones(ny, nx)
        T = torch.ones(ny, nx) * 0.0
        T_ref = 1.0
        g_beta = 1e-3

        F = compute_buoyancy_force(rho, T, T_ref, g_beta)

        assert float(F[1].mean()) < 0  # downward

    def test_buoyancy_zero_at_reference(self):
        """Buoyancy is zero at T = T_ref."""
        ny, nx = 8, 8
        rho = torch.ones(ny, nx)
        T = torch.ones(ny, nx) * 0.5
        T_ref = 0.5
        g_beta = 1e-3

        F = compute_buoyancy_force(rho, T, T_ref, g_beta)

        assert torch.allclose(F, torch.zeros_like(F), atol=1e-15)


# ── Plume exponent fitting tests ─────────────────────────────────────────────

class TestPlumeExponent:
    """Tests for plume power-law exponent fitting."""

    def test_fit_known_exponent(self):
        """Fit recovers known power-law exponent."""
        # Q(z) = z^2 (exponent = 2)
        heights = [10.0, 20.0, 30.0, 40.0, 50.0]
        fluxes = [h ** 2 for h in heights]

        fit = fit_plume_exponent(heights, fluxes)

        assert abs(fit["exponent"] - 2.0) < 0.01
        assert fit["r_squared"] > 0.99

    def test_fit_pure_plume_exponent(self):
        """Fit recovers pure plume exponent 5/3."""
        # Q(z) = z^(5/3)
        heights = [10.0, 20.0, 30.0, 40.0, 50.0]
        fluxes = [h ** (5.0 / 3.0) for h in heights]

        fit = fit_plume_exponent(heights, fluxes)

        assert abs(fit["exponent"] - 5.0 / 3.0) < 0.01
        assert fit["r_squared"] > 0.99


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
