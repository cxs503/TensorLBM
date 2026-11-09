"""Pytest tests for IN65-N03 scalar transport + Boussinesq verification.

Tests cover all acceptance criteria:
  G1: 1-D advection-diffusion Gaussian analytical anchor (error ≤ 2%)
  G2: Mass conservation ΣC relative residual ≤ 1e-12
  G3: Buoyant plume self-similarity Q(z) ∝ z^(5/3), exponent ∈ [1.6, 1.75]
  G4: Negative control g'=0 → Q(z) linear, exponent ∈ [0.95, 1.05]
  G5: LFL contour length vs wind speed: 5 levels strictly monotonic decreasing
  G6: Grid convergence ≤ 3%

Plus unit tests for D2Q5 lattice, scalar operations, and Boussinesq coupling.
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
from tensorlbm.scalar_verification_in65 import (
    run_1d_advection_diffusion,
    run_mass_conservation_test,
    run_buoyant_plume_self_similarity,
    run_buoyant_plume_lbm_check,
    run_negative_control_no_buoyancy,
    run_lfl_wind_speed_sweep,
    run_lfl_grid_convergence,
)


# ═══════════════════════════════════════════════════════════════════════════════
# D2Q5 Lattice Tests
# ═══════════════════════════════════════════════════════════════════════════════

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
        assert OPPOSITE5[0] == 0
        assert OPPOSITE5[1] == 2
        assert OPPOSITE5[2] == 1
        assert OPPOSITE5[3] == 4
        assert OPPOSITE5[4] == 3

    def test_cs2_value(self):
        """D2Q5 cs² = 1/3."""
        assert abs(CS2_D2Q5 - 1.0 / 3.0) < 1e-15


# ═══════════════════════════════════════════════════════════════════════════════
# Scalar Equilibrium Tests
# ═══════════════════════════════════════════════════════════════════════════════

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

        assert torch.allclose(C_recovered, C, atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════════
# Scalar Collision Tests
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# Scalar Streaming Tests
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# Diffusivity Tests
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# Boussinesq Buoyancy Tests
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# Plume Exponent Fitting Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPlumeExponent:
    """Tests for plume power-law exponent fitting."""

    def test_fit_known_exponent(self):
        """Fit recovers known power-law exponent."""
        heights = [10.0, 20.0, 30.0, 40.0, 50.0]
        fluxes = [h ** 2 for h in heights]

        fit = fit_plume_exponent(heights, fluxes)

        assert abs(fit["exponent"] - 2.0) < 0.01
        assert fit["r_squared"] > 0.99

    def test_fit_pure_plume_exponent(self):
        """Fit recovers pure plume exponent 5/3."""
        heights = [10.0, 20.0, 30.0, 40.0, 50.0]
        fluxes = [h ** (5.0 / 3.0) for h in heights]

        fit = fit_plume_exponent(heights, fluxes)

        assert abs(fit["exponent"] - 5.0 / 3.0) < 0.01
        assert fit["r_squared"] > 0.99


# ═══════════════════════════════════════════════════════════════════════════════
# IN65-N03 Verification Gate Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestIN65Verification:
    """Tests for IN65-N03 verification gates (acceptance criteria)."""

    def test_G1_1d_advection_diffusion_l2_error(self):
        """G1: 1-D advection-diffusion Gaussian L2 error ≤ 2%."""
        # Use nx=256 to avoid periodic wrap-around at large n_steps
        result = run_1d_advection_diffusion(nx=256, n_steps=1000)
        assert result["pass_l2"], f"L2 error {result['l2_relative_error']:.4f} >= 0.02"

    def test_G1_1d_advection_diffusion_peak_error(self):
        """G1: 1-D advection-diffusion Gaussian peak error ≤ 2%."""
        result = run_1d_advection_diffusion(nx=256, n_steps=1000)
        assert result["pass_peak"], f"Peak error {result['peak_error_pct']:.2f}% >= 2%"

    def test_G2_mass_conservation(self):
        """G2: Mass conservation ΣC relative residual ≤ 1e-12."""
        result = run_mass_conservation_test(nx=64, ny=64, n_steps=2000)
        assert result["pass"], f"Max drift {result['max_relative_drift']:.2e} >= 1e-12"

    def test_G3_buoyant_plume_exponent(self):
        """G3: Buoyant plume Q(z) ∝ z^(5/3), exponent ∈ [1.6, 1.75]."""
        result = run_buoyant_plume_self_similarity()
        assert result["pass"], f"Exponent {result['exponent']:.3f} not in [1.6, 1.75]"
        assert 1.6 <= result["exponent"] <= 1.75

    def test_G3b_lbm_buoyancy_sanity(self):
        """G3b: LBM buoyancy drives upward flow."""
        result = run_buoyant_plume_lbm_check()
        assert result["buoyancy_drives_upward"], "Buoyancy did not drive upward flow"

    def test_G4_negative_control_exponent(self):
        """G4: Negative control g'=0 → Q(z) linear, exponent ∈ [0.95, 1.05]."""
        result = run_negative_control_no_buoyancy()
        assert result["pass"], f"Exponent {result['exponent']:.3f} not in [0.95, 1.05]"
        assert 0.95 <= result["exponent"] <= 1.05

    def test_G5_lfl_wind_speed_monotonicity(self):
        """G5: LFL contour length strictly decreases with wind speed."""
        result = run_lfl_wind_speed_sweep(mass_rate=10.0)
        assert result["pass"], f"Not strictly decreasing: {result['lengths']}"
        assert result["is_strictly_decreasing"]

    def test_G6_lfl_grid_convergence(self):
        """G6: LFL grid convergence ≤ 3%."""
        result = run_lfl_grid_convergence()
        assert result["pass"], f"Max change {result['max_relative_change']:.4f} >= 0.03"
        assert result["max_relative_change"] < 0.03


# ═══════════════════════════════════════════════════════════════════════════════
# Additional Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """Integration tests for scalar transport + Boussinesq coupling."""

    def test_scalar_diffusion_isotropic(self):
        """Scalar diffusion is isotropic (same in x and y)."""
        ny, nx = 32, 32
        tau_s = 0.8
        D = scalar_diffusivity(tau_s)

        # Initialize line source in x
        C_x = torch.zeros(ny, nx)
        C_x[ny // 2, :] = 1.0
        g_x = scalar_equilibrium(C_x, torch.zeros(ny, nx), torch.zeros(ny, nx))

        # Initialize line source in y
        C_y = torch.zeros(ny, nx)
        C_y[:, nx // 2] = 1.0
        g_y = scalar_equilibrium(C_y, torch.zeros(ny, nx), torch.zeros(ny, nx))

        # Evolve both
        ux = torch.zeros(ny, nx)
        uy = torch.zeros(ny, nx)
        for _ in range(500):
            C_x = scalar_macroscopic(g_x)
            g_x = scalar_collision(g_x, C_x, ux, uy, tau_s)
            g_x = scalar_stream(g_x)

            C_y = scalar_macroscopic(g_y)
            g_y = scalar_collision(g_y, C_y, ux, uy, tau_s)
            g_y = scalar_stream(g_y)

        # Check variance is similar (isotropic diffusion)
        var_x = float(((torch.arange(nx) - nx / 2) ** 2 * scalar_macroscopic(g_x).sum(dim=0)).sum())
        var_y = float(((torch.arange(ny) - ny / 2) ** 2 * scalar_macroscopic(g_y).sum(dim=1)).sum())

        # Should be within 20% (not exact due to discrete effects)
        ratio = var_x / var_y if var_y > 0 else float("inf")
        assert 0.8 < ratio < 1.2, f"Anisotropy ratio {ratio:.3f}"

    def test_boussinesq_force_magnitude(self):
        """Buoyancy force magnitude scales with g_beta and ΔT."""
        ny, nx = 8, 8
        rho = torch.ones(ny, nx)
        T = torch.ones(ny, nx) * 2.0
        T_ref = 0.0

        F1 = compute_buoyancy_force(rho, T, T_ref, 1e-3)
        F2 = compute_buoyancy_force(rho, T, T_ref, 2e-3)

        # Force should scale linearly with g_beta
        ratio = float(F2[1].mean() / F1[1].mean())
        assert abs(ratio - 2.0) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
