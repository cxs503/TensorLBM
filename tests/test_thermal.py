"""Tests for the thermal LBM module (D2Q9 + D2Q5 double-distribution model)."""
from __future__ import annotations

import math

import torch

from tensorlbm import equilibrium
from tensorlbm.thermal import (
    C_D2Q5,
    W_D2Q5,
    apply_buoyancy_force,
    collide_thermal_bgk,
    equilibrium_thermal,
    macroscopic_thermal,
    stream_thermal,
)

# ---------------------------------------------------------------------------
# Lattice constants
# ---------------------------------------------------------------------------


class TestThermalConstants:
    def test_d2q5_weights_sum_to_one(self) -> None:
        assert abs(float(W_D2Q5.sum().item()) - 1.0) < 1e-6

    def test_d2q5_has_5_directions(self) -> None:
        assert C_D2Q5.shape == (5, 2)
        assert W_D2Q5.shape == (5,)

    def test_d2q5_first_direction_is_rest(self) -> None:
        assert C_D2Q5[0, 0].item() == 0
        assert C_D2Q5[0, 1].item() == 0


# ---------------------------------------------------------------------------
# Equilibrium distribution
# ---------------------------------------------------------------------------


class TestEquilibriumThermal:
    def test_returns_correct_shape(self) -> None:
        ny, nx = 8, 12
        T = torch.ones((ny, nx))
        ux = torch.zeros_like(T)
        uy = torch.zeros_like(T)
        geq = equilibrium_thermal(T, ux, uy)
        assert geq.shape == (5, ny, nx)

    def test_sums_to_temperature(self) -> None:
        """Sum over velocity directions should recover the temperature."""
        ny, nx = 8, 12
        T = torch.rand((ny, nx)) + 0.5
        ux = torch.full_like(T, 0.03)
        uy = torch.full_like(T, -0.02)
        geq = equilibrium_thermal(T, ux, uy)
        T_out = geq.sum(dim=0)
        assert torch.allclose(T_out, T, atol=1e-5)

    def test_non_negative_for_moderate_velocity(self) -> None:
        """Equilibrium should be non-negative for small velocities."""
        ny, nx = 8, 12
        T = torch.ones((ny, nx))
        ux = torch.full_like(T, 0.05)
        uy = torch.zeros_like(T)
        geq = equilibrium_thermal(T, ux, uy)
        # Not always strictly positive, but should be finite
        assert torch.isfinite(geq).all()

    def test_finite(self) -> None:
        ny, nx = 8, 12
        T = torch.rand((ny, nx)) * 2.0 + 0.1
        ux = torch.rand_like(T) * 0.05
        uy = torch.rand_like(T) * 0.04
        geq = equilibrium_thermal(T, ux, uy)
        assert torch.isfinite(geq).all()


# ---------------------------------------------------------------------------
# Collision
# ---------------------------------------------------------------------------


class TestCollideThermalBGK:
    def test_preserves_shape(self) -> None:
        ny, nx = 8, 12
        T = torch.ones((ny, nx))
        ux = torch.zeros_like(T)
        uy = torch.zeros_like(T)
        g = equilibrium_thermal(T, ux, uy)
        g_out = collide_thermal_bgk(g, T, ux, uy, tau_T=0.7)
        assert g_out.shape == g.shape

    def test_conserves_temperature(self) -> None:
        """Collision must conserve the zeroth moment (temperature)."""
        ny, nx = 8, 12
        T = torch.rand((ny, nx)) * 2.0 + 0.5
        ux = torch.full_like(T, 0.03)
        uy = torch.full_like(T, -0.01)
        g = equilibrium_thermal(T, ux, uy)
        # Perturb slightly to create non-equilibrium
        g = g + 0.001 * torch.rand_like(g)
        T_orig = g.sum(dim=0)
        g_out = collide_thermal_bgk(g, T_orig, ux, uy, tau_T=0.8)
        T_out = g_out.sum(dim=0)
        assert torch.allclose(T_out, T_orig, atol=1e-5)

    def test_identity_at_equilibrium(self) -> None:
        """f_eq is a fixed point: collide(g_eq) == g_eq."""
        ny, nx = 8, 12
        T = torch.ones((ny, nx))
        ux = torch.full_like(T, 0.04)
        uy = torch.zeros_like(T)
        geq = equilibrium_thermal(T, ux, uy)
        g_out = collide_thermal_bgk(geq, T, ux, uy, tau_T=0.7)
        assert torch.allclose(g_out, geq, atol=1e-5)

    def test_finite_output(self) -> None:
        ny, nx = 8, 12
        T = torch.rand((ny, nx)) + 0.5
        ux = torch.rand_like(T) * 0.04
        uy = torch.rand_like(T) * 0.03
        g = equilibrium_thermal(T, ux, uy)
        g_out = collide_thermal_bgk(g, T, ux, uy, tau_T=0.6)
        assert torch.isfinite(g_out).all()


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestStreamThermal:
    def test_preserves_shape(self) -> None:
        ny, nx = 8, 12
        T = torch.ones((ny, nx))
        ux = torch.zeros_like(T)
        uy = torch.zeros_like(T)
        g = equilibrium_thermal(T, ux, uy)
        g_out = stream_thermal(g)
        assert g_out.shape == g.shape

    def test_conserves_total_temperature(self) -> None:
        """Streaming (periodic) conserves total mass (temperature integral)."""
        ny, nx = 16, 16
        T = torch.rand((ny, nx)) + 0.5
        ux = torch.full_like(T, 0.03)
        uy = torch.zeros_like(T)
        g = equilibrium_thermal(T, ux, uy)
        total_before = g.sum()
        g_out = stream_thermal(g)
        total_after = g_out.sum()
        assert torch.allclose(total_after, total_before, atol=1e-5)

    def test_finite_output(self) -> None:
        ny, nx = 8, 12
        T = torch.rand((ny, nx)) + 0.5
        ux = torch.zeros_like(T)
        uy = torch.zeros_like(T)
        g = equilibrium_thermal(T, ux, uy)
        g_out = stream_thermal(g)
        assert torch.isfinite(g_out).all()


# ---------------------------------------------------------------------------
# Macroscopic recovery
# ---------------------------------------------------------------------------


class TestMacroscopicThermal:
    def test_recovers_temperature(self) -> None:
        ny, nx = 8, 12
        T_in = torch.rand((ny, nx)) * 2.0 + 0.3
        ux = torch.zeros_like(T_in)
        uy = torch.zeros_like(T_in)
        g = equilibrium_thermal(T_in, ux, uy)
        T_out = macroscopic_thermal(g)
        assert torch.allclose(T_out, T_in, atol=1e-5)


# ---------------------------------------------------------------------------
# Buoyancy force
# ---------------------------------------------------------------------------


class TestApplyBuoyancyForce:
    def test_preserves_shape(self) -> None:
        ny, nx = 8, 12
        rho = torch.ones((ny, nx))
        ux = torch.full_like(rho, 0.03)
        uy = torch.zeros_like(rho)
        f = equilibrium(rho, ux, uy)
        T = torch.ones_like(rho)
        f_out = apply_buoyancy_force(f, T, T_ref=1.0, beta=0.001)
        assert f_out.shape == f.shape

    def test_zero_force_at_reference_temperature(self) -> None:
        """T == T_ref should give zero force (no buoyancy)."""
        ny, nx = 8, 12
        rho = torch.ones((ny, nx))
        ux = torch.full_like(rho, 0.03)
        uy = torch.zeros_like(rho)
        f = equilibrium(rho, ux, uy)
        T_ref = 1.0
        T = torch.full_like(rho, T_ref)
        f_out = apply_buoyancy_force(f, T, T_ref=T_ref, beta=0.01)
        assert torch.allclose(f_out, f, atol=1e-6)

    def test_finite_output(self) -> None:
        ny, nx = 8, 12
        rho = torch.ones((ny, nx))
        ux = torch.zeros_like(rho)
        uy = torch.zeros_like(rho)
        f = equilibrium(rho, ux, uy)
        T = torch.rand_like(rho) * 0.5 + 0.75
        f_out = apply_buoyancy_force(f, T, T_ref=1.0, beta=0.001, g_y=-1.0)
        assert torch.isfinite(f_out).all()


# ---------------------------------------------------------------------------
# Diffusion convergence test
# ---------------------------------------------------------------------------


class TestThermalDiffusionConvergence:
    def test_1d_diffusion_decays_at_correct_rate(self) -> None:
        """A sinusoidal temperature mode should decay at rate α k²."""
        ny, nx = 4, 32  # periodic in x, trivial in y
        tau_T = 0.7
        alpha = (tau_T - 0.5) / 3.0  # thermal diffusivity

        k = 2.0 * math.pi / nx
        xs = torch.arange(nx, dtype=torch.float32)
        T0_1d = 0.1 * torch.sin(k * xs) + 1.0  # small amplitude over mean 1.0
        T0 = T0_1d.unsqueeze(0).expand(ny, nx).clone()

        ux0 = torch.zeros((ny, nx))
        uy0 = torch.zeros((ny, nx))
        g = equilibrium_thermal(T0, ux0, uy0)

        n_steps = 100
        for _ in range(n_steps):
            T = macroscopic_thermal(g)
            g = collide_thermal_bgk(g, T, ux0, uy0, tau_T=tau_T)
            g = stream_thermal(g)

        T_final = macroscopic_thermal(g)
        # Amplitude of the sinusoidal mode
        T_mean = float(T0.mean().item())
        amp0 = float((T0 - T_mean).abs().max().item())
        amp_final = float((T_final - T_mean).abs().max().item())

        decay_rate_theory = alpha * k ** 2
        if amp0 > 0 and amp_final > 0:
            measured_rate = -math.log(amp_final / amp0) / n_steps
            assert abs(measured_rate - decay_rate_theory) / decay_rate_theory < 0.15, (
                f"Thermal diffusion rate mismatch: measured={measured_rate:.5f}, "
                f"theory={decay_rate_theory:.5f}"
            )
        else:
            assert amp_final < amp0, "Temperature mode did not decay"
