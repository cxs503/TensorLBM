"""Contract tests for the thermal common module.

Verifies the composable DDF thermal LBM step (D3Q7 + D3Q19/D3Q27),
buoyancy coupling, and conjugate heat transfer.

Contract tests verify operator algebra (shape, finite, mass/temperature
conservation, equilibrium identity), NOT thermal physics correctness.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.d3q27 import equilibrium27, macroscopic27
from tensorlbm.thermal_common import (
    C_D3Q7,
    W_D3Q7,
    apply_buoyancy_3d,
    conjugate_ht_step,
    thermal_collide_bgk_3d,
    thermal_equilibrium_3d,
    thermal_macroscopic_3d,
    thermal_step,
    thermal_stream_3d,
)

TAU_T = 0.8  # > 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f3d19(nz=4, ny=6, nx=8, u_mag=0.04) -> torch.Tensor:
    rho = torch.rand((nz, ny, nx)) + 0.5
    ux = torch.rand_like(rho) * u_mag
    uy = torch.rand_like(rho) * u_mag
    uz = torch.rand_like(rho) * u_mag
    return equilibrium3d(rho, ux, uy, uz)


def _f3d27(nz=4, ny=6, nx=8, u_mag=0.04) -> torch.Tensor:
    rho = torch.rand((nz, ny, nx)) + 0.5
    ux = torch.rand_like(rho) * u_mag
    uy = torch.rand_like(rho) * u_mag
    uz = torch.rand_like(rho) * u_mag
    return equilibrium27(rho, ux, uy, uz)


def _g_thermal(nz=4, ny=6, nx=8, T_mag=1.0, u_mag=0.04) -> torch.Tensor:
    T = torch.rand((nz, ny, nx)) * T_mag + 0.5
    ux = torch.rand_like(T) * u_mag
    uy = torch.rand_like(T) * u_mag
    uz = torch.rand_like(T) * u_mag
    return thermal_equilibrium_3d(T, ux, uy, uz)


# ---------------------------------------------------------------------------
# D3Q7 lattice constants
# ---------------------------------------------------------------------------

class TestD3Q7Constants:
    def test_weights_sum_to_one(self) -> None:
        assert abs(float(W_D3Q7.sum().item()) - 1.0) < 1e-6

    def test_has_7_directions(self) -> None:
        assert C_D3Q7.shape == (7, 3)
        assert W_D3Q7.shape == (7,)

    def test_rest_direction_is_first(self) -> None:
        assert C_D3Q7[0, 0].item() == 0
        assert C_D3Q7[0, 1].item() == 0
        assert C_D3Q7[0, 2].item() == 0


# ---------------------------------------------------------------------------
# Equilibrium
# ---------------------------------------------------------------------------

class TestThermalEquilibrium:
    def test_shape(self) -> None:
        nz, ny, nx = 4, 6, 8
        T = torch.ones((nz, ny, nx))
        ux = torch.zeros_like(T)
        uy = torch.zeros_like(T)
        uz = torch.zeros_like(T)
        geq = thermal_equilibrium_3d(T, ux, uy, uz)
        assert geq.shape == (7, nz, ny, nx)

    def test_sums_to_temperature(self) -> None:
        nz, ny, nx = 4, 6, 8
        T = torch.rand((nz, ny, nx)) + 0.5
        ux = torch.full_like(T, 0.03)
        uy = torch.full_like(T, -0.02)
        uz = torch.full_like(T, 0.01)
        geq = thermal_equilibrium_3d(T, ux, uy, uz)
        T_out = geq.sum(dim=0)
        assert torch.allclose(T_out, T, atol=1e-5)

    def test_finite(self) -> None:
        nz, ny, nx = 4, 6, 8
        T = torch.rand((nz, ny, nx)) * 2.0 + 0.1
        ux = torch.rand_like(T) * 0.05
        uy = torch.rand_like(T) * 0.04
        uz = torch.rand_like(T) * 0.03
        geq = thermal_equilibrium_3d(T, ux, uy, uz)
        assert torch.isfinite(geq).all()


# ---------------------------------------------------------------------------
# Collision
# ---------------------------------------------------------------------------

class TestThermalCollision:
    def test_preserves_shape(self) -> None:
        g = _g_thermal()
        T = thermal_macroscopic_3d(g)
        ux = torch.zeros_like(T)
        uy = torch.zeros_like(T)
        uz = torch.zeros_like(T)
        g_out = thermal_collide_bgk_3d(g, T, ux, uy, uz, tau_T=TAU_T)
        assert g_out.shape == g.shape

    def test_conserves_temperature(self) -> None:
        g = _g_thermal()
        T = thermal_macroscopic_3d(g)
        ux = torch.full_like(T, 0.03)
        uy = torch.full_like(T, -0.01)
        uz = torch.zeros_like(T)
        # Perturb to create non-equilibrium
        g = g + 0.001 * torch.rand_like(g)
        T_orig = g.sum(dim=0)
        g_out = thermal_collide_bgk_3d(g, T_orig, ux, uy, uz, tau_T=TAU_T)
        T_out = g_out.sum(dim=0)
        assert torch.allclose(T_out, T_orig, atol=1e-5)

    def test_identity_at_equilibrium(self) -> None:
        nz, ny, nx = 4, 6, 8
        T = torch.ones((nz, ny, nx))
        ux = torch.full_like(T, 0.04)
        uy = torch.zeros_like(T)
        uz = torch.full_like(T, -0.01)
        geq = thermal_equilibrium_3d(T, ux, uy, uz)
        g_out = thermal_collide_bgk_3d(geq, T, ux, uy, uz, tau_T=TAU_T)
        assert torch.allclose(g_out, geq, atol=1e-5)

    def test_finite_output(self) -> None:
        g = _g_thermal()
        T = thermal_macroscopic_3d(g)
        ux = torch.rand_like(T) * 0.04
        uy = torch.rand_like(T) * 0.03
        uz = torch.rand_like(T) * 0.02
        g_out = thermal_collide_bgk_3d(g, T, ux, uy, uz, tau_T=0.6)
        assert torch.isfinite(g_out).all()


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

class TestThermalStreaming:
    def test_preserves_shape(self) -> None:
        g = _g_thermal()
        g_out = thermal_stream_3d(g)
        assert g_out.shape == g.shape

    def test_conserves_total_temperature(self) -> None:
        g = _g_thermal(nz=8, ny=8, nx=8)
        total_before = g.sum()
        g_out = thermal_stream_3d(g)
        total_after = g_out.sum()
        assert torch.allclose(total_after, total_before, atol=1e-5)

    def test_finite_output(self) -> None:
        g = _g_thermal()
        g_out = thermal_stream_3d(g)
        assert torch.isfinite(g_out).all()


# ---------------------------------------------------------------------------
# Macroscopic recovery
# ---------------------------------------------------------------------------

class TestThermalMacroscopic:
    def test_recovers_temperature(self) -> None:
        nz, ny, nx = 4, 6, 8
        T_in = torch.rand((nz, ny, nx)) * 2.0 + 0.3
        ux = torch.zeros_like(T_in)
        uy = torch.zeros_like(T_in)
        uz = torch.zeros_like(T_in)
        g = thermal_equilibrium_3d(T_in, ux, uy, uz)
        T_out = thermal_macroscopic_3d(g)
        assert torch.allclose(T_out, T_in, atol=1e-5)


# ---------------------------------------------------------------------------
# Buoyancy force
# ---------------------------------------------------------------------------

class TestBuoyancyForce:
    @pytest.mark.parametrize("lattice", ["D3Q19", "D3Q27"])
    def test_preserves_shape(self, lattice: str) -> None:
        f = _f3d19() if lattice == "D3Q19" else _f3d27()
        T = torch.ones(f.shape[1:])
        f_out = apply_buoyancy_3d(f, T, T_ref=1.0, beta=0.001, lattice=lattice)
        assert f_out.shape == f.shape

    @pytest.mark.parametrize("lattice", ["D3Q19", "D3Q27"])
    def test_zero_force_at_reference_temperature(self, lattice: str) -> None:
        f = _f3d19() if lattice == "D3Q19" else _f3d27()
        T_ref = 1.0
        T = torch.full_like(f[0], T_ref)
        f_out = apply_buoyancy_3d(f, T, T_ref=T_ref, beta=0.01, lattice=lattice)
        assert torch.allclose(f_out, f, atol=1e-6)

    @pytest.mark.parametrize("lattice", ["D3Q19", "D3Q27"])
    def test_finite_output(self, lattice: str) -> None:
        f = _f3d19() if lattice == "D3Q19" else _f3d27()
        T = torch.rand(f.shape[1:]) * 0.5 + 0.75
        f_out = apply_buoyancy_3d(f, T, T_ref=1.0, beta=0.001, lattice=lattice)
        assert torch.isfinite(f_out).all()

    def test_rejects_unknown_lattice(self) -> None:
        f = _f3d19()
        T = torch.ones(f.shape[1:])
        with pytest.raises(ValueError, match="lattice"):
            apply_buoyancy_3d(f, T, T_ref=1.0, beta=0.001, lattice="D2Q9")


# ---------------------------------------------------------------------------
# Combined thermal_step
# ---------------------------------------------------------------------------

class TestThermalStep:
    @pytest.mark.parametrize("lattice", ["D3Q19", "D3Q27"])
    def test_shape(self, lattice: str) -> None:
        f = _f3d19() if lattice == "D3Q19" else _f3d27()
        g = _g_thermal()
        f_out, g_out, T_out = thermal_step(f, g, tau_T=TAU_T, lattice=lattice)
        assert f_out.shape == f.shape
        assert g_out.shape == g.shape
        assert T_out.shape == f.shape[1:]

    @pytest.mark.parametrize("lattice", ["D3Q19", "D3Q27"])
    def test_finite(self, lattice: str) -> None:
        f = _f3d19() if lattice == "D3Q19" else _f3d27()
        g = _g_thermal()
        f_out, g_out, T_out = thermal_step(f, g, tau_T=TAU_T, lattice=lattice)
        assert torch.isfinite(f_out).all()
        assert torch.isfinite(g_out).all()
        assert torch.isfinite(T_out).all()

    @pytest.mark.parametrize("lattice", ["D3Q19", "D3Q27"])
    def test_no_buoyancy_preserves_mass(self, lattice: str) -> None:
        """With beta=0, f should be unchanged (no buoyancy coupling)."""
        f = _f3d19() if lattice == "D3Q19" else _f3d27()
        g = _g_thermal()
        f_out, _, _ = thermal_step(f, g, tau_T=TAU_T, lattice=lattice, beta=0.0)
        assert torch.allclose(f_out, f, atol=1e-7)

    @pytest.mark.parametrize("lattice", ["D3Q19", "D3Q27"])
    def test_temperature_conservation(self, lattice: str) -> None:
        """Thermal step should conserve total temperature (periodic)."""
        f = _f3d19() if lattice == "D3Q19" else _f3d27()
        g = _g_thermal()
        T_before = thermal_macroscopic_3d(g)
        _, _, T_after = thermal_step(f, g, tau_T=TAU_T, lattice=lattice)
        assert torch.allclose(T_after.sum(), T_before.sum(), atol=1e-4)

    @pytest.mark.parametrize("lattice", ["D3Q19", "D3Q27"])
    def test_buoyancy_modifies_f(self, lattice: str) -> None:
        """With beta>0 and T≠T_ref, f should be modified."""
        f = _f3d19() if lattice == "D3Q19" else _f3d27()
        g = _g_thermal()
        # Set T away from T_ref
        T_field = torch.full(f.shape[1:], 2.0)
        g = thermal_equilibrium_3d(T_field, torch.zeros_like(T_field),
                                   torch.zeros_like(T_field), torch.zeros_like(T_field))
        f_out, _, _ = thermal_step(f, g, tau_T=TAU_T, lattice=lattice,
                                   beta=0.01, T_ref=1.0)
        assert not torch.allclose(f_out, f, atol=1e-7)

    def test_rejects_unknown_lattice(self) -> None:
        f = _f3d19()
        g = _g_thermal()
        with pytest.raises(ValueError, match="lattice"):
            thermal_step(f, g, tau_T=TAU_T, lattice="D2Q9")


# ---------------------------------------------------------------------------
# Conjugate heat transfer
# ---------------------------------------------------------------------------

class TestConjugateHT:
    def test_shape_2d(self) -> None:
        ny, nx = 8, 12
        T_f = torch.ones((ny, nx))
        T_s = torch.ones((ny, nx))
        mask = torch.zeros((ny, nx), dtype=torch.bool)
        T_f_out, T_s_out = conjugate_ht_step(T_f, T_s, mask, alpha_s=0.1)
        assert T_f_out.shape == T_f.shape
        assert T_s_out.shape == T_s.shape

    def test_shape_3d(self) -> None:
        nz, ny, nx = 4, 6, 8
        T_f = torch.ones((nz, ny, nx))
        T_s = torch.ones((nz, ny, nx))
        mask = torch.zeros((nz, ny, nx), dtype=torch.bool)
        T_f_out, T_s_out = conjugate_ht_step(T_f, T_s, mask, alpha_s=0.1)
        assert T_f_out.shape == T_f.shape
        assert T_s_out.shape == T_s.shape

    def test_no_solid_no_change_to_fluid(self) -> None:
        """With no solid cells, fluid temperature should be unchanged."""
        nz, ny, nx = 4, 6, 8
        T_f = torch.rand((nz, ny, nx))
        T_s = torch.rand((nz, ny, nx))
        mask = torch.zeros((nz, ny, nx), dtype=torch.bool)
        T_f_out, _ = conjugate_ht_step(T_f, T_s, mask, alpha_s=0.1)
        assert torch.allclose(T_f_out, T_f, atol=1e-7)

    def test_solid_diffusion_decays(self) -> None:
        """A sinusoidal solid temperature mode should decay."""
        nz, ny, nx = 1, 1, 32
        mask = torch.ones((nz, ny, nx), dtype=torch.bool)
        k = 2.0 * math.pi / nx
        xs = torch.arange(nx, dtype=torch.float32)
        T_s = 0.1 * torch.sin(k * xs).unsqueeze(0).unsqueeze(0) + 1.0
        T_f = torch.ones_like(T_s)
        alpha_s = 0.1
        n_steps = 50
        for _ in range(n_steps):
            T_f, T_s = conjugate_ht_step(T_f, T_s, mask, alpha_s=alpha_s)
        # Amplitude should decay
        T_mean = float(T_s.mean().item())
        amp = float((T_s - T_mean).abs().max().item())
        assert amp < 0.1, f"Solid temperature did not decay: amp={amp}"

    def test_interface_coupling(self) -> None:
        """Interface cells should have averaged temperature."""
        nz, ny, nx = 1, 1, 8
        mask = torch.zeros((nz, ny, nx), dtype=torch.bool)
        mask[..., 4:] = True  # right half is solid
        T_f = torch.ones((nz, ny, nx)) * 2.0
        T_s = torch.ones((nz, ny, nx)) * 1.0
        T_f_out, T_s_out = conjugate_ht_step(T_f, T_s, mask, alpha_s=0.0, k_ratio=1.0)
        # At interface (k_ratio=1), T_int = 0.5*T_f + 0.5*T_s = 1.5
        # Fluid cell at index 3 (adjacent to solid at 4)
        assert abs(float(T_f_out[..., 3].item()) - 1.5) < 1e-5

    def test_finite_output(self) -> None:
        nz, ny, nx = 4, 6, 8
        T_f = torch.rand((nz, ny, nx))
        T_s = torch.rand((nz, ny, nx))
        mask = torch.zeros((nz, ny, nx), dtype=torch.bool)
        mask[..., 0] = True
        T_f_out, T_s_out = conjugate_ht_step(T_f, T_s, mask, alpha_s=0.1, k_ratio=5.0)
        assert torch.isfinite(T_f_out).all()
        assert torch.isfinite(T_s_out).all()


# ---------------------------------------------------------------------------
# Composability: thermal_step with different collision outputs
# ---------------------------------------------------------------------------

class TestThermalComposability:
    @pytest.mark.parametrize("lattice", ["D3Q19", "D3Q27"])
    def test_compose_with_bgk(self, lattice: str) -> None:
        """thermal_step should accept BGK-collided f without error."""
        from tensorlbm.solver3d import collide_bgk3d, stream3d
        from tensorlbm.d3q27 import collide_bgk27, stream27

        f = _f3d19() if lattice == "D3Q19" else _f3d27()
        g = _g_thermal()
        T = thermal_macroscopic_3d(g)
        tau = 0.7
        for _ in range(3):
            if lattice == "D3Q19":
                f = collide_bgk3d(f, tau)
                f = stream3d(f)
            else:
                f = collide_bgk27(f, tau)
                f = stream27(f)
            f, g, T = thermal_step(f, g, tau_T=TAU_T, lattice=lattice)
        assert torch.isfinite(f).all()
        assert torch.isfinite(g).all()
        assert torch.isfinite(T).all()
