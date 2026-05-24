"""Tests for 3-D multiphase LBM models: Color-Gradient and Free-Energy (D3Q19).

Verifies:
    - Color-Gradient 3D: output shapes, total-mass conservation, finite values,
      uniform-density stability, solid-mask correctness
    - Free-Energy 3D: output shapes, total-density conservation, finite values,
      init_free_energy_g_3d shape and finite values
"""
from __future__ import annotations

import torch

from tensorlbm import (
    color_gradient_step_3d,
    equilibrium3d,
    free_energy_step_3d,
    init_free_energy_g_3d,
    stream3d,
)

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_two_component_3d(
    nz: int = 5, ny: int = 6, nx: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equilibrium distributions for two components with different densities."""
    rho1 = torch.ones((nz, ny, nx), device=DEVICE)
    rho2 = torch.full((nz, ny, nx), 0.5, device=DEVICE)
    zero = torch.zeros((nz, ny, nx), device=DEVICE)
    return equilibrium3d(rho1, zero, zero, zero), equilibrium3d(rho2, zero, zero, zero)


def _make_phase_field_3d(nz: int = 5, ny: int = 6, nx: int = 8) -> torch.Tensor:
    """A tanh interface phase field in the x-direction."""
    x = torch.linspace(-3.0, 3.0, nx)
    phi = torch.tanh(x).view(1, 1, nx).expand(nz, ny, nx).contiguous()
    return phi.to(DEVICE)


# ---------------------------------------------------------------------------
# Color-Gradient 3D
# ---------------------------------------------------------------------------

class TestColorGradient3D:
    def test_output_shape(self) -> None:
        f_r, f_b = _make_two_component_3d()
        f_r_out, f_b_out = color_gradient_step_3d(f_r, f_b)
        assert f_r_out.shape == f_r.shape
        assert f_b_out.shape == f_b.shape

    def test_total_mass_conservation(self) -> None:
        """Total mass (both components) must be conserved by collision."""
        f_r, f_b = _make_two_component_3d()
        total_before = (f_r + f_b).sum()
        f_r_out, f_b_out = color_gradient_step_3d(f_r, f_b)
        assert torch.allclose((f_r_out + f_b_out).sum(), total_before, atol=1e-4)

    def test_finite_values(self) -> None:
        f_r, f_b = _make_two_component_3d()
        f_r_out, f_b_out = color_gradient_step_3d(f_r, f_b)
        assert torch.isfinite(f_r_out).all()
        assert torch.isfinite(f_b_out).all()

    def test_uniform_density_stability(self) -> None:
        """Equal uniform densities → no spurious interface, no blow-up."""
        nz, ny, nx = 5, 6, 8
        rho_half = torch.full((nz, ny, nx), 0.5, device=DEVICE)
        zero = torch.zeros_like(rho_half)
        f_r = equilibrium3d(rho_half, zero, zero, zero)
        f_b = equilibrium3d(rho_half, zero, zero, zero)
        for _ in range(5):
            f_r, f_b = color_gradient_step_3d(f_r, f_b, tau=1.0, A=0.04)
        assert torch.isfinite(f_r).all()
        assert torch.isfinite(f_b).all()

    def test_streaming_preserves_mass(self) -> None:
        f_r, f_b = _make_two_component_3d()
        m_before = (f_r + f_b).sum()
        f_r, f_b = color_gradient_step_3d(f_r, f_b)
        f_r = stream3d(f_r)
        f_b = stream3d(f_b)
        assert torch.allclose((f_r + f_b).sum(), m_before, atol=1e-3)

    def test_solid_mask_preserves_wall_cells(self) -> None:
        """Solid cells must be untouched by the CG collision step."""
        nz, ny, nx = 5, 6, 8
        solid = torch.zeros((nz, ny, nx), dtype=torch.bool, device=DEVICE)
        solid[:, 0, :] = True  # bottom wall

        rho1 = torch.ones((nz, ny, nx), device=DEVICE)
        rho2 = torch.full((nz, ny, nx), 0.5, device=DEVICE)
        zero = torch.zeros_like(rho1)
        f_r = equilibrium3d(rho1, zero, zero, zero)
        f_b = equilibrium3d(rho2, zero, zero, zero)
        f_r_pre, f_b_pre = f_r.clone(), f_b.clone()

        f_r_out, f_b_out = color_gradient_step_3d(f_r, f_b, solid_mask=solid)

        # Solid cells must remain identical to pre-collision
        assert torch.allclose(f_r_out[:, solid], f_r_pre[:, solid], atol=1e-6)
        assert torch.allclose(f_b_out[:, solid], f_b_pre[:, solid], atol=1e-6)

    def test_gravity_does_not_break_finiteness(self) -> None:
        f_r, f_b = _make_two_component_3d()
        f_r_out, f_b_out = color_gradient_step_3d(f_r, f_b, gz=-1e-4)
        assert torch.isfinite(f_r_out).all()
        assert torch.isfinite(f_b_out).all()


# ---------------------------------------------------------------------------
# Free-Energy 3D (Phase-Field model)
# ---------------------------------------------------------------------------

class TestFreeEnergy3D:
    def test_init_g_shape(self) -> None:
        nz, ny, nx = 5, 6, 8
        phi = _make_phase_field_3d(nz, ny, nx)
        g = init_free_energy_g_3d(phi)
        assert g.shape == (19, nz, ny, nx)

    def test_init_g_finite(self) -> None:
        phi = _make_phase_field_3d()
        g = init_free_energy_g_3d(phi)
        assert torch.isfinite(g).all()

    def test_init_g_zeroth_moment_equals_phi(self) -> None:
        """Sum over directions of g must recover the phase field."""
        phi = _make_phase_field_3d()
        g = init_free_energy_g_3d(phi)
        phi_recovered = g.sum(dim=0)
        assert torch.allclose(phi_recovered, phi, atol=1e-5)

    def test_output_shapes(self) -> None:
        nz, ny, nx = 5, 6, 8
        rho = torch.ones((nz, ny, nx), device=DEVICE)
        zero = torch.zeros_like(rho)
        phi = _make_phase_field_3d(nz, ny, nx)
        f = equilibrium3d(rho, zero, zero, zero)
        g = init_free_energy_g_3d(phi)
        f_out, g_out = free_energy_step_3d(f, g)
        assert f_out.shape == (19, nz, ny, nx)
        assert g_out.shape == (19, nz, ny, nx)

    def test_total_density_conservation(self) -> None:
        """Total density (zeroth moment of f) must be conserved."""
        nz, ny, nx = 5, 6, 8
        rho = torch.ones((nz, ny, nx), device=DEVICE)
        zero = torch.zeros_like(rho)
        phi = _make_phase_field_3d(nz, ny, nx)
        f = equilibrium3d(rho, zero, zero, zero)
        g = init_free_energy_g_3d(phi)
        mass_before = f.sum()
        f_out, _ = free_energy_step_3d(f, g)
        assert torch.allclose(f_out.sum(), mass_before, atol=1e-4)

    def test_finite_output(self) -> None:
        nz, ny, nx = 5, 6, 8
        rho = torch.ones((nz, ny, nx), device=DEVICE)
        zero = torch.zeros_like(rho)
        phi = _make_phase_field_3d(nz, ny, nx)
        f = equilibrium3d(rho, zero, zero, zero)
        g = init_free_energy_g_3d(phi)
        f_out, g_out = free_energy_step_3d(f, g)
        assert torch.isfinite(f_out).all()
        assert torch.isfinite(g_out).all()

    def test_boussinesq_buoyancy_runs_without_error(self) -> None:
        nz, ny, nx = 5, 6, 8
        rho = torch.ones((nz, ny, nx), device=DEVICE)
        zero = torch.zeros_like(rho)
        phi = _make_phase_field_3d(nz, ny, nx)
        f = equilibrium3d(rho, zero, zero, zero)
        g = init_free_energy_g_3d(phi)
        f_out, g_out = free_energy_step_3d(
            f, g, gz=-1e-4, rho_heavy=2.0, rho_light=0.2,
        )
        assert torch.isfinite(f_out).all()
        assert torch.isfinite(g_out).all()

    def test_init_g_with_velocity(self) -> None:
        nz, ny, nx = 5, 6, 8
        phi = _make_phase_field_3d(nz, ny, nx)
        ux = torch.full_like(phi, 0.03)
        uy = torch.zeros_like(phi)
        uz = torch.zeros_like(phi)
        g = init_free_energy_g_3d(phi, ux=ux, uy=uy, uz=uz)
        assert g.shape == (19, nz, ny, nx)
        assert torch.isfinite(g).all()
        phi_recovered = g.sum(dim=0)
        assert torch.allclose(phi_recovered, phi, atol=1e-5)
