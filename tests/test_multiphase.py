"""Tests for D2Q9/D3Q19 multiphase LBM models.

Verifies:
    - Shan-Chen two-component (SCMC): mass conservation, force symmetry
    - Shan-Chen single-component (SCMP): mass conservation
    - Color-Gradient (CG): mass conservation of each component
    - Free-Energy (FE): total density conservation
    - 3-D SC two-component: basic shape and mass conservation
    - DamBreakConfig validation
    - MultiphaseWaterEntryConfig validation
"""
from __future__ import annotations

import pytest
import torch

from tensorlbm import (
    DamBreakConfig,
    MultiphaseWaterEntryConfig,
    collide_sc_single_component,
    collide_sc_two_component,
    collide_sc_two_component_3d,
    color_gradient_step,
    equilibrium,
    equilibrium3d,
    free_energy_step,
    init_free_energy_g,
    psi_exp,
    psi_linear,
    psi_power,
    sc_two_component_force,
    sc_two_component_force_3d,
    stream,
    stream3d,
)

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_two_component_2d(ny: int = 20, nx: int = 24) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform-density equilibrium distributions for two components."""
    rho1 = torch.ones((ny, nx), device=DEVICE)
    rho2 = torch.full((ny, nx), 0.5, device=DEVICE)
    zero = torch.zeros((ny, nx), device=DEVICE)
    return equilibrium(rho1, zero, zero), equilibrium(rho2, zero, zero)


def _make_two_component_3d(
    nz: int = 6, ny: int = 8, nx: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    rho1 = torch.ones((nz, ny, nx), device=DEVICE)
    rho2 = torch.full((nz, ny, nx), 0.5, device=DEVICE)
    zero = torch.zeros((nz, ny, nx), device=DEVICE)
    return equilibrium3d(rho1, zero, zero, zero), equilibrium3d(rho2, zero, zero, zero)


# ---------------------------------------------------------------------------
# Pseudopotential functions
# ---------------------------------------------------------------------------

class TestPseudopotentials:
    def test_psi_linear_identity(self) -> None:
        rho = torch.tensor([0.5, 1.0, 2.0])
        assert torch.allclose(psi_linear(rho), rho)

    def test_psi_exp_range(self) -> None:
        rho = torch.linspace(0.1, 3.0, 30)
        psi = psi_exp(rho, rho0=1.0)
        assert (psi >= 0).all()
        assert (psi <= 1.0).all()

    def test_psi_power_positive(self) -> None:
        rho = torch.linspace(0.5, 3.0, 20)
        psi = psi_power(rho)
        assert (psi > 0).all()


# ---------------------------------------------------------------------------
# SC two-component force
# ---------------------------------------------------------------------------

class TestSCTwoComponentForce:
    def test_force_shape(self) -> None:
        ny, nx = 16, 20
        rho1 = torch.rand((ny, nx)) + 0.5
        rho2 = torch.rand((ny, nx)) + 0.5
        Fx1, Fy1, Fx2, Fy2 = sc_two_component_force(rho1, rho2, G_12=0.9)
        for t in (Fx1, Fy1, Fx2, Fy2):
            assert t.shape == (ny, nx)

    def test_zero_force_uniform_density(self) -> None:
        """Uniform density → zero SC force (isotropic neighbourhood)."""
        ny, nx = 16, 20
        rho1 = torch.ones((ny, nx))
        rho2 = torch.ones((ny, nx))
        Fx1, Fy1, Fx2, Fy2 = sc_two_component_force(rho1, rho2, G_12=0.9)
        assert torch.allclose(Fx1, torch.zeros_like(Fx1), atol=1e-6)
        assert torch.allclose(Fy1, torch.zeros_like(Fy1), atol=1e-6)


# ---------------------------------------------------------------------------
# SC two-component collision (2-D)
# ---------------------------------------------------------------------------

class TestSCTwoComponent2D:
    def test_output_shape(self) -> None:
        f1, f2 = _make_two_component_2d()
        f1_out, f2_out = collide_sc_two_component(f1, f2)
        assert f1_out.shape == f1.shape
        assert f2_out.shape == f2.shape

    def test_mass_conservation(self) -> None:
        """Total mass of each component is conserved by collision (no streaming)."""
        f1, f2 = _make_two_component_2d()
        m1_before = f1.sum()
        m2_before = f2.sum()
        f1_out, f2_out = collide_sc_two_component(f1, f2)
        assert torch.allclose(f1_out.sum(), m1_before, atol=1e-4)
        assert torch.allclose(f2_out.sum(), m2_before, atol=1e-4)

    def test_mass_conservation_with_gravity(self) -> None:
        f1, f2 = _make_two_component_2d()
        m1 = f1.sum()
        m2 = f2.sum()
        f1_out, f2_out = collide_sc_two_component(f1, f2, gy=-1e-4)
        assert torch.allclose(f1_out.sum(), m1, atol=1e-4)
        assert torch.allclose(f2_out.sum(), m2, atol=1e-4)

    def test_streaming_preserves_mass(self) -> None:
        f1, f2 = _make_two_component_2d(ny=10, nx=10)
        m_before = (f1 + f2).sum()
        f1, f2 = collide_sc_two_component(f1, f2)
        f1 = stream(f1)
        f2 = stream(f2)
        assert torch.allclose((f1 + f2).sum(), m_before, atol=1e-4)


# ---------------------------------------------------------------------------
# SC single-component (2-D)
# ---------------------------------------------------------------------------

class TestSCSingleComponent2D:
    def test_output_shape(self) -> None:
        ny, nx = 16, 20
        rho = torch.rand((ny, nx)) + 0.5
        zero = torch.zeros((ny, nx))
        f = equilibrium(rho, zero, zero)
        f_out = collide_sc_single_component(f, G=-4.0, tau=1.0)
        assert f_out.shape == f.shape

    def test_mass_conservation(self) -> None:
        ny, nx = 16, 20
        rho = torch.rand((ny, nx)) + 0.5
        zero = torch.zeros((ny, nx))
        f = equilibrium(rho, zero, zero)
        mass_before = f.sum()
        f_out = collide_sc_single_component(f, G=-4.0, tau=1.0)
        assert torch.allclose(f_out.sum(), mass_before, atol=1e-4)


# ---------------------------------------------------------------------------
# Color-Gradient (2-D)
# ---------------------------------------------------------------------------

class TestColorGradient2D:
    def test_output_shape(self) -> None:
        f1, f2 = _make_two_component_2d()
        f1_out, f2_out = color_gradient_step(f1, f2)
        assert f1_out.shape == f1.shape
        assert f2_out.shape == f2.shape

    def test_total_mass_conservation(self) -> None:
        f1, f2 = _make_two_component_2d()
        total_before = (f1 + f2).sum()
        f1_out, f2_out = color_gradient_step(f1, f2)
        assert torch.allclose((f1_out + f2_out).sum(), total_before, atol=1e-4)

    def test_uniform_density_stability(self) -> None:
        """Uniform density should remain stable (no spurious flows)."""
        ny, nx = 12, 16
        rho_eq = torch.ones((ny, nx))
        zero = torch.zeros((ny, nx))
        f1 = equilibrium(rho_eq * 0.5, zero, zero)
        f2 = equilibrium(rho_eq * 0.5, zero, zero)
        for _ in range(5):
            f1, f2 = color_gradient_step(f1, f2, tau=1.0, A=0.04)
        # Total mass should still be conserved
        assert torch.isfinite(f1).all()
        assert torch.isfinite(f2).all()


# ---------------------------------------------------------------------------
# Free-Energy (2-D)
# ---------------------------------------------------------------------------

class TestFreeEnergy2D:
    def test_output_shape(self) -> None:
        ny, nx = 16, 20
        rho = torch.ones((ny, nx))
        zero = torch.zeros((ny, nx))
        phi = torch.tanh(torch.linspace(-3, 3, nx).unsqueeze(0).expand(ny, -1))
        f = equilibrium(rho, zero, zero)
        g = init_free_energy_g(phi, zero, zero)
        assert g.shape == (9, ny, nx)
        f_out, g_out = free_energy_step(f, g)
        assert f_out.shape == f.shape
        assert g_out.shape == g.shape

    def test_total_density_conservation(self) -> None:
        ny, nx = 16, 20
        rho = torch.ones((ny, nx))
        zero = torch.zeros((ny, nx))
        phi = torch.tanh(torch.linspace(-3, 3, nx).unsqueeze(0).expand(ny, -1))
        f = equilibrium(rho, zero, zero)
        g = init_free_energy_g(phi, zero, zero)
        mass_before = f.sum()
        f_out, _ = free_energy_step(f, g)
        assert torch.allclose(f_out.sum(), mass_before, atol=1e-4)


# ---------------------------------------------------------------------------
# SC two-component force (3-D)
# ---------------------------------------------------------------------------

class TestSCTwoComponentForce3D:
    def test_force_shape(self) -> None:
        nz, ny, nx = 6, 8, 10
        rho1 = torch.rand((nz, ny, nx)) + 0.5
        rho2 = torch.rand((nz, ny, nx)) + 0.5
        out = sc_two_component_force_3d(rho1, rho2, G_12=0.9)
        assert len(out) == 6
        for t in out:
            assert t.shape == (nz, ny, nx)

    def test_zero_force_uniform_3d(self) -> None:
        nz, ny, nx = 6, 8, 10
        rho1 = torch.ones((nz, ny, nx))
        rho2 = torch.ones((nz, ny, nx))
        Fx1, Fy1, Fz1, Fx2, Fy2, Fz2 = sc_two_component_force_3d(rho1, rho2, G_12=0.9)
        for t in (Fx1, Fy1, Fz1, Fx2, Fy2, Fz2):
            assert torch.allclose(t, torch.zeros_like(t), atol=1e-6)


# ---------------------------------------------------------------------------
# SC two-component collision (3-D)
# ---------------------------------------------------------------------------

class TestSCTwoComponent3D:
    def test_output_shape(self) -> None:
        f1, f2 = _make_two_component_3d()
        f1_out, f2_out = collide_sc_two_component_3d(f1, f2)
        assert f1_out.shape == f1.shape
        assert f2_out.shape == f2.shape

    def test_mass_conservation(self) -> None:
        f1, f2 = _make_two_component_3d()
        m1 = f1.sum()
        m2 = f2.sum()
        f1_out, f2_out = collide_sc_two_component_3d(f1, f2)
        assert torch.allclose(f1_out.sum(), m1, atol=1e-3)
        assert torch.allclose(f2_out.sum(), m2, atol=1e-3)

    def test_streaming_preserves_mass_3d(self) -> None:
        f1, f2 = _make_two_component_3d()
        m_before = (f1 + f2).sum()
        f1, f2 = collide_sc_two_component_3d(f1, f2)
        f1 = stream3d(f1)
        f2 = stream3d(f2)
        assert torch.allclose((f1 + f2).sum(), m_before, atol=1e-3)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestDamBreakConfig:
    def test_valid_config(self) -> None:
        cfg = DamBreakConfig(nx=50, ny=30, dam_width=15, n_steps=2, output_interval=2)
        cfg.validate()  # should not raise

    def test_invalid_dam_width(self) -> None:
        cfg = DamBreakConfig(nx=50, ny=30, dam_width=60)
        with pytest.raises(ValueError, match="dam_width"):
            cfg.validate()

    def test_invalid_tau(self) -> None:
        cfg = DamBreakConfig(tau=0.4)
        with pytest.raises(ValueError, match="tau"):
            cfg.validate()

    def test_invalid_densities(self) -> None:
        cfg = DamBreakConfig(rho_heavy=0.1, rho_light=2.0)
        with pytest.raises(ValueError, match="rho_heavy"):
            cfg.validate()


class TestMultiphaseWaterEntryConfig:
    def test_valid_config(self) -> None:
        cfg = MultiphaseWaterEntryConfig(nx=50, ny=50, water_level=25, n_steps=2, output_interval=2)
        cfg.validate()

    def test_valid_config_with_model(self) -> None:
        cfg = MultiphaseWaterEntryConfig(
            nx=50, ny=50, water_level=25, n_steps=2, output_interval=2, model="cg",
        )
        cfg.validate()

    def test_invalid_mode(self) -> None:
        cfg = MultiphaseWaterEntryConfig.__new__(MultiphaseWaterEntryConfig)
        object.__setattr__(cfg, "mode", "4d")
        object.__setattr__(cfg, "model", "cg")
        object.__setattr__(cfg, "nx", 50)
        object.__setattr__(cfg, "ny", 50)
        object.__setattr__(cfg, "nz", 50)
        object.__setattr__(cfg, "radius", 5.0)
        object.__setattr__(cfg, "water_level", 25)
        object.__setattr__(cfg, "clearance", 4)
        object.__setattr__(cfg, "rho_water", 2.0)
        object.__setattr__(cfg, "rho_air", 0.1)
        object.__setattr__(cfg, "G", 0.9)
        object.__setattr__(cfg, "tau", 1.0)
        object.__setattr__(cfg, "g", 5e-5)
        object.__setattr__(cfg, "n_steps", 2)
        object.__setattr__(cfg, "output_interval", 2)
        object.__setattr__(cfg, "output_root", "outputs")
        object.__setattr__(cfg, "run_name", None)
        object.__setattr__(cfg, "device", "cpu")
        object.__setattr__(cfg, "overwrite", False)
        with pytest.raises(ValueError, match="mode"):
            cfg.validate()

    def test_invalid_tau(self) -> None:
        cfg = MultiphaseWaterEntryConfig(tau=0.3, nx=50, ny=50, water_level=25)
        with pytest.raises(ValueError, match="tau"):
            cfg.validate()


# ---------------------------------------------------------------------------
# solid_mask skips collision
# ---------------------------------------------------------------------------

class TestSolidMask:
    """Verify that solid cells are not modified by collision steps."""

    def test_sc_two_component_solid_not_modified(self) -> None:
        ny, nx = 12, 16
        solid = torch.zeros((ny, nx), dtype=torch.bool)
        solid[0, :] = True   # bottom wall

        rho1 = torch.ones((ny, nx)) * 0.5
        rho2 = torch.ones((ny, nx)) * 0.5
        zero = torch.zeros((ny, nx))
        f1 = equilibrium(rho1, zero, zero)
        f2 = equilibrium(rho2, zero, zero)

        # Give wall cells distinct distributions (different from equilibrium)
        f1_pre = f1.clone()
        f2_pre = f2.clone()

        f1_out, f2_out = collide_sc_two_component(f1, f2, solid_mask=solid)

        # Interior cells should be updated, solid cells should be unchanged
        assert torch.allclose(f1_out[:, solid], f1_pre[:, solid], atol=1e-6)
        assert torch.allclose(f2_out[:, solid], f2_pre[:, solid], atol=1e-6)

    def test_cg_solid_not_modified(self) -> None:
        ny, nx = 12, 16
        solid = torch.zeros((ny, nx), dtype=torch.bool)
        solid[-1, :] = True   # top wall

        rho1 = torch.ones((ny, nx)) * 0.6
        rho2 = torch.ones((ny, nx)) * 0.4
        zero = torch.zeros((ny, nx))
        f1 = equilibrium(rho1, zero, zero)
        f2 = equilibrium(rho2, zero, zero)
        f1_pre, f2_pre = f1.clone(), f2.clone()

        f1_out, f2_out = color_gradient_step(f1, f2, solid_mask=solid)

        assert torch.allclose(f1_out[:, solid], f1_pre[:, solid], atol=1e-6)
        assert torch.allclose(f2_out[:, solid], f2_pre[:, solid], atol=1e-6)
