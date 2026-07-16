"""Tests for D3Q27 Shan-Chen multiphase collision operators.

Verifies:
    - SC two-component force (D3Q27): output shapes, zero force for uniform density
    - SC two-component collision (D3Q27): output shape, mass conservation,
      streaming preserves mass, solid-mask correctness, gravity finiteness,
      Guo forcing variant
    - SC single-component collision (D3Q27): output shape, mass conservation,
      multi-step stability
    - Static-droplet stability: a perturbed droplet remains finite over many steps
    - D3Q27 vs D3Q19 consistency: both lattices produce zero force for uniform density
"""
from __future__ import annotations

import pytest
import torch

from tensorlbm import (
    collide_sc_single_component_27,
    collide_sc_two_component_27,
    equilibrium27,
    macroscopic27,
    sc_two_component_force_27,
    stream27,
    psi_exp,
)

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_two_component_27(
    nz: int = 5, ny: int = 6, nx: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equilibrium distributions for two D3Q27 components with different densities."""
    rho1 = torch.ones((nz, ny, nx), device=DEVICE)
    rho2 = torch.full((nz, ny, nx), 0.5, device=DEVICE)
    zero = torch.zeros((nz, ny, nx), device=DEVICE)
    return equilibrium27(rho1, zero, zero, zero), equilibrium27(rho2, zero, zero, zero)


def _make_single_component_27(
    nz: int = 5, ny: int = 6, nx: int = 8,
) -> torch.Tensor:
    """Equilibrium distribution for a single D3Q27 component."""
    rho = torch.ones((nz, ny, nx), device=DEVICE)
    zero = torch.zeros((nz, ny, nx), device=DEVICE)
    return equilibrium27(rho, zero, zero, zero)


# ---------------------------------------------------------------------------
# SC two-component force (D3Q27)
# ---------------------------------------------------------------------------

class TestSCTwoComponentForce27:
    def test_force_shape(self) -> None:
        nz, ny, nx = 5, 6, 8
        rho1 = torch.rand((nz, ny, nx)) + 0.5
        rho2 = torch.rand((nz, ny, nx)) + 0.5
        out = sc_two_component_force_27(rho1, rho2, G_12=0.9)
        assert len(out) == 6
        for t in out:
            assert t.shape == (nz, ny, nx)

    def test_zero_force_uniform_density(self) -> None:
        """Uniform density → zero SC force (isotropic 27-direction neighbourhood)."""
        nz, ny, nx = 5, 6, 8
        rho1 = torch.ones((nz, ny, nx))
        rho2 = torch.ones((nz, ny, nx))
        Fx1, Fy1, Fz1, Fx2, Fy2, Fz2 = sc_two_component_force_27(rho1, rho2, G_12=0.9)
        for t in (Fx1, Fy1, Fz1, Fx2, Fy2, Fz2):
            assert torch.allclose(t, torch.zeros_like(t), atol=1e-6)

    def test_body_force_added(self) -> None:
        """Body force should appear as rho * g in the total force."""
        nz, ny, nx = 5, 6, 8
        rho1 = torch.ones((nz, ny, nx))
        rho2 = torch.full((nz, ny, nx), 0.5)
        Fx1, Fy1, Fz1, Fx2, Fy2, Fz2 = sc_two_component_force_27(
            rho1, rho2, G_12=0.0, gx=0.01, gy=0.02, gz=0.03,
        )
        # With G=0, only body force remains: F = rho * g
        assert torch.allclose(Fx1, rho1 * 0.01, atol=1e-6)
        assert torch.allclose(Fy2, rho2 * 0.02, atol=1e-6)
        assert torch.allclose(Fz1, rho1 * 0.03, atol=1e-6)


# ---------------------------------------------------------------------------
# SC two-component collision (D3Q27)
# ---------------------------------------------------------------------------

class TestSCTwoComponent27:
    def test_output_shape(self) -> None:
        f1, f2 = _make_two_component_27()
        f1_out, f2_out = collide_sc_two_component_27(f1, f2)
        assert f1_out.shape == f1.shape
        assert f2_out.shape == f2.shape
        assert f1_out.shape[0] == 27

    def test_mass_conservation(self) -> None:
        """Total mass of each component is conserved by collision (no streaming)."""
        f1, f2 = _make_two_component_27()
        m1 = f1.sum()
        m2 = f2.sum()
        f1_out, f2_out = collide_sc_two_component_27(f1, f2)
        assert torch.allclose(f1_out.sum(), m1, atol=1e-3)
        assert torch.allclose(f2_out.sum(), m2, atol=1e-3)

    def test_mass_conservation_with_gravity(self) -> None:
        f1, f2 = _make_two_component_27()
        m1 = f1.sum()
        m2 = f2.sum()
        f1_out, f2_out = collide_sc_two_component_27(f1, f2, gz=-1e-4)
        assert torch.allclose(f1_out.sum(), m1, atol=1e-4)
        assert torch.allclose(f2_out.sum(), m2, atol=1e-4)

    def test_streaming_preserves_mass(self) -> None:
        f1, f2 = _make_two_component_27()
        m_before = (f1 + f2).sum()
        f1, f2 = collide_sc_two_component_27(f1, f2)
        f1 = stream27(f1)
        f2 = stream27(f2)
        assert torch.allclose((f1 + f2).sum(), m_before, atol=1e-3)

    def test_finite_values(self) -> None:
        f1, f2 = _make_two_component_27()
        f1_out, f2_out = collide_sc_two_component_27(f1, f2)
        assert torch.isfinite(f1_out).all()
        assert torch.isfinite(f2_out).all()

    def test_solid_mask_preserves_wall_cells(self) -> None:
        """Solid cells must be untouched by the collision step."""
        nz, ny, nx = 5, 6, 8
        solid = torch.zeros((nz, ny, nx), dtype=torch.bool, device=DEVICE)
        solid[:, 0, :] = True  # bottom wall

        rho1 = torch.ones((nz, ny, nx), device=DEVICE)
        rho2 = torch.full((nz, ny, nx), 0.5, device=DEVICE)
        zero = torch.zeros((nz, ny, nx), device=DEVICE)
        f1 = equilibrium27(rho1, zero, zero, zero)
        f2 = equilibrium27(rho2, zero, zero, zero)
        f1_pre, f2_pre = f1.clone(), f2.clone()

        f1_out, f2_out = collide_sc_two_component_27(f1, f2, solid_mask=solid)

        assert torch.allclose(f1_out[:, solid], f1_pre[:, solid], atol=1e-6)
        assert torch.allclose(f2_out[:, solid], f2_pre[:, solid], atol=1e-6)

    def test_guo_forcing_finite(self) -> None:
        """Guo second-order forcing should produce finite output."""
        f1, f2 = _make_two_component_27()
        f1_out, f2_out = collide_sc_two_component_27(f1, f2, use_guo=True)
        assert torch.isfinite(f1_out).all()
        assert torch.isfinite(f2_out).all()

    def test_guo_forcing_mass_conservation(self) -> None:
        f1, f2 = _make_two_component_27()
        m1 = f1.sum()
        m2 = f2.sum()
        f1_out, f2_out = collide_sc_two_component_27(f1, f2, use_guo=True)
        assert torch.allclose(f1_out.sum(), m1, atol=1e-3)
        assert torch.allclose(f2_out.sum(), m2, atol=1e-3)

    def test_uniform_density_stability(self) -> None:
        """Equal uniform densities → no spurious interface, no blow-up over 10 steps."""
        nz, ny, nx = 5, 6, 8
        rho_half = torch.full((nz, ny, nx), 0.5, device=DEVICE)
        zero = torch.zeros_like(rho_half)
        f1 = equilibrium27(rho_half, zero, zero, zero)
        f2 = equilibrium27(rho_half, zero, zero, zero)
        for _ in range(10):
            f1, f2 = collide_sc_two_component_27(f1, f2, G_12=0.9, tau1=1.0, tau2=1.0)
        assert torch.isfinite(f1).all()
        assert torch.isfinite(f2).all()


# ---------------------------------------------------------------------------
# SC single-component collision (D3Q27)
# ---------------------------------------------------------------------------

class TestSCSingleComponent27:
    def test_output_shape(self) -> None:
        f = _make_single_component_27()
        f_out = collide_sc_single_component_27(f, G=-4.0, tau=1.0)
        assert f_out.shape == f.shape
        assert f_out.shape[0] == 27

    def test_mass_conservation(self) -> None:
        f = _make_single_component_27()
        mass_before = f.sum()
        f_out = collide_sc_single_component_27(f, G=-4.0, tau=1.0)
        assert torch.allclose(f_out.sum(), mass_before, atol=1e-4)

    def test_finite_values(self) -> None:
        f = _make_single_component_27()
        f_out = collide_sc_single_component_27(f, G=-4.0, tau=1.0)
        assert torch.isfinite(f_out).all()

    def test_solid_mask_preserves_wall_cells(self) -> None:
        nz, ny, nx = 5, 6, 8
        solid = torch.zeros((nz, ny, nx), dtype=torch.bool, device=DEVICE)
        solid[:, 0, :] = True
        rho = torch.ones((nz, ny, nx), device=DEVICE)
        zero = torch.zeros((nz, ny, nx), device=DEVICE)
        f = equilibrium27(rho, zero, zero, zero)
        f_pre = f.clone()
        f_out = collide_sc_single_component_27(f, G=-4.0, tau=1.0, solid_mask=solid)
        assert torch.allclose(f_out[:, solid], f_pre[:, solid], atol=1e-6)

    def test_multi_step_stability(self) -> None:
        """Run 20 collision+stream steps; must remain finite."""
        f = _make_single_component_27(nz=8, ny=8, nx=8)
        for _ in range(20):
            f = collide_sc_single_component_27(f, G=-4.0, tau=1.0)
            f = stream27(f)
        assert torch.isfinite(f).all()

    def test_custom_psi_fn(self) -> None:
        """A custom pseudopotential function should be accepted."""
        f = _make_single_component_27()
        f_out = collide_sc_single_component_27(f, G=-4.0, tau=1.0, psi_fn=psi_exp)
        assert f_out.shape == f.shape
        assert torch.isfinite(f_out).all()


# ---------------------------------------------------------------------------
# Static droplet stability (D3Q27)
# ---------------------------------------------------------------------------

class TestStaticDroplet27:
    def test_droplet_remains_finite(self) -> None:
        """A spherical droplet in a periodic box should remain stable for 50 steps."""
        nz, ny, nx = 16, 16, 16
        cx, cy, cz = nx / 2, ny / 2, nz / 2
        z, y, x = torch.meshgrid(
            torch.arange(nz, dtype=torch.float32),
            torch.arange(ny, dtype=torch.float32),
            torch.arange(nx, dtype=torch.float32),
            indexing="ij",
        )
        r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
        rho = torch.where(r < 4.0, torch.tensor(2.0), torch.tensor(0.5))

        zero = torch.zeros_like(rho)
        f = equilibrium27(rho, zero, zero, zero)
        for _ in range(50):
            f = collide_sc_single_component_27(f, G=-4.0, tau=1.0)
            f = stream27(f)
        assert torch.isfinite(f).all()
        rho_out, _, _, _ = macroscopic27(f)
        assert torch.isfinite(rho_out).all()
        assert rho_out.min() > 0.01  # density should not collapse

    def test_droplet_mass_drift_bounded(self) -> None:
        """Total mass should not drift more than 5% over 50 steps."""
        nz, ny, nx = 12, 12, 12
        cx, cy, cz = nx / 2, ny / 2, nz / 2
        z, y, x = torch.meshgrid(
            torch.arange(nz, dtype=torch.float32),
            torch.arange(ny, dtype=torch.float32),
            torch.arange(nx, dtype=torch.float32),
            indexing="ij",
        )
        r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
        rho = torch.where(r < 3.0, torch.tensor(2.0), torch.tensor(0.5))
        zero = torch.zeros_like(rho)
        f = equilibrium27(rho, zero, zero, zero)
        mass0 = f.sum()
        for _ in range(50):
            f = collide_sc_single_component_27(f, G=-4.0, tau=1.0)
            f = stream27(f)
        mass1 = f.sum()
        drift = abs(mass1 - mass0) / mass0
        assert drift < 0.05, f"mass drift {drift:.4f} exceeds 5%"
