"""Contract tests for the passive-scalar common module.

Verifies the composable D3Q7 passive-scalar LBM step (D3Q7 + D3Q19/D3Q27).

Contract tests verify operator algebra (shape, finite, scalar conservation,
equilibrium identity, source term), NOT scalar transport physics correctness.
"""
from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.d3q27 import equilibrium27, macroscopic27
from tensorlbm.passive_scalar_common import (
    passive_scalar_step,
    scalar_collide_bgk_3d,
    scalar_equilibrium_3d,
    scalar_macroscopic_3d,
    scalar_stream_3d,
)

TAU_D = 0.8  # > 0.5


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


def _g_scalar(nz=4, ny=6, nx=8, phi_mag=1.0, u_mag=0.04) -> torch.Tensor:
    phi = torch.rand((nz, ny, nx)) * phi_mag + 0.5
    ux = torch.rand_like(phi) * u_mag
    uy = torch.rand_like(phi) * u_mag
    uz = torch.rand_like(phi) * u_mag
    return scalar_equilibrium_3d(phi, ux, uy, uz)


# ---------------------------------------------------------------------------
# Equilibrium
# ---------------------------------------------------------------------------

class TestScalarEquilibrium:
    def test_shape(self) -> None:
        nz, ny, nx = 4, 6, 8
        phi = torch.ones((nz, ny, nx))
        ux = torch.zeros_like(phi)
        uy = torch.zeros_like(phi)
        uz = torch.zeros_like(phi)
        geq = scalar_equilibrium_3d(phi, ux, uy, uz)
        assert geq.shape == (7, nz, ny, nx)

    def test_sums_to_scalar(self) -> None:
        nz, ny, nx = 4, 6, 8
        phi = torch.rand((nz, ny, nx)) + 0.5
        ux = torch.full_like(phi, 0.03)
        uy = torch.full_like(phi, -0.02)
        uz = torch.full_like(phi, 0.01)
        geq = scalar_equilibrium_3d(phi, ux, uy, uz)
        phi_out = geq.sum(dim=0)
        assert torch.allclose(phi_out, phi, atol=1e-5)

    def test_finite(self) -> None:
        nz, ny, nx = 4, 6, 8
        phi = torch.rand((nz, ny, nx)) * 2.0 + 0.1
        ux = torch.rand_like(phi) * 0.05
        uy = torch.rand_like(phi) * 0.04
        uz = torch.rand_like(phi) * 0.03
        geq = scalar_equilibrium_3d(phi, ux, uy, uz)
        assert torch.isfinite(geq).all()


# ---------------------------------------------------------------------------
# Collision
# ---------------------------------------------------------------------------

class TestScalarCollision:
    def test_preserves_shape(self) -> None:
        g = _g_scalar()
        phi = scalar_macroscopic_3d(g)
        ux = torch.zeros_like(phi)
        uy = torch.zeros_like(phi)
        uz = torch.zeros_like(phi)
        g_out = scalar_collide_bgk_3d(g, phi, ux, uy, uz, tau_d=TAU_D)
        assert g_out.shape == g.shape

    def test_conserves_scalar(self) -> None:
        """Collision must conserve the zeroth moment (scalar concentration)."""
        g = _g_scalar()
        phi = scalar_macroscopic_3d(g)
        ux = torch.full_like(phi, 0.03)
        uy = torch.full_like(phi, -0.01)
        uz = torch.zeros_like(phi)
        # Perturb to create non-equilibrium
        g = g + 0.001 * torch.rand_like(g)
        phi_orig = g.sum(dim=0)
        g_out = scalar_collide_bgk_3d(g, phi_orig, ux, uy, uz, tau_d=TAU_D)
        phi_out = g_out.sum(dim=0)
        assert torch.allclose(phi_out, phi_orig, atol=1e-5)

    def test_identity_at_equilibrium(self) -> None:
        """g_eq is a fixed point: collide(g_eq) == g_eq."""
        nz, ny, nx = 4, 6, 8
        phi = torch.ones((nz, ny, nx))
        ux = torch.full_like(phi, 0.04)
        uy = torch.zeros_like(phi)
        uz = torch.full_like(phi, -0.01)
        geq = scalar_equilibrium_3d(phi, ux, uy, uz)
        g_out = scalar_collide_bgk_3d(geq, phi, ux, uy, uz, tau_d=TAU_D)
        assert torch.allclose(g_out, geq, atol=1e-5)

    def test_finite_output(self) -> None:
        g = _g_scalar()
        phi = scalar_macroscopic_3d(g)
        ux = torch.rand_like(phi) * 0.04
        uy = torch.rand_like(phi) * 0.03
        uz = torch.rand_like(phi) * 0.02
        g_out = scalar_collide_bgk_3d(g, phi, ux, uy, uz, tau_d=0.6)
        assert torch.isfinite(g_out).all()

    def test_source_adds_scalar(self) -> None:
        """Source term should increase total scalar."""
        g = _g_scalar()
        phi = scalar_macroscopic_3d(g)
        ux = torch.zeros_like(phi)
        uy = torch.zeros_like(phi)
        uz = torch.zeros_like(phi)
        source = torch.ones_like(phi) * 0.01
        g_out = scalar_collide_bgk_3d(g, phi, ux, uy, uz, tau_d=TAU_D, source=source)
        phi_out = scalar_macroscopic_3d(g_out)
        # Total scalar should increase by source.sum()
        assert phi_out.sum() > phi.sum()


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

class TestScalarStreaming:
    def test_preserves_shape(self) -> None:
        g = _g_scalar()
        g_out = scalar_stream_3d(g)
        assert g_out.shape == g.shape

    def test_conserves_total_scalar(self) -> None:
        g = _g_scalar(nz=8, ny=8, nx=8)
        total_before = g.sum()
        g_out = scalar_stream_3d(g)
        total_after = g_out.sum()
        assert torch.allclose(total_after, total_before, atol=1e-5)

    def test_finite_output(self) -> None:
        g = _g_scalar()
        g_out = scalar_stream_3d(g)
        assert torch.isfinite(g_out).all()


# ---------------------------------------------------------------------------
# Macroscopic recovery
# ---------------------------------------------------------------------------

class TestScalarMacroscopic:
    def test_recovers_scalar(self) -> None:
        nz, ny, nx = 4, 6, 8
        phi_in = torch.rand((nz, ny, nx)) * 2.0 + 0.3
        ux = torch.zeros_like(phi_in)
        uy = torch.zeros_like(phi_in)
        uz = torch.zeros_like(phi_in)
        g = scalar_equilibrium_3d(phi_in, ux, uy, uz)
        phi_out = scalar_macroscopic_3d(g)
        assert torch.allclose(phi_out, phi_in, atol=1e-5)


# ---------------------------------------------------------------------------
# Combined passive_scalar_step
# ---------------------------------------------------------------------------

class TestPassiveScalarStep:
    @pytest.mark.parametrize("lattice", ["D3Q19", "D3Q27"])
    def test_shape(self, lattice: str) -> None:
        f = _f3d19() if lattice == "D3Q19" else _f3d27()
        g = _g_scalar()
        g_out, phi_out = passive_scalar_step(f, g, tau_d=TAU_D, lattice=lattice)
        assert g_out.shape == g.shape
        assert phi_out.shape == f.shape[1:]

    @pytest.mark.parametrize("lattice", ["D3Q19", "D3Q27"])
    def test_finite(self, lattice: str) -> None:
        f = _f3d19() if lattice == "D3Q19" else _f3d27()
        g = _g_scalar()
        g_out, phi_out = passive_scalar_step(f, g, tau_d=TAU_D, lattice=lattice)
        assert torch.isfinite(g_out).all()
        assert torch.isfinite(phi_out).all()

    @pytest.mark.parametrize("lattice", ["D3Q19", "D3Q27"])
    def test_scalar_conservation(self, lattice: str) -> None:
        """Scalar step should conserve total scalar (periodic, no source)."""
        f = _f3d19() if lattice == "D3Q19" else _f3d27()
        g = _g_scalar()
        phi_before = scalar_macroscopic_3d(g)
        _, phi_after = passive_scalar_step(f, g, tau_d=TAU_D, lattice=lattice)
        assert torch.allclose(phi_after.sum(), phi_before.sum(), atol=1e-4)

    @pytest.mark.parametrize("lattice", ["D3Q19", "D3Q27"])
    def test_source_increases_scalar(self, lattice: str) -> None:
        """Source term should increase total scalar."""
        f = _f3d19() if lattice == "D3Q19" else _f3d27()
        g = _g_scalar()
        phi_before = scalar_macroscopic_3d(g)
        source = torch.ones_like(phi_before) * 0.01
        _, phi_after = passive_scalar_step(f, g, tau_d=TAU_D, lattice=lattice,
                                            source=source)
        assert phi_after.sum() > phi_before.sum()

    def test_rejects_unknown_lattice(self) -> None:
        f = _f3d19()
        g = _g_scalar()
        with pytest.raises(ValueError, match="lattice"):
            passive_scalar_step(f, g, tau_d=TAU_D, lattice="D2Q9")


# ---------------------------------------------------------------------------
# Composability
# ---------------------------------------------------------------------------

class TestComposability:
    @pytest.mark.parametrize("lattice", ["D3Q19", "D3Q27"])
    def test_compose_with_collision(self, lattice: str) -> None:
        """passive_scalar_step should compose with collision + streaming."""
        from tensorlbm.solver3d import collide_bgk3d, stream3d
        from tensorlbm.d3q27 import collide_bgk27, stream27

        f = _f3d19() if lattice == "D3Q19" else _f3d27()
        g = _g_scalar()
        phi = scalar_macroscopic_3d(g)
        tau = 0.7
        for _ in range(3):
            if lattice == "D3Q19":
                f = collide_bgk3d(f, tau)
                f = stream3d(f)
            else:
                f = collide_bgk27(f, tau)
                f = stream27(f)
            g, phi = passive_scalar_step(f, g, tau_d=TAU_D, lattice=lattice)
        assert torch.isfinite(g).all()
        assert torch.isfinite(phi).all()

    @pytest.mark.parametrize("lattice", ["D3Q19", "D3Q27"])
    def test_compose_with_thermal(self, lattice: str) -> None:
        """passive_scalar_step should compose with thermal_step."""
        from tensorlbm.thermal_common import thermal_step

        f = _f3d19() if lattice == "D3Q19" else _f3d27()
        g_scalar = _g_scalar()
        g_thermal = _g_scalar()  # reuse for thermal
        phi = scalar_macroscopic_3d(g_scalar)
        T = scalar_macroscopic_3d(g_thermal)
        tau = 0.7
        for _ in range(3):
            f, g_thermal, T = thermal_step(f, g_thermal, tau_T=0.8, lattice=lattice)
            g_scalar, phi = passive_scalar_step(f, g_scalar, tau_d=TAU_D, lattice=lattice)
        assert torch.isfinite(g_scalar).all()
        assert torch.isfinite(phi).all()
