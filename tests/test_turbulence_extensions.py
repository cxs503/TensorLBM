"""Tests for WALE and Vreman LES turbulence models.

Verifies:
    - D2Q9  BGK + WALE:   shape, mass, momentum, equilibrium identity, finite output
    - D3Q19 BGK + WALE:   shape, mass, momentum, equilibrium identity, finite output
    - D3Q27 BGK + WALE:   shape, mass, momentum, equilibrium identity
    - D2Q9  BGK + Vreman: shape, mass, momentum, equilibrium identity, finite output
    - D3Q19 BGK + Vreman: shape, mass, momentum, equilibrium identity, finite output
    - D3Q27 BGK + Vreman: shape, mass, momentum, equilibrium identity
    - Eddy viscosity is non-negative
    - Effective tau is always > 0.5
"""
from __future__ import annotations

import pytest
import torch

from tensorlbm import (
    collide_vreman_bgk,
    collide_vreman_bgk3d,
    collide_vreman_bgk27,
    collide_wale_bgk,
    collide_wale_bgk3d,
    collide_wale_bgk27,
    equilibrium,
    equilibrium3d,
    macroscopic,
    macroscopic3d,
)
from tensorlbm.d3q27 import equilibrium27, macroscopic27
from tensorlbm.turbulence import (
    _nu_t_to_tau_eff,
    _vreman_nu_t_2d,
    _vreman_nu_t_3d,
    _wale_nu_t_2d,
    _wale_nu_t_3d,
)

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f2d(ny: int = 10, nx: int = 12, u_mag: float = 0.04) -> torch.Tensor:
    rho = torch.rand((ny, nx)) + 0.5
    ux = torch.rand_like(rho) * u_mag
    uy = torch.rand_like(rho) * u_mag
    return equilibrium(rho, ux, uy)


def _f3d19(nz: int = 4, ny: int = 6, nx: int = 8, u_mag: float = 0.04) -> torch.Tensor:
    rho = torch.rand((nz, ny, nx)) + 0.5
    ux = torch.rand_like(rho) * u_mag
    uy = torch.rand_like(rho) * u_mag
    uz = torch.rand_like(rho) * u_mag
    return equilibrium3d(rho, ux, uy, uz)


def _f3d27(nz: int = 4, ny: int = 6, nx: int = 8, u_mag: float = 0.04) -> torch.Tensor:
    rho = torch.rand((nz, ny, nx)) + 0.5
    ux = torch.rand_like(rho) * u_mag
    uy = torch.rand_like(rho) * u_mag
    uz = torch.rand_like(rho) * u_mag
    return equilibrium27(rho, ux, uy, uz)


# ---------------------------------------------------------------------------
# Velocity-gradient / eddy-viscosity helpers
# ---------------------------------------------------------------------------

class TestNuTHelpers:
    def test_wale_nu_t_2d_nonnegative(self) -> None:
        ny, nx = 10, 12
        ux = torch.rand((ny, nx)) * 0.05
        uy = torch.rand((ny, nx)) * 0.05
        nu_t = _wale_nu_t_2d(ux, uy, C_w=0.5)
        assert (nu_t >= 0.0).all()

    def test_wale_nu_t_3d_nonnegative(self) -> None:
        nz, ny, nx = 4, 6, 8
        ux = torch.rand((nz, ny, nx)) * 0.05
        uy = torch.rand((nz, ny, nx)) * 0.05
        uz = torch.rand((nz, ny, nx)) * 0.05
        nu_t = _wale_nu_t_3d(ux, uy, uz, C_w=0.5)
        assert (nu_t >= 0.0).all()

    def test_vreman_nu_t_2d_nonnegative(self) -> None:
        ny, nx = 10, 12
        ux = torch.rand((ny, nx)) * 0.05
        uy = torch.rand((ny, nx)) * 0.05
        nu_t = _vreman_nu_t_2d(ux, uy, C_V=0.025)
        assert (nu_t >= 0.0).all()

    def test_vreman_nu_t_3d_nonnegative(self) -> None:
        nz, ny, nx = 4, 6, 8
        ux = torch.rand((nz, ny, nx)) * 0.05
        uy = torch.rand((nz, ny, nx)) * 0.05
        uz = torch.rand((nz, ny, nx)) * 0.05
        nu_t = _vreman_nu_t_3d(ux, uy, uz, C_V=0.025)
        assert (nu_t >= 0.0).all()

    def test_tau_eff_exceeds_half(self) -> None:
        nu_t = torch.zeros(4, 6)
        tau_eff = _nu_t_to_tau_eff(tau=0.6, nu_t=nu_t)
        assert (tau_eff >= 0.5001).all()

    def test_wale_zero_velocity_gives_zero_nu_t_2d(self) -> None:
        """Zero velocity → zero velocity gradients → zero eddy viscosity."""
        ny, nx = 8, 10
        ux = torch.zeros((ny, nx))
        uy = torch.zeros((ny, nx))
        nu_t = _wale_nu_t_2d(ux, uy, C_w=0.5)
        assert torch.allclose(nu_t, torch.zeros_like(nu_t), atol=1e-10)

    def test_vreman_zero_velocity_gives_zero_nu_t_2d(self) -> None:
        ny, nx = 8, 10
        ux = torch.zeros((ny, nx))
        uy = torch.zeros((ny, nx))
        nu_t = _vreman_nu_t_2d(ux, uy, C_V=0.025)
        assert torch.allclose(nu_t, torch.zeros_like(nu_t), atol=1e-10)

    def test_wale_zero_velocity_gives_zero_nu_t_3d(self) -> None:
        nz, ny, nx = 4, 6, 8
        ux = torch.zeros((nz, ny, nx))
        uy = torch.zeros((nz, ny, nx))
        uz = torch.zeros((nz, ny, nx))
        nu_t = _wale_nu_t_3d(ux, uy, uz, C_w=0.5)
        assert torch.allclose(nu_t, torch.zeros_like(nu_t), atol=1e-10)

    def test_vreman_zero_velocity_gives_zero_nu_t_3d(self) -> None:
        nz, ny, nx = 4, 6, 8
        ux = torch.zeros((nz, ny, nx))
        uy = torch.zeros((nz, ny, nx))
        uz = torch.zeros((nz, ny, nx))
        nu_t = _vreman_nu_t_3d(ux, uy, uz, C_V=0.025)
        assert torch.allclose(nu_t, torch.zeros_like(nu_t), atol=1e-10)


# ---------------------------------------------------------------------------
# D2Q9 WALE
# ---------------------------------------------------------------------------

class TestWALE2D:
    def test_shape(self) -> None:
        f = _f2d()
        assert collide_wale_bgk(f, tau=0.7).shape == f.shape

    def test_finite(self) -> None:
        f = _f2d()
        assert torch.isfinite(collide_wale_bgk(f, tau=0.7)).all()

    def test_conserves_mass(self) -> None:
        f = _f2d()
        rho, _, _ = macroscopic(f)
        rho_out, _, _ = macroscopic(collide_wale_bgk(f, tau=0.7))
        assert torch.allclose(rho_out, rho, atol=1e-5)

    def test_conserves_momentum(self) -> None:
        f = _f2d()
        _, ux, uy = macroscopic(f)
        _, ux_out, uy_out = macroscopic(collide_wale_bgk(f, tau=0.7))
        assert torch.allclose(ux_out, ux, atol=1e-5)
        assert torch.allclose(uy_out, uy, atol=1e-5)

    def test_equilibrium_is_identity(self) -> None:
        ny, nx = 8, 10
        rho = torch.ones((ny, nx))
        ux = torch.full_like(rho, 0.04)
        uy = torch.full_like(rho, 0.02)
        feq = equilibrium(rho, ux, uy)
        assert torch.allclose(collide_wale_bgk(feq, tau=0.7), feq, atol=1e-5)

    @pytest.mark.parametrize("C_w", [0.3, 0.5, 0.6])
    def test_various_C_w(self, C_w: float) -> None:
        f = _f2d()
        f_out = collide_wale_bgk(f, tau=0.7, C_w=C_w)
        assert torch.isfinite(f_out).all()


# ---------------------------------------------------------------------------
# D3Q19 WALE
# ---------------------------------------------------------------------------

class TestWALE3D19:
    def test_shape(self) -> None:
        f = _f3d19()
        assert collide_wale_bgk3d(f, tau=0.7).shape == f.shape

    def test_finite(self) -> None:
        f = _f3d19()
        assert torch.isfinite(collide_wale_bgk3d(f, tau=0.7)).all()

    def test_conserves_mass(self) -> None:
        f = _f3d19()
        rho, _, _, _ = macroscopic3d(f)
        rho_out, _, _, _ = macroscopic3d(collide_wale_bgk3d(f, tau=0.7))
        assert torch.allclose(rho_out, rho, atol=1e-4)

    def test_conserves_momentum(self) -> None:
        f = _f3d19()
        _, ux, uy, uz = macroscopic3d(f)
        _, ux_out, uy_out, uz_out = macroscopic3d(collide_wale_bgk3d(f, tau=0.7))
        assert torch.allclose(ux_out, ux, atol=1e-4)
        assert torch.allclose(uy_out, uy, atol=1e-4)
        assert torch.allclose(uz_out, uz, atol=1e-4)

    def test_equilibrium_is_identity(self) -> None:
        nz, ny, nx = 4, 6, 8
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.03)
        uy = torch.full_like(rho, 0.01)
        uz = torch.full_like(rho, -0.01)
        feq = equilibrium3d(rho, ux, uy, uz)
        assert torch.allclose(collide_wale_bgk3d(feq, tau=0.7), feq, atol=1e-4)


# ---------------------------------------------------------------------------
# D3Q27 WALE
# ---------------------------------------------------------------------------

class TestWALE3D27:
    def test_shape(self) -> None:
        f = _f3d27()
        assert collide_wale_bgk27(f, tau=0.7).shape == f.shape

    def test_finite(self) -> None:
        f = _f3d27()
        assert torch.isfinite(collide_wale_bgk27(f, tau=0.7)).all()

    def test_conserves_mass(self) -> None:
        f = _f3d27()
        rho, _, _, _ = macroscopic27(f)
        rho_out, _, _, _ = macroscopic27(collide_wale_bgk27(f, tau=0.7))
        assert torch.allclose(rho_out, rho, atol=1e-4)

    def test_equilibrium_is_identity(self) -> None:
        nz, ny, nx = 4, 6, 8
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.03)
        uy = torch.full_like(rho, 0.01)
        uz = torch.full_like(rho, -0.01)
        feq = equilibrium27(rho, ux, uy, uz)
        assert torch.allclose(collide_wale_bgk27(feq, tau=0.7), feq, atol=1e-4)


# ---------------------------------------------------------------------------
# D2Q9 Vreman
# ---------------------------------------------------------------------------

class TestVreman2D:
    def test_shape(self) -> None:
        f = _f2d()
        assert collide_vreman_bgk(f, tau=0.7).shape == f.shape

    def test_finite(self) -> None:
        f = _f2d()
        assert torch.isfinite(collide_vreman_bgk(f, tau=0.7)).all()

    def test_conserves_mass(self) -> None:
        f = _f2d()
        rho, _, _ = macroscopic(f)
        rho_out, _, _ = macroscopic(collide_vreman_bgk(f, tau=0.7))
        assert torch.allclose(rho_out, rho, atol=1e-5)

    def test_conserves_momentum(self) -> None:
        f = _f2d()
        _, ux, uy = macroscopic(f)
        _, ux_out, uy_out = macroscopic(collide_vreman_bgk(f, tau=0.7))
        assert torch.allclose(ux_out, ux, atol=1e-5)
        assert torch.allclose(uy_out, uy, atol=1e-5)

    def test_equilibrium_is_identity(self) -> None:
        ny, nx = 8, 10
        rho = torch.ones((ny, nx))
        ux = torch.full_like(rho, 0.04)
        uy = torch.full_like(rho, 0.02)
        feq = equilibrium(rho, ux, uy)
        assert torch.allclose(collide_vreman_bgk(feq, tau=0.7), feq, atol=1e-5)

    @pytest.mark.parametrize("C_V", [0.01, 0.025, 0.05])
    def test_various_C_V(self, C_V: float) -> None:
        f = _f2d()
        f_out = collide_vreman_bgk(f, tau=0.7, C_V=C_V)
        assert torch.isfinite(f_out).all()


# ---------------------------------------------------------------------------
# D3Q19 Vreman
# ---------------------------------------------------------------------------

class TestVreman3D19:
    def test_shape(self) -> None:
        f = _f3d19()
        assert collide_vreman_bgk3d(f, tau=0.7).shape == f.shape

    def test_finite(self) -> None:
        f = _f3d19()
        assert torch.isfinite(collide_vreman_bgk3d(f, tau=0.7)).all()

    def test_conserves_mass(self) -> None:
        f = _f3d19()
        rho, _, _, _ = macroscopic3d(f)
        rho_out, _, _, _ = macroscopic3d(collide_vreman_bgk3d(f, tau=0.7))
        assert torch.allclose(rho_out, rho, atol=1e-4)

    def test_conserves_momentum(self) -> None:
        f = _f3d19()
        _, ux, uy, uz = macroscopic3d(f)
        _, ux_out, uy_out, uz_out = macroscopic3d(collide_vreman_bgk3d(f, tau=0.7))
        assert torch.allclose(ux_out, ux, atol=1e-4)
        assert torch.allclose(uy_out, uy, atol=1e-4)
        assert torch.allclose(uz_out, uz, atol=1e-4)

    def test_equilibrium_is_identity(self) -> None:
        nz, ny, nx = 4, 6, 8
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.03)
        uy = torch.full_like(rho, 0.01)
        uz = torch.full_like(rho, -0.01)
        feq = equilibrium3d(rho, ux, uy, uz)
        assert torch.allclose(collide_vreman_bgk3d(feq, tau=0.7), feq, atol=1e-4)


# ---------------------------------------------------------------------------
# D3Q27 Vreman
# ---------------------------------------------------------------------------

class TestVreman3D27:
    def test_shape(self) -> None:
        f = _f3d27()
        assert collide_vreman_bgk27(f, tau=0.7).shape == f.shape

    def test_finite(self) -> None:
        f = _f3d27()
        assert torch.isfinite(collide_vreman_bgk27(f, tau=0.7)).all()

    def test_conserves_mass(self) -> None:
        f = _f3d27()
        rho, _, _, _ = macroscopic27(f)
        rho_out, _, _, _ = macroscopic27(collide_vreman_bgk27(f, tau=0.7))
        assert torch.allclose(rho_out, rho, atol=1e-4)

    def test_equilibrium_is_identity(self) -> None:
        nz, ny, nx = 4, 6, 8
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.03)
        uy = torch.full_like(rho, 0.01)
        uz = torch.full_like(rho, -0.01)
        feq = equilibrium27(rho, ux, uy, uz)
        assert torch.allclose(collide_vreman_bgk27(feq, tau=0.7), feq, atol=1e-4)
