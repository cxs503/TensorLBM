"""Unit tests for the turbulence module (Smagorinsky LES collision operators)."""
from __future__ import annotations

import pytest
import torch

from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.turbulence import collide_smagorinsky_bgk, collide_smagorinsky_bgk3d


# ---------------------------------------------------------------------------
# D2Q9 – collide_smagorinsky_bgk
# ---------------------------------------------------------------------------

def test_smagorinsky_bgk_shape() -> None:
    rho = torch.ones(8, 16)
    ux = torch.full((8, 16), 0.05)
    uy = torch.zeros(8, 16)
    f = equilibrium(rho, ux, uy)
    f_out = collide_smagorinsky_bgk(f, tau_0=0.6, C_s=0.1)
    assert f_out.shape == f.shape


def test_smagorinsky_bgk_equilibrium_unchanged() -> None:
    """At equilibrium f_neq = 0, so the collision must leave f unchanged."""
    rho = torch.ones(8, 16)
    ux = torch.full((8, 16), 0.05)
    uy = torch.zeros(8, 16)
    f = equilibrium(rho, ux, uy)
    f_out = collide_smagorinsky_bgk(f, tau_0=0.65, C_s=0.1)
    assert torch.allclose(f_out, f, atol=1e-5)


def test_smagorinsky_bgk_conserves_mass() -> None:
    """Collision must preserve local density (collision invariant)."""
    torch.manual_seed(42)
    f = torch.rand(9, 8, 16) * 0.1 + 0.05
    rho_pre, ux_pre, uy_pre = macroscopic(f)
    f_out = collide_smagorinsky_bgk(f, tau_0=0.65, C_s=0.12)
    rho_post, _, _ = macroscopic(f_out)
    assert torch.allclose(rho_pre, rho_post, atol=1e-5)


def test_smagorinsky_bgk_conserves_momentum() -> None:
    """Collision must preserve local momentum (collision invariant)."""
    torch.manual_seed(7)
    f = torch.rand(9, 8, 16) * 0.1 + 0.05
    rho_pre, ux_pre, uy_pre = macroscopic(f)
    f_out = collide_smagorinsky_bgk(f, tau_0=0.65, C_s=0.12)
    rho_post, ux_post, uy_post = macroscopic(f_out)
    assert torch.allclose(ux_pre * rho_pre, ux_post * rho_post, atol=1e-5)
    assert torch.allclose(uy_pre * rho_pre, uy_post * rho_post, atol=1e-5)


def test_smagorinsky_bgk_finite() -> None:
    """Output must be finite for well-posed input."""
    torch.manual_seed(3)
    f = torch.rand(9, 12, 20) * 0.1 + 0.05
    f_out = collide_smagorinsky_bgk(f, tau_0=0.7, C_s=0.15)
    assert torch.isfinite(f_out).all()


@pytest.mark.parametrize("C_s", [0.0, 0.1, 0.18])
def test_smagorinsky_bgk_various_cs(C_s: float) -> None:
    """collide_smagorinsky_bgk should work for the full range of C_s."""
    rho = torch.ones(6, 10)
    ux = torch.full((6, 10), 0.04)
    uy = torch.zeros(6, 10)
    f = equilibrium(rho, ux, uy)
    f_out = collide_smagorinsky_bgk(f, tau_0=0.6, C_s=C_s)
    assert f_out.shape == f.shape
    assert torch.isfinite(f_out).all()


# ---------------------------------------------------------------------------
# D3Q19 – collide_smagorinsky_bgk3d
# ---------------------------------------------------------------------------

def test_smagorinsky_bgk3d_shape() -> None:
    rho = torch.ones(4, 8, 16)
    ux = torch.full((4, 8, 16), 0.05)
    uy = torch.zeros(4, 8, 16)
    uz = torch.zeros(4, 8, 16)
    f = equilibrium3d(rho, ux, uy, uz)
    f_out = collide_smagorinsky_bgk3d(f, tau_0=0.6, C_s=0.1)
    assert f_out.shape == f.shape


def test_smagorinsky_bgk3d_equilibrium_unchanged() -> None:
    """At equilibrium f_neq = 0, so the collision must leave f unchanged."""
    rho = torch.ones(4, 8, 16)
    ux = torch.full((4, 8, 16), 0.05)
    uy = torch.zeros(4, 8, 16)
    uz = torch.zeros(4, 8, 16)
    f = equilibrium3d(rho, ux, uy, uz)
    f_out = collide_smagorinsky_bgk3d(f, tau_0=0.6, C_s=0.1)
    assert torch.allclose(f_out, f, atol=1e-5)


def test_smagorinsky_bgk3d_conserves_mass() -> None:
    torch.manual_seed(42)
    f = torch.rand(19, 4, 8, 16) * 0.05 + 0.02
    rho_pre, _, _, _ = macroscopic3d(f)
    f_out = collide_smagorinsky_bgk3d(f, tau_0=0.65, C_s=0.12)
    rho_post, _, _, _ = macroscopic3d(f_out)
    assert torch.allclose(rho_pre, rho_post, atol=1e-5)


def test_smagorinsky_bgk3d_conserves_momentum() -> None:
    torch.manual_seed(7)
    f = torch.rand(19, 4, 8, 16) * 0.05 + 0.02
    rho_pre, ux_pre, uy_pre, uz_pre = macroscopic3d(f)
    f_out = collide_smagorinsky_bgk3d(f, tau_0=0.65, C_s=0.12)
    rho_post, ux_post, uy_post, uz_post = macroscopic3d(f_out)
    assert torch.allclose(ux_pre * rho_pre, ux_post * rho_post, atol=1e-5)
    assert torch.allclose(uy_pre * rho_pre, uy_post * rho_post, atol=1e-5)
    assert torch.allclose(uz_pre * rho_pre, uz_post * rho_post, atol=1e-5)


def test_smagorinsky_bgk3d_finite() -> None:
    torch.manual_seed(3)
    f = torch.rand(19, 4, 8, 12) * 0.05 + 0.02
    f_out = collide_smagorinsky_bgk3d(f, tau_0=0.7, C_s=0.15)
    assert torch.isfinite(f_out).all()


@pytest.mark.parametrize("C_s", [0.0, 0.1, 0.18])
def test_smagorinsky_bgk3d_various_cs(C_s: float) -> None:
    rho = torch.ones(4, 6, 8)
    ux = torch.full((4, 6, 8), 0.04)
    uy = torch.zeros(4, 6, 8)
    uz = torch.zeros(4, 6, 8)
    f = equilibrium3d(rho, ux, uy, uz)
    f_out = collide_smagorinsky_bgk3d(f, tau_0=0.6, C_s=C_s)
    assert f_out.shape == f.shape
    assert torch.isfinite(f_out).all()
