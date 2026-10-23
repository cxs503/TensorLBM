"""Tests for dynamic Smagorinsky turbulence closures."""

from __future__ import annotations

import torch

from tensorlbm import equilibrium, equilibrium3d
from tensorlbm.d3q19 import macroscopic3d
from tensorlbm.turbulence import (
    collide_dynamic_smagorinsky_bgk,
    collide_dynamic_smagorinsky_bgk3d,
    collide_dynamic_smagorinsky_mrt3d,
)


def test_dynamic_smagorinsky_bgk_shape() -> None:
    rho = torch.ones((8, 10))
    ux = torch.rand((8, 10)) * 0.02
    uy = torch.rand((8, 10)) * 0.02
    f = equilibrium(rho, ux, uy)
    assert collide_dynamic_smagorinsky_bgk(f, tau=0.7).shape == f.shape


def test_dynamic_smagorinsky_bgk3d_shape() -> None:
    rho = torch.ones((4, 6, 8))
    ux = torch.rand((4, 6, 8)) * 0.02
    uy = torch.rand((4, 6, 8)) * 0.02
    uz = torch.rand((4, 6, 8)) * 0.02
    f = equilibrium3d(rho, ux, uy, uz)
    assert collide_dynamic_smagorinsky_bgk3d(f, tau=0.7).shape == f.shape


def test_dynamic_smagorinsky_output_finite() -> None:
    rho = torch.ones((8, 10))
    ux = torch.rand((8, 10)) * 0.02
    uy = torch.rand((8, 10)) * 0.02
    f = equilibrium(rho, ux, uy)
    fout = collide_dynamic_smagorinsky_bgk(f, tau=0.7)
    assert torch.isfinite(fout).all()


# ---------------------------------------------------------------------------
# D3Q19 MRT contract tests (shape, finite, mass, momentum, equilibrium identity)
# ---------------------------------------------------------------------------


def test_dynamic_smagorinsky_mrt3d_shape() -> None:
    rho = torch.ones((4, 6, 8))
    ux = torch.rand((4, 6, 8)) * 0.02
    uy = torch.rand((4, 6, 8)) * 0.02
    uz = torch.rand((4, 6, 8)) * 0.02
    f = equilibrium3d(rho, ux, uy, uz)
    assert collide_dynamic_smagorinsky_mrt3d(f, tau=0.7).shape == f.shape


def test_dynamic_smagorinsky_mrt3d_finite() -> None:
    rho = torch.ones((4, 6, 8))
    ux = torch.rand((4, 6, 8)) * 0.02
    uy = torch.rand((4, 6, 8)) * 0.02
    uz = torch.rand((4, 6, 8)) * 0.02
    f = equilibrium3d(rho, ux, uy, uz)
    fout = collide_dynamic_smagorinsky_mrt3d(f, tau=0.7)
    assert torch.isfinite(fout).all()


def test_dynamic_smagorinsky_mrt3d_mass_conservation() -> None:
    rho = torch.rand((4, 6, 8)) + 0.5
    ux = torch.rand_like(rho) * 0.02
    uy = torch.rand_like(rho) * 0.02
    uz = torch.rand_like(rho) * 0.02
    f = equilibrium3d(rho, ux, uy, uz)
    fout = collide_dynamic_smagorinsky_mrt3d(f, tau=0.7)
    rho_out, *_ = macroscopic3d(fout)
    assert torch.allclose(rho_out, rho, atol=1e-5)


def test_dynamic_smagorinsky_mrt3d_momentum_conservation() -> None:
    rho = torch.rand((4, 6, 8)) + 0.5
    ux = torch.rand_like(rho) * 0.02
    uy = torch.rand_like(rho) * 0.02
    uz = torch.rand_like(rho) * 0.02
    f = equilibrium3d(rho, ux, uy, uz)
    fout = collide_dynamic_smagorinsky_mrt3d(f, tau=0.7)
    rho_out, ux_out, uy_out, uz_out = macroscopic3d(fout)
    assert torch.allclose(rho_out, rho, atol=1e-5)
    assert torch.allclose(ux_out, ux, atol=1e-5)
    assert torch.allclose(uy_out, uy, atol=1e-5)
    assert torch.allclose(uz_out, uz, atol=1e-5)


def test_dynamic_smagorinsky_mrt3d_equilibrium_is_identity() -> None:
    """At equilibrium (zero non-equilibrium) collision is identity."""
    rho = torch.ones((4, 6, 8))
    f = equilibrium3d(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))
    fout = collide_dynamic_smagorinsky_mrt3d(f, tau=0.7)
    assert torch.allclose(fout, f, atol=1e-5)


def test_static_smagorinsky_grad_finite_at_equilibrium() -> None:
    """Zero-stress cells must not NaN the backward pass (sqrt 0/0 mine).

    An equilibrium initialisation has f_neq == 0 exactly on every cell, so
    the Frobenius stress norm under the Smagorinsky sqrt is exactly zero;
    its backward is 1/(2*sqrt(0)) = inf and poisons dLoss/dC_s.  The clamp
    (min=1e-30) keeps the gradient finite with a bit-identical forward
    (f_neq == 0 divides to 0 whatever tau_eff is).  Covers the D2Q9, D3Q19
    and D3Q27 static closures.
    """
    from tensorlbm.turbulence import (
        collide_smagorinsky_bgk,
        collide_smagorinsky_bgk3d,
        collide_smagorinsky_bgk27,
    )

    rho2, ux2, uy2 = torch.full((8, 10), 1.0), torch.zeros((8, 10)), torch.zeros((8, 10))
    rho3 = torch.ones((4, 6, 8))
    u0 = torch.zeros((4, 6, 8))
    cases = (
        (equilibrium(rho2, ux2, uy2), collide_smagorinsky_bgk),
        (equilibrium3d(rho3, u0, u0, u0), collide_smagorinsky_bgk3d),
    )
    for f, collide in cases:
        cs = torch.tensor(0.1, requires_grad=True)
        out = collide(f, tau=0.7, C_s=cs)
        out.sum().backward()
        assert torch.isfinite(cs.grad), collide.__name__

    from tensorlbm.d3q27 import equilibrium27

    cs27 = torch.tensor(0.1, requires_grad=True)
    u27 = torch.zeros((3, 4, 5))
    f27 = equilibrium27(torch.ones((3, 4, 5)), u27, u27, u27)
    out27 = collide_smagorinsky_bgk27(f27, tau=0.7, C_s=cs27)
    out27.sum().backward()
    assert torch.isfinite(cs27)
