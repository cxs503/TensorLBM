"""Contract tests for the dynamic Smagorinsky MRT (D3Q19) and BGK (D3Q27) closures.

These verify operator algebra (shape, finiteness, mass/momentum conservation,
equilibrium fixed-point), NOT turbulence physics correctness.
"""
from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.d3q27 import equilibrium27, macroscopic27
from tensorlbm.turbulence import (
    collide_dynamic_smagorinsky_bgk27,
    collide_dynamic_smagorinsky_bgk3d,
    collide_dynamic_smagorinsky_mrt3d,
)


# ---------------------------------------------------------------------------
# D3Q19 dynamic Smagorinsky MRT
# ---------------------------------------------------------------------------

def _make_d3q19_state(seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    shape = (4, 6, 8)
    rho = 1.0 + 0.01 * torch.rand(shape)
    ux = 0.02 * (2.0 * torch.rand(shape) - 1.0)
    uy = 0.02 * (2.0 * torch.rand(shape) - 1.0)
    uz = 0.02 * (2.0 * torch.rand(shape) - 1.0)
    return equilibrium3d(rho, ux, uy, uz)


def test_dyn_smag_mrt3d_shape() -> None:
    f = _make_d3q19_state()
    assert collide_dynamic_smagorinsky_mrt3d(f, tau=0.7).shape == f.shape


def test_dyn_smag_mrt3d_output_finite() -> None:
    f = _make_d3q19_state()
    fout = collide_dynamic_smagorinsky_mrt3d(f, tau=0.7)
    assert torch.isfinite(fout).all()


def test_dyn_smag_mrt3d_preserves_mass() -> None:
    f = _make_d3q19_state()
    fout = collide_dynamic_smagorinsky_mrt3d(f, tau=0.7)
    rho_before = macroscopic3d(f)[0]
    rho_after = macroscopic3d(fout)[0]
    torch.testing.assert_close(rho_after, rho_before, rtol=1e-6, atol=1e-7)


def test_dyn_smag_mrt3d_preserves_momentum() -> None:
    f = _make_d3q19_state()
    fout = collide_dynamic_smagorinsky_mrt3d(f, tau=0.7)
    _, ux_b, uy_b, uz_b = macroscopic3d(f)
    _, ux_a, uy_a, uz_a = macroscopic3d(fout)
    torch.testing.assert_close(ux_a, ux_b, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(uy_a, uy_b, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(uz_a, uz_b, rtol=1e-6, atol=1e-7)


def test_dyn_smag_mrt3d_equilibrium_fixed_point() -> None:
    """Equilibrium distribution must be a collision fixed point."""
    f = _make_d3q19_state()
    fout = collide_dynamic_smagorinsky_mrt3d(f, tau=0.7)
    torch.testing.assert_close(fout, f, rtol=1e-6, atol=1e-7)


def test_dyn_smag_mrt3d_relaxes_non_equilibrium() -> None:
    """A non-equilibrium perturbation must move toward equilibrium."""
    f = _make_d3q19_state()
    f_neq = f + 1e-3 * torch.randn_like(f)
    fout = collide_dynamic_smagorinsky_mrt3d(f_neq, tau=0.7)
    # Post-collision should be closer to equilibrium than pre-collision
    feq = f
    err_before = (f_neq - feq).abs().mean()
    err_after = (fout - feq).abs().mean()
    assert err_after < err_before


def test_dyn_smag_mrt3d_accepts_mrt_rates() -> None:
    """MRT relaxation rates should be accepted as keyword arguments."""
    f = _make_d3q19_state()
    fout = collide_dynamic_smagorinsky_mrt3d(
        f, tau=0.7, s_e=1.19, s_eps=1.4, s_q=1.2
    )
    assert fout.shape == f.shape
    assert torch.isfinite(fout).all()


# ---------------------------------------------------------------------------
# D3Q27 dynamic Smagorinsky BGK
# ---------------------------------------------------------------------------

def _make_d3q27_state(seed: int = 99) -> torch.Tensor:
    torch.manual_seed(seed)
    shape = (4, 6, 8)
    rho = 1.0 + 0.01 * torch.rand(shape)
    ux = 0.02 * (2.0 * torch.rand(shape) - 1.0)
    uy = 0.02 * (2.0 * torch.rand(shape) - 1.0)
    uz = 0.02 * (2.0 * torch.rand(shape) - 1.0)
    return equilibrium27(rho, ux, uy, uz)


def test_dyn_smag_bgk27_shape() -> None:
    f = _make_d3q27_state()
    assert collide_dynamic_smagorinsky_bgk27(f, tau=0.7).shape == f.shape


def test_dyn_smag_bgk27_output_finite() -> None:
    f = _make_d3q27_state()
    fout = collide_dynamic_smagorinsky_bgk27(f, tau=0.7)
    assert torch.isfinite(fout).all()


def test_dyn_smag_bgk27_preserves_mass() -> None:
    f = _make_d3q27_state()
    fout = collide_dynamic_smagorinsky_bgk27(f, tau=0.7)
    rho_before = macroscopic27(f)[0]
    rho_after = macroscopic27(fout)[0]
    torch.testing.assert_close(rho_after, rho_before, rtol=1e-6, atol=1e-7)


def test_dyn_smag_bgk27_preserves_momentum() -> None:
    f = _make_d3q27_state()
    fout = collide_dynamic_smagorinsky_bgk27(f, tau=0.7)
    _, ux_b, uy_b, uz_b = macroscopic27(f)
    _, ux_a, uy_a, uz_a = macroscopic27(fout)
    torch.testing.assert_close(ux_a, ux_b, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(uy_a, uy_b, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(uz_a, uz_b, rtol=1e-6, atol=1e-7)


def test_dyn_smag_bgk27_equilibrium_fixed_point() -> None:
    """Equilibrium distribution must be a collision fixed point."""
    f = _make_d3q27_state()
    fout = collide_dynamic_smagorinsky_bgk27(f, tau=0.7)
    torch.testing.assert_close(fout, f, rtol=1e-6, atol=1e-7)


def test_dyn_smag_bgk27_relaxes_non_equilibrium() -> None:
    """A non-equilibrium perturbation must move toward equilibrium."""
    f = _make_d3q27_state()
    f_neq = f + 1e-3 * torch.randn_like(f)
    fout = collide_dynamic_smagorinsky_bgk27(f_neq, tau=0.7)
    feq = f
    err_before = (f_neq - feq).abs().mean()
    err_after = (fout - feq).abs().mean()
    assert err_after < err_before


def test_dyn_smag_bgk27_consistent_with_bgk3d_on_equilibrium() -> None:
    """Both dynamic BGK variants should leave equilibrium unchanged."""
    torch.manual_seed(7)
    shape = (3, 5, 7)
    rho = torch.ones(shape)
    ux = 0.01 * torch.rand(shape)
    uy = 0.01 * torch.rand(shape)
    uz = 0.01 * torch.rand(shape)
    f19 = equilibrium3d(rho, ux, uy, uz)
    f27 = equilibrium27(rho, ux, uy, uz)
    out19 = collide_dynamic_smagorinsky_bgk3d(f19, tau=0.7)
    out27 = collide_dynamic_smagorinsky_bgk27(f27, tau=0.7)
    torch.testing.assert_close(out19, f19, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(out27, f27, rtol=1e-6, atol=1e-7)
