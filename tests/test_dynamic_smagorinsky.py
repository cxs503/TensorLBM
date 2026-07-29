"""Tests for dynamic Smagorinsky turbulence closures."""
from __future__ import annotations

import torch

from tensorlbm import equilibrium, equilibrium3d
from tensorlbm.turbulence import (
    collide_dynamic_smagorinsky_bgk,
    collide_dynamic_smagorinsky_bgk3d,
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
