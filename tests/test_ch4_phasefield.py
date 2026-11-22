"""Unit tests for the CH4 single-distribution Cahn-Hilliard phase-field LBE."""

from __future__ import annotations

import pytest
import torch

from tensorlbm.ch4_phasefield import (
    _c_on,
    ch4_coupled_step,
    ch4_init,
    ch4_step,
    chemical_potential,
    central_grad,
    laplacian,
)


def test_central_grad_constant():
    """The central gradient of a constant field is zero."""
    x = torch.full((8, 8, 8), 3.7)
    gx, gy, gz = central_grad(x)
    assert gx.abs().max() == 0
    assert gy.abs().max() == 0
    assert gz.abs().max() == 0


def test_central_grad_linear():
    """The central gradient of a linear field is the slope (interior)."""
    n = 16
    z, y, x = torch.meshgrid(
        torch.arange(n), torch.arange(n), torch.arange(n), indexing="ij"
    )
    f = x.to(torch.float64)
    gx, gy, gz = central_grad(f)
    assert abs(float(gx[4:-4, 4:-4, 4:-4].mean()) - 1.0) < 1e-12
    assert gy[4:-4, 4:-4, 4:-4].abs().max() < 1e-12
    assert gz[4:-4, 4:-4, 4:-4].abs().max() < 1e-12


def test_laplacian_quadratic():
    """The Laplacian of x^2+y^2+z^2 is 6 (interior)."""
    n = 16
    z, y, x = torch.meshgrid(
        torch.arange(n), torch.arange(n), torch.arange(n), indexing="ij"
    )
    f = (x**2 + y**2 + z**2).to(torch.float64)
    lap = laplacian(f)
    assert abs(float(lap[4:-4, 4:-4, 4:-4].mean()) - 6.0) < 1e-12


def test_chemical_potential_bulk_zero():
    """The chemical potential is zero in the bulk (phi = +/- 1)."""
    phi = torch.ones(4, 4, 4)
    mu = chemical_potential(phi, 0.1, 0.02)
    assert mu.abs().max() < 1e-12
    phi = -torch.ones(4, 4, 4)
    mu = chemical_potential(phi, 0.1, 0.02)
    assert mu.abs().max() < 1e-12


def test_ch_step_conservative():
    """The Cahn-Hilliard step conserves the liquid volume."""
    torch.manual_seed(0)
    n = 24
    R = 8.0
    w = 1.5
    phi = torch.zeros(n, n, n)
    z, y, x = torch.meshgrid(
        torch.arange(n), torch.arange(n), torch.arange(n), indexing="ij"
    )
    r = ((z - n / 2) ** 2 + (y - n / 2) ** 2 + (x - n / 2) ** 2).sqrt()
    phi = torch.tanh((r - R) / w)
    A, kappa = 1e-5, 1e-5 * w * w
    M, tau_g = 0.1, 0.8
    _, g = ch4_init(phi, A, kappa, M, 1.0, 100.0)
    vol0 = float(((phi + 1) / 2).sum())
    for _ in range(50):
        g, phi = ch4_step(g, phi, A, kappa, M, tau_g)
    vol = float(((phi + 1) / 2).sum())
    assert abs(vol / vol0 - 1) < 1e-6


def test_static_droplet_spurious_flow_small():
    """A static droplet develops only a small spurious current."""
    torch.manual_seed(0)
    n = 24
    R = 8.0
    w = 1.5
    phi = torch.zeros(n, n, n)
    z, y, x = torch.meshgrid(
        torch.arange(n), torch.arange(n), torch.arange(n), indexing="ij"
    )
    r = ((z - n / 2) ** 2 + (y - n / 2) ** 2 + (x - n / 2) ** 2).sqrt()
    phi = torch.tanh((r - R) / w)
    A, kappa = 1e-5, 1e-5 * w * w
    M, tau_g, tau_f = 0.1, 0.8, 0.5
    f, g = ch4_init(phi, A, kappa, M, 1.0, 100.0)
    c = _c_on(phi.device)
    for _ in range(100):
        f, g, phi = ch4_coupled_step(
            f, g, phi, A, kappa, M, tau_g, tau_f, 0.0, 0.01, 1.0, 100.0
        )
    rho_f = f.sum(0)
    mom = (f * c[:, 0].view(19, 1, 1, 1)).sum(0)
    u = mom / rho_f.clamp(min=1e-12)
    umax = float(u.abs().max())
    assert umax < 1e-2


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
