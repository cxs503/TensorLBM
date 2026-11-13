"""XVAL gates for the OSS (sailfish) pressure-tensor-equilibrium transplant.

Anchor: ``XVAL-LBM-PRESSURE-TENSOR-EQUILIBRIUM-TRANSPLANT`` (mission m18195).

These tests are the machine-checkable half of the cross-validation: they assert
the *algebraic* identities that the transplant must satisfy exactly, and the
*physical* gates (surface tension vs the closed-form ``(2/3) sqrt(2 kappa A)``)
that must hold to better than the owner 15 % precision standard.  The numbers
the report quotes come from ``artifacts/xval_pressure_tensor_equilibrium/``, not
from here.

Reference: ``sailfish-team/sailfish`` @ b7140c5,
``sym_equilibrium.py:16-49`` / ``:51-83``, ``lb_binary.py:224-259``,
``templates/finite_difference_optimized.mako:46-48``; the vendored copy and the
symbolic derivation live under
``xval/XVAL-LBM-PRESSURE-TENSOR-EQUILIBRIUM-TRANSPLANT/``.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.phasefield.pressure_tensor_equilibrium import (
    C,
    WXX_OSS,
    WYY_OSS,
    analytic_sigma,
    chemical_potential,
    grad_lap,
    interface_width,
    order_param_equilibrium,
    oss_d2q9_coefficients,
    pressure_from_populations,
    pressure_tensor,
    quiescent_droplet,
    quiescent_flat_interface,
    sigma_from_flat_dip,
    fluid_equilibrium,
)


def _c(name: str, shape: tuple[int, ...]) -> torch.Tensor:
    vec = torch.tensor([c[0 if name == "x" else 1] for c in C], dtype=torch.float64)
    return vec.view(-1, *([1] * len(shape)))


def _random_state(ny: int = 12, nx: int = 10, seed: int = 18195):
    g = torch.Generator().manual_seed(seed)
    phi = torch.tanh(0.3 * torch.randn(ny, nx, generator=g, dtype=torch.float64))
    ux = 0.05 * torch.randn(ny, nx, generator=g, dtype=torch.float64)
    uy = 0.05 * torch.randn(ny, nx, generator=g, dtype=torch.float64)
    rho = 1.0 + 0.1 * torch.rand(ny, nx, generator=g, dtype=torch.float64)
    return rho, ux, uy, phi


def test_coefficient_tables_match_oss_source():
    """``lb_binary.py:224-259``: wi = 1/3 (axes) / 1/12 (diagonals), and
    ``sum_{i>0} wi_i c_ia c_ib = delta_ab`` (the factor 3 over the BGK weights)."""
    wi, wxx, wyy, wxy = oss_d2q9_coefficients()
    assert torch.equal(wi, torch.tensor([1 / 3] * 4 + [1 / 12] * 4, dtype=torch.float64))
    axes = torch.tensor([[c[0], c[1]] for c in C[1:]], dtype=torch.float64)
    second = torch.einsum("i,ia,ib->ab", wi, axes, axes)
    assert torch.allclose(second, torch.eye(2, dtype=torch.float64), atol=1e-15)
    # the anisotropic tables: sum_i wi wxx cx^2 = 5/24, sum_i wi wxx cy^2 = -1/8
    assert abs(float(torch.sum(wi * wxx * axes[:, 0] ** 2)) - 5 / 24) < 1e-15
    assert abs(float(torch.sum(wi * wxx * axes[:, 1] ** 2)) + 1 / 8) < 1e-15
    assert abs(float(torch.sum(wi * wyy * axes[:, 1] ** 2)) - 5 / 24) < 1e-15
    assert abs(float(torch.sum(wi * wxy * axes[:, 0] * axes[:, 1])) - 1 / 12) < 1e-15


def test_mass_and_phase_are_conserved_by_the_equilibrium():
    rho, ux, uy, phi = _random_state()
    f = fluid_equilibrium(rho, ux, uy, phi, A=0.1, kappa=0.2)
    assert torch.allclose(f.sum(dim=0), rho, atol=1e-14)
    mu = chemical_potential(phi, A=0.1, kappa=0.2)
    g = order_param_equilibrium(phi, ux, uy, mu, Gamma=1.0)
    assert torch.allclose(g.sum(dim=0), phi, atol=1e-14)
    # first moment of the order-parameter equilibrium is phi u (OSS closure)
    assert torch.allclose((g * _c("x", phi.shape)).sum(dim=0), phi * ux, atol=1e-14)
    assert torch.allclose((g * _c("y", phi.shape)).sum(dim=0), phi * uy, atol=1e-14)


def test_second_moment_is_the_free_energy_pressure_tensor():
    """``sum_i f_i^eq c_ia c_ib == P_ab + rho u_a u_b`` -- the whole point of the
    transplant (residual is machine precision, not an approximation)."""
    rho, ux, uy, phi = _random_state()
    A, kappa = 0.1, 0.2
    f = fluid_equilibrium(rho, ux, uy, phi, A=A, kappa=kappa)
    pxx, pyy, pxy = pressure_from_populations(f)
    pxx_a, pyy_a, pxy_a = pressure_tensor(phi, A=A, kappa=kappa, rho=rho)
    assert float((pxx - pxx_a).abs().max()) < 1e-14
    assert float((pyy - pyy_a).abs().max()) < 1e-14
    assert float((pxy - pxy_a).abs().max()) < 1e-14
    # and the raw second moment (without the momentum-flux subtraction)
    raw_xx = (f * _c("x", phi.shape) ** 2).sum(dim=0)
    assert float((raw_xx - pxx_a - rho * ux * ux).abs().max()) < 1e-14


def _lap_error(n: int, periods: int = 4) -> float:
    """Relative max-norm error of the nine-point Laplacian on ``sin(2 pi periods x / n)``."""
    k = 2 * math.pi * periods / n
    x = torch.arange(n, dtype=torch.float64)
    ph = torch.sin(k * x).view(1, -1).expand(n, n).contiguous()
    gx, gy, lap = grad_lap(ph)
    exact = -(k**2) * ph
    err = float((lap - exact).abs().max()) / float(exact.abs().max())
    # y-invariant field -> no y gradient at machine precision
    assert float(gy.abs().max()) < 1e-14
    exact_gx = k * torch.cos(k * x).view(1, -1).expand(n, n)
    gerr = float((gx - exact_gx).abs().max()) / float(exact_gx.abs().max())
    assert gerr < 0.15, gerr  # O(h^2), constant ~2x the Laplacian one
    return err


def test_nine_point_operators_are_second_order_isotropic():
    """The OSS operators (``finite_difference_optimized.mako:46-48``) are the
    standard D2Q9 isotropic stencil: on a resolved Fourier mode the Laplacian
    error must fall as ``O(h^2)`` under grid refinement."""
    e32, e64, e128 = _lap_error(32), _lap_error(64), _lap_error(128)
    assert e32 > e64 > e128
    assert 3.0 < e32 / e64 < 5.0, (e32, e64)   # second order => ratio ~4
    assert 3.0 < e64 / e128 < 5.0, (e64, e128)
    assert e128 < 5e-3


@pytest.mark.parametrize("xi_target", [2.0, 3.0])
def test_flat_interface_surface_tension_within_15pct(xi_target):
    A = 0.1
    kappa = A * xi_target**2 / 2.0
    r = quiescent_flat_interface(A=A, kappa=kappa, ny=96, nx=8)
    assert r.closure_residual < 1e-12
    assert abs(r.rel_error) < 0.15, r
    assert abs(r.sigma_analytic - analytic_sigma(A, kappa)) < 1e-15


def test_surface_tension_error_decreases_with_interface_resolution():
    """Convergence: the sigma error is a discretisation effect, not a bias."""
    A = 0.1
    errors = []
    for xi_t in (1.0, 2.0, 4.0):
        kappa = A * xi_t**2 / 2.0
        errors.append(abs(quiescent_flat_interface(A=A, kappa=kappa, ny=96, nx=8).rel_error))
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 0.03


def test_normal_stress_is_flat_and_improves_with_resolution():
    A = 0.1
    res = [quiescent_flat_interface(A=A, kappa=A * x**2 / 2.0, ny=96, nx=8).dpnn_residual for x in (1.0, 3.0)]
    assert res[1] < res[0]
    # residual must stay a small fraction of the pressure scale A
    assert res[1] < 0.05 * A


def test_droplet_surface_tension_and_zero_bulk_pressure_jump():
    """Curved interface: sigma within 15 %; the *bulk* pressure jump is identically
    zero for the symmetric double well (``pb(+1) = pb(-1) = rho/3 + A/4``), which is
    why no Laplace percentage is claimed for this model variant."""
    A = 0.1
    kappa = A * 3.0**2 / 2.0
    r = quiescent_droplet(A=A, kappa=kappa, radius=16.0)
    assert abs(r.rel_error) < 0.15, r
    assert abs(r.dp_bulk) < 1e-3
    assert abs(r.dp_curvature_bound) > 50 * abs(r.dp_bulk)


def test_interface_width_and_surface_tension_analytic_anchors():
    A, kappa = 0.1, 0.2
    assert abs(interface_width(A, kappa) - math.sqrt(2 * kappa / A)) < 1e-15
    assert abs(analytic_sigma(A, kappa) - (2 / 3) * math.sqrt(2 * kappa * A)) < 1e-16


def test_flat_dip_window_does_not_double_count():
    """The periodic band carries two interfaces; the windowed measurement must
    return one of them (the double-counting trap that inflated sigma by 2x)."""
    A = 0.1
    kappa = A * 3.0**2 / 2.0
    ny = 96
    from tensorlbm.phasefield.pressure_tensor_equilibrium import flat_interface_profile

    phi = flat_interface_profile(ny, 8, A=A, kappa=kappa)
    rho = torch.ones_like(phi)
    zero = torch.zeros_like(phi)
    f = fluid_equilibrium(rho, zero, zero, phi, A=A, kappa=kappa)
    pxx, pyy, _ = pressure_from_populations(f)
    one = sigma_from_flat_dip(pxx, pyy, axis="y", center=ny // 4, half_width=12)
    both = sigma_from_flat_dip(pxx, pyy, axis="y", center=ny // 2, half_width=ny // 2)
    assert abs(both / one - 2.0) < 0.1, (one, both)


def test_determinism():
    A = 0.1
    kappa = 0.2
    a = quiescent_flat_interface(A=A, kappa=kappa, ny=64, nx=8)
    b = quiescent_flat_interface(A=A, kappa=kappa, ny=64, nx=8)
    assert a == b