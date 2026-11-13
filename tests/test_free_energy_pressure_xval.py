"""Invariants of the sailfish free-energy binary model transplant.

These tests assert *identities and analytic anchors*, not "the code ran":

1. the anisotropic second-moment coefficients reproduce the moment sums that
   ``sym_equilibrium.py`` relies on (``lb_binary.py:249-268`` values);
2. the nine-point operators reproduce the analytic gradient/Laplacian of low
   order polynomials (the sailfish ``finite_difference_optimized.mako`` branch);
3. the equilibrium closes exactly:
   ``sum_i f_i^eq c_ia c_ib = P_ab + rho u_a u_b`` with ``P`` the free-energy
   pressure tensor -- machine precision, the algebra gate of the anchor;
4. the model conserves mass and order parameter (``f_0 = rho - sum``,
   ``g_0 = phi - sum``) and streaming is a pure shift on a periodic grid;
5. the flat-interface surface tension measured from the *populations* stays
   inside the 15% owner band against ``sigma = (2/3) sqrt(2 kappa A)``;
6. the Laplace jump of a static droplet is inside the 15% band against
   ``sigma / R`` -- the acceptance KPI of the anchor.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.phasefield.free_energy_pressure import (  # noqa: E402
    CX,
    CY,
    FreeEnergyBinaryParams,
    anisotropic_coefficients,
    bulk_pressure,
    chemical_potential,
    collide_and_stream,
    equilibrium_f,
    equilibrium_g,
    grad_lap,
    initial_condition,
    macroscopic,
    mean_pressure,
    periodic_distance,
    pressure_tensor_analytic,
    pressure_tensor_measured,
    stream,
    tanh_profile,
)


def _params(**kw) -> FreeEnergyBinaryParams:
    base = dict(A=0.1, kappa=0.2, Gamma=0.2, tau_f=1.0, tau_phi=1.5)
    base.update(kw)
    return FreeEnergyBinaryParams(**base)


def test_coefficients_match_sailfish_moment_sums():
    wi, wxx, wyy, wxy = anisotropic_coefficients()
    cx = torch.tensor(CX, dtype=torch.float64)[1:]
    cy = torch.tensor(CY, dtype=torch.float64)[1:]
    W = lambda v: torch.tensor(v, dtype=torch.float64)  # noqa: E731
    wxx, wyy, wxy = W(wxx), W(wyy), W(wxy)

    # values quoted from lb_binary.py:249-268 (axis 1/3 / diag 1/12)
    assert wi[0] == pytest.approx(1.0 / 3.0)
    assert wi[4] == pytest.approx(1.0 / 12.0)

    # moment sums the kappa term needs: sum wxx ex^2 = 1/2, sum wyy ex^2 = -1/2,
    # sum wxy ex ey = 1, and first moments vanish
    assert float((wxx * cx**2).sum()) == pytest.approx(0.5, abs=1e-14)
    assert float((wyy * cx**2).sum()) == pytest.approx(-0.5, abs=1e-14)
    assert float((wxy * cx * cy).sum()) == pytest.approx(1.0, abs=1e-14)
    assert float((wxx * cx).sum()) == pytest.approx(0.0, abs=1e-14)
    assert float((wyy * cy).sum()) == pytest.approx(0.0, abs=1e-14)
    # measured, not assumed: sum_i wi_i wxx_i = 7/72 != 0, i.e. the kappa term
    # contributes kappa |grad phi|^2 / 3 to sum_i f_i (absorbed by f_0 = rho - sum)
    wi_t = torch.tensor(wi, dtype=torch.float64)
    assert float((wi_t * wxx).sum()) == pytest.approx(7.0 / 72.0, abs=1e-14)
    assert float((wi_t * wyy).sum()) == pytest.approx(7.0 / 72.0, abs=1e-14)
    assert float((wi_t * wxy).sum()) == pytest.approx(0.0, abs=1e-14)


def test_nine_point_operators_on_polynomials():
    n = 16
    x = torch.arange(n, dtype=torch.float64).view(1, -1)
    y = torch.arange(n, dtype=torch.float64).view(-1, 1)
    f = torch.sin(2 * math.pi * x / n) * torch.cos(2 * math.pi * y / n)
    gx, gy, lap = grad_lap(f)
    exact_lap = -2 * (2 * math.pi / n) ** 2 * f
    # O(h^2) on the interior points only (periodic wrap is exact for this mode)
    assert float((lap - exact_lap).abs().max()) < 3e-2 * float(exact_lap.abs().max())  # O(h^2) band

    lin = 0.3 * x + 0.7 * y
    gx, gy, lap = grad_lap(lin)
    assert float((gx - 0.3).abs().max()) < 1e-12
    assert float((gy - 0.7).abs().max()) < 1e-12
    assert float(lap.abs().max()) < 1e-12


def test_equilibrium_closes_to_pressure_tensor_at_machine_precision():
    p = _params()
    n = 24
    torch.manual_seed(0)
    x = torch.arange(n, dtype=torch.float64).view(1, -1)
    y = torch.arange(n, dtype=torch.float64).view(-1, 1)
    phi = tanh_profile(periodic_distance(x.expand(n, n).contiguous(), n / 2, n), 0.0, p.xi)
    rho = 1.0 + 0.02 * torch.rand(n, n, dtype=torch.float64)
    ux = 0.02 * torch.rand(n, n, dtype=torch.float64)
    uy = -0.01 * torch.rand(n, n, dtype=torch.float64)

    gx, gy, lap = grad_lap(phi)
    feq = equilibrium_f(rho, ux, uy, phi, gx, gy, lap, p.A, p.kappa)
    Pm = pressure_tensor_measured(feq)
    Pa = pressure_tensor_analytic(rho, phi, gx, gy, lap, p.A, p.kappa)
    target = (rho * ux * ux + Pa[0], rho * uy * uy + Pa[1], rho * ux * uy + Pa[2])
    scale = max(float(t.abs().max()) for t in target)
    resid = max(float((Pm[i] - target[i]).abs().max()) for i in range(3))
    assert resid / scale < 1e-12, f"closure residual {resid/scale:.3e}"


def test_equilibrium_conserves_rho_and_phi():
    p = _params()
    n = 16
    phi = torch.tanh(torch.linspace(-1, 1, n, dtype=torch.float64)).expand(n, n).contiguous()
    rho = torch.full((n, n), 1.0, dtype=torch.float64)
    ux = 0.01 * torch.ones((n, n), dtype=torch.float64)
    uy = torch.zeros((n, n), dtype=torch.float64)
    gx, gy, lap = grad_lap(phi)
    feq = equilibrium_f(rho, ux, uy, phi, gx, gy, lap, p.A, p.kappa)
    mu = chemical_potential(phi, lap, p.A, p.kappa)
    geq = equilibrium_g(phi, ux, uy, mu, p.Gamma)
    assert float((feq.sum(0) - rho).abs().max()) < 1e-14
    assert float((geq.sum(0) - phi).abs().max()) < 1e-14
    r, u, v, ph = macroscopic(feq, geq)
    assert float((u - ux).abs().max()) < 1e-13
    assert float((ph - phi).abs().max()) < 1e-14


def test_streaming_is_a_shift():
    n = 8
    f = torch.zeros(9, n, n, dtype=torch.float64)
    f[1, 3, 4] = 1.0  # population moving in +x
    out = stream(f)
    assert out[1, 3, 5] == pytest.approx(1.0)
    assert float(out.sum()) == pytest.approx(1.0)
    f2 = torch.zeros(9, n, n, dtype=torch.float64)
    f2[5, 7, 7] = 1.0  # (1,1) corner population wraps on both axes
    out2 = stream(f2)
    assert out2[5, 0, 0] == pytest.approx(1.0)


def test_flat_band_surface_tension_inside_owner_band():
    p = _params(kappa=0.45)  # xi = 3
    nx, ny, band = 64, 4, 16
    x = torch.arange(nx, dtype=torch.float64).view(1, -1)
    signed = (band / 2.0) - (x - nx / 2.0).abs()
    phi0 = tanh_profile(signed, 0.0, p.xi).expand(ny, nx).contiguous()
    f, g = initial_condition(phi0, p)
    for _ in range(600):
        f, g = collide_and_stream(f, g, p)
    rho, ux, uy, phi = macroscopic(f, g)
    assert torch.isfinite(rho).all()
    Pxx, Pyy, _ = pressure_tensor_measured(f)
    sigma_band = float((Pxx - Pyy).mean(dim=0).sum()) / 2.0  # two interfaces
    assert abs(sigma_band - p.sigma) / p.sigma < 0.15, (
        f"sigma_band={sigma_band:.5f} analytic={p.sigma:.5f}")


def test_droplet_laplace_jump_inside_owner_band():
    p = _params()  # xi = 2
    n, R = 64, 10.0
    grid = torch.arange(n, dtype=torch.float64)
    c = n / 2.0
    dx = periodic_distance(grid.view(1, -1).expand(n, n).contiguous(), c, n)
    dy = periodic_distance(grid.view(-1, 1).expand(n, n).contiguous(), c, n)
    r = torch.sqrt(dx * dx + dy * dy)
    phi0 = -tanh_profile(r, R, p.xi)
    f, g = initial_condition(phi0, p)
    for _ in range(2500):
        f, g = collide_and_stream(f, g, p)
    rho, ux, uy, phi = macroscopic(f, g)
    assert torch.isfinite(rho).all()
    pr = mean_pressure(f)
    p_in = float(pr[r <= R - 3 * p.xi].mean())
    p_out = float(pr[r >= R + 3 * p.xi].mean())
    dp = p_in - p_out
    assert abs(dp - p.sigma / R) / (p.sigma / R) < 0.15, (
        f"dp={dp:.6e} analytic={p.sigma / R:.6e}")