"""Wall boundary conditions for the pressure-evolution + Allen-Cahn port (XVAL R2).

Round 1 (m18194) had periodic streaming only.  These tests pin the new boundary
treatment -- halfway bounce-back with the moving-wall momentum correction, the
neutral-wetting phase mirror, and the two-row wall band that the reference
model's mixed (2-cell) directional stencil needs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm.phasefield import pf_pressure_evolution as M

DT = torch.float64
WALL_T = 2


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _channel(ny: int, nx: int, thick: int = WALL_T) -> torch.Tensor:
    solid = torch.zeros((ny, nx), dtype=torch.bool)
    solid[:thick, :] = True
    solid[ny - thick:, :] = True
    return solid


def _y_eff(ny: int, thick: int = WALL_T) -> torch.Tensor:
    return torch.arange(thick, ny - thick, dtype=DT) - (thick - 0.5)


def _tanh_layer(ny: int, nx: int, y_int: float, width: float) -> torch.Tensor:
    y = torch.arange(ny, dtype=DT).view(-1, 1)
    return (0.5 * (1.0 - torch.tanh(2.0 * (y - y_int) / width))).expand(ny, nx).contiguous()


def _solve_diffuse_reference(phi_col, rho_h, rho_l, nu_h, nu_l, force):
    frac = np.clip(phi_col, 0.0, 1.0)
    rho = rho_l + (rho_h - rho_l) * frac
    w_l, w_h = 1.0 / (3.0 * nu_l), 1.0 / (3.0 * nu_h)
    mu = rho / (3.0 * (w_l + (w_h - w_l) * frac))
    n = frac.size
    a = np.zeros((n, n))
    b = np.full(n, -force)
    for j in range(n):
        lo = mu[j] if j == 0 else 0.5 * (mu[j] + mu[j - 1])
        hi = mu[j] if j == n - 1 else 0.5 * (mu[j] + mu[j + 1])
        d_lo = 0.5 if j == 0 else 1.0
        d_hi = 0.5 if j == n - 1 else 1.0
        a[j, j] = -(lo / d_lo + hi / d_hi)
        if j > 0:
            a[j, j - 1] = lo / d_lo
        if j < n - 1:
            a[j, j + 1] = hi / d_hi
    return np.linalg.solve(a, b)


# ---------------------------------------------------------------------------
# analytic references (pure functions, no solver)
# ---------------------------------------------------------------------------


def test_layered_profile_limits_and_continuity():
    H, a = 30.0, 15.0
    y = torch.linspace(0.0, H, 301, dtype=DT)
    # equal layers -> single-phase parabola
    same = M.layered_poiseuille_profile(y, H, a, 0.1, 0.1, 1e-4, 1e-4)
    ref = M.poiseuille_profile(y, H, 1e-4, 0.1)
    assert float((same - ref).abs().max()) < 1e-15
    # no-slip at both walls
    assert abs(float(same[0])) < 1e-15 and abs(float(same[-1])) < 1e-15
    # two layers: u continuous and the shear flux kappa du/dy continuous at a
    k1, k2 = 0.05, 0.2
    u = M.layered_poiseuille_profile(torch.linspace(0, H, 30001, dtype=DT), H, a, k1, k2,
                                     1e-4, 1e-4)
    uu = u.view(-1).numpy()
    yv = np.linspace(0, H, 30001)
    i = int(np.argmin(np.abs(yv - a)))
    assert abs(uu[i - 1] - uu[i + 1]) < 0.02 * uu.max()          # u continuous
    g1 = (uu[i - 1] - uu[i - 5]) / (yv[i - 1] - yv[i - 5])       # du/dy below
    g2 = (uu[i + 5] - uu[i + 1]) / (yv[i + 5] - yv[i + 1])       # du/dy above
    assert abs(k1 * g1 - k2 * g2) < 0.02 * max(abs(k1 * g1), abs(k2 * g2))
    # heavier viscosity layer carries the smaller curvature
    assert abs(uu[: int(0.2 * len(uu))].max()) < uu.max()


def test_couette_profile_is_linear():
    H = 30.0
    y = torch.linspace(0, H, 31, dtype=DT)
    u = M.couette_profile(y, H, 0.02)
    assert abs(float(u[0])) < 1e-15 and abs(float(u[-1]) - 0.02) < 1e-15
    assert float((u - 0.02 * y / H).abs().max()) < 1e-15


# ---------------------------------------------------------------------------
# boundary kernels
# ---------------------------------------------------------------------------


def test_wall_bounce_back_is_at_rest_without_moving_wall():
    """A uniform ``phi = 0`` channel at rest must stay exactly at rest."""
    ny, nx = 16, 4
    sol = M.PFPressureEvolution(shape=(ny, nx), rho_h=1.0, rho_l=1.0, phi_h=0.0,
                                phi_l=0.0, viscosity_h=0.1, viscosity_l=0.1)
    sol.set_walls(_channel(ny, nx))
    sol.initialize(torch.zeros((ny, nx), dtype=DT))
    sol.run(50)
    u = sol.velocity()
    sl = slice(WALL_T, ny - WALL_T)
    assert float(u[0][sl].abs().max()) < 1e-16
    assert float(u[1][sl].abs().max()) < 1e-16


def test_wall_phase_mirror_has_zero_normal_gradient():
    ny, nx = 32, 4
    sol = M.PFPressureEvolution(shape=(ny, nx), rho_h=1.5, rho_l=1.0, sigma=0.05,
                                width=4.0, mobility=0.05, viscosity_h=0.05,
                                viscosity_l=0.2)
    sol.set_walls(_channel(ny, nx))
    sol.initialize(_tanh_layer(ny, nx, 15.5, 4.0))
    sol.run(40)
    phi = sol.phi
    fluid = sol.fluid
    # every wall node holds the mirror of its adjacent fluid node
    for y, yb in ((WALL_T, WALL_T - 1), (ny - WALL_T - 1, ny - WALL_T)):
        d = (phi[yb] - phi[y]).abs().max()
        assert float(d) < 1e-9, (y, yb, float(d))
    # and the phase mass over the fluid is conserved to round-off
    assert sol.mass_error() < 1e-11
    assert fluid.sum() == ny * nx - 2 * WALL_T * nx


def test_moving_wall_couette_matches_analytic():
    ny, nx, nu, U = 20, 4, 0.1, 0.01
    sol = M.PFPressureEvolution(shape=(ny, nx), rho_h=1.0, rho_l=1.0, phi_h=0.0,
                                phi_l=0.0, viscosity_h=nu, viscosity_l=nu)
    uw = (torch.zeros((ny, nx), dtype=DT), torch.zeros((ny, nx), dtype=DT))
    uw[0][-WALL_T:, :] = U
    sol.set_walls(_channel(ny, nx), uw)
    sl = slice(WALL_T, ny - WALL_T)
    height = float(ny - 2 * WALL_T)
    y = _y_eff(ny)
    ui = torch.zeros((ny, nx), dtype=DT)
    ui[sl, :] = (U * y / height).view(-1, 1).expand(y.numel(), nx)
    sol.initialize(torch.zeros((ny, nx), dtype=DT), u0=(ui, torch.zeros_like(ui)))
    sol.run(2000)
    prof = sol.velocity()[0][sl, :].mean(1).numpy()
    ref = np.array(M.couette_profile(y, height, U).tolist())
    err = np.abs(prof - ref).max() / ref.max()
    assert err < 0.05, err
    # sign check: the flow follows the wall (+x), it is not reversed
    assert prof.mean() > 0.0


def test_no_slip_poiseuille_matches_analytic():
    ny, nx, nu, F = 24, 4, 0.1, 1e-5
    sol = M.PFPressureEvolution(shape=(ny, nx), rho_h=1.0, rho_l=1.0, phi_h=0.0,
                                phi_l=0.0, viscosity_h=nu, viscosity_l=nu)
    sol.set_walls(_channel(ny, nx))
    sl = slice(WALL_T, ny - WALL_T)
    height = float(ny - 2 * WALL_T)
    y = _y_eff(ny)
    body = (torch.full((ny, nx), F, dtype=DT), torch.zeros((ny, nx), dtype=DT))
    ui = torch.zeros((ny, nx), dtype=DT)
    ui[sl, :] = M.poiseuille_profile(y, height, F, nu).view(-1, 1).expand(y.numel(), nx)
    sol.initialize(torch.zeros((ny, nx), dtype=DT), u0=(ui, torch.zeros_like(ui)), body=body)
    sol.run(2500, body=body)
    prof = sol.velocity()[0][sl, :].mean(1).numpy()
    ref = np.array(M.poiseuille_profile(y, height, F, nu).tolist())
    assert abs(prof.max() - ref.max()) / ref.max() < 0.10, (prof.max(), ref.max())


def test_layered_poiseuille_against_diffuse_and_sharp_reference():
    ny, nx, W, F = 32, 4, 4.0, 1e-5
    rho_h, rho_l, nu_h, nu_l = 1.5, 1.0, 0.05, 0.2
    sol = M.PFPressureEvolution(shape=(ny, nx), rho_h=rho_h, rho_l=rho_l, sigma=0.05,
                                width=W, mobility=0.05, viscosity_h=nu_h,
                                viscosity_l=nu_l)
    sol.set_walls(_channel(ny, nx))
    sl = slice(WALL_T, ny - WALL_T)
    height = float(ny - 2 * WALL_T)
    y = _y_eff(ny)
    phi0 = _tanh_layer(ny, nx, height / 2.0 + WALL_T - 0.5, W)
    body = (torch.full((ny, nx), F, dtype=DT), torch.zeros((ny, nx), dtype=DT))
    sharp = np.array(M.layered_poiseuille_profile(
        y, height, height / 2.0, rho_h * nu_h, rho_l * nu_l, F, F).tolist())
    ui = torch.zeros((ny, nx), dtype=DT)
    ui[sl, :] = torch.as_tensor(sharp, dtype=DT).view(-1, 1).expand(y.numel(), nx)
    sol.initialize(phi0, u0=(ui, torch.zeros_like(ui)), body=body)
    sol.run(3000, body=body)
    prof = sol.velocity()[0][sl, :].mean(1).numpy()
    diffuse = _solve_diffuse_reference(sol.phi[sl, :].mean(1).numpy(),
                                       rho_h, rho_l, nu_h, nu_l, F)
    e_dif = np.abs(prof - diffuse).max() / sharp.max()
    e_sharp = np.abs(prof - sharp).max() / sharp.max()
    assert e_dif < 0.06, e_dif
    assert e_sharp < 0.06, e_sharp


def test_mixed_stencil_reads_through_a_one_row_wall():
    """The defect behind the wall band thickness (kernel level, cheap).

    ``directional_mixed`` is a *2-cell* stencil.  With a one-row wall the mirror
    fills only one row, so at the first fluid node the ``+2 c_i`` sample of the
    directions pointing into the wall is taken from the periodic wrap (the far,
    opposite phase) instead of the mirrored phase -- an O(1) phi error that pumps
    a spurious normal velocity into the channel (0.023 u_ref after 3000 steps,
    measured; see the R2 report).  Two wall rows remove it.
    """
    from tensorlbm.phasefield.cac_h_population import get_stencil

    st = get_stencil("d2q9")
    ny, nx = 32, 4
    y = torch.arange(ny, dtype=DT).view(-1, 1)
    phi = (0.5 * (1.0 - torch.tanh(2.0 * (y - 15.5) / 4.0))).expand(ny, nx).contiguous()

    def _mirror(thick: int) -> torch.Tensor:
        """Fill ``thick`` wall rows, outward from the fluid (neutral wetting)."""
        out = phi.clone()
        lo, hi = thick, ny - thick - 1          # first / last fluid rows
        for k in range(thick):
            out[lo - 1 - k] = out[lo - k]
            out[hi + 1 + k] = out[hi + k]
        return out

    one = _mirror(1)
    two = _mirror(2)
    d1 = M.directional_mixed(one, st)
    d2 = M.directional_mixed(two, st)
    # first fluid row of each discretisation: y = 1 (one-row mirror) vs y = 2
    # (two-row mirror).  In the one-row case the ``+2 c_i`` sample of the
    # wall-facing directions leaves the domain and lands on the periodic copy of
    # the opposite phase; in the two-row case it lands on the mirrored phase.
    gap_first = float((d1[:, 1] - d2[:, 2]).abs().max())
    assert gap_first > 1e-2, gap_first
    # one row further in the stencil is inside the band in both cases
    gap_second = float((d1[:, 2] - d2[:, 3]).abs().max())
    assert gap_second < 1e-2, gap_second
    assert gap_second < 0.1 * gap_first, (gap_first, gap_second)


def test_two_row_wall_band_keeps_the_normal_velocity_small():
    """End-to-end guard: with the two-row band no spurious normal flow develops."""
    ny, nx, F = 32, 4, 1e-5
    sol = M.PFPressureEvolution(shape=(ny, nx), rho_h=1.5, rho_l=1.0, sigma=0.05,
                                width=4.0, mobility=0.05, viscosity_h=0.05,
                                viscosity_l=0.2)
    sol.set_walls(_channel(ny, nx))
    sl = slice(WALL_T, ny - WALL_T)
    height = float(ny - 2 * WALL_T)
    y = _y_eff(ny)
    phi0 = _tanh_layer(ny, nx, height / 2.0 + WALL_T - 0.5, 4.0)
    ref = M.layered_poiseuille_profile(y, height, height / 2.0, 1.5 * 0.05, 1.0 * 0.2, F, F)
    ui = torch.zeros((ny, nx), dtype=DT)
    ui[sl, :] = ref.view(-1, 1).expand(y.numel(), nx)
    body = (torch.full((ny, nx), F, dtype=DT), torch.zeros((ny, nx), dtype=DT))
    sol.initialize(phi0, u0=(ui, torch.zeros_like(ui)), body=body)
    sol.run(1200, body=body)
    uy = float(sol.velocity()[1][sl].abs().max())
    assert uy < 3e-4, uy
