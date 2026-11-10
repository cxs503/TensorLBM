"""Tests for the D3Q19 yield-stress kernel with moving walls (P82W5-N04).

Covers
------
* exact Bingham Couette--Poiseuille solution (plug thickness closed form, ODE
  residual of the profile, admissibility of the pure-Couette degeneracy);
* AL (regularisation-free) duct reference vs the Newtonian Fourier series and
  vs the 2D channel in the wide-aspect limit;
* moving-wall half-way bounce-back + Ladd correction against the Newtonian
  Couette/Poiseuille solutions;
* the discrete momentum ledger identity ``dP = M_body + M_wall`` and the
  steady-state force balance;
* mass conservation and mask geometry.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.tensorlbm.yield_stress_analytical_moving_wall import (
    bingham_channel_flow_rate,
    bingham_channel_profile,
    bingham_couette_poiseuille_plug,
    bingham_couette_poiseuille_profile,
    bingham_plane_couette_regime,
    bingham_rectangular_duct_reference,
    bingham_square_duct_reference,
    newtonian_square_duct_series,
    rectangular_duct_shape_factor,
)
from src.tensorlbm.yield_stress_moving_wall_3d import (
    apply_moving_wall_bounce_back,
    bounce_back_arrival_mask,
    build_channel_walls,
    build_duct_walls,
    run_channel_3d,
    run_duct_3d,
)

torch.set_num_threads(1)
DTYPE = torch.float64


# ---------------------------------------------------------------------------
# analytical references
# ---------------------------------------------------------------------------
def test_couette_poiseuille_plug_thickness_closed_form():
    H, G, mu, ty = 32.0, 2e-6, 0.1, 4e-6
    plug = bingham_couette_poiseuille_plug(H, G, ty, mu, 1.28e-3)
    assert plug["regime"] == "plug"
    # closed form delta_p = 2*tau_y/G
    assert plug["delta_p"] == pytest.approx(2.0 * ty / G, rel=1e-12)
    assert 0.0 < plug["y1"] < plug["y2"] < H


def test_couette_poiseuille_profile_satisfies_bingham_ode():
    """In the yielded layers du/dy = tau/mu_0, in the plug du/dy = 0."""
    H, G, mu, ty, Uw = 32, 2e-6, 0.1, 4e-6, 1.28e-3
    plug = bingham_couette_poiseuille_plug(H, G, ty, mu, Uw)
    y = np.linspace(0.0, H, 4001)
    u = bingham_couette_poiseuille_profile(H, G, ty, mu, Uw, y=y)
    du = np.gradient(u, y)
    tau = plug["tau_0"] - G * y
    # wall conditions
    assert u[0] == pytest.approx(0.0, abs=1e-12)
    assert u[-1] == pytest.approx(Uw, rel=1e-10)
    # yielded layers: Bingham law u' = (tau -+ tau_y)/mu_0
    band = 2e-2 * H          # exclude the kink cells at the yield surfaces
    lo = y < plug["y1"] - band
    hi = y > plug["y2"] + band
    assert np.allclose(du[lo], (tau[lo] - ty) / mu, rtol=1e-3, atol=1e-11)
    assert np.allclose(du[hi], (tau[hi] + ty) / mu, rtol=1e-3, atol=1e-11)
    # plug: |tau| <= tau_y and no deformation
    eps = 1e-3 * H
    mid = (y > plug["y1"] + eps) & (y < plug["y2"] - eps)
    assert np.abs(tau[mid]).max() <= ty * (1 + 1e-12)
    assert np.abs(du[mid]).max() < 1e-7


def test_plane_couette_degeneracy():
    H, mu, ty = 16.0, 0.1, 4e-6
    # sub-yield wall stress -> no admissible steady flow (not a partial plug)
    sub = bingham_plane_couette_regime(H, ty, mu, 0.0001)
    assert sub["regime"] == "no_steady_solution"
    assert sub["tau_wall"] < ty
    # post-yield -> whole gap yields, linear profile, delta_p = 0
    post = bingham_plane_couette_regime(H, ty, mu, 0.02)
    assert post["regime"] == "fully_yielded"
    assert post["delta_p"] == 0.0


def test_newtonian_duct_series_shape_factor():
    G, mu = 1e-5, 0.1
    fs = []
    for N in (16, 32, 64):
        u = newtonian_square_duct_series(N, G, mu)
        fs.append(u.sum() / (G * N**4 / (12 * mu)))
    assert fs[0] > fs[1] > fs[2] > 0.4217
    assert fs[-1] == pytest.approx(0.4217, abs=2e-3)
    assert rectangular_duct_shape_factor(1.0) == pytest.approx(0.4217, abs=1e-4)


def test_al_reference_matches_newtonian_and_channel_limit():
    G, mu = 1e-5, 0.1
    ref = bingham_square_duct_reference(16, G, 0.0, mu, refine=4, n_iter=800)
    u_series = newtonian_square_duct_series(16, G, mu)
    assert np.abs(ref["u"] - u_series).max() / u_series.max() < 5e-3
    # wide aspect ratio -> per-width flow rate approaches the 2D channel value
    wide = bingham_rectangular_duct_reference(16, 64, G, 2e-5, mu, refine=2,
                                              n_iter=1500)
    q2d = bingham_channel_flow_rate(16, G, 2e-5, mu)
    assert wide["Q"] / 64 < q2d
    assert wide["Q"] / 64 > 0.75 * q2d


# ---------------------------------------------------------------------------
# geometry / mask
# ---------------------------------------------------------------------------
def test_arrival_mask_counts_planar_wall():
    wall = build_channel_walls(4, 2, 2)
    mask = bounce_back_arrival_mask(wall)
    # first fluid layer at y=1: 5 links arrive from the bottom wall per cell
    assert int(mask[:, :, 1, :].sum()) == 5 * 2 * 2
    # last fluid layer at y=H: 5 links arrive from the top wall
    assert int(mask[:, :, 4, :].sum()) == 5 * 2 * 2
    # no arrival links in the interior
    assert int(mask[:, :, 2, :].sum()) == 0


def test_duct_arrival_mask_corners():
    wall = build_duct_walls(4, 1)
    mask = bounce_back_arrival_mask(wall)
    # corner fluid cell (z=1,y=1) receives from both walls: links with
    # c_y=+1 or c_z=+1 pointing out of a solid neighbour
    corner = int(mask[:, 1, 1, 0].sum())
    edge = int(mask[:, 1, 2, 0].sum())
    assert corner == 5 + 5 - 1     # overlap: (0,1,1) counted once
    assert edge == 5


# ---------------------------------------------------------------------------
# moving wall: Newtonian analytic anchors
# ---------------------------------------------------------------------------
def test_newtonian_plane_couette_moving_wall_second_order():
    H, u_w, mu = 12, 0.02, 0.1
    lin = u_w * (np.arange(1, H + 1) - 0.5) / H
    init = torch.zeros((2, H + 2, 2), dtype=DTYPE)
    init[:, 1:-1, :] = torch.as_tensor(lin, dtype=DTYPE).view(1, H, 1)
    res = run_channel_3d(H, nx=2, nz=2, u_wall=(u_w, 0.0, 0.0), moving_top=True,
                         tau_y=0.0, mu_0=mu, m=0.0, max_iter=20000, tol=1e-11,
                         check_every=100, u_init=init)
    prof = res["ux_profile"].numpy()
    assert np.abs(prof - lin).max() / u_w < 5e-3
    led = res["ledger"]
    # pure Couette has no body force: the moving wall injects, the static wall
    # removes; both are O(1e-4) and must balance (the *sum* M_wall is O(0) there,
    # so the discrete identity dP = M_body + M_wall is checked in the duct test)
    assert 0.97 <= led["couette_balance"] <= 1.03
    assert abs(led["M_wall_window"]) < 1e-9


def test_newtonian_duct_poiseuille_and_force_balance():
    N, G, mu = 16, 1e-5, 0.1
    u_ref = newtonian_square_duct_series(N, G, mu)
    init = torch.zeros((N + 2, N + 2, 2), dtype=DTYPE)
    init[1:-1, 1:-1, :] = torch.as_tensor(u_ref, dtype=DTYPE).view(N, N, 1)
    res = run_duct_3d(N, nx=2, ax=G, tau_y=0.0, mu_0=mu, m=0.0,
                      max_iter=30000, tol=1e-10, check_every=100, u_init=init)
    u = res["ux_cross_iyz"].numpy()
    assert np.abs(u - u_ref).max() / u_ref.max() < 8e-3
    led = res["ledger"]
    assert abs(led["closure_window"] - 1.0) < 1e-6      # exact discrete identity
    assert abs(led["force_balance"] - 1.0) < 1e-2


def test_mass_conservation_bounce_back():
    wall = build_channel_walls(6, 2, 2)
    mask = bounce_back_arrival_mask(wall)
    fluid = ~wall
    rho = torch.ones(wall.shape, dtype=DTYPE)
    z = torch.zeros_like(rho)
    from src.tensorlbm.d3q19 import equilibrium3d
    from src.tensorlbm.solver3d import stream3d
    from src.tensorlbm.yield_stress_moving_wall_3d import _collide_newtonian_guo

    f = equilibrium3d(rho, z, z, z)
    m0 = float((f * fluid).sum())
    for _ in range(50):
        f_post = _collide_newtonian_guo(f, 0.1, 1e-5)
        f_stream = stream3d(f_post)
        f, _ = apply_moving_wall_bounce_back(
            f_stream, f_post, mask,
            u_wall=(0.01, 0.0, 0.0),
            moving_mask=(mask & torch.roll(_wall_of(mask, wall), 0)))
        f = torch.where(wall.unsqueeze(0), equilibrium3d(rho, z, z, z), f)
    m1 = float((f * fluid).sum())
    assert abs(m1 - m0) / m0 < 1e-12


def _wall_of(mask, wall):
    """Helper: moving links of the top wall (arrival mask rows whose source is
    the top wall layer), used only inside the mass-conservation test."""
    moving = torch.zeros_like(mask)
    for i in range(19):
        moving[i] = mask[i] & wall[-1, :, :].unsqueeze(0)
    return moving


# ---------------------------------------------------------------------------
# Bingham plug: constant plug velocity, monotone thickness
# ---------------------------------------------------------------------------
def test_bingham_channel_plug_is_flat_and_ledger_closes():
    H, G, mu, ty = 12, 1e-5, 0.1, 1.5e-5
    y_p = ty / G
    u_ex = bingham_channel_profile(H, G, ty, mu)
    init = torch.zeros((2, H + 2, 2), dtype=DTYPE)
    init[:, 1:-1, :] = torch.as_tensor(u_ex, dtype=DTYPE).view(1, H, 1)
    res = run_channel_3d(H, nx=2, nz=2, ax=G, tau_y=ty, mu_0=mu, m=1e5,
                         max_iter=30000, tol=1e-10, check_every=100,
                         nu_max=mu * (1 + 1e5 * ty / mu) * 10,
                         u_init=init, use_neq=False)
    prof = res["ux_profile"].numpy()
    y = np.arange(1, H + 1) - 0.5
    plug = np.abs(y - H / 2) < y_p - 0.5
    assert plug.sum() >= 2
    u_plug = prof[plug]
    assert u_plug.std() / abs(u_plug.mean()) < 5e-2      # nearly rigid plug
    assert abs(ledger_closed(res)) < 1e-6


def ledger_closed(res):
    return res["ledger"]["closure_window"] - 1.0