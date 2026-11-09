"""Tests for the rotating-frame interior-flow module.

These are unit-level (no long integration): every analytic anchor is checked
against an independent route (finite differences of the pressure integral, a
direct FD solve of the rotating NS system, the library's force assembly), and
the angular-momentum identity gets a *negative* control so that a sign error
cannot pass silently.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm.rotating_frame import centrifugal_accel, coriolis_accel
from tensorlbm.rotating_interior import (
    RotatingAnnulusD2Q9,
    angular_momentum_residual,
    circular_couette,
    couette_pressure_rotating_frame,
    fd_rotating_channel,
    fit_channel_ekman_delta,
    fit_couette_AB,
    invert_effective_radii,
    rotating_channel_analytic,
    stream_function_field,
)


# --------------------------------------------------------------------------- #
# analytic anchors
# --------------------------------------------------------------------------- #
def test_couette_profile_and_effective_radius_inversion():
    ri, ro, wi = 12.0, 24.0, 0.004
    r = np.linspace(ri, ro, 25)
    c = circular_couette(r, ri, ro, wi)
    # no-slip at both walls
    assert abs(c["A"] * ri + c["B"] / ri - wi * ri) < 1e-14
    assert abs(c["A"] * ro + c["B"] / ro) < 1e-14
    # (A, B) recovered from the exact profile, effective radii are the nominal ones
    fit = fit_couette_AB(r, c["u_theta"])
    assert fit["A"] == pytest.approx(c["A"], rel=1e-10)
    assert fit["B"] == pytest.approx(c["B"], rel=1e-10)
    eff = invert_effective_radii(fit["A"], fit["B"], wi)
    assert eff["ri_eff"] == pytest.approx(ri, rel=1e-9)
    assert eff["ro_eff"] == pytest.approx(ro, rel=1e-9)


def test_couette_pressure_integral_and_frame_terms():
    """dp/dr must equal rho (u^2/r + 2 Omega u + Omega^2 r) and the Coriolis and
    centrifugal terms must each be present with the right size."""
    ri, ro, wi, wf = 12.0, 24.0, 0.004, 0.002
    r = np.linspace(ri + 1e-6, ro, 4000)
    for w in (0.0, wf):
        p = couette_pressure_rotating_frame(r, ri, ro, wi, omega_frame=w)
        dpdr = np.gradient(p, r)
        u = circular_couette(r, ri, ro, wi)["u_theta"]
        expect = u * u / r + 2.0 * w * u + w * w * r
        assert np.max(np.abs(dpdr - expect)) < 1e-3 * np.max(np.abs(expect))
    # frame terms are a measurable fraction of the total (the anchor discriminates)
    p_w = couette_pressure_rotating_frame(r, ri, ro, wi, omega_frame=wf)
    p_0 = couette_pressure_rotating_frame(r, ri, ro, wi, omega_frame=0.0)
    rel = np.max(np.abs(p_w - p_0)) / np.max(np.abs(p_w - p_w[0]))
    assert rel > 0.5


def test_channel_analytic_matches_fd_solve():
    G, Om, nu, h = 6.25e-5, 1.5625e-3, 0.1, 24.0
    fd = fd_rotating_channel(G, Om, nu, h, n=1201)
    ux, uz = rotating_channel_analytic(fd["y"], h, G, Om, nu)
    scale = np.max(np.abs(uz))
    assert np.max(np.abs(fd["ux"] - ux)) / scale < 1e-4
    assert np.max(np.abs(fd["uz"] - uz)) / scale < 1e-4
    # core value is the geostrophic balance 2 Omega u_z = G
    i = len(uz) // 2
    assert abs(2.0 * Om * fd["uz"][i] - G) / G < 0.12  # h/delta = 3 => O(e^{-3}) correction


def test_channel_analytic_satisfies_rotating_ns():
    """Residual of nu ux'' - 2 Omega u_z = -G and nu uz'' + 2 Omega u_x = 0."""
    G, Om, nu, h = 6.25e-5, 1.5625e-3, 0.1, 24.0
    y = np.linspace(-h + 1e-3, h - 1e-3, 6001)
    ux, uz = rotating_channel_analytic(y, h, G, Om, nu)
    d2x = np.gradient(np.gradient(ux, y), y)
    d2z = np.gradient(np.gradient(uz, y), y)
    sl = slice(2, -2)  # one-sided np.gradient is useless on the end nodes
    assert np.max(np.abs(nu * d2x - 2 * Om * uz + G)[sl]) / G < 2e-6
    assert np.max(np.abs(nu * d2z + 2 * Om * ux)[sl]) / G < 2e-6


def test_ekman_delta_fit_is_blind_and_accurate():
    G, Om, nu, h = 6.25e-5, 1.5625e-3, 0.1, 32.0
    y = np.linspace(-h, h, 129)
    _, uz = rotating_channel_analytic(y, h, G, Om, nu)
    fit = fit_channel_ekman_delta(y, uz, h, G, Om, nu)
    assert abs(fit["delta_fit"] - fit["delta_theory"]) / fit["delta_theory"] < 0.01
    # a genuinely 20 % larger viscosity in the *data* (delta -> sqrt(1.2) delta)
    # must be recovered: the fit is blind to the nu argument, only the profile counts
    _, uz2 = rotating_channel_analytic(y, h, G, Om, nu * 1.2)
    fit2 = fit_channel_ekman_delta(y, uz2, h, G, Om, nu)
    assert fit2["delta_fit"] / fit["delta_theory"] == pytest.approx(math.sqrt(1.2), rel=0.02)


# --------------------------------------------------------------------------- #
# solver plumbing
# --------------------------------------------------------------------------- #
def test_annulus_force_matches_library_accel():
    """The solver's numpy force assembly must equal the library's torch helpers."""
    sim = RotatingAnnulusD2Q9((40, 40), 0.65, 8.0, 16.0, 0.004, omega_frame=0.002)
    rng = np.random.default_rng(0)
    ux = rng.normal(scale=0.01, size=sim.shape)
    uy = rng.normal(scale=0.01, size=sim.shape)
    ax, ay = sim._force(ux, uy)
    cx, cy, _ = coriolis_accel(
        torch.as_tensor(ux), torch.as_tensor(uy), torch.zeros(sim.shape), (0.0, 0.0, 0.002)
    )
    fx, fy, _ = centrifugal_accel(
        torch.as_tensor(sim.dx), torch.as_tensor(sim.dy), None, (0.0, 0.0, 0.002)
    )
    ex = (cx + fx).numpy()
    ey = (cy + fy).numpy()
    assert np.max(np.abs(ax - ex)) < 1e-15
    assert np.max(np.abs(ay - ey)) < 1e-15


def test_annulus_moving_wall_drag_direction():
    """Inner cylinder rotating counter-clockwise must drag the fluid the same way.

    Catches a sign error in the Ladd momentum correction, which would reverse
    the drag (the profile test alone would then fail by 200 %, not by a few %).
    """
    sim = RotatingAnnulusD2Q9((40, 40), 0.65, 8.0, 16.0, 0.01, omega_frame=0.0)
    sim.step(400)
    prof = sim.azimuthal_profile()
    r = np.array(prof["r_bin"])
    u = np.array(prof["u_theta"])
    cnt = np.array(prof["count"])
    sel = (cnt > 0) & np.isfinite(u) & (r > 9) & (r < 15)
    assert np.all(u[sel] > 0.0)
    ur = np.array(prof["u_rad"])[sel]
    assert np.all(np.isfinite(ur))
    assert np.max(np.abs(ur)) < 0.2 * np.max(u[sel])


def test_annulus_mass_and_symmetry_after_a_few_steps():
    sim = RotatingAnnulusD2Q9((40, 40), 0.65, 8.0, 16.0, 0.004, omega_frame=0.002)
    m0 = sim.f[:, ~sim.solid].sum()
    sim.step(50)
    m1 = sim.f[:, ~sim.solid].sum()
    # stationary-wall bounce-back is mass neutral; the Ladd moving-wall term on a
    # *curved* wall leaks O(1e-7/step) (documented in the module README)
    assert abs(m1 - m0) / m0 < 1e-3


# --------------------------------------------------------------------------- #
# angular-momentum identity: positive and negative controls
# --------------------------------------------------------------------------- #
def test_angular_momentum_identity_divergence_free():
    ny = nx = 48
    ux, uy = stream_function_field(ny, nx, seed=15802, shear=0.01)
    yy, xx = np.meshgrid(np.arange(ny, dtype=float), np.arange(nx, dtype=float), indexing="ij")
    # rotation axis at the grid corner: an arbitrary but *non-degenerate* choice
    # (any centred moment of a periodic stream-function field vanishes, which
    #  would make the residual 0/0 instead of a test)
    x, y = xx, yy
    div = (np.roll(ux, -1, 1) - np.roll(ux, 1, 1)) / 2 + (np.roll(uy, -1, 0) - np.roll(uy, 1, 0)) / 2
    assert np.max(np.abs(div)) < 1e-12  # exactly divergence-free by construction
    for om in (0.002, 0.02):
        res = angular_momentum_residual(np.ones((ny, nx)), ux, uy, x, y, om)
        assert abs(res["torque_coriolis_plus_centrifugal"]) > 1e-2  # non-trivial
        assert res["residual_rel"] < 1e-12


def test_angular_momentum_identity_detects_a_sign_error():
    """Negative control: flipping Omega must break the balance at O(1)."""
    ny = nx = 32
    ux, uy = stream_function_field(ny, nx, seed=3, shear=0.02)
    yy, xx = np.meshgrid(np.arange(ny, dtype=float), np.arange(nx, dtype=float), indexing="ij")
    x, y = xx, yy  # axis at the corner (see the test above)
    om = 0.002
    res_ok = angular_momentum_residual(np.ones((ny, nx)), ux, uy, x, y, om)
    # emulate an implementation with the wrong sign: force from -Omega, budget
    # with +Omega (exactly what a sign bug in coriolis_accel would produce)
    Fx, Fy, _ = coriolis_accel(
        torch.as_tensor(ux), torch.as_tensor(uy), torch.zeros_like(torch.as_tensor(ux)),
        (0.0, 0.0, -om),
    )
    T_wrong = float(np.sum(x * Fy.numpy() - y * Fx.numpy()))
    dI = 2.0 * float(np.sum(x * ux + y * uy))
    resid_wrong = abs(T_wrong + om * dI) / max(abs(T_wrong), abs(om * dI))
    assert res_ok["residual_rel"] < 1e-12
    assert resid_wrong > 0.5


def test_stream_function_field_scales_with_shear():
    ux0, uy0 = stream_function_field(16, 16, seed=1, shear=0.0)
    ux1, uy1 = stream_function_field(16, 16, seed=1, shear=0.05)
    xx = np.arange(16)[None, :]
    assert np.allclose(ux1, ux0)  # shear enters u_y only (divergence-free)
    assert np.allclose(uy1 - uy0, 0.05 * (1.0 + np.cos(2 * np.pi * xx / 16)))
    assert np.sum(np.arange(16)[:, None] * (uy1 - uy0)) != 0.0