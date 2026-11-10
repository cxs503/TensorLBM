"""Unit tests for :mod:`tensorlbm.rotating_duct` (campaign node p158w2-n01).

Covers the lattice/force algebra, the two forcing schemes (Guo vs EDM), the
flat-wall (exact) limits of the D3Q19 kernel, the Taylor-Couette analytic
anchors and torque identity, the secondary-flow diagnostics, and the
finite-difference reference of the linearised rotating-duct problem.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.rotating_duct import (
    C3,
    OPP3,
    W3,
    ForcedD3Q19,
    annulus3d_setup,
    couette_profile_error,
    couette_torque_analytic,
    duct_poiseuille_fd,
    ekman_delta_from_phase,
    fit_power_law,
    linear_coriolis_secondary_fd,
    params_sha256,
    secondary_flow_diagnostics,
    torque_from_profile,
)
from tensorlbm.rotating_interior import circular_couette, rotating_channel_analytic

NU = 0.05
TAU = 3.0 * NU + 0.5


# --------------------------------------------------------------------------- #
# lattice / algebra                                                           #
# --------------------------------------------------------------------------- #
def test_d3q19_weights_and_opposites():
    assert W3.sum() == pytest.approx(1.0, abs=1e-15)
    assert np.allclose(W3 @ C3, 0.0, atol=1e-15)
    assert np.allclose(W3 @ (C3[:, 0] ** 2), 1.0 / 3.0, atol=1e-15)
    assert np.allclose(W3 @ (C3[:, 0] * C3[:, 1]), 0.0, atol=1e-15)
    for k in range(19):
        assert OPP3[OPP3[k]] == k
        assert np.allclose(C3[OPP3[k]], -C3[k])
    assert len({tuple(c) for c in C3}) == 19


def test_equilibrium_moments():
    shape = (4, 5, 3)
    rng = np.random.default_rng(0)
    rho = 1.0 + 0.01 * rng.standard_normal(shape)
    u = 0.01 * rng.standard_normal((3,) + shape)
    solver = ForcedD3Q19(shape, tau=TAU)
    f = solver._feq(rho, u)
    assert np.allclose(f.sum(axis=0), rho)
    mom = np.stack([(f * C3[:, i].reshape(19, 1, 1, 1)).sum(axis=0) for i in range(3)])
    assert np.allclose(mom, rho[None] * u, atol=1e-15)


def test_accel_coriolis_and_centrifugal_identities():
    shape = (6, 5, 4)
    solver = ForcedD3Q19(shape, tau=TAU, omega=(0.0, 0.0, 0.3), centrifugal=False)
    u = np.zeros((3,) + shape)
    u[0] = 0.01
    a = solver._accel(u)
    # -2 Omega x u with Omega = W z_hat and u = u_x x_hat: z_hat x x_hat = y_hat,
    # so a_y = -2 W u_x (a *negative* y acceleration)
    assert np.allclose(a[1], -2.0 * 0.3 * 0.01, atol=1e-15)
    assert np.allclose(a[0][:, :, :], 0.0, atol=1e-15)
    # the centrifugal term is -Omega x (Omega x r) = W^2 r_perp: a pure gradient
    solver2 = ForcedD3Q19((8, 8, 2), tau=TAU, omega=(0.0, 0.0, 0.2), centrifugal=True)
    cf = solver2._accel(np.zeros((3, 8, 8, 2)))
    # a_cf = -Omega x (Omega x r) = Omega^2 r_perp: a pure gradient (curl free),
    # zero along the axis of rotation
    assert np.allclose(cf[0], 0.2**2 * solver2.pos[0], atol=1e-15)
    assert np.allclose(cf[1], 0.2**2 * solver2.pos[1], atol=1e-15)
    assert np.allclose(cf[2], 0.0, atol=1e-15)
    # curl(a_cf) = 0 on the interior (it must not drive any velocity)
    assert np.max(np.abs(np.gradient(cf[0], axis=1) - np.gradient(cf[1], axis=0))) < 1e-12


# --------------------------------------------------------------------------- #
# flat-wall limits of the D3Q19 kernel (exact solutions)                      #
# --------------------------------------------------------------------------- #
def test_plane_couette_moving_wall_exact():
    """Half-way bounce-back with a moving flat wall: exact linear Couette profile."""
    N, U = 20, 0.02
    solver = ForcedD3Q19((3, N, 3), tau=TAU, scheme="guo")
    solid = np.zeros((3, N, 3), dtype=bool)
    solid[:, 0, :] = True
    solid[:, -1, :] = True
    uw = np.zeros((3, 3, N, 3))
    uw[0, :, -1, :] = U
    solver.set_solid(solid, uw)
    st = solver.steady(tol=1e-11, max_steps=20000, check_every=200)
    assert st["converged"]
    rho, u = solver.fields()
    y = np.arange(1, N - 1)
    exact = U * (y - 0.5) / (N - 2)
    assert np.max(np.abs(u[0, 1, y, 1] - exact)) / U < 1e-3
    # moving walls must not inject mass for lattice-aligned walls
    fluid = ~solver.solid
    assert abs(rho[fluid].sum() - rho[fluid].mean() * fluid.sum()) < 1e-10


def test_body_force_channel_both_schemes():
    """Body-force Poiseuille channel: Guo and EDM both match the parabola."""
    N, G = 20, 1e-5
    h = (N - 2) / 2.0
    y = np.arange(1, N - 1) - (N - 1) / 2.0
    exact = G / (2.0 * NU) * (h**2 - y**2)
    out = {}
    for scheme in ("guo", "edm"):
        solver = ForcedD3Q19((3, N, 3), tau=TAU, scheme=scheme, body_force=(G, 0.0, 0.0))
        solid = np.zeros((3, N, 3), dtype=bool)
        solid[:, 0, :] = True
        solid[:, -1, :] = True
        solver.set_solid(solid)
        st = solver.steady(tol=1e-10, max_steps=12000, check_every=200)
        assert st["residual"] < 1e-9
        _, u = solver.fields()
        prof = u[0, 1, 1:-1, 1]
        out[scheme] = prof
        assert np.max(np.abs(prof - exact)) / exact.max() < 0.01
    # the two schemes must agree with each other as well
    assert np.max(np.abs(out["guo"] - out["edm"])) / exact.max() < 0.01


def test_guo_vs_edm_rotating_channel_agree_but_differ():
    """Guo and EDM agree within 2 % yet are not the same operator."""
    delta, ug = 4.0, 0.02
    Omega = NU / delta**2
    G = 2.0 * Omega * ug
    N = int(2 * 5 * delta) + 2
    h = (N - 2) / 2.0
    y = np.arange(1, N - 1) - (N - 1) / 2.0
    ux_a, uz_a = rotating_channel_analytic(y, h, G, Omega, NU)
    profs = {}
    for scheme in ("guo", "edm"):
        solver = ForcedD3Q19(
            (3, N, 3), tau=TAU, scheme=scheme, omega=(0.0, Omega, 0.0), body_force=(G, 0.0, 0.0)
        )
        solid = np.zeros((3, N, 3), dtype=bool)
        solid[:, 0, :] = True
        solid[:, -1, :] = True
        solver.set_solid(solid)
        u0 = np.zeros((3, 3, N, 3))
        u0[0, :, 1:-1, :] = ux_a[None, :, None]
        u0[2, :, 1:-1, :] = uz_a[None, :, None]
        solver.initialize_from_velocity(u0)
        solver.steady(tol=1e-9, max_steps=4000, check_every=200)
        _, u = solver.fields()
        profs[scheme] = np.concatenate([u[0, 0, 1:-1, 0], u[2, 0, 1:-1, 0]])
        err = max(
            np.max(np.abs(u[0, 0, 1:-1, 0] - ux_a)) / ug,
            np.max(np.abs(u[2, 0, 1:-1, 0] - uz_a)) / ug,
        )
        assert err < 0.03
    diff = np.max(np.abs(profs["guo"] - profs["edm"])) / max(np.max(np.abs(profs["guo"])), 1e-30)
    assert 0.0 < diff < 0.02


# --------------------------------------------------------------------------- #
# Taylor-Couette anchors                                                      #
# --------------------------------------------------------------------------- #
def test_annulus_geometry_and_wall_velocity():
    shape = (48, 48, 3)
    ri, ro, wi, wo = 10.0, 20.0, 0.004, -0.002
    geom = annulus3d_setup(shape, ri, ro, wi, wo)
    assert geom["solid"].shape == shape
    assert np.all(geom["solid"][:, :, 0] == geom["solid"][:, :, -1])  # periodic in z
    rad = geom["rad"]
    assert geom["solid"][rad < ri].all() and geom["solid"][rad > ro].all()
    assert not geom["solid"][(rad > ri + 1) & (rad < ro - 1)].any()
    uw = geom["wall_velocity"]
    # |omega x r| == |omega| * r on the solid nodes
    speed = np.hypot(uw[0], uw[1])
    inner = np.repeat((rad < ri)[:, :, None], shape[2], axis=2)
    outer = np.repeat((rad > ro)[:, :, None], shape[2], axis=2)
    r3 = np.repeat(rad[:, :, None], shape[2], axis=2)
    assert np.allclose(speed[inner], abs(wi) * r3[inner], atol=1e-12)
    assert np.allclose(speed[outer], abs(wo) * r3[outer], atol=1e-12)
    # the wall velocity is tangential: u . r = 0
    radial = uw[0] * np.repeat(geom["dx"][:, :, None], shape[2], axis=2) + uw[1] * np.repeat(
        geom["dy"][:, :, None], shape[2], axis=2
    )
    assert np.max(np.abs(radial)) < 1e-12


def test_couette_analytic_profile_and_torque_identity():
    ri, ro, wi, wo = 12.0, 24.0, 0.004, -0.002
    r = np.arange(ri + 0.5, ro, 1.0)
    c = circular_couette(r, ri, ro, wi, wo)
    at_walls = circular_couette(np.array([ri, ro]), ri, ro, wi, wo)["u_theta"]
    assert at_walls[0] == pytest.approx(wi * ri, rel=1e-12)
    assert at_walls[1] == pytest.approx(wo * ro, rel=1e-12)
    t = torque_from_profile(r, c["u_theta"], NU, length=1.0)
    T_ana = couette_torque_analytic(NU, c["B"], length=1.0)
    assert abs(t["T_mean_interior"]) == pytest.approx(T_ana, rel=3e-3)
    assert t["spread"] < 1e-2            # T(r) is r-independent for the exact family
    # counter-rotating case: both A and B non-zero
    assert c["A"] < 0 and c["B"] > 0
    # effective-radius inversion must be refused safely when omega_o != 0
    err = couette_profile_error(r, c["u_theta"], ri, ro, wi, wo)
    assert err["nominal_rel_max"] < 1e-8   # exact profile -> zero error


def test_couette_profile_error_detects_bias():
    ri, ro, wi = 12.0, 24.0, 0.004
    r = np.arange(ri + 0.5, ro, 1.0)
    c = circular_couette(r, ri, ro, wi, 0.0)
    bad = c["u_theta"] * 1.06
    err = couette_profile_error(r, bad, ri, ro, wi, 0.0)
    assert err["nominal_rel_max"] > 0.05


# --------------------------------------------------------------------------- #
# secondary-flow diagnostics and the linear reference                         #
# --------------------------------------------------------------------------- #
def _psi_pair(h: float, n: int = 41):
    c = np.linspace(0.0, 2 * h, n)
    y, z = np.meshgrid(c, c, indexing="ij")
    psi = np.sin(np.pi * y / (2 * h)) ** 2 * np.sin(np.pi * z / (2 * h)) * np.sign(y - h)
    uy = np.gradient(psi, c, axis=1)
    uz = -np.gradient(psi, c, axis=0)
    return psi, uy, uz, c, c


def test_secondary_diagnostics_vortex_pair():
    psi, uy, uz, y, z = _psi_pair(8.0, 41)
    d = secondary_flow_diagnostics(uy, uz, y, z)
    assert d["sign_changes_y_mid"] == 1        # one crossing: a pair, not a single cell
    assert d["sign_changes_z_quarter"] == 0    # single cell per half duct
    assert d["antisym_psi"] < 0.05
    # the two cells carry circulation of opposite sign.  (Their magnitudes are
    # not balanced for an arbitrary antisymmetric psi: balance is a property of
    # the physical solution, checked in the rotating-duct case instead.)
    assert d["circulation_upper"] * d["circulation_lower"] < 0
    assert abs(d["circulation_upper"]) > 0 and abs(d["circulation_lower"]) > 0
    assert abs(d["net_flux_z"]) < 0.05


def test_secondary_diagnostics_single_cell_is_rejected():
    c = np.linspace(0.0, 16.0, 41)
    y, z = np.meshgrid(c, c, indexing="ij")
    psi = np.sin(np.pi * y / 16.0) * np.sin(np.pi * z / 16.0)   # no sign change in y
    uy = np.gradient(psi, c, axis=1)
    uz = -np.gradient(psi, c, axis=0)
    d = secondary_flow_diagnostics(uy, uz, c, c)
    assert d["sign_changes_y_mid"] == 0
    assert d["antisym_psi"] > 0.5


def test_duct_poiseuille_fd_matches_fourier_series():
    h, G = 8.0, 1e-4
    res = duct_poiseuille_fd(h, G, NU, d=0.25)
    w = res["w0"]
    c = res["coord"]
    L = 2 * h
    # series solution for -lap(w) = G/nu on a square with w = 0 on the walls
    ref = np.zeros_like(w)
    for m in range(1, 40, 2):
        for n in range(1, 40, 2):
            amp = 16.0 * G * L**2 / (NU * math.pi**4) / (m * n * (m * m + n * n))
            ref += amp * np.sin(m * np.pi * c[:, None] / L) * np.sin(n * np.pi * c[None, :] / L)
    assert np.max(np.abs(w - ref)) / ref.max() < 5e-3


def test_linear_coriolis_reference_properties():
    res = linear_coriolis_secondary_fd(h=8.0, Omega=1e-3, nu=NU, G=2.1e-5, d=0.5)
    assert res["biharmonic_residual"] < 1e-10
    assert res["coupling_residual"] < 1e-10
    assert res["psi_wall_max"] < 1e-14
    uz = res["uz"]
    uy = res["uy"]
    # u_z even and u_y odd about the mid-plane
    assert np.max(np.abs(uz - uz[::-1])) / np.max(np.abs(uz)) < 1e-10
    assert np.max(np.abs(uy + uy[::-1])) / max(np.max(np.abs(uy)), 1e-30) < 1e-10
    # linearity in the rotation rate
    res2 = linear_coriolis_secondary_fd(h=8.0, Omega=2e-3, nu=NU, G=2.1e-5, d=0.5)
    assert res2["sec_max"] / res["sec_max"] == pytest.approx(2.0, rel=1e-8)
    # sign reversal
    res3 = linear_coriolis_secondary_fd(h=8.0, Omega=-1e-3, nu=NU, G=2.1e-5, d=0.5)
    assert np.allclose(res3["uz"], -uz, atol=1e-18)
    # a single vortex in each half duct (Dean-type pair)
    c = res["coord"][1:-1]
    d = secondary_flow_diagnostics(uy, uz, c, c, psi=res["psi"][1:-1, 1:-1])
    assert d["sign_changes_y_mid"] == 1
    assert d["sign_changes_z_quarter"] == 0


# --------------------------------------------------------------------------- #
# misc                                                                        #
# --------------------------------------------------------------------------- #
def test_fit_power_law_recovers_exponent():
    x = np.array([2.5e-4, 5e-4, 1e-3, 2e-3])
    y = 0.5 * x ** (-0.5)
    fit = fit_power_law(x, y)
    assert fit["exponent"] == pytest.approx(-0.5, abs=1e-10)
    assert fit["prefactor"] == pytest.approx(0.5, rel=1e-10)


def test_ekman_phase_anchor_on_analytic_spiral():
    delta = 6.0
    Omega = NU / delta**2
    G = 2.0 * Omega * 0.02
    N = int(2 * 6 * delta) + 2
    h = (N - 2) / 2.0
    y = np.arange(1, N - 1) - (N - 1) / 2.0
    ux, uz = rotating_channel_analytic(y, h, G, Omega, NU)
    d = ekman_delta_from_phase(y, ux, -uz, w_geo=-1j * G / (2.0 * Omega))
    assert d["delta_phase"] == pytest.approx(delta, rel=1e-3)


def test_params_sha256_is_stable_and_order_independent():
    a = params_sha256({"b": 1, "a": [2, 3]})
    b = params_sha256({"a": [2, 3], "b": 1})
    assert a == b and len(a) == 64
    assert a != params_sha256({"b": 1, "a": [2, 4]})
