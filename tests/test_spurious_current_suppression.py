"""Unit tests for :mod:`tensorlbm.verification.spurious_current`.

Node ``p362w2-n06``.  The tests pin down, in order:

* lattice primitives (weights, moments, streaming, equilibrium),
* the two collision operators (BGK/MRT conservation, and MRT == BGK in the
  uniform-rate limit),
* the two force discretisations (uniform-field null, periodic net force zero,
  agreement of the lattice sum with ``c_s^2 grad`` for a smooth field, 4th-order
  accuracy of the geometric stencil),
* the measurements (EOS pressure, Laplace fit, interface thickness, Pareto,
  convergence order),
* the physical acceptance controls that are cheap enough for CI: mass and
  momentum conservation over a short run and ``sigma = 0 => U_spurious = 0``.

The full ``U_spurious`` / Laplace / grid-convergence campaign lives in
``benchmarks/spurious_current_suppression/run.py`` (minutes, not seconds); these
tests are the fast regression guard on the machinery that campaign relies on.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm.verification import spurious_current as sc  # noqa: E402


# ---------------------------------------------------------------------------
# Lattice primitives
# ---------------------------------------------------------------------------


def test_weights_sum_to_one():
    assert sum(sc.WEIGHTS) == pytest.approx(1.0, abs=1e-15)


def test_lattice_isotropy_moments():
    w = np.asarray(sc.WEIGHTS)
    ex = np.asarray(sc.EX, dtype=float)
    ey = np.asarray(sc.EY, dtype=float)
    assert float((w * ex).sum()) == pytest.approx(0.0, abs=1e-15)
    assert float((w * ey).sum()) == pytest.approx(0.0, abs=1e-15)
    assert float((w * ex * ey).sum()) == pytest.approx(0.0, abs=1e-15)
    assert float((w * ex**2).sum()) == pytest.approx(sc.CS2, rel=1e-14)
    assert float((w * ey**2).sum()) == pytest.approx(sc.CS2, rel=1e-14)
    assert float((w * (ex**2 + ey**2) ** 2).sum()) == pytest.approx(8.0 / 9.0, rel=1e-14)


def test_equilibrium_recovers_density_and_momentum():
    rng = np.random.default_rng(0)
    rho = 1.0 + 0.1 * rng.random((8, 9))
    ux = 0.02 * (rng.random((8, 9)) - 0.5)
    uy = 0.02 * (rng.random((8, 9)) - 0.5)
    f = sc.equilibrium(rho, ux, uy)
    r, ur, vr = sc.macroscopic(f)
    assert np.allclose(r, rho, rtol=0, atol=1e-14)
    assert np.allclose(ur, ux, rtol=0, atol=1e-13)
    assert np.allclose(vr, uy, rtol=0, atol=1e-13)


def test_equilibrium_at_rest_is_exactly_rest():
    f = sc.equilibrium(np.full((6, 6), 0.7), np.zeros((6, 6)), np.zeros((6, 6)))
    r, ux, uy = sc.macroscopic(f)
    assert np.allclose(r, 0.7, atol=1e-15)
    assert np.abs(ux).max() == 0.0
    assert np.abs(uy).max() == 0.0


def test_streaming_moves_populations_along_their_direction():
    f = np.zeros((9, 7, 7))
    f[1, 3, 3] = 1.0  # direction +x
    f[3, 3, 3] = 2.0  # direction +y
    out = sc.stream(f)
    assert out[1, 3, 4] == 1.0
    assert out[3, 4, 3] == 2.0
    assert out.sum() == pytest.approx(f.sum())


def test_streaming_conserves_mass_and_momentum():
    rng = np.random.default_rng(1)
    f = rng.random((9, 10, 10))
    out = sc.stream(f)
    assert out.sum() == pytest.approx(f.sum(), rel=1e-15)
    for ex, ey in ((1, 0), (0, 1), (-1, 0), (0, -1)):
        m_before = (f * (np.asarray(sc.EX) * ex + np.asarray(sc.EY) * ey)[:, None, None]).sum()
        m_after = (out * (np.asarray(sc.EX) * ex + np.asarray(sc.EY) * ey)[:, None, None]).sum()
        assert m_after == pytest.approx(m_before, rel=1e-14, abs=1e-12)


# ---------------------------------------------------------------------------
# Collision operators
# ---------------------------------------------------------------------------


def test_bgk_preserves_mass_and_momentum_without_force():
    rng = np.random.default_rng(2)
    f = sc.equilibrium(1.0 + 0.05 * rng.random((6, 6)), np.zeros((6, 6)), np.zeros((6, 6)))
    f = f + 0.01 * (rng.random(f.shape) - 0.5)
    zero = np.zeros((6, 6))
    out = sc.collide(f, zero, zero, tau=0.8, collision="bgk")
    assert out.sum(axis=0) == pytest.approx(f.sum(axis=0), rel=1e-14)
    assert (out * sc.EX9).sum(axis=0) == pytest.approx((f * sc.EX9).sum(axis=0), abs=1e-13)
    assert (out * sc.EY9).sum(axis=0) == pytest.approx((f * sc.EY9).sum(axis=0), abs=1e-13)


def test_mrt_moment_matrices_are_inverse():
    assert np.allclose(sc.MOMENT_MATRIX @ sc.MOMENT_MATRIX_INV, np.eye(9), atol=1e-12)


def test_mrt_reduces_to_bgk_in_the_uniform_rate_limit():
    rng = np.random.default_rng(3)
    f = sc.equilibrium(1.0 + 0.05 * rng.random((6, 6)), np.zeros((6, 6)), np.zeros((6, 6)))
    f = f + 0.01 * (rng.random(f.shape) - 0.5)
    zero = np.zeros((6, 6))
    bgk = sc.collide(f, zero, zero, tau=1.0, collision="bgk")
    mrt = sc.collide(
        f, zero, zero, tau=1.0, collision="mrt", magic=0.25, s_eps=1.0, s_q=1.0
    )
    assert np.abs(bgk - mrt).max() < 1e-14


def test_mrt_conserves_mass_and_momentum_without_force():
    rng = np.random.default_rng(4)
    f = sc.equilibrium(1.0 + 0.05 * rng.random((6, 7)), np.zeros((6, 7)), np.zeros((6, 7)))
    f = f + 0.01 * (rng.random(f.shape) - 0.5)
    zero = np.zeros((6, 7))
    out = sc.collide(f, zero, zero, tau=1.0, collision="mrt")
    assert out.sum(axis=0) == pytest.approx(f.sum(axis=0), rel=1e-13)
    assert (out * sc.EX9).sum(axis=0) == pytest.approx((f * sc.EX9).sum(axis=0), abs=1e-12)
    assert (out * sc.EY9).sum(axis=0) == pytest.approx((f * sc.EY9).sum(axis=0), abs=1e-12)


def test_magic_parameter_mapping():
    for s_nu in (0.9, 1.0, 1.5):
        s_e = sc.magic_s_e(s_nu, 3.0 / 16.0)
        lam = (1.0 / s_nu - 0.5) * (1.0 / s_e - 0.5)
        assert lam == pytest.approx(3.0 / 16.0, rel=1e-12)
    with pytest.raises(ValueError):
        sc.magic_s_e(2.0)


def test_edm_forcing_reduces_to_bgk_without_force():
    """With F = 0 the exact-difference increment vanishes and EDM == BGK."""
    rng = np.random.default_rng(5)
    f = sc.equilibrium(1.0 + 0.05 * rng.random((6, 6)), np.zeros((6, 6)), np.zeros((6, 6)))
    f = f + 0.01 * (rng.random(f.shape) - 0.5)
    zero = np.zeros((6, 6))
    bgk = sc.collide(f, zero, zero, tau=0.9, collision="bgk")
    edm = sc.collide(f, zero, zero, tau=0.9, collision="bgk", forcing="edm")
    assert np.abs(bgk - edm).max() < 1e-15


def test_edm_forcing_is_rejected_for_mrt():
    f = sc.equilibrium(np.ones((4, 4)), np.zeros((4, 4)), np.zeros((4, 4)))
    zero = np.zeros((4, 4))
    with pytest.raises(ValueError):
        sc.collide(f, zero, zero, collision="mrt", forcing="edm")


def test_edm_forcing_runs_a_stable_droplet():
    run = sc.run_static_droplet(
        G12=2.5, n=32, radius=8.0, width=2.0, steps=120, forcing="edm",
        keep_fields=False, check_every=60,
    )
    assert math.isfinite(run["u_max"])
    assert run["rho1_centre"] > run["rho1_far"]
    assert run["converged"] in (True, False)


def test_unknown_schemes_are_rejected():
    f = sc.equilibrium(np.ones((4, 4)), np.zeros((4, 4)), np.zeros((4, 4)))
    zero = np.zeros((4, 4))
    with pytest.raises(ValueError):
        sc.collide(f, zero, zero, collision="nope")
    with pytest.raises(ValueError):
        sc.interaction_force(np.ones((4, 4)), np.ones((4, 4)), 2.5, scheme="nope")


# ---------------------------------------------------------------------------
# Force discretisations
# ---------------------------------------------------------------------------


def test_both_force_schemes_vanish_on_a_uniform_field():
    rho = np.full((8, 8), 0.6)
    for scheme in ("lattice", "geometric"):
        fx, fy = sc.interaction_force(rho, rho, 2.5, scheme=scheme)
        assert np.abs(fx).max() == 0.0
        assert np.abs(fy).max() == 0.0


def test_internal_force_has_zero_net_over_the_periodic_domain():
    r = sc.grid_radius(40)
    psi1 = 0.5 * (1.0 - np.tanh((r - 9.0) / 1.5))
    psi2 = 1.0 - psi1
    for scheme in ("lattice", "geometric"):
        fx1, fy1 = sc.interaction_force(psi1, psi2, 2.5, scheme=scheme)
        fx2, fy2 = sc.interaction_force(psi2, psi1, 2.5, scheme=scheme)
        assert abs(fx1.sum() + fx2.sum()) < 1e-10
        assert abs(fy1.sum() + fy2.sum()) < 1e-10


def test_lattice_gradient_is_cs2_times_the_continuum_gradient():
    # Smooth periodic field: lattice sum == c_s^2 grad + O(h^2) anisotropic error.
    n = 64
    idx = np.arange(n) * 2.0 * math.pi / n
    yy, xx = np.meshgrid(idx, idx, indexing="ij")
    field = np.sin(xx) * np.cos(yy)
    gx, gy = sc.lattice_gradient(field)
    # Gradients here are per lattice index (unit spacing), not per radian.
    k = 2.0 * math.pi / n
    exact_x = k * np.cos(xx) * np.cos(yy)
    exact_y = -k * np.sin(xx) * np.sin(yy)
    err_x = np.abs(gx - sc.CS2 * exact_x).max() / np.abs(exact_x).max()
    err_y = np.abs(gy - sc.CS2 * exact_y).max() / np.abs(exact_y).max()
    assert err_x < 2e-2
    assert err_y < 2e-2


def test_geometric_gradient_is_fourth_order_accurate():
    # d/dx sin(kx) with the 2-scale 4th-order stencil: error scales like h^4.
    def max_err(n):
        idx = np.arange(n) * 2.0 * math.pi / n
        yy, xx = np.meshgrid(idx, idx, indexing="ij")
        k = 2.0 * math.pi / n
        gx, _ = sc.gradient_4th(np.sin(xx))
        return np.abs(gx - k * np.cos(xx)).max()

    e1, e2 = max_err(32), max_err(64)
    assert e2 < e1 / 10.0          # h^4 => 16x per doubling, allow head-room
    assert e1 < 1e-5


def test_geometric_force_matches_lattice_sum_for_a_thick_interface():
    r = sc.grid_radius(64)
    psi = 0.5 * (1.0 - np.tanh((r - 20.0) / 4.0))
    gx_lat, _ = sc.lattice_gradient(psi)
    gx_geo, _ = sc.gradient_4th(psi)
    scale = np.abs(gx_lat).max()
    assert np.abs(gx_lat - sc.CS2 * gx_geo).max() / scale < 0.05


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


def test_eos_pressure_lowers_pressure_where_components_coexist():
    pure = sc.eos_pressure(np.full((3, 3), 0.9), np.zeros((3, 3)), 2.5)
    mixed = sc.eos_pressure(np.full((3, 3), 0.5), np.full((3, 3), 0.5), 2.5)
    assert mixed.mean() < pure.mean()


def test_laplace_fit_recovers_a_known_surface_tension():
    sigma, radii = 0.05, [20.0, 30.0, 40.0]
    dp = [sigma / r for r in radii]
    fit = sc.laplace_fit(radii, dp)
    assert fit["sigma"] == pytest.approx(sigma, rel=1e-12)
    assert abs(fit["intercept"]) < 1e-15
    assert fit["max_rel_error"] < 1e-12
    fit0 = sc.laplace_fit(radii, dp, through_origin=True)
    assert fit0["sigma"] == pytest.approx(sigma, rel=1e-12)


def test_laplace_fit_reports_the_residual_of_a_bad_point():
    fit = sc.laplace_fit([20.0, 30.0, 40.0], [0.0025, 0.001667, 0.00125 * 1.2])
    assert fit["max_rel_error"] > 0.05


def test_interface_width_recovers_the_tanh_width():
    n, r0, w = 160, 40.0, 2.0
    r = sc.grid_radius(n)
    run = {"rho1": 0.1 + 0.8 * 0.5 * (1.0 - np.tanh((r - r0) / w))}
    got = sc.interface_width_10_90(run)
    assert got["width_10_90"] == pytest.approx(2.0 * math.atanh(0.8) * w, rel=0.06)
    assert got["w_eq"] == pytest.approx(w, rel=0.06)
    assert got["radius_fit"] == pytest.approx(r0, abs=1.5)


def test_pareto_front_known_case():
    pts = [(1.0, 3.0), (2.0, 2.0), (3.0, 1.0), (2.5, 3.0)]
    assert sc.pareto_front(pts) == [0, 1, 2]
    assert sc.pareto_front([(1.0, 1.0), (1.0, 1.0)]) == [0, 1]


def test_convergence_order_first_order_sequence():
    vals = [1.0, 1.5, 1.75]  # u_inf - C h
    out = sc.convergence_order(vals, [1.0, 0.5, 0.25])
    assert out["monotone"] is True
    assert out["order"] == pytest.approx(1.0, rel=1e-12)


def test_convergence_order_flags_non_monotone_data():
    out = sc.convergence_order([1.0, 0.9, 0.5], [1.0, 0.5, 0.25])
    assert out["monotone"] is False
    assert math.isnan(out["order"])


def test_methods_registry_covers_all_scheme_combinations():
    assert len(sc.METHODS) >= 4
    combos = {(m["collision"], m["force"]) for m in sc.METHODS.values()}
    assert ("bgk", "lattice") in combos
    assert ("mrt", "lattice") in combos
    assert ("bgk", "geometric") in combos
    assert any(cfg.get("forcing") == "edm" for cfg in sc.METHODS.values())
    for cfg in sc.METHODS.values():
        assert set(cfg) >= {"collision", "force", "label"}


# ---------------------------------------------------------------------------
# Physical controls (cheap enough for CI)
# ---------------------------------------------------------------------------


def test_sigma_zero_gives_exactly_zero_spurious_current():
    """No interface (uniform mixture) and G12 = 0 => no capillary force => u = 0."""
    run = sc.run_static_droplet(
        G12=0.0,
        n=32,
        radius=8.0,
        width=2.0,
        steps=200,
        uniform=True,
        keep_fields=False,
        check_every=50,
    )
    assert run["u_max"] < 1e-10


def test_mass_is_conserved_by_the_coupled_step():
    run = sc.run_static_droplet(
        G12=2.5, n=36, radius=9.0, width=2.0, steps=150, keep_fields=False,
        check_every=75,
    )
    n = 36
    rho1_init = 0.14 + 0.78 * 0.5 * (1.0 - np.tanh((sc.grid_radius(n) - 9.0) / 2.0))
    rho2_init = 0.86 - 0.78 * 0.5 * (1.0 - np.tanh((sc.grid_radius(n) - 9.0) / 2.0))
    assert run["mass1"] == pytest.approx(rho1_init.sum(), rel=1e-9)
    assert run["mass2"] == pytest.approx(rho2_init.sum(), rel=1e-9)
    assert abs(run["rho_tot_drift"]) < 1e-9 * n * n


def test_net_momentum_stays_zero_for_the_isolated_droplet():
    run = sc.run_static_droplet(
        G12=2.5, n=36, radius=9.0, width=2.0, steps=150, keep_fields=True,
        check_every=75,
    )
    rho = run["rho1"] + run["rho2"]
    px = float((rho * run["ux"]).sum())
    py = float((rho * run["uy"]).sum())
    assert abs(px) < 1e-8
    assert abs(py) < 1e-8


def test_relaxed_droplet_has_positive_laplace_pressure_jump():
    run = sc.run_static_droplet(
        G12=2.5, n=64, radius=16.0, width=2.0, steps=300, keep_fields=True,
        check_every=100,
    )
    p = sc.droplet_pressure(run)
    assert p["dp"] > 0.0
    assert p["p_in"] > p["p_out"]
