"""Regression + validation tests for ``tensorlbm.periodic_channel_dpdx``.

The module ports the OSS ``single-phase-periodic-x-plus-constant-dpdx-driving``
channel (``joconnor22/LIFE``): x-periodic, constant body force ``dp/dx``,
two wall closures (halfway bounce-back / regularised) and two force-injection
conventions (split Guo / Guo with the ``1 - omega/2`` weight and ``+F/2``
velocity shift).

Round 1 of the xval left the port on an honest FAIL (both arms diverged after a
few thousand steps).  The two defects were:

1. ``_shift`` streamed with the opposite sign, so populations travelled against
   their own lattice velocity and the wall closures read the wrong neighbour
   row (``test_streaming_moves_population_along_its_velocity``);
2. ``equilibrium`` used ``1 + 3cu + 1.5cu^2 - 1.5u^2`` instead of the
   second-order Hermite form ``... + 4.5cu^2 ...``, which breaks the
   equilibrium's own moments — ``sum f = 0.9975 rho`` for ``u = 0.05``
   (``test_equilibrium_moments``).

Both are pinned below, together with the physics the module exists for: the
steady state must be the exact plane-Poiseuille parabola, mass must be
conserved, and the wall-less control lane must fail the accuracy gate.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlbm.periodic_channel_dpdx import (
    CS2,
    D2Q9_C,
    D2Q9_OPP,
    D2Q9_W,
    FORCE_GUO_HALF_FROZEN,
    FORCE_SPLIT_GUO,
    WALL_HALFWAY_BB,
    WALL_NONE,
    WALL_REGULARISED,
    DpdxChannel,
    _shift,
    dpdx_to_lattice_force,
    equilibrium,
    life_force_xy,
    poiseuille_analytic,
)

TAU = 0.5375
NU = (TAU - 0.5) / 3.0
C = np.array(D2Q9_C, dtype=np.float64)


def lattice_force(ny: int, u_target: float, wall: str) -> float:
    h = float(ny - 2) if wall == WALL_HALFWAY_BB else float(ny - 1)
    return 8.0 * NU * u_target / h ** 2


# ---------------------------------------------------------------------------
# lattice bookkeeping
# ---------------------------------------------------------------------------
def test_lattice_table_is_d2q9():
    assert len(D2Q9_C) == len(D2Q9_W) == 9
    assert pytest.approx(1.0) == sum(D2Q9_W)
    # symmetry, weight symmetry, and a real opposite table
    for i, (cx, cy) in enumerate(D2Q9_C):
        assert (-cx, -cy) in D2Q9_C
        j = D2Q9_OPP[i]
        assert D2Q9_C[j] == (-cx, -cy)
        assert D2Q9_W[j] == pytest.approx(D2Q9_W[i])
    assert pytest.approx(0.0) == float(np.dot([w * c[0] for w, c in zip(D2Q9_W, D2Q9_C)], np.ones(9)))


# ---------------------------------------------------------------------------
# defect 1 — streaming sign
# ---------------------------------------------------------------------------
def test_streaming_moves_population_along_its_velocity():
    bump = np.zeros((5, 6))
    bump[2, 3] = 1.0
    for i, (cx, cy) in enumerate(D2Q9_C):
        if (cx, cy) == (0, 0):
            continue
        idx = np.argwhere(_shift(bump, cx, cy) == 1.0)
        assert len(idx) == 1
        # pull form: f_i(x, t+1) = f_i_post(x - c, t)  =>  displacement +c
        assert (int(idx[0][0]), int(idx[0][1])) == (2 + cx, 3 + cy), (
            f"direction {cx},{cy} streamed to {idx[0]} — sign flipped")


# ---------------------------------------------------------------------------
# defect 2 — equilibrium moments
# ---------------------------------------------------------------------------
def test_equilibrium_moments():
    rho = np.full((2, 3), 1.3)
    ux = np.full((2, 3), 0.05)
    uy = np.full((2, 3), -0.02)
    f = equilibrium(rho, ux, uy)
    assert f.shape == (9, 2, 3)
    flat = f.reshape(9, -1)
    assert np.allclose(f.sum(axis=0), rho, rtol=0, atol=1e-15)
    assert np.allclose(C[:, 0] @ flat, (rho * ux).ravel(), rtol=0, atol=1e-16)
    assert np.allclose(C[:, 1] @ flat, (rho * uy).ravel(), rtol=0, atol=1e-16)
    # second moment: 2 rho cs^2 + rho |u|^2
    second = (C[:, 0] ** 2 + C[:, 1] ** 2) @ flat
    exact = 2.0 * CS2 * rho + rho * (ux * ux + uy * uy)
    assert np.allclose(second, exact.ravel(), rtol=0, atol=1e-15)
    # the round-1 form violated the mass moment by 0.25 % at u = 0.05
    assert abs(f.sum(axis=0) - rho).max() < 1e-14


# ---------------------------------------------------------------------------
# conservative operators
# ---------------------------------------------------------------------------
def test_guo_source_is_mass_neutral():
    from tensorlbm.periodic_channel_dpdx import guo_source
    rng = np.random.default_rng(7)
    u = rng.normal(scale=0.05, size=(2, 4, 5))
    force = np.zeros((2, 4, 5))
    force[0] = 1e-4
    force[1] = -3e-5
    weight = 1.0 - 0.5 / TAU
    src = guo_source(u, force, weight)
    # zero mass moment ...
    assert np.abs(src.sum(axis=0)).max() < 1e-18
    # ... and momentum moment = weight * F (Guo 2002, second-order source)
    mom = np.stack([np.tensordot(C[:, 0], src, axes=(0, 0)),
                    np.tensordot(C[:, 1], src, axes=(0, 0))])
    assert np.abs(mom - weight * force).max() < 1e-18


@pytest.mark.parametrize("wall,scheme", [(WALL_REGULARISED, FORCE_GUO_HALF_FROZEN),
                                         (WALL_HALFWAY_BB, FORCE_SPLIT_GUO)])
def test_mass_conserved_in_transient(wall, scheme):
    ny = 17
    k = DpdxChannel(2, ny, TAU, lattice_force(ny, 0.02, wall),
                    wall=wall, force_scheme=scheme)
    m0 = k.mass()
    for _ in range(300):
        k.step()
    assert abs((k.mass() - m0) / m0) < 1e-12
    assert np.abs(k.rho - 1.0).max() < 1e-9


# ---------------------------------------------------------------------------
# physics: steady plane Poiseuille
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("wall,scheme", [(WALL_REGULARISED, FORCE_GUO_HALF_FROZEN),
                                         (WALL_HALFWAY_BB, FORCE_SPLIT_GUO)])
def test_steady_state_matches_poiseuille(wall, scheme):
    """Steady state must be the exact parabola — gate: < 3 % (owner std < 15 %)."""
    ny, u_target = 17, 0.02
    force = lattice_force(ny, u_target, wall)
    k = DpdxChannel(2, ny, TAU, force, wall=wall, force_scheme=scheme)
    rec = k.run_until_steady(tol=1e-9, max_steps=200_000, check_every=200)
    assert rec["converged"], rec
    _, u_ana, y_w, h = poiseuille_analytic(ny, NU, force, wall=wall)
    u = k.profile()
    rel_l2 = np.linalg.norm(u - u_ana) / np.linalg.norm(u_ana)
    rel_umax = abs(u.max() - u_ana.max()) / u_ana.max()
    assert rel_l2 < 0.03, f"rel-L2 {rel_l2:.3%} vs parabola"
    assert rel_umax < 0.03, f"u_max error {rel_umax:.3%}"
    # shape invariant: the sampled profile must have the analytic parabola's
    # shape (u_mean/u_max of the *discrete* sample, not the continuum 2/3 —
    # the fluid rows never sit on the walls)
    assert abs(u.mean() / u.max() - u_ana.mean() / u_ana.max()) < 3e-3
    # walls at the closure-specific origin, not the fluid row
    assert y_w == (0.5 if wall == WALL_HALFWAY_BB else 0.0)
    assert h == (float(ny - 2) if wall == WALL_HALFWAY_BB else float(ny - 1))


def test_forced_channel_is_not_uniform_control():
    """Control lane: no y walls -> the solution cannot be a parabola."""
    ny, u_target = 17, 0.02
    force = lattice_force(ny, u_target, WALL_HALFWAY_BB)
    k = DpdxChannel(2, ny, TAU, force, wall=WALL_NONE, force_scheme=FORCE_SPLIT_GUO)
    for _ in range(2000):
        k.step()
    u = k.profile()
    _, u_ana, _, _ = poiseuille_analytic(ny, NU, force, wall=WALL_REGULARISED)
    rel_l2 = np.linalg.norm(u - u_ana) / np.linalg.norm(u_ana)
    assert rel_l2 > 0.15, "the gate has no teeth: a wall-less channel passed"


# ---------------------------------------------------------------------------
# unit mapping dp/dx [Pa/m] -> lattice force
# ---------------------------------------------------------------------------
def test_dpdx_unit_mapping():
    dx, dt, rho_p, nu_p, h_p = 0.02, 5.0e-4, 1.0, 1e-6, 0.64
    dpdx = 1.0
    f_std = dpdx_to_lattice_force(dpdx, dx=dx, dt=dt, rho_scale=rho_p)
    f_life = dpdx_to_lattice_force(dpdx, dx=dx, dt=dt, rho_scale=rho_p,
                                   convention="life_compat")
    assert f_life == pytest.approx(life_force_xy(dpdx, dx=dx, dt=dt, rho_scale=rho_p))
    # LIFE's expression is the standard one divided by dx (round-1 finding)
    assert f_life / f_std == pytest.approx(1.0 / dx, rel=1e-12)
    # the standard convention reproduces the physical u_max after scaling
    nu_lb = nu_p * dt / dx ** 2
    h_lb = h_p / dx
    u_phys = dpdx * h_p ** 2 / (8.0 * rho_p * nu_p)
    u_lb = f_std * h_lb ** 2 / (8.0 * nu_lb)
    assert u_lb * dx / dt == pytest.approx(u_phys, rel=1e-12)
    with pytest.raises(ValueError):
        dpdx_to_lattice_force(dpdx, dx=dx, dt=dt, convention="nope")


def test_invalid_configuration_rejected():
    with pytest.raises(ValueError):
        DpdxChannel(2, 8, TAU, 1e-5, wall="slip")
    with pytest.raises(ValueError):
        DpdxChannel(2, 8, TAU, 1e-5, force_scheme="psi")
    with pytest.raises(ValueError):
        DpdxChannel(2, 3, TAU, 1e-5)