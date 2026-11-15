"""Round-4 XVAL tests: the **transient** (impulsive-start) solution of the
periodic-x + constant-dp/dx channel and the solver's fidelity to it.

Rounds 1-3 verified the steady state only.  A steady benchmark is blind to time
errors in the driving: a constant ``O(dt)`` offset in the reported velocity, or
a lag in the force injection, leaves the steady parabola unchanged.  These
tests pin the fourth verification axis:

* the analytic start-up solution itself (reduces to the steady parabola, starts
  from rest, is monotone, satisfies ``du/dt = g + nu d2u/dy2``, and gives the
  uniform initial acceleration ``u = g t``),
* the solver's whole start-up history against it (gate ``< 3 %`` of the steady
  L2 norm, second order in the wall distance),
* and the *closed-form* velocity offset each force convention reports in the
  first steps -- ``(1 - omega)/2 * g`` for the OSS ``guo_half_frozen`` arm,
  exactly zero for the repo's ``split_guo`` arm.  That offset is invisible in
  the steady state, which is why it needed this round.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlbm.periodic_channel_dpdx import (
    DpdxChannel,
    FORCE_GUO_HALF_FROZEN,
    FORCE_SPLIT_GUO,
    WALL_HALFWAY_BB,
    WALL_REGULARISED,
    observed_order,
    poiseuille_analytic,
    poiseuille_transient,
)

TAU = 0.5375
NU = (TAU - 0.5) / 3.0
G = 1.0e-3
GATE = 0.03


def h_for(ny: int, wall: str) -> float:
    return float(ny - 1) if wall == WALL_REGULARISED else float(ny - 2)


def g_for(ny: int, wall: str, u_target: float = 0.01) -> float:
    return 8.0 * NU * u_target / h_for(ny, wall) ** 2


# ---------------------------------------------------------------------------
# the analytic start-up solution
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("wall", [WALL_HALFWAY_BB, WALL_REGULARISED])
def test_transient_reduces_to_the_steady_parabola(wall):
    h = h_for(33, wall)
    t_late = 5.0 * h * h / NU          # tau = 5: first mode damped by e^-49
    _, u, y_w, H, n_modes = poiseuille_transient(33, NU, G, t_late, wall=wall)
    _, u_ss, y_w_ss, H_ss = poiseuille_analytic(33, NU, G, wall=wall)
    assert (y_w, H) == (y_w_ss, H_ss)
    assert np.abs(u - u_ss).max() <= 1e-13 * np.abs(u_ss).max()
    assert n_modes >= 1


@pytest.mark.parametrize("wall", [WALL_HALFWAY_BB, WALL_REGULARISED])
def test_transient_starts_from_rest_and_grows_monotonically(wall):
    ny = 33
    h = h_for(ny, wall)
    _, u_ss, _, _ = poiseuille_analytic(ny, NU, G, wall=wall)
    prev = np.zeros_like(u_ss)
    for tau in (1e-3, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        _, u, _, _, _ = poiseuille_transient(ny, NU, G, tau * h * h / NU, wall=wall)
        # starts from rest, grows towards the steady profile, never overshoots
        assert np.all(u >= prev - 1e-15)
        assert np.all(u <= u_ss + 1e-12 * u_ss.max())
        prev = u
    assert np.linalg.norm(prev - u_ss) <= 1e-3 * np.linalg.norm(u_ss)


def test_transient_initial_acceleration_is_uniform():
    """For tau -> 0 the exact solution is the uniform acceleration u = g t."""
    ny, wall = 33, WALL_HALFWAY_BB
    h = h_for(ny, wall)
    for tau in (1e-4, 1e-3):
        t = tau * h * h / NU
        _, u, _, _, n_modes = poiseuille_transient(ny, NU, G, t, wall=wall)
        ratio = u[6:-6] / (G * t)
        assert np.allclose(ratio, 1.0, atol=1e-6), (tau, ratio.min(), ratio.max())
        assert n_modes > 1


def test_transient_satisfies_the_momentum_equation():
    """Central differences in t and y must reproduce du/dt = g + nu d2u/dy2."""
    ny, wall = 33, WALL_HALFWAY_BB
    h = h_for(ny, wall)
    t0 = 0.1 * h * h / NU
    dt = 1e-4 * h * h / NU
    _, u, _, _, _ = poiseuille_transient(ny, NU, G, t0, wall=wall)
    _, u_p, _, _, _ = poiseuille_transient(ny, NU, G, t0 + dt, wall=wall)
    _, u_m, _, _, _ = poiseuille_transient(ny, NU, G, t0 - dt, wall=wall)
    dut = (u_p - u_m) / (2.0 * dt)
    d2u = np.roll(u, -1) - 2.0 * u + np.roll(u, 1)
    residual = dut[1:-1] - (G + NU * d2u[1:-1])
    # truncation is O(dy^2 (pi/H)^2) with dy = 1 lattice unit; measured 4e-4
    assert np.abs(residual).max() < 2e-3 * G


def test_transient_series_converges_and_guards_bad_input():
    ny, wall = 33, WALL_HALFWAY_BB
    h = h_for(ny, wall)
    for tau in (0.02, 0.05, 0.2, 1.0):
        t = tau * h * h / NU
        _, u_adapt, _, _, n_modes = poiseuille_transient(ny, NU, G, t, wall=wall)
        _, u_fixed, _, _, n_fixed = poiseuille_transient(
            ny, NU, G, t, wall=wall, n_modes=41)
        assert n_fixed == 41 and n_modes <= 41
        assert np.linalg.norm(u_adapt - u_fixed) <= 1e-12 * np.linalg.norm(u_adapt)
    with pytest.raises(ValueError):
        poiseuille_transient(ny, NU, G, 0.0, wall=wall)
    with pytest.raises(ValueError):
        poiseuille_transient(ny, NU, G, 1e-12, wall=wall, max_modes=5)
    assert poiseuille_transient(ny, NU, 0.0, 0.1 * h * h / NU)[1].max() == 0.0


# ---------------------------------------------------------------------------
# the solver against the start-up history
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("wall,scheme", [(WALL_REGULARISED, FORCE_GUO_HALF_FROZEN),
                                         (WALL_HALFWAY_BB, FORCE_SPLIT_GUO)])
def test_solver_follows_the_exact_start_up(wall, scheme):
    ny = 17
    h = h_for(ny, wall)
    force = g_for(ny, wall)
    ch = DpdxChannel(2, ny, TAU, force, wall=wall, force_scheme=scheme)
    u_ss = poiseuille_analytic(ny, NU, force, wall=wall)[1]
    norm_ss = np.linalg.norm(u_ss)
    m0 = ch.mass()
    for tau in (0.05, 0.1, 0.2):
        nsteps = int(round(tau * h * h / NU))
        while ch.steps < nsteps:
            ch.step()
        _, ua, _, _, _ = poiseuille_transient(ny, NU, force, float(ch.steps), wall=wall)
        u = ch.profile()
        rel = np.linalg.norm(u - ua) / norm_ss
        assert rel < GATE, f"tau={tau}: {rel:.2%} of the steady norm"
        # the growing centreline value must track the analytic history too
        assert abs(u.max() / u_ss.max() - ua.max() / u_ss.max()) < 0.01
    assert abs((ch.mass() - m0) / m0) < 1e-10


def test_transient_error_is_second_order_in_resolution():
    """At fixed nondimensional time the transient error must converge at p ~ 2."""
    wall, scheme = WALL_HALFWAY_BB, FORCE_SPLIT_GUO
    errs, hs = [], []
    for ny in (17, 33):
        h = h_for(ny, wall)
        force = g_for(ny, wall)
        ch = DpdxChannel(2, ny, TAU, force, wall=wall, force_scheme=scheme)
        nsteps = int(round(0.1 * h * h / NU))
        while ch.steps < nsteps:
            ch.step()
        _, ua, _, _, _ = poiseuille_transient(ny, NU, force, float(ch.steps), wall=wall)
        u_ss = poiseuille_analytic(ny, NU, force, wall=wall)[1]
        errs.append(np.linalg.norm(ch.profile() - ua) / np.linalg.norm(u_ss))
        hs.append(h)
    p = observed_order(errs[0], errs[1], hs[0], hs[1])
    assert p > 1.5, f"observed transient order {p:.2f}"


@pytest.mark.parametrize("tau_relax", [TAU, 0.6, 0.8])
def test_reported_velocity_offset_matches_the_force_convention(tau_relax):
    """The first steps report ``u = g t + offset`` with a *constant* offset.

    OSS ``guo_half_frozen`` (source weight ``1 - omega/2`` plus a ``+F/2``
    velocity shift) reports ``(1 - omega)/2 * g``; the repo's ``split_guo``
    (full weight, no shift) reports exactly zero.  Both are invisible in the
    steady state -- this is the round-4 finding.
    """
    ny = 17
    nu = (tau_relax - 0.5) / 3.0
    omega = 1.0 / tau_relax
    for wall, scheme, expected in (
            (WALL_REGULARISED, FORCE_GUO_HALF_FROZEN, 0.5 * (1.0 - omega)),
            (WALL_HALFWAY_BB, FORCE_SPLIT_GUO, 0.0)):
        h = h_for(ny, wall)
        force = 8.0 * nu * 0.01 / h ** 2
        ch = DpdxChannel(2, ny, tau_relax, force, wall=wall, force_scheme=scheme)
        for _ in range(6):
            ch.step()
            u = ch.profile()
            offset = u[u.size // 2] / force - ch.steps
            assert abs(offset - expected) < 1e-6, (wall, ch.steps, offset, expected)