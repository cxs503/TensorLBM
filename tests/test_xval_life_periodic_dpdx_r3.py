"""Round-3 XVAL tests: order-of-accuracy bookkeeping, measured wall origin, and
the physics the round-3 driver relies on.

Context: round 2 of ``XVAL-LBM-SINGLE-PHASE-PERIODIC-X-PLUS-CONSTANT-DP``
reported an "observed grid order" of 1.15 (regularised closure) / 1.23
(halfway bounce-back) and blamed an ``O(dx)`` wall-origin error for the
shortfall.  Recomputing from round 2's *own* artifact shows the shortfall was an
estimator bug: the order was formed as ``log2(e_17 / e_25)`` while the
refinement of the wall distance ``H`` was ``16 -> 24`` (1.5x, not 2x).  The
correct order is ``1.97 / 2.00``.

These tests pin (a) the resolution-explicit estimator and the exact size of the
``log2`` error, (b) that the steady state of the port is an *exact* parabola --
the fit residual is at round-off, so the error against the nominal-geometry
analytic solution is purely a geometric wall offset -- and (c) second-order
convergence of the port to the exact Poiseuille solution, which is what makes
the ``<15 %`` gate meaningful.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from tensorlbm.periodic_channel_dpdx import (
    DpdxChannel,
    FORCE_SPLIT_GUO,
    WALL_HALFWAY_BB,
    WALL_REGULARISED,
    fit_parabola,
    observed_order,
    poiseuille_analytic,
)

TAU = 0.5375
NU = (TAU - 0.5) / 3.0
GATE = 0.15
R2_ARTIFACT = (pathlib.Path(__file__).resolve().parents[1]
               / "benchmarks" / "cross_validation" / "periodic_channel_dpdx" / "out"
               / "xval_life_periodic_dpdx_r2.json")


def h_for(ny: int, wall: str) -> float:
    return float(ny - 1) if wall == WALL_REGULARISED else float(ny - 2)


def g_for(ny: int, u_target: float, wall: str) -> float:
    return 8.0 * NU * u_target / h_for(ny, wall) ** 2


def steady(ny: int, u_target: float, *, tol: float = 1e-9):
    wall = WALL_HALFWAY_BB
    ch = DpdxChannel(nx=2, ny=ny, tau=TAU, force=g_for(ny, u_target, wall),
                     wall=wall, force_scheme=FORCE_SPLIT_GUO)
    info = ch.run_until_steady(tol=tol, max_steps=200_000, check_every=200)
    return ch, info


# ---------------------------------------------------------------------------
# 1. the estimator
# ---------------------------------------------------------------------------
def test_observed_order_recovers_a_known_power_law():
    h = np.array([8.0, 16.0, 32.0])
    for p in (1.0, 2.0, 3.5):
        e = 0.7 * h ** (-p)
        assert observed_order(e[0], e[1], h[0], h[1]) == pytest.approx(p, rel=1e-12)
        assert observed_order(e[1], e[2], h[1], h[2]) == pytest.approx(p, rel=1e-12)


def test_log2_on_a_non_doubling_refinement_understates_the_order():
    """The round-2 defect, reproduced on round 2's own numbers."""
    e_coarse, e_fine = 0.006607597379530949, 0.002969142736804209
    h_coarse, h_fine = 16.0, 24.0                      # ny 17 -> 25, regularised
    correct = observed_order(e_coarse, e_fine, h_coarse, h_fine)
    buggy = float(np.log2(e_coarse / e_fine))
    assert buggy == pytest.approx(1.1540793314996773, rel=1e-12)   # r2 reported
    assert correct == pytest.approx(1.9730, abs=5e-4)
    # the damage factor is exactly ln2/ln(H_fine/H_coarse)
    assert correct / buggy == pytest.approx(np.log(2.0) / np.log(1.5), rel=1e-12)
    assert buggy < 0.8 * correct          # a second-order scheme mislabelled


def test_observed_order_rejects_meaningless_inputs():
    with pytest.raises(ValueError):
        observed_order(0.0, 1e-3, 8.0, 16.0)
    with pytest.raises(ValueError):
        observed_order(1e-3, -1e-3, 8.0, 16.0)
    with pytest.raises(ValueError):
        # not a refinement: "finer" grid must have the larger resolution length
        observed_order(1e-2, 1e-3, 16.0, 8.0)
    with pytest.raises(ValueError):
        observed_order(1e-2, 1e-3, 16.0, 16.0)


# ---------------------------------------------------------------------------
# 2. the measured wall origin
# ---------------------------------------------------------------------------
def test_fit_parabola_recovers_an_exact_poiseuille_profile():
    y = np.arange(1, 33, dtype=np.float64)
    y_wall, h, u_max = 0.5, 32.0, 0.037
    s = (y - y_wall) / h
    u = 4.0 * u_max * s * (1.0 - s)
    fit = fit_parabola(u, y)
    assert fit["y_wall"] == pytest.approx(y_wall, abs=1e-10)
    assert fit["h"] == pytest.approx(h, abs=1e-10)
    assert fit["u_max"] == pytest.approx(u_max, rel=1e-12)
    assert fit["rel_l2_residual"] < 1e-14


def test_fit_parabola_rejects_non_channel_profiles():
    y = np.arange(6, dtype=np.float64)
    with pytest.raises(ValueError):
        fit_parabola(np.ones(6), y)                       # flat: a = 0
    with pytest.raises(ValueError):
        fit_parabola(np.ones(5), y[:5])                   # wrong length pairing


# ---------------------------------------------------------------------------
# 3. physics the round-3 claims rest on
# ---------------------------------------------------------------------------
def test_steady_state_is_an_exact_parabola_with_a_measured_wall_origin():
    ch, info = steady(25, 0.01)
    assert info["converged"]
    u = ch.profile()
    y = ch.y_centres()
    fit = fit_parabola(u, y)
    s, u_ana, y_wall_nom, h_nom = poiseuille_analytic(
        25, NU, g_for(25, 0.01, WALL_HALFWAY_BB), wall=WALL_HALFWAY_BB)
    # the discrete steady state is parabolic to round-off: whatever error
    # remains versus the analytic solution is geometry, not shape
    assert fit["rel_l2_residual"] < 1e-9
    assert np.max(np.abs(u - u_ana)) / np.max(np.abs(u_ana)) < 0.05
    # wall origin is measured, and it is the closure's midpoint to O(dx)
    assert fit["y_wall"] == pytest.approx(y_wall_nom, abs=0.1)
    assert fit["h"] == pytest.approx(h_nom, abs=0.2)
    assert fit["y_vertex"] == pytest.approx(0.5 * (fit["y_wall"] + fit["y_wall_top"]),
                                           abs=1e-9)


def test_accuracy_gate_and_mass_conservation():
    """<15 % gate (and the stricter <3 %) on the repo's production closure."""
    ch, info = steady(17, 0.02)
    assert info["converged"]
    assert ch.rel_l2_vs_analytic() < 0.03
    assert ch.rel_l2_vs_analytic() < GATE
    assert abs(ch.mass() / (ch.rho0 * 2 * (17 - 2)) - 1.0) < 1e-9


def test_second_order_convergence_between_two_resolutions():
    """H: 7 -> 15 (ny 9 -> 17) at fixed u_max = 0.01.

    Two points only -- enough to fail an order loss (the round-2 report claimed
    1.15), not enough to certify the asymptotic rate; the five-point sweep
    lives in ``benchmarks/cross_validation/periodic_channel_dpdx/run_xval_r3.py``.
    """
    ch_a, info_a = steady(9, 0.01)
    ch_b, info_b = steady(17, 0.01)
    assert info_a["converged"] and info_b["converged"]
    e_a = ch_a.rel_l2_vs_analytic()
    e_b = ch_b.rel_l2_vs_analytic()
    p = observed_order(e_a, e_b, h_for(9, WALL_HALFWAY_BB),
                       h_for(17, WALL_HALFWAY_BB))
    assert 1.8 <= p <= 2.3, f"observed order {p:.3f} is not second order"


# ---------------------------------------------------------------------------
# 4. evidence chain: the round-2 order number is corrected by round 3
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not R2_ARTIFACT.exists(), reason="round-2 artifact absent")
def test_round2_artifact_supports_second_order_once_the_estimator_is_fixed():
    r2 = json.loads(R2_ARTIFACT.read_text())
    rows = sorted(r2["steady"].values(), key=lambda v: (v["arm"], v["ny"]))
    seen = {}
    for a, b in zip(rows, rows[1:]):
        if a["arm"] != b["arm"]:
            continue
        assert b["H"] > a["H"]
        p = observed_order(a["rel_l2_vs_analytic"], b["rel_l2_vs_analytic"],
                           a["H"], b["H"])
        assert 1.8 <= p <= 2.3, (a["arm"], p)
        seen[a["arm"]] = p
    assert seen, "no refinement pair found in the round-2 artifact"