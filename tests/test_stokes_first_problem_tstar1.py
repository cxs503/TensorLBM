"""Acceptance / regression test for case pool LBM-STOKES-FIRST-PROBLEM.

Criterion (case pool): pointwise relative error of the non-dimensional profile
``u/U = erfc(eta)`` for ``eta in [0, 2]`` at ``t* = 1.0`` must be below 2%.

Cheap smoke configuration (``delta = sqrt(nu t) = 12.5`` lattice units at
``t* = 0.25``) so the whole file runs in a few seconds; the full evidence
(resolution study, Ma scan, TRT/BGK and top-BC variants, figures) lives in
``benchmarks/verified/stokes_first_problem/scan_tstar1/``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "verified" / "stokes_first_problem"
sys.path.insert(0, str(BENCH))

from run_tstar1 import DELTA_REF, analytic_profile, erfc, run_case  # noqa: E402

H = 100
NU = 0.1  # tau = 0.8
TAU = 3.0 * NU + 0.5
TSTAR = 0.25
U = 1e-3


@pytest.fixture(scope="module")
def case() -> dict:
    return run_case(H, TAU, U, TSTAR, record_tstars=(TSTAR,))


def test_erfc_helper_matches_stdlib() -> None:
    xs = np.array([0.0, 0.25, 1.0, 2.0])
    assert np.allclose(erfc(xs), [1.0 - math.erf(v) for v in xs], atol=1e-14)


def test_analytic_profile_is_similarity_solution() -> None:
    nu, t = NU, 1250.0
    eta, u = analytic_profile(np.array([0.0, math.sqrt(nu * t)]), 1.0, nu, t)
    assert np.allclose(eta, [0.0, 0.5])
    # y = 2 sqrt(nu t) is the eta = 1 point
    _, u1 = analytic_profile(np.array([2 * math.sqrt(nu * t)]), 1.0, nu, t)
    assert u1[0] == pytest.approx(math.erfc(1.0), rel=1e-12)
    assert u[1] == pytest.approx(math.erfc(0.5), rel=1e-12)


def test_pointwise_error_within_2pct(case: dict) -> None:
    p = case["profiles"][0]
    assert p["delta_lb"] == pytest.approx(math.sqrt(NU * case["nsteps"]), rel=1e-9)
    assert p["max_rel_err_pct_eta0.1_2"] < 2.0
    assert p["max_rel_err_pct_native_nodes"] < 2.0
    # Full window incl. the wall-plane endpoint: at this cheap resolution
    # (delta = 12.5) that endpoint is a coordinate gap worth exactly
    # 1 - erfc(1/(4 delta)) = 2.26% (see the artifact test below).  The full-window
    # < 2% claim needs delta >= 25 and is asserted in the evidence bundle
    # scan_tstar1/summary.json (all_pass_2pct == true).
    assert p["max_rel_err_pct_eta0_2"] < 3.0


def test_eta_zero_gap_is_the_half_way_wall_artifact(case: dict) -> None:
    """eta = 0 has no node: the half-way wall sits at y = 0.5, so the first fluid
    node is at eta_1 = 1/(4 delta) and the clamped value differs from erfc(0) by
    exactly 1 - erfc(eta_1).  This pins the origin of the largest residual error."""
    p = case["profiles"][0]
    delta = p["delta_lb"]
    eta1 = 0.5 / (2.0 * delta)
    assert p["max_rel_err_pct_eta0_2"] == pytest.approx(100.0 * (1.0 - math.erfc(eta1)), rel=5e-3)


def test_wall_node_matches_analytic_at_wall_plane(case: dict) -> None:
    p = case["profiles"][0]
    u_wall_node = p["u_over_U_native"][0]
    u_ana = p["u_over_U_ana_native"][0]
    assert p["eta_native"][0] == pytest.approx(1.0 / (4.0 * p["delta_lb"]), abs=1e-5)
    assert u_wall_node == pytest.approx(u_ana, rel=1e-4)


def test_solution_is_exactly_self_similar_in_U() -> None:
    """Unidirectional, solenoidal, pressure-free flow: the quadratic equilibrium
    terms carry no first moment, so rho == 1 and u/U is independent of Ma.  This
    is what makes Ma a non-issue for this case (and why so few steps are needed)."""
    lo = run_case(H, TAU, 1e-3, TSTAR, record_tstars=(TSTAR,))
    hi = run_case(H, TAU, 5e-2, TSTAR, record_tstars=(TSTAR,))
    a = np.array(lo["profiles"][0]["u_over_U_native"])
    b = np.array(hi["profiles"][0]["u_over_U_native"])
    assert np.max(np.abs(a - b)) < 1e-5
    assert hi["rho_max_dev_from_1"] < 1e-6


def test_default_reference_length_documented() -> None:
    # t* = 1 <=> delta = delta_ref; the eta window [0,2] must stay inside the domain
    assert DELTA_REF == 25.0
    assert 4 * DELTA_REF < 200  # H = 200 -> top wall at eta = 4