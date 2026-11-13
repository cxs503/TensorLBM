"""Acceptance tests for the case-pool case CASE-LBM-KOVASZNAY-EXACT.

Four layers (fast: the live runs use ny=32 and a few thousand steps; the
ny=128 anchor run lives in the committed artifact, not in CI):

1. ``test_lambda_root_and_paper_correction`` -- the decay rate is the negative
   root of ``lam^2 - Re lam - 4 pi^2 = 0`` and ``lam(Re=40) = -0.9637`` (the
   "lam ~= -0.5 at Re=40" figure that older task sheets quote is wrong; it
   corresponds to Re ~= 78.5).
2. ``test_reference_field_is_exact_solution`` -- closed-form N-S residual and
   continuity at machine precision, so the reference cannot silently drift.
3. ``test_outlet_band_dominates_the_full_field_error`` -- a live run must show
   the interior (outlet-free) error far below the full-field error, which is the
   quantitative claim this case entry makes.
4. ``test_committed_artifact_acceptance`` -- the committed
   ``artifacts/kovasznay_2d/results.json`` must carry a PASS verdict on the
   anchor criterion at N >= 128 together with its own provenance.

Run: PYTHONPATH=src python -m pytest -q tests/test_case_pool_kovasznay_2d.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CASE = REPO / "cases" / "kovasznay_2d"
for p in (REPO / "src", CASE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import analytic as A  # noqa: E402

ARTIFACT = REPO / "artifacts" / "kovasznay_2d" / "results.json"


def test_lambda_root_and_paper_correction() -> None:
    for re in (10.0, 40.0, 100.0):
        lam = A.kovasznay_lambda(re)
        assert lam < 0.0, "the physical branch is the decaying (negative) root"
        # lam^2 - Re lam - 4 pi^2 = 0
        assert abs(lam * lam - re * lam - 4.0 * 3.141592653589793**2) < 1e-9 * max(1.0, re**2)
    lam40 = A.kovasznay_lambda(40.0)
    assert lam40 == pytest.approx(-0.9637405, abs=1e-6)
    # the correction quoted in the docs: lam = -0.5 belongs to Re ~= 78.5
    re_for_half = -0.5 - 4.0 * 3.141592653589793**2 / (-0.5)  # Re = lam - 4pi^2/lam
    assert re_for_half == pytest.approx(78.46, abs=0.05)


def test_reference_field_is_exact_solution() -> None:
    res = A.make_field(64, re=40.0).ns_residual()
    assert res["res_div_max"] < 1e-12, res
    assert res["res_mom_x_max"] < 1e-12, res
    assert res["res_mom_y_max"] < 1e-12, res
    assert abs(res["compat_lhs"]) < 1e-8
    # nu recovered from the compatibility relation must equal 1/Re
    assert res["nu_from_compat"] == pytest.approx(1.0 / 40.0, rel=1e-9)


def test_outlet_band_dominates_the_full_field_error() -> None:
    """Live run (ny=32, xmax=3): interior error << full-field error."""
    from run import simulate

    m = simulate(32, xmax=3.0, steps=3000, threads=4)
    assert m["finite"] and m["steady"] in (True, False)
    full_v = m["full"]["v_l2_rel_pct"]
    int_v = m["interior"]["v_l2_rel_pct"]
    assert 2.0 < full_v < 5.0, f"v full-field L2 out of expected band: {full_v}"
    assert int_v < 0.5, f"interior v L2 should be far below the gate: {int_v}"
    assert full_v / int_v > 5.0, "outlet band must dominate the full-field error"
    # u is better behaved than v but must obey the same ordering
    assert m["interior"]["u_l2_rel_pct"] < 0.5
    # the interior window really excludes a full y-period at the outlet
    lo, hi = m["interior"]["cols"]
    assert lo == 1 and hi == m["nx"] - m["ny"]


def test_shorter_domain_is_worse() -> None:
    """xmax is the single variable that controls the boundary term."""
    from run import simulate

    short = simulate(32, xmax=2.0, steps=2000, threads=4)["full"]["v_l2_rel_pct"]
    long_ = simulate(32, xmax=3.0, steps=2000, threads=4)["full"]["v_l2_rel_pct"]
    assert short > 2.0 * long_, f"expected a strong domain-length effect: {short} vs {long_}"


def test_committed_artifact_acceptance() -> None:
    """The committed evidence must pass the anchor criterion, with provenance."""
    assert ARTIFACT.exists(), f"missing case-pool evidence: {ARTIFACT}"
    d = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    gate = d["gate"]
    assert gate["anchor"]["tolerance_pct"] == 3.0
    cfg = gate["config"]
    assert cfg["ny"] >= 128, "anchor criterion is judged at N >= 128"
    assert cfg["re"] == 40.0
    assert gate["u_l2_rel_pct"] < 3.0 and gate["v_l2_rel_pct"] < 3.0, gate
    assert gate["verdict"] == "PASS"
    assert gate["extrap"] == "none", "no extrapolation is allowed on the gate quantity"
    assert cfg["steps"] >= 6000 and cfg["steady"] is True
    # the falsifiable claims this entry adds, pinned so a silent regression fails
    assert d["xmax_scaling"]["64"]["slope_v_full"] < -1.5, "outlet term must decay with xmax"
    assert d["order_of_accuracy"]["u"]["interior_grid_independent"] is True
    assert d["mach_refinement_ny64"]["u"]["ratio_halving"] > 2.5, "interior floor must be O(Ma^2)"