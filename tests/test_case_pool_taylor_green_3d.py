"""Acceptance tests for the case-pool case CASE-TAYLOR-GREEN-3D (TensorLBM).

Three layers, mirroring the pool convention:

1. ``test_series_reference_reproduces_closed_forms`` -- the reference generator is
   pinned against two *independent* closed forms (z-invariant Taylor-Green and the
   Beltrami/ABC flow): where an exact solution exists the series must reproduce it
   to machine precision.  ``test_pool_contract_*`` pins the declared contract
   (refs.json) against the code, so neither can drift silently.
2. ``test_case_meets_pool_tolerance`` -- actually runs the pool case (coarse grid)
   and asserts every predeclared quantity inside its band.
3. ``test_committed_artifact_is_evidence`` -- the committed ``artifacts/summary.json``
   must carry a passing verdict, the predeclared thresholds and per-quantity
   ref/got numbers (guards against "artifact without evidence"), and
   ``test_verdict_logic_rejects_out_of_band`` guards the opposite failure mode
   (a verdict that cannot fail).

Run: PYTHONPATH=src:benchmarks pytest -q tests/test_case_pool_taylor_green_3d.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "benchmarks"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from case_pool.taylor_green_3d import run_case as rc  # noqa: E402

CASE_DIR = REPO / "benchmarks" / "case_pool" / "taylor_green_3d"
ART = CASE_DIR / "artifacts"


# --------------------------------------------------------------------------- #
# layer 0: the pool actually owns this case
# --------------------------------------------------------------------------- #
def test_pool_driver_discovers_case() -> None:
    """The case is reachable through the zero-semantics pool driver (no registry edit)."""
    from case_pool import run_pool

    assert "taylor_green_3d" in run_pool.discover()


def test_pool_contract_matches_ledger_tolerance() -> None:
    """refs.json (the declared contract) and the code must agree, and match the ledger band."""
    refs = json.loads((CASE_DIR / "refs.json").read_text())
    assert refs["case"] == rc.CASE
    assert refs["primary_reference"]["kind"] == "ANALYTIC_SERIES"
    assert refs["primary_reference"]["label_zh"] == "级数精确解"
    tol = refs["adopted_numeric_reference"]["tolerance_pct"]
    assert tol == rc.TOL_PCT == 8.0
    for qid in ("Q1_energy_ratio_rel_err_pct", "Q2_enstrophy_ratio_rel_err_pct",
                "Q3_velocity_l2_rel_err_pct", "Q4_vorticity_l2_rel_err_pct"):
        assert rc.THRESHOLDS[qid][0] == tol, qid
    assert all(v[1] in ("le", "ge") for v in rc.THRESHOLDS.values())


# --------------------------------------------------------------------------- #
# layer 1: the reference is self-consistent before it is used as a reference
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.physics
def test_series_reference_reproduces_closed_forms() -> None:
    """Where a closed form exists the series must reproduce it exactly.

    z-invariant mode: E(t)/E(0) = exp(-2*nu*|k|^2*t) with |k|^2 = 2k^2.
    Beltrami (ABC):   E(t)/E(0) = exp(-2*nu*k^2*t).
    Both are exact NS solutions, so any mismatch means the recursion, the
    projection or the normalisation of the reference is wrong -- independently of
    any LBM run.
    """
    anchor = rc.closed_form_anchor(n=16, order=8)
    assert anchor["z_invariant_rel_err"] < 1e-12
    assert anchor["abc_beltrami_rel_err"] < 1e-12
    assert anchor["max_rel_err"] < rc.THRESHOLDS["Q0_series_vs_closed_form_rel"][0]
    # the "closed forms" are genuine decays, i.e. the invariant is not vacuous
    assert 0.5 < anchor["z_invariant_ratio_closed_form"] < 1.0
    assert 0.5 < anchor["abc_beltrami_ratio_closed_form"] < 1.0


@pytest.mark.unit
@pytest.mark.physics
def test_reference_is_order_and_workgrid_converged() -> None:
    """The reference must be converged far tighter than the 8 % band it judges."""
    r = rc.leg_reference_convergence(n=16, variants=((8, 48), (10, 48), (10, 56)))
    assert r["order_convergence_rel"] < rc.THRESHOLDS["Q8_series_order_convergence_rel"][0]
    assert r["workgrid_invariance_rel"] < rc.THRESHOLDS["Q9_series_workgrid_invariance_rel"][0]


@pytest.mark.unit
@pytest.mark.physics
def test_classic_field_needs_a_series_reference() -> None:
    """The case premise: the z-dependent field's convective term is NOT a gradient."""
    from tensorlbm.tg3d_series import (
        abc_flow,
        convective_acceleration,
        taylor_green_classic,
        taylor_green_extruded,
    )

    a = 0.05
    classic = convective_acceleration(taylor_green_classic(a), box=32.0, n=32)
    extruded = convective_acceleration(taylor_green_extruded(a), box=32.0, n=32)
    assert classic["curl_over_adv"] > 0.1          # rotational -> no closed form
    assert extruded["curl_over_adv"] < 1e-10       # gradient -> closed form exists
    # ... while the Beltrami (ABC) companion is 3-D and z-dependent yet still exact,
    # because curl u = k u makes (u.grad)u a pure gradient as well
    abc = convective_acceleration(abc_flow(a), box=32.0, n=32)
    assert abc["curl_over_adv"] < 1e-10


# --------------------------------------------------------------------------- #
# layer 2: the LBM path meets the predeclared band
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.benchmark
@pytest.mark.physics
def test_case_meets_pool_tolerance() -> None:
    """Coarse-grid acceptance run: every predeclared quantity inside its band."""
    s = rc.build_verdict(grid=rc.SMOKE_GRID, with_grid=False, with_ref_conv=False,
                         save_field=False)
    got = {q["id"]: q for q in s["quantities"]}
    assert set(got) == {"Q0_series_vs_closed_form_rel", "Q1_energy_ratio_rel_err_pct",
                        "Q2_enstrophy_ratio_rel_err_pct", "Q3_velocity_l2_rel_err_pct",
                        "Q4_vorticity_l2_rel_err_pct", "Q5_series_truncation_rel",
                        "Q6_rho_deviation"}
    failed = [q for q in s["quantities"] if not q["pass"]]
    assert not failed, [(q["id"], q["got"], q["tolerance"]) for q in failed]
    # the run must be a real 3-D one: w != 0 in the IC, enstrophy decaying too
    assert got["Q2_enstrophy_ratio_rel_err_pct"]["got"] < 8.0
    assert s["legs"]["primary"]["n"] == rc.SMOKE_GRID
    assert s["legs"]["primary"]["ma"] > 0.05


# --------------------------------------------------------------------------- #
# layer 3: the committed evidence bundle
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.regression
def test_verdict_logic_rejects_out_of_band() -> None:
    """A verdict that cannot fail is not a verdict."""
    # direct check of the decision rule, both directions
    bad = rc._quantity("Q1_energy_ratio_rel_err_pct", 0.78, 8.01, "%")
    assert bad["pass"] is False
    ok = rc._quantity("Q1_energy_ratio_rel_err_pct", 0.78, 7.99, "%")
    assert ok["pass"] is True
    low = rc._quantity("Q7_observed_order", 1.5, 1.49, "-")
    assert low["pass"] is False
    high = rc._quantity("Q7_observed_order", 1.5, 1.51, "-")
    assert high["pass"] is True
    assert rc._quantity("Q6_rho_deviation", 0.0, 0.02, "rel")["pass"] is False


@pytest.mark.unit
@pytest.mark.benchmark
def test_committed_artifact_is_evidence() -> None:
    """The committed summary.json must carry the verdict AND its numbers."""
    path = ART / "summary.json"
    assert path.is_file(), "run run_case.py (not --smoke) to produce artifacts/summary.json"
    s = json.loads(path.read_text())

    assert s["case"] == rc.CASE
    assert s["protocol"]["tol_pct"] == rc.TOL_PCT
    assert s["protocol"]["grid_primary"] == rc.PRIMARY_GRID
    assert s["protocol"]["thresholds"], "thresholds must be recorded in the artifact"

    qs = {q["id"]: q for q in s["quantities"]}
    assert {"Q1_energy_ratio_rel_err_pct", "Q2_enstrophy_ratio_rel_err_pct",
            "Q3_velocity_l2_rel_err_pct", "Q4_vorticity_l2_rel_err_pct"} <= set(qs)
    for q in qs.values():
        # every quantity must bind a reference number to a measured number
        assert isinstance(q["ref"], (int, float)) and isinstance(q["got"], (int, float))
        assert q["tolerance"] > 0 and q["direction"] in ("le", "ge")
        bound = q["got"] <= q["tolerance"] if q["direction"] == "le" \
            else q["got"] >= q["tolerance"]
        assert q["pass"] is bound, q["id"]

    if s["verdict"]["pass"]:
        assert s["verdict"]["failed"] == []
        assert abs(qs["Q1_energy_ratio_rel_err_pct"]["got"]) < rc.TOL_PCT
        assert qs["Q7_observed_order"]["got"] >= rc.ORDER_MIN
        assert len(s["protocol"]["grids_ladder"]) >= 2
        # decay must be real, not a no-op: both ratios strictly below 1
        p = s["legs"]["primary"]
        assert p["energy_ratio_lbm"] < 1.0 and p["enstrophy_ratio_lbm"] < 1.0
    else:
        # an honest FAIL must name the failing quantities (no silent pass)
        assert s["verdict"]["failed"]

# --------------------------------------------------------------------------- #
# layer 1b: the *field-level* reference must be reproducible, and the series
#           must be cross-checked against an independent discretisation
# --------------------------------------------------------------------------- #
def test_reference_field_is_reproducible_across_work_grids() -> None:
    """Q11: the series *field* must not depend on the working grid nfft.

    Energy/enstrophy are modal sums and are nfft-invariant to 1e-16, so they
    cannot detect a field-level reconstruction problem: at the library default
    work grid (nfft = 5*order) the same series returns fields differing by
    several percent on an n = 24/32 lattice.  With nfft >= 3n (de-aliasing band
    covering the evaluation band) the field reproduces itself.
    """
    chk = rc.field_workgrid_check(n=rc.SMOKE_GRID)
    assert chk["field_workgrid_rel"] <= rc.FIELD_WORKGRID_REL
    assert chk["field_workgrid_rel"] < 0.1 * rc.TOL_PCT
    # the default work grid is *not* covered, and the defect is recorded
    assert chk["default_nfft_level"] is not None
    for lv in chk["levels"]:
        assert lv["nfft"] >= rc.FIELD_NFFT_FACTOR * rc.SMOKE_GRID
        assert lv["dealias_margin"] > 0.0


def test_dns_cross_check_agrees_with_series() -> None:
    """Q10: series vs an independent pseudo-spectral DNS of the same problem.

    The closed-form anchor is nonlinearly trivial (both closed forms have a zero
    or gradient-only convective term), so it cannot test the recursion's
    convective term.  The DNS can, and shares no code path with the series.
    """
    leg = rc.leg_dns(n=rc.SMOKE_GRID, substeps=(1, 2))
    assert leg["series_vs_dns_rel"] <= rc.DNS_XCHECK_REL
    # genuinely two different computations (not the same object twice)
    assert leg["series_vs_dns_rel"] > 0.0
    # the residual is attributed: refining the DNS time step barely moves it
    assert leg["dt_refinement_rel"] <= leg["series_vs_dns_rel"]
    assert leg["reynolds"] == rc.REYNOLDS


def test_verdict_fails_when_cross_checks_break() -> None:
    """The new gates must be able to fail, else they are decoration."""
    anchor = {"max_rel_err": 0.0}
    refconv = {"order_convergence_rel": 0.0, "workgrid_invariance_rel": 0.0,
               "field_workgrid": {"field_workgrid_rel": 1e-9}}
    primary = {"n": rc.SMOKE_GRID, "energy_ratio_series": 1.0,
               "energy_ratio_rel_err_pct": 0.0, "enstrophy_ratio_series": 1.0,
               "enstrophy_ratio_rel_err_pct": 0.0, "u_l2_ref": 1.0,
               "velocity_l2_rel_err_pct": 0.0, "vorticity_l2_rel_err_pct": 0.0,
               "series_truncation_rel": 0.0, "rho_deviation": 0.0, "history": []}
    good = {"series_vs_dns_rel": 1e-9}
    ok = rc.build_verdict(grid=rc.SMOKE_GRID, anchor=anchor, refconv=refconv,
                          primary=primary, grid_leg=None, dns=good)
    assert ok["verdict"]["pass"]
    bad_dns = dict(good, series_vs_dns_rel=1.0)
    v1 = rc.build_verdict(grid=rc.SMOKE_GRID, anchor=anchor, refconv=refconv,
                          primary=primary, grid_leg=None, dns=bad_dns)
    assert v1["verdict"]["failed"] == ["Q10_series_vs_dns_rel"]
    bad_fw = {"order_convergence_rel": 0.0, "workgrid_invariance_rel": 0.0,
              "field_workgrid": {"field_workgrid_rel": 6e-2}}
    v2 = rc.build_verdict(grid=rc.SMOKE_GRID, anchor=anchor, refconv=bad_fw,
                          primary=primary, grid_leg=None, dns=good)
    assert v2["verdict"]["failed"] == ["Q11_series_field_workgrid_rel"]


def test_committed_artifact_carries_cross_check_evidence() -> None:
    """The committed evidence must show Q10/Q11 numbers and the leg they came from."""
    s = json.loads((ART / "summary.json").read_text())
    qids = {q["id"] for q in s["quantities"]}
    assert {"Q10_series_vs_dns_rel", "Q11_series_field_workgrid_rel"} <= qids
    assert s["legs"]["dns"]["levels"], "dns leg has no per-level rows"
    assert s["legs"]["reference_convergence"]["field_workgrid"]["levels"]
    for q in s["quantities"]:
        if q["id"] in ("Q10_series_vs_dns_rel", "Q11_series_field_workgrid_rel"):
            assert q["got"] <= q["tolerance"]
