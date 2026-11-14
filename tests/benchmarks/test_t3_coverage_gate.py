"""CI contract for the T3 nine-case coverage gate (GAP-LBM-T3-BENCH).

These tests are fast and solver-free: they join the frozen matrix verdicts with
the T3 slot contract and assert that the headline coverage number is (a) what
the contract produces, (b) frozen on disk, and (c) not inflated by substitute
evidence.  They also pin the turbulent-channel gate reproduction, so the
recorded FAIL number cannot drift away from its own evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.validation_matrix import t3_coverage

REPO_ROOT = Path(__file__).resolve().parents[2]
VIRTUAL_MATRIX_SLOTS = {
    "cavity",
    "cylinder",
    "sphere",
    "channel",
    "taylor_green",
    "static_droplet",
    "rayleigh_taylor",
    "dam_break",
    "ship",
}


@pytest.fixture(scope="module")
def report() -> dict:
    return t3_coverage.coverage()


def test_contract_declares_the_nine_lineages() -> None:
    matrix = t3_coverage.load_matrix()
    contract = t3_coverage.load_contract()
    slots = {s["slot"] for s in contract["slots"]}
    assert slots == VIRTUAL_MATRIX_SLOTS
    assert {c["slot"] for c in matrix["cases"]} == VIRTUAL_MATRIX_SLOTS
    attrs = contract["case_attributes"]
    ids = {c["id"] for c in matrix["cases"]}
    assert ids == set(attrs)
    for cid, a in attrs.items():
        assert a["physics_class"] and a["level_basis"]
        assert a["components"], f"{cid}: no components"


def test_every_slot_requires_a_declared_physics_class(report: dict) -> None:
    for row in report["slots"]:
        assert row["requires"]["physics_class"]
        assert row["components"], row["slot"]
        assert row["state"] in ("verified", "partial", "gap")


def test_laminar_channel_is_substitute_only(report: dict) -> None:
    channel = next(r for r in report["slots"] if r["slot"] == "channel")
    assert channel["state"] == "gap"
    subs = {s["case"] for s in channel["substitute_evidence"]}
    assert "channel_core_path_ci" in subs
    # the substitute is a genuine pass, just the wrong physics for this lineage
    assert all(s["physics_class"] != "wall_bounded_turbulent" for s in channel["substitute_evidence"])


def test_failed_and_pending_cases_never_credit_a_slot(report: dict) -> None:
    for row in report["slots"]:
        credited = {
            e["case"]
            for comp in row["components"].values()
            for e in comp["evidence"]
        }
        blocked = {b["case"] for b in row["blockers"]}
        assert not (credited & blocked), f"{row['slot']}: {credited & blocked}"


def test_resistance_proxy_does_not_verify_the_ship_lineage(report: dict) -> None:
    ship = next(r for r in report["slots"] if r["slot"] == "ship")
    assert ship["state"] == "partial"
    assert ship["components"]["resistance"]["state"] == "proxy"
    assert ship["components"]["wave"]["state"] == "missing"


def test_headline_is_not_the_case_count(report: dict) -> None:
    matrix = t3_coverage.load_matrix()
    passed_cases = sum(1 for c in matrix["cases"] if c.get("declared_status") != "failed")
    headline = report["headline"]
    assert headline["slots"] == 9
    assert headline["verified"] + headline["partial"] + headline["gap"] == 9
    # 6/9 verified is the honest slot number; it must not silently become 8/9
    assert headline["verified"] == 6
    assert passed_cases >= headline["verified"]


def test_frozen_coverage_matches_fresh_computation() -> None:
    frozen = json.loads((REPO_ROOT / "benchmarks/validation_matrix/T3_COVERAGE.json").read_text(encoding="utf-8"))
    fresh = t3_coverage.coverage()
    for key in ("headline", "slots", "matrix_version"):
        assert frozen[key] == fresh[key], f"T3_COVERAGE.json drift in {key}; rerun t3_coverage.py"


def test_status_matrix_markdown_matches_fresh_verdicts() -> None:
    """Doc drift gate: the published status counts must equal the machine state."""
    from benchmarks.validation_matrix import runner

    matrix = runner.load_matrix()
    verdicts = [runner.evaluate_case(c) for c in matrix["cases"]]
    counts = {s: sum(1 for v in verdicts if v["status"] == s) for s in ("passed", "failed", "pending")}
    text = (REPO_ROOT / "benchmarks/validation_matrix/STATUS_MATRIX.md").read_text(encoding="utf-8")
    assert f"passed {counts['passed']} / failed {counts['failed']} / pending {counts['pending']}" in text


def test_channel_gate_number_reproduces_from_its_evidence() -> None:
    from benchmarks.validation_matrix import channel_budget

    out = channel_budget.build()
    rep = out["reproduction"]
    assert rep["pass"] is True
    assert rep["delta_pp"] <= channel_budget.REPRO_TOL_PP
    assert rep["recomputed_rms_pct"] == pytest.approx(rep["recorded_rms_pct"], abs=0.01)
    # normalisation sensitivity is material and must stay visible
    conv = out["uplus_conventions"]
    assert conv["rel_offset0.5_measured_u_tau"]["rms_pct"] > 2.0 * conv["rel_offset0.5_target_u_tau"]["rms_pct"]
    # budget diagnostic: the outer balance closes much better than the full-range one
    ranges = out["budget_diagnostic"]["ranges"]
    assert ranges["core[10,ny-11]"]["residual_pct_of_tau_w"] < ranges["full[1,ny-2]"]["residual_pct_of_tau_w"]


def test_channel_measured_ladder_is_visible_in_coverage(report: dict) -> None:
    """The slot record must show the measured 3-level FAIL ladder, not 'unresolved'."""
    channel = next(r for r in report["slots"] if r["slot"] == "channel")
    ladder = channel["measured_ladder"]
    levels = [lv["level"] for lv in ladder["levels"]]
    assert levels == ["L1", "L2", "L3"]
    rms = [lv["rms_y30_100_pct"] for lv in ladder["levels"]]
    assert rms == sorted(rms, reverse=True), "the FAIL ladder is monotone decreasing in the artifact"
    assert min(rms) > 3.0, "the best level must stay outside the 3% gate"
    assert ladder["gate_reproduction"]["delta_pp"] <= 0.01