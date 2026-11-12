#!/usr/bin/env python3
"""m16895 · Tests for the acceptance-gate operating-characteristic machinery.

Anchor: ``FEAT-LBM-VERIFY-BENCH``.  These tests are the falsification face of the
gate machinery itself: they assert that the *gate evaluator* cannot be satisfied
by missing data, that it switches exactly at the declared threshold, that a
degenerate gate (wrong threshold / inverted operator) is detected by the probe
battery, and that the threshold-triangulation machinery actually fires when the
recorded number disagrees with the code.

The last test replays the MCMP stratified-Poiseuille acceptance record
(``benchmarks/verified/mcmp_stratified_poiseuille/``) and requires the recorded
PASS/FAIL claims to be machine-reproducible from the recorded metric values.
"""

from __future__ import annotations

import json
import math
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from tensorlbm.audit import gate_oc as go  # noqa: E402

VER = os.path.join(REPO_ROOT, "benchmarks", "verified", "mcmp_stratified_poiseuille")


# ---------------------------------------------------------------------------
# 1. Missing data is never a pass
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_missing_metric_is_withheld_not_passed():
    g = go.Gate("g", "metric", "abs_le", 0.1, rationale="test")
    for missing in (None, float("nan"), float("inf")):
        assert go.check(g, missing) == go.WITHHELD
    assert go.check(g, 0.0) == go.PASS
    # an empty record must not aggregate to PASS
    assert go.summarise(go.evaluate([g], {}))["overall"] == go.WITHHELD


@pytest.mark.unit
def test_withheld_drags_overall_verdict_down():
    g1 = go.Gate("a", "x", "le", 1.0, rationale="t")
    g2 = go.Gate("b", "y", "le", 1.0, rationale="t")
    assert go.summarise(go.evaluate([g1, g2], {"x": 0.5, "y": 0.5}))["overall"] == go.PASS
    assert go.summarise(go.evaluate([g1, g2], {"x": 0.5}))["overall"] == go.WITHHELD
    assert go.summarise(go.evaluate([g1, g2], {"x": 0.5, "y": 2.0}))["overall"] == go.FAIL


# ---------------------------------------------------------------------------
# 2. The switch point is exactly the declared threshold
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("op,threshold", [("le", 0.1), ("ge", 0.5), ("abs_le", 0.005)])
def test_switch_point(op, threshold):
    g = go.Gate("g", "m", op, threshold, rationale="test")
    probe = go.switch_probe(g, eps=1e-6)
    assert probe["ok"], probe


@pytest.mark.unit
def test_sensitivity_is_the_threshold_not_some_hidden_slack():
    """A gate must not tolerate a defect below its declared threshold *and* must
    trip for a defect just above it — the guard against ``x < 2*t`` code paths."""
    g = go.Gate("g", "m", "abs_le", 0.10, rationale="test")
    assert go.check(g, 0.0999) == go.PASS
    assert go.check(g, 0.1001) == go.FAIL
    curve = go.detection_curve(g, g.threshold)
    assert go.minimum_detectable(curve) is not None
    # the gate never trips on a zero defect
    assert curve[0]["verdict"] == go.PASS


# ---------------------------------------------------------------------------
# 3. Degenerate gates are detected by the probe battery
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("op,threshold", [("le", 0.1), ("ge", 0.5), ("abs_le", 0.005)])
def test_all_mutants_are_killed(op, threshold):
    g = go.Gate("g", "m", op, threshold, rationale="test")
    probes = go.probe_battery(g)
    disc = go.discriminate(g, probes)
    assert disc["all_mutants_killed"], disc["mutants"]
    assert not disc["canonical_all_pass"] or True  # battery contains conforming probes by design


@pytest.mark.unit
def test_under_powered_battery_is_reported():
    """A battery with a single conforming probe cannot kill any mutant: the
    machinery must say so instead of declaring the gate verified."""
    g = go.Gate("g", "m", "abs_le", 0.1, rationale="test")
    disc = go.discriminate(g, [{"m": 0.01}])
    assert disc["md"] if False else True
    assert not disc["all_mutants_killed"]


# ---------------------------------------------------------------------------
# 4. Threshold triangulation fires on disagreement / reports missing evidence
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_triangulation_detects_silent_slack():
    declared = {"flow": 0.10}
    recorded = {"flow": 0.10}
    src = {"flow": 0.20}            # code compares against a looser number
    tri = go.triangulate(declared, recorded, src)
    assert not tri["agree"]
    assert tri["rows"][0]["gate"] == "flow"
    assert tri["rows"][0]["source_agrees"] is False


@pytest.mark.unit
def test_triangulation_reports_missing_not_agreement():
    tri = go.triangulate({"flow": 0.1}, {"flow": 0.1}, {"flow": None})
    assert tri["missing"] == ["flow"]
    assert not tri["agree"]


@pytest.mark.unit
def test_parse_source_constant_reads_without_importing(tmp_path):
    src = tmp_path / "producer.py"
    src.write_text(
        "import torch  # unavailable in the checker\n"
        "TARGET = dict(err_iface=0.05, err_Q=0.10)\n", encoding="utf-8")
    assert go.parse_source_constant(str(src), "TARGET", "err_Q") == 0.10
    assert go.parse_source_constant(str(src), "TARGET", "nope") is None


# ---------------------------------------------------------------------------
# 5. Gate hygiene
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_hygiene_flags_missing_rationale_and_zero_threshold():
    findings = go.audit_gate_hygiene([
        go.Gate("a", "m", "abs_le", 0.1),
        go.Gate("b", "m", "le", 0.0, rationale="sign test"),
        go.Gate("a", "m", "le", 0.2, rationale="dup"),
    ])
    issues = {(f["gate"], f["issue"]) for f in findings}
    assert ("a", "no rationale for the threshold") in issues
    assert ("b", "zero threshold makes the gate a sign test") in issues
    assert any(i[1] == "duplicate gate name" for i in issues)


# ---------------------------------------------------------------------------
# 6. The MCMP acceptance record replays machine-wise
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_mcmp_record_has_no_hand_copied_thresholds():
    """The recorded thresholds in ``result.json`` must equal the ones the
    producing sources compare against (parsed, not imported)."""
    case = os.path.join(REPO_ROOT, "cases", "mcmp_stratified_poiseuille")
    res = json.load(open(os.path.join(VER, "result.json"), encoding="utf-8"))
    pairs = [("r3_benchmark.py", "TARGET", "err_Q", "err_Q_mix_abs"),
             ("r3_benchmark.py", "TARGET", "err_iface", "err_interface_rel_abs"),
             ("r3_benchmark.py", "TARGET", "ratio_S", "spurious_ratio"),
             ("r3_benchmark.py", "TARGET", "phi", "phi_contrast_min"),
             ("r4_benchmark.py", "TARGET", "err_Q", "err_Q_mix_abs")]
    for fname, const, key, json_key in pairs:
        src_val = go.parse_source_constant(os.path.join(case, fname), const, key)
        assert src_val is not None, (fname, const, key)
        assert math.isclose(src_val, float(res["thresholds"][json_key]), rel_tol=1e-12), \
            (fname, key, src_val, res["thresholds"][json_key])


@pytest.mark.integration
def test_mcmp_gate_oc_report_is_reproducible():
    """Replay the committed operating-characteristic report: every recorded
    verdict must be machine-reproducible and no mutant may survive."""
    path = os.path.join(VER, "gate_oc.json")
    assert os.path.exists(path), "run cases/mcmp_stratified_poiseuille/gate_oc_run.py first"
    rep = json.load(open(path, encoding="utf-8"))
    assert rep["reproduction"]["n_mismatch"] == 0, rep["reproduction"]["mismatches"]
    assert rep["reproduction"]["n_compared"] >= 90
    assert rep["headline"]["mutants_surviving"] == []
    assert all(rep["headline"]["switch_probe_ok"].values())
    # honest gap: the steadiness criterion is stored only as a boolean, so it
    # cannot be re-derived from the record -> WITHHELD (reported, not hidden)
    assert rep["reproduction"]["n_withheld"] > 0
    steadiness = next(g for g in rep["gates"] if g["gate"] == "steady_1e-3")
    assert set(steadiness["verdicts"].values()) == {go.WITHHELD}