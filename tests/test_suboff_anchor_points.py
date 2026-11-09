"""Acceptance tests for the SUBOFF anchor-point pack (paper-exp lbm-suboff-tau, item A).

Artefact: ``benchmarks/verified/suboff_anchor/anchor_points.json``.
Builder:  ``benchmarks/verified/suboff_anchor/build_anchor_points.py``.

These tests enforce the mission's traceability discipline mechanically so that a
later edit cannot silently degrade it:

* every external number in the pack is parsed out of a verbatim quote that equals
  the cited line of an in-repo file (machine-checked hit rate must be 100%);
* the committed ``anchor_points.json`` is exactly reproducible from the builder;
* no float literal equal to a derived quantity (C_T / C_F / Re / (1+k) / h) exists
  in the builder (no third hand copy);
* Re = U*L/nu reproduces the transcribed 4-significant-digit ladder to <=0.1%;
* the ITTC-78 C_F recompute residual stays below 1e-12 and the (1+k) cross-anchor
  consistency below 1%;
* the reachability verdict is honest: the binary conclusion and the similarity
  error budget are present with numbers, and no PASS may be claimed for a
  measured value that breaks its predeclared threshold.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PACK = REPO / "benchmarks" / "verified" / "suboff_anchor"
BUILDER = PACK / "build_anchor_points.py"
ART = PACK / "anchor_points.json"
TRANS = PACK / "sources" / "anchor_points_transcriptions.json"
MEAS = PACK / "sources" / "cluster_mlups_measurement.json"


def _load_builder():
    spec = importlib.util.spec_from_file_location("suboff_anchor_points_builder", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder():
    return _load_builder()


@pytest.fixture(scope="module")
def art(builder):
    return builder.build()


def test_pack_files_exist():
    assert BUILDER.is_file()
    assert ART.is_file()
    assert TRANS.is_file()
    assert MEAS.is_file()


def test_committed_artifact_is_exactly_reproducible(art):
    committed = json.loads(ART.read_text(encoding="utf-8"))
    assert committed == json.loads(json.dumps(art, ensure_ascii=False)), (
        "anchor_points.json is stale — rerun "
        "python3 benchmarks/verified/suboff_anchor/build_anchor_points.py"
    )


def test_c1_at_least_five_points_with_verbatim_locators(art):
    pts = art["anchor_points"]
    assert len(pts) >= 5
    for p in pts:
        assert p["quotes"], p["point_id"]
        for q in p["quotes"]:
            assert q["file"] and q["line"] > 0
            assert q["locator"]
    hit = art["external_quotes"]["hit_rate_pct"]
    assert hit == 100.0
    assert art["criteria_summary"]["C1"] == "PASS"


def test_quotes_are_verbatim_lines_of_the_cited_files(art):
    for q in art["external_quotes"]["quotes"].values():
        path = REPO / q["file"]
        lines = path.read_text(encoding="utf-8").split("\n")
        assert lines[q["line"] - 1] == q["text"], q["id"]
        assert q["machine_check"]["regex_hit"] is True


def test_c2_reynolds_consistency(art):
    devs = [abs(p["re_rel_dev_pct"]) for p in art["anchor_points"]]
    assert max(devs) <= 0.1
    assert art["criteria_summary"]["C2"] == "PASS"
    # the retired two-significant-digit anchor must stay retired and explained
    ret = art["reynolds_consistency"]["retired_two_sig_fig_anchor"]
    assert ret["verdict"].startswith("RETIRED")
    assert abs(ret["rel_dev_pct"]) > 0.1      # the retired anchor genuinely misses the bar


def test_c3a_ittc78_residual_below_1e_12(art):
    resid = float(art["ittc78_cf_recompute"]["max_abs_residual"])
    assert resid < 1e-12
    # independent re-evaluation of the closed form at the lowest anchor
    p0 = art["anchor_points"][0]
    coef = art["conventions"]["ittc78_cf"]["coefficient"]
    shift = art["conventions"]["ittc78_cf"]["log10_shift"]
    expo = art["conventions"]["ittc78_cf"]["exponent"]
    assert abs(coef / (__import__("math").log10(p0["Re_computed"]) - shift) ** expo
               - p0["CF_ittc78"]) < 1e-12


def test_c3b_one_plus_k_cross_anchor_consistency(art):
    cross = art["one_plus_k_consistency"]["cross_anchor"]
    assert cross["max_abs_rel_diff_pct"] <= 1.0
    assert len(cross["pairs"]) >= 4
    for pr in cross["unpaired"]:
        assert pr["nearest_speed_match_rel_pct"] > cross["pairing_tolerance_pct"]
    assert art["criteria_summary"]["C3b"] == "PASS"


def test_c4_reachability_is_binary_and_quantified(art):
    reach = art["reachability"]
    bc = reach["binary_conclusion"]
    assert bc["verdict"] in {"NOT REACHABLE at credible resolution",
                             "REACHABLE within budget"}
    assert isinstance(bc["REACHABLE_AT_CREDIBLE_RESOLUTION"], bool)
    assert bc["credible_wmles_hours_optimistic_rate"] > 0
    # verdict must agree with the numbers and the predeclared budget
    assert bc["REACHABLE_AT_CREDIBLE_RESOLUTION"] == (
        bc["credible_wmles_hours_optimistic_rate"] <= bc["budget_hours"])
    # the finest executable grid in budget must be reported with h and compute
    for key, d in reach["finest_executable_grid_within_budget"].items():
        assert d["h_m"] > 0 and d["cells"] > 0 and d["wall_hours"] > 0
        assert d["verdict"]


def test_c4b_similarity_budget_formula_and_numbers(art):
    sb = art["similarity_error_budget"]
    assert "E_line" in sb["formula"] and "E_k" in sb["formula"]
    assert sb["evaluations"]
    for ev in sb["evaluations"]:
        assert ev["similarity_correction_pct_of_CT"] < 0      # low-Re sim over-predicts
        assert ev["E_total_quantified_pct"] > 0
        # honest reporting: the conversion path may not be claimed to support 3%
        assert ev["supports_band_pct"] == (ev["E_total_quantified_pct"] <= 3.0)
    assert art["criteria_summary"]["C4"] == "PASS"


def test_no_hand_copied_constants(art):
    assert art["no_hand_copy"]["pass"] is True
    assert art["no_hand_copy"]["violations"] == []


def test_verdicts_are_honest(art):
    """A FAIL must not be hidden: every criterion's measured value is re-checked."""
    thr = art["criteria_predeclared"]["thresholds"]
    by_id = {c["id"]: c for c in art["criteria"]}
    if by_id["C2"]["measured"]["max_abs_re_rel_dev_pct"] > thr["c2_re_rel_dev_max_pct"]:
        assert by_id["C2"]["verdict"] == "FAIL"
    if float(by_id["C3a"]["measured"]["max_abs_residual"]) >= thr["c3_cf_residual_max_abs"]:
        assert by_id["C3a"]["verdict"] == "FAIL"
    if by_id["C3b"]["measured"]["max_abs_rel_diff_pct"] > thr["c3_one_plus_k_consistency_max_pct"]:
        assert by_id["C3b"]["verdict"] == "FAIL"
    assert art["overall_verdict"] == ("PASS" if all(c["verdict"] == "PASS"
                                                    for c in art["criteria"]) else "FAIL")