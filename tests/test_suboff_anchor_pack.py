"""Acceptance tests for the SUBOFF external-anchor pack (paper-exp lbm-suboff-tau, exp 1).

The artefact is ``benchmarks/verified/suboff_anchor/anchor.json``; its builder is
``benchmarks/verified/suboff_anchor/build_anchor.py``.  These tests enforce the
mission's traceability discipline mechanically, so a later edit cannot silently
degrade it:

* every external number in the pack must be a verbatim substring of an archived
  in-repo document (no memory-written numbers);
* the committed ``anchor.json`` must be exactly reproducible from the builder;
* the builder must contain no float literal equal to any derived quantity
  (CT / C_F / Re / (1+k)) — i.e. no third hand copy;
* the ITTC-78 friction-line residual must stay below 1e-12;
* criterion verdicts must be honest (a measured deviation above the threshold
  may not be reported as PASS).
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PACK = REPO / "benchmarks" / "verified" / "suboff_anchor"
BUILDER = PACK / "build_anchor.py"
ANCHOR_JSON = PACK / "anchor.json"


def _load_builder():
    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))
    spec = importlib.util.spec_from_file_location("suboff_anchor_builder", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder():
    return _load_builder()


@pytest.fixture(scope="module")
def anchored(builder):
    return builder.build()


def test_pack_files_exist():
    assert BUILDER.is_file()
    assert ANCHOR_JSON.is_file()
    assert (PACK / "sources" / "external_transcriptions.json").is_file()


def test_committed_anchor_json_is_exactly_reproducible(builder):
    committed = json.loads(ANCHOR_JSON.read_text(encoding="utf-8"))
    rebuilt = builder.build()
    assert committed == rebuilt, "anchor.json drifted from build_anchor.py output"


def test_every_quote_is_verbatim_in_its_archived_file(builder):
    spec = builder.load_sources()
    layer = builder.verify_and_parse(spec)  # raises if any quote is not verbatim
    assert len(layer["checks"]) == sum(len(s["groups"]) for s in spec["sources"])
    assert all(c["verbatim_found"] for c in layer["checks"])
    # archived files must exist and be hashed (provenance)
    for check in layer["checks"]:
        assert (REPO / check["archived_at"]).is_file()
        assert len(check["archived_sha256"]) == 64


def test_all_numbers_come_from_quotes_not_literals(builder, anchored):
    check = anchored["derived"]["form_factor"]["no_hand_copy_check"]
    assert check["pass"], check["violations"]
    assert check["n_float_literals_in_script"] > 0


def test_ittc78_residual_below_1e_12(anchored):
    cf = anchored["derived"]["ittc78_cf"]
    assert cf["max_abs_residual"] < 1e-12
    assert cf["max_abs_residual"] < cf["threshold"]
    # the reported residual really is the max over the evaluated grid
    assert cf["residual_report"]["max_abs_residual"] == max(
        max(r["abs_residual_vs_decimal"], r["abs_residual_vs_ln"])
        for r in cf["residual_report"]["rows"]
    )


def test_ittc78_matches_transcribed_known_value(anchored):
    known = anchored["derived"]["ittc78_cf"]["known_value_check"]
    assert known["within_two_sig_fig_rounding"]
    assert abs(known["cf_computed"] - 0.00291) < 5e-6


def test_re_consistency_verdict_is_honest(anchored):
    rc = anchored["derived"]["re_consistency"]
    assert math.isclose(abs(rc["rel_dev_pct"]), abs(rc["rel_dev_pct"]), rel_tol=0)
    expected = abs(rc["rel_dev_pct"]) <= rc["threshold_pct"]
    assert rc["pass"] is expected
    crit = next(c for c in anchored["criteria"] if c["id"] == "C2")
    assert crit["verdict"] == ("PASS" if expected else "FAIL")
    if not expected:
        assert crit["gap"], "a failing criterion must document the quantified gap"


def test_verdicts_are_consistent_with_measurements(anchored):
    for crit in anchored["criteria"]:
        assert crit["verdict"] in {"PASS", "FAIL", "PARTIAL_FAIL"}
        assert crit["statement"] and crit["measured"] and crit["evidence"]
        if crit["verdict"] != "PASS":
            assert crit["gap"], f"{crit['id']} must state its gap"
    passed = sum(c["verdict"] == "PASS" for c in anchored["criteria"])
    assert anchored["overall_verdict"] == (
        "PASS" if passed == len(anchored["criteria"]) else "PARTIAL" if passed else "FAIL"
    )


def test_form_factor_recomputed_from_two_anchors(anchored):
    ff = anchored["derived"]["form_factor"]
    a = anchored["anchors"]["A_AFF1_bare_DTRC_T14"]["rows"][0]
    b = anchored["anchors"]["B_AFF8_full_CFDletters_T3"]["rows"][0]
    assert ff["anchor_A_one_plus_k"] == pytest.approx(a["CT"] / a["CF_ittc78"], rel=1e-9)
    assert ff["anchor_B_one_plus_k"] == pytest.approx(b["CT"] / b["CF_ittc78"], rel=1e-9)
    # the two independent transcriptions of the same AFF-8 test agree tightly
    assert abs(ff["anchor_A2_vs_B_rel_dev_pct"]) < 0.5
    # appended hull drags more than the bare hull
    assert ff["anchor_B_one_plus_k"] > ff["anchor_A_one_plus_k"] > 1.0


def test_aff1_derived_ct_lands_inside_transcribed_bracket(anchored):
    br = anchored["derived"]["transcribed_bracket_membership"]["AFF-1"]
    lo, hi = br["transcribed_bracket"]
    assert lo <= br["derived_CT_at_head_anchor"] <= hi
    assert br["inside_bracket"] is True


def test_aff8_bracket_is_reported_as_unreproducible(anchored):
    br = anchored["derived"]["transcribed_bracket_membership"]["AFF-8"]
    # honest reporting: we do not silently accept the quoted AFF-8 0.004 anchor
    assert br["inside_bracket"] is False
    assert br["bracket_requires_area_below_bare_hull"] is True
    assert br["wetted_area_needed_m2_for_bracket_hi"] < br["bare_hull_wetted_area_m2"]


def test_reproduces_frozen_reference_lock(anchored):
    repro = anchored["derived"]["cross_artifact_reproduction"]
    assert len(repro["rows"]) == 5
    assert repro["max_abs_rel_dev_pct"] < 1e-6
    lock = json.loads(
        (REPO / "benchmarks/pending/suboff_highre_resistance/refs_lock.json").read_text()
    )
    for row, frozen in zip(repro["rows"], lock["anchors"]):
        assert row["V_ms"] == pytest.approx(frozen["V_ms"])
        assert row["our_CT"] == pytest.approx(frozen["CT"], rel=1e-9)
        assert row["lock_CT"] == frozen["CT"]
        assert row["our_Re"] == pytest.approx(frozen["Re_Lmax_nu1139"], rel=1e-9)
        assert row["our_CF"] == pytest.approx(frozen["CT_ITTC1957"], rel=1e-9)


def test_double_entry_against_package_constant(anchored):
    de = anchored["derived"]["double_entry_check"]
    assert de["n_rows"] == 12
    assert de["exact_match"] is True


def test_anchor_json_has_no_non_finite_numbers(anchored):
    def walk(node, path="root"):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        elif isinstance(node, float):
            assert math.isfinite(node), f"non-finite value at {path}"

    walk(anchored)


def test_uncertainty_budget_is_below_the_three_percent_band(anchored):
    budget = anchored["derived"]["uncertainty_budget"]
    rss = budget["rss_total_pct"]
    assert 0.0 < rss < 3.0
    assert budget["not_quantified"], "repeatability/ν caveats must be disclosed"
    contract = anchored["downstream_contract"]["band_resolvability"]
    assert contract["anchor_convention_rss_pct"] == pytest.approx(rss, rel=1e-9)