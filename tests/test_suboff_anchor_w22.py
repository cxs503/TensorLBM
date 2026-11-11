"""W22 (m14504) — invariants for the SUBOFF AFF-8 anchor transcription + ITTC-57 closure.

The tests do not re-assert the artefact's own numbers against themselves where a
stronger independent check is available:
  * readback of transcribed values is re-implemented here and applied to the sources;
  * the friction line is checked against analytically known values;
  * the Prohaska estimator is checked on a synthetic ladder with a known intercept;
  * the closure arithmetic is recomputed from the artefact's own transcribed leaf values.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CASE = REPO / "benchmarks/verified/suboff_anchor_w22"
TRANS_PATH = CASE / "anchor_transcription.json"
CLOS_PATH = CASE / "ittc57_closure.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def artifacts():
    if not (TRANS_PATH.is_file() and CLOS_PATH.is_file()):
        pytest.fail("W22 artifacts missing — run build_anchor_transcription.py then "
                    "build_ittc57_closure.py")
    return (json.loads(TRANS_PATH.read_text(encoding="utf-8")),
            json.loads(CLOS_PATH.read_text(encoding="utf-8")))


def resolve(doc, pointer: str):
    if pointer == "":
        return doc
    node = doc
    for raw in pointer.split("/")[1:]:
        tok = raw.replace("~1", "/").replace("~0", "~")
        node = node[int(tok)] if isinstance(node, list) else node[tok]
    return node


# ------------------------------------------------------------------ transcription
def test_transcription_completeness_and_no_duplicate_pointers(artifacts):
    trans, _ = artifacts
    assert trans["completeness"]["completeness_pct"] == 100.0
    assert trans["completeness"]["n_missing_or_mismatched"] == 0
    assert trans["duplicate_pointers"] == []
    assert trans["n_leaves_total"] == len(trans["leaves"])
    assert trans["n_leaves_total"] == trans["completeness"]["n_expected"]
    for src in trans["sources"]:
        assert src["n_readback_mismatch"] == 0
        assert src["n_scalar_leaves"] > 0
        assert len(src["sha256"]) == 64


def test_readback_reimplemented_against_sources(artifacts):
    """Independent readback: re-open each source and compare a stride sample."""
    trans, _ = artifacts
    by_source = {}
    for leaf in trans["leaves"]:
        by_source.setdefault(leaf["source_id"], []).append(leaf)
    json_docs = {}
    md_lines = {}
    checked = 0
    for sid, leaves in by_source.items():
        sample = leaves[:: max(1, len(leaves) // 25)][:25]
        for leaf in sample:
            path = REPO / leaf["source_path"]
            if path.suffix == ".json":
                if sid not in json_docs:
                    json_docs[sid] = json.loads(path.read_text(encoding="utf-8"))
                got = resolve(json_docs[sid], leaf["pointer"])
                assert type(got) is type(leaf["value"]), (sid, leaf["pointer"])
                assert got == leaf["value"], (sid, leaf["pointer"])
            else:
                if sid not in md_lines:
                    md_lines[sid] = path.read_text(encoding="utf-8").splitlines()
                loc = leaf["locator"]
                raw = md_lines[sid][loc["line"] - 1][loc["col"] - 1:
                                                    loc["col"] - 1 + len(loc["raw"])]
                assert raw == loc["raw"], (sid, leaf["pointer"])
                assert float(raw) == leaf["value"], (sid, leaf["pointer"])
            checked += 1
    assert checked >= 50


def test_transcription_builder_is_deterministic(tmp_path):
    out = tmp_path / "trans.json"
    env = dict(os.environ, W22_TRANSCRIPTION_OUT=str(out))
    proc = subprocess.run(
        [sys.executable, str(CASE / "build_anchor_transcription.py")],
        cwd=REPO, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    fresh = json.loads(out.read_text(encoding="utf-8"))
    committed = json.loads(TRANS_PATH.read_text(encoding="utf-8"))
    assert fresh["leaves"] == committed["leaves"]
    assert fresh["sources"] == committed["sources"]


# --------------------------------------------------------------------- ITTC-1957
def test_ittc57_matches_analytic_values():
    mod = load_module("w22_closure", CASE / "build_ittc57_closure.py")
    assert mod.cf_ittc57_log10(1.0e5) == pytest.approx(0.075 / 9.0, rel=0, abs=1e-15)
    assert mod.cf_ittc57_log10(1.0e6) == pytest.approx(0.0046875, rel=0, abs=1e-15)
    assert mod.cf_ittc57_log10(1.0e7) == pytest.approx(0.003, rel=0, abs=1e-15)
    assert mod.cf_ittc57_log10(100.0) == math.inf          # documented pole
    for re in (1.0e4, 1.0e5, 1.0e6, 1.0e7):
        assert mod.cf_ittc57_ln(re) == pytest.approx(mod.cf_ittc57_log10(re), rel=1e-14)
        assert mod.cf_ittc57_decimal(re) == pytest.approx(mod.cf_ittc57_log10(re), rel=1e-14)


def test_prohaska_recovers_synthetic_intercept():
    mod = load_module("w22_closure2", CASE / "build_ittc57_closure.py")
    a_true, b_true = 1.32, 0.05
    pts = []
    for i, fr in enumerate((0.4, 0.55, 0.7, 0.85, 1.0)):
        cf = 0.003 * (1.0 + 0.05 * i)
        pts.append({"Fr": fr, "CF": cf, "CT": cf * (a_true + b_true * fr ** 4)})
    fit = mod.prohaska_fit(pts, 4.0)
    assert fit["intercept"] == pytest.approx(a_true, rel=1e-12)
    assert fit["slope"] == pytest.approx(b_true, rel=1e-10)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-12)


# ------------------------------------------------------------------------ closure
def _transcribed(trans: dict, sid: str, pointer: str):
    for leaf in trans["leaves"]:
        if leaf["source_id"] == sid and leaf["pointer"] == pointer:
            return leaf["value"]
    raise KeyError(f"{sid}{pointer}")


def test_closure_arithmetic_recomputed_from_transcribed_leaves(artifacts):
    trans, clos = artifacts
    cal = clos["closure"]["caliber_A_primary"]
    assert cal["C_W"] == 0.0
    for row in cal["per_tier"]:
        tier = row["tier"]
        c_t = _transcribed(trans, "S2", f"/tiers/{tier}/cd_tot_window")
        assert c_t == row["C_T"]
        assert row["C_T_predicted"] == pytest.approx((1.0 + cal["k"]) * cal["C_F"], rel=0, abs=0)
        residual = (c_t - (1.0 + cal["k"]) * cal["C_F"]) / c_t * 100.0
        assert residual == pytest.approx(row["residual_pct"], rel=1e-12, abs=1e-12)
        assert row["required_C_W_to_close"] < 0.0          # physically impossible to close
        assert abs(row["residual_pct"]) > clos["criteria_predeclared"]["C3_closure_residual_pct"]
        assert row["pass_3pct"] is False


def test_closure_fails_but_secondary_calibers_are_disclosed(artifacts):
    _, clos = artifacts
    assert clos["closure"]["caliber_A_primary"]["all_pass"] is False
    assert clos["closure"]["caliber_B"]["all_pass"] is False
    assert clos["closure"]["caliber_C"]["all_pass"] is False
    # definitional caliber must be flagged non-evidential and numerically zero
    d = clos["closure"]["caliber_D_definitional"]
    assert d["evidential"] is False
    assert d["max_abs_residual_pct"] < 1e-6
    # C_T caliber sensitivity must not be omitted
    sens = clos["closure"]["C_T_caliber_sensitivity"]
    assert len(sens) == 6
    assert all(x["abs_residual_pct"] > 3.0 for x in sens)


def test_case_reference_is_laminar_and_ittc_not_applicable(artifacts):
    trans, clos = artifacts
    assert "Blasius" in _transcribed(trans, "S7", "/reference/formula")
    note = _transcribed(trans, "S7", "/reference/cross_checks_recorded_not_applicable")
    assert "ittc1957" in note
    gap = clos["case_caliber_gap"]
    assert gap["Re_case"] == 1000.0
    assert gap["rel_dev_pct"] == pytest.approx(78.592, abs=5e-3)


def test_k_two_calibers_are_reported_on_both_bases(artifacts):
    _, clos = artifacts
    k = clos["k_identifiability"]
    rec = k["record"]["primary"]
    pro = k["prohaska"]["k_prohaska"]
    assert k["prohaska"]["diff_pct_k_basis"] == pytest.approx(
        100.0 * (pro - rec) / rec, rel=1e-10)
    assert k["prohaska"]["diff_pct_one_plus_k_basis"] == pytest.approx(
        100.0 * ((1 + pro) - (1 + rec)) / (1 + rec), rel=1e-10)
    c4 = [c for c in clos["criteria"] if c["id"] == "C4"][0]
    assert c4["verdict"] == "FAIL"                       # literal k basis, predeclared primary
    assert abs(k["prohaska"]["diff_pct_k_basis"]) > 5.0
    assert abs(k["prohaska"]["diff_pct_one_plus_k_basis"]) < 5.0   # disclosed, not hidden
    # window sensitivity must be published in full (not just the passing window)
    windows = k["prohaska"]["window_sensitivity"]
    assert {w["n_points"] for w in windows} == {3, 4}
    assert any(w["pass_k_basis"] for w in windows) and any(not w["pass_k_basis"] for w in windows)


def test_double_entry_of_recorded_form_factor(artifacts):
    _, clos = artifacts
    rec = clos["k_identifiability"]["record"]
    assert rec["double_entry_rel_dev_pct"] == pytest.approx(0.0, abs=1e-6)
    assert rec["primary"] == pytest.approx(1.255210948127971 - 1.0, rel=0, abs=1e-12)


def test_cw_zero_claim_is_evidence_backed(artifacts):
    _, clos = artifacts
    cw = clos["closure"]["C_W_assumption"]
    assert cw["value_used"] == 0.0
    assert cw["n_hits_in_transcription"] == 0
    assert "Blasius" in cw["structural_argument"]


def test_overall_verdict_is_honest(artifacts):
    _, clos = artifacts
    verdicts = {c["id"]: c["verdict"] for c in clos["criteria"]}
    assert verdicts["C1"] == "PASS"
    assert verdicts["C2"] == "PASS"
    assert verdicts["C3"] == "FAIL"
    assert clos["overall_verdict"] == "HONEST-FAIL"
    assert clos["honest_fail_summary"]["failed"]
    # no criterion may be silently missing from the adjudication table
    assert set(verdicts) == {"C1", "C2", "C2'", "C3", "C4"}


def test_numbers_in_closure_come_from_the_transcription(artifacts):
    """Every number the closure uses must be either a transcribed leaf or a value derived
    from transcribed leaves by a stated formula (nothing typed in by hand)."""
    trans, clos = artifacts
    pool = {leaf["value"] for leaf in trans["leaves"] if isinstance(leaf["value"], float)}
    cal = clos["closure"]["caliber_A_primary"]

    # 1. C_T of every grid is a transcribed leaf, verbatim
    for row in cal["per_tier"]:
        assert row["C_T"] in pool

    # 2. C_F of caliber A is the friction line evaluated at the transcribed case Re
    mod = load_module("w22_closure3", CASE / "build_ittc57_closure.py")
    assert cal["C_F"] == mod.cf_ittc57_log10(clos["case"]["Re_case"])

    # 3. k of caliber A is (1+k) - 1 of the transcribed anchor record
    for ref in cal["k_refs"]:
        # primary record carries 15 digits, the independent double entry carries 11:
        # the tolerance is the transcription precision of the coarser record, not a fudge
        assert ref["value"] - 1.0 == pytest.approx(cal["k"], rel=0, abs=1e-9)
        assert _transcribed(trans, ref["source_id"], ref["pointer"]) == ref["value"]

    # 4. every declared ref resolves to the same value in the transcription
    for refs in (cal["C_F_refs"], cal["k_refs"]):
        for ref in refs:
            assert _transcribed(trans, ref["source_id"], ref["pointer"]) == ref["value"]

    # 5. the case reference C_F is itself reproduced by the recorded Blasius formula
    assert clos["case"]["blasius_check"]["rel_dev_pct"] == pytest.approx(0.0, abs=1e-12)
