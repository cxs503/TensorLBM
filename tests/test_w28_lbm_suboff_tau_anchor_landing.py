"""W28 guard: the DTRC SUBOFF anchor lives on main, is graded honestly, and the
three-tier GCI verdict cannot be silently overwritten.

Mission paper-exp-lbm-suboff-tau-13 (m19903).  These tests re-derive every W28
criterion from the frozen files in the worktree -- they do not trust the recorded
artifact.  A later round that edits the numbers, relabels the anchor as A/B, or
flips the honest FAIL to PASS without re-running the grids will fail here.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BUILDER = REPO / "papers/artifacts/w28/lbm-suboff-tau/build_anchor_landing.py"
ARTIFACT = REPO / "papers/artifacts/w28/lbm-suboff-tau/anchor_landing.json"
ANCHOR = REPO / "benchmarks/anchors/lbm-suboff-tau.json"

BRANCH_BLOB_SHA256 = "b452bccd7d35ce1c294b159342b2d0e0fbb75c6fd5c574975df732abe7b54f88"
LADDER_DIR = REPO / "benchmarks/verified/suboff_sail_resistance"
TIER_LS = (96, 112, 128)
ARMS = {"A_e2_standard_with_sail": "ey2_L",
        "B_formal_mix50_with_sail": "formal_L",
        "C_diag_bare_mix50": "diag_bare_L"}


@pytest.fixture(scope="module")
def w28():
    assert BUILDER.exists(), "W28 builder missing"
    spec = importlib.util.spec_from_file_location("w28_build_anchor_landing", BUILDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert ARTIFACT.exists(), "W28 artifact missing"
    assert ANCHOR.exists(), "anchor not landed on main"
    return mod, json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _strip(landed: bytes, off: int, text: str) -> bytes:
    tb = text.encode("utf-8")
    assert landed[off:off + len(tb)] == tb
    return landed[:off] + landed[off + len(tb):]


# --------------------------------------------------------------------------------------
# criterion 1 · the anchor is on main and byte-identical to the W27 branch blob
# --------------------------------------------------------------------------------------
def test_anchor_present_in_main_worktree(w28):
    _, art = w28
    assert art["landing"]["target"] == "benchmarks/anchors/lbm-suboff-tau.json"
    assert art["landing"]["source_ref"] == "agent/m18960"
    doc = json.loads(ANCHOR.read_text(encoding="utf-8"))
    assert doc["anchor_id"] == "lbm-suboff-tau"
    assert len(doc["anchor_points"]) == 6


def test_strip_recovers_the_branch_blob_byte_for_byte(w28):
    _, art = w28
    landed = ANCHOR.read_bytes()
    ins = art["landing"]["amendment"]
    blob = _strip(landed, ins["insert_offset_bytes"], ins["inserted_text"])
    assert hashlib.sha256(blob).hexdigest() == BRANCH_BLOB_SHA256
    assert ins["existing_bytes_changed"] == 0
    assert art["landing"]["plain_copy_sha256"] == BRANCH_BLOB_SHA256
    assert len(blob) == art["landing"]["branch_blob_bytes"]


def test_every_amendment_is_itemised(w28):
    _, art = w28
    items = art["landing"]["amendment"]["diff_itemization"]
    assert len(items) == 3
    added = [i for i in items if i["change"] == "insert"]
    assert len(added) == 2
    assert all(i["at_byte_offset"] == art["landing"]["amendment"]["insert_offset_bytes"] for i in added)
    assert art["landing"]["amendment"]["inserted_keys"] == ["anchor_grade", "grade_rationale"]
    assert any(i["change"] == "none" for i in items)
    # landed file parses and still carries every branch key
    landed_doc = json.loads(ANCHOR.read_text(encoding="utf-8"))
    blob_doc = json.loads(_strip(
        ANCHOR.read_bytes(),
        art["landing"]["amendment"]["insert_offset_bytes"],
        art["landing"]["amendment"]["inserted_text"],
    ).decode("utf-8"))
    assert set(blob_doc) <= set(landed_doc)
    assert set(landed_doc) - set(blob_doc) == {"anchor_grade", "grade_rationale"}


def test_anchor_quoted_inputs_still_match_main(w28):
    _, art = w28
    rows = art["landing"]["anchor_input_sha256_recheck"]
    assert rows and all(r["match"] for r in rows), "an archive the anchor quotes changed on main"


# --------------------------------------------------------------------------------------
# criterion 2 · grading is honest
# --------------------------------------------------------------------------------------
def test_grade_is_secondary_and_never_A_or_B(w28):
    doc = json.loads(ANCHOR.read_text(encoding="utf-8"))
    assert doc["status"] == "TRANSCRIBED"
    assert doc["acquisition_status"] == "WITHHELD"
    assert doc["anchor_grade"] == "TRANSCRIBED-SECONDARY"
    rat = doc["grade_rationale"]
    assert rat["grade_letter"] == "C"
    assert rat["anchor_grade"] not in ("A", "B")
    assert rat["claim_boundary"]["may_be_cited_as_published_experimental_benchmark"] is False
    assert all(d["acquisition_status"] == "WITHHELD" for d in rat["primary_documents"])


def test_grade_follows_the_rule_not_the_mood(w28):
    mod, _ = w28
    doc = json.loads(ANCHOR.read_text(encoding="utf-8"))
    expected = mod.GRADE_FOR_STATE[(doc["status"], doc["acquisition_status"])]
    assert doc["anchor_grade"] == expected
    # a WITHHELD primary can never produce an A/B grade through this table
    for state, grade in mod.GRADE_FOR_STATE.items():
        if state[1] == "WITHHELD":
            assert grade not in ("A", "B")


# --------------------------------------------------------------------------------------
# criterion 3 · ITTC-1957 independent recomputation closes to 1e-9
# --------------------------------------------------------------------------------------
def test_ittc57_two_paths_close(w28):
    mod, art = w28
    doc = json.loads(ANCHOR.read_text(encoding="utf-8"))
    worst = 0.0
    for p in doc["anchor_points"]:
        a = mod.cf_ittc57_double(p["Re"])
        b = float(mod.cf_ittc57_decimal(p["Re"]))
        worst = max(worst, abs(a - p["C_F"]["value"]), abs(b - p["C_F"]["value"]))
    assert worst <= 1e-9
    assert worst == pytest.approx(art["ittc57"]["max_abs_residual_double_vs_registered"], abs=1e-18)
    assert art["criteria"][2]["verdict"] == "PASS"


def test_registered_CT_and_Re_reconstruct(w28):
    """Dimensional/algebraic layer: C_T and Re must fall out of the registered R_T, V, S, nu."""
    doc = json.loads(ANCHOR.read_text(encoding="utf-8"))
    L = doc["geometry"]["L_max_m"]
    S = doc["geometry"]["WSA_m2"]
    nu = doc["conventions"]["nu_m2_s"]
    kn = doc["conventions"]["kn_to_ms"]
    for p in doc["anchor_points"]:
        v = p["speed_knots"] * kn
        assert (v * L / nu) == pytest.approx(p["Re"], rel=1e-12)
        ct = p["resistance_n"] / (0.5 * 1000.0 * v ** 2 * S)
        assert ct == pytest.approx(p["C_T"], rel=1e-12)


# --------------------------------------------------------------------------------------
# criterion 4 · three-tier GCI guard (independent recomputation from frozen archives)
# --------------------------------------------------------------------------------------
def _read_cd(arm: str, L: int) -> float:
    rel = LADDER_DIR / ("%s%d" % (arm, L)) / ("result_L%d.json" % L)
    return json.loads(rel.read_text(encoding="utf-8"))["Cd_total"]


def test_gci_guard_recomputes_from_frozen_archives(w28):
    mod, art = w28
    r21 = TIER_LS[2] / TIER_LS[1]
    r32 = TIER_LS[1] / TIER_LS[0]
    recorded = {row["ladder"]["key"]: row["ladder"]["gci"] for row in art["ladders"]}
    for key, arm in ARMS.items():
        g = mod.gci_block(_read_cd(arm, 96), _read_cd(arm, 112), _read_cd(arm, 128), r21, r32)
        assert g["p_observed"] == pytest.approx(recorded[key]["p_observed"], rel=1e-12)
        assert g["gci_fine_pct"] == pytest.approx(recorded[key]["gci_fine_pct"], rel=1e-12)
        # the pre-declared criterion: GCI < 5% AND p >= 1.8
        assert g["pass_both"] is False
        assert g["gci_fine_pct"] >= mod.GCI_MAX_PCT or g["p_observed"] < mod.P_MIN


def test_gci_verdict_is_not_silently_flipped(w28):
    """The honest FAIL must stay FAIL while the grids are unchanged."""
    _, art = w28
    c4 = next(c for c in art["criteria"] if c["id"].startswith("C4"))
    assert c4["verdict"] == "FAIL"
    assert all(v["pass_both"] is False for v in c4["per_ladder"].values())
    for key, v in c4["per_ladder"].items():
        assert v["gci_fine_pct"] > 5.0, key
        assert v["p_observed"] < 1.8, key


def test_unequal_ratio_is_actually_used(w28):
    """The equal-ratio shortcut would fake p ~ 2.2; the real ratios give p < 1.8."""
    _, art = w28
    for row in art["ladders"]:
        g = row["ladder"]["gci"]
        assert g["h_ratio_r21"] == pytest.approx(128 / 112)
        assert g["h_ratio_r32"] == pytest.approx(112 / 96)
        shortcut = g["uniform_r_assumption"]
        assert abs(shortcut["p_observed"]) > 2.0          # the trap
        assert g["p_observed"] < 1.8                       # the honest value


# --------------------------------------------------------------------------------------
# criterion 5 · point-wise deviation table
# --------------------------------------------------------------------------------------
def test_deviation_table_covers_every_tier_and_anchor_point(w28):
    _, art = w28
    doc = json.loads(ANCHOR.read_text(encoding="utf-8"))
    n_points = len(doc["anchor_points"])
    assert {row["ladder"]["key"] for row in art["ladders"]} == set(ARMS)
    for row in art["ladders"]:
        tiers = row["transposition"]["tiers"]
        assert [t["tier_L"] for t in tiers] == list(TIER_LS)
        for t in tiers:
            assert len(t["points"]) == n_points
            for q in t["points"]:
                assert q["rel_dev_pct_OP1"] == pytest.approx(
                    100.0 * (q["Cd_OP1_constant_form_factor"] / q["C_T_anchor"] - 1.0), rel=1e-12)


def test_operator_dependence_is_disclosed_not_hidden(w28):
    _, art = w28
    c5 = next(c for c in art["criteria"] if c["id"].startswith("C5"))
    assert c5["verdict"] == "FAIL"
    assert any(v["any_point_operator_dependent"] for v in c5["per_ladder"].values())
    assert c5["disclosure"].startswith("OP0 identity is reported for scale only")


# --------------------------------------------------------------------------------------
# artifact-level invariants
# --------------------------------------------------------------------------------------
def test_overall_verdict_matches_the_criteria(w28):
    _, art = w28
    verdicts = {c["verdict"] for c in art["criteria"]}
    assert verdicts <= {"PASS", "FAIL", "WITHHELD"}
    assert art["overall_verdict"] == ("PASS" if verdicts == {"PASS"} else "HONEST_FAIL")
    assert art["overall_verdict"] == "HONEST_FAIL"


def test_prereg_is_declared_and_thresholds_untouched(w28):
    _, art = w28
    pre = art["prereg"]
    assert pre["declared_before_computation"] is True
    assert pre["thresholds_predeclared"]["gci_fine_pct_max"] == 5.0
    assert pre["thresholds_predeclared"]["p_observed_min"] == 1.8
    assert pre["thresholds_predeclared"]["rel_dev_abs_pct_max"] == 15.0
    assert pre["thresholds_predeclared"]["ittc57_residual_max_abs"] == 1e-9
    assert pre["stopping_rule"].startswith("no threshold may be relaxed")


def test_builder_reproduces_the_artifact(w28):
    """--check mode: recompute everything and diff against the stored artifact."""
    mod, _ = w28
    mod.check()

def test_w27_verifier_still_accepts_the_amended_anchor(w28):
    """The W27 builder's own --check (read-only) must still pass on the amended file."""
    import subprocess
    import sys

    res = subprocess.run(
        [sys.executable, str(REPO / "benchmarks/anchors/lbm_suboff_tau_anchor.py"), "--check"],
        capture_output=True, text=True, timeout=300,
    )
    assert res.returncode == 0, res.stdout + res.stderr


def test_regeneration_hazard_is_documented(w28):
    _, art = w28
    note = art["landing"]["regeneration_note"]
    assert "drops the gate-v3 grading block" in note
    assert "build_anchor_landing.py" in note
