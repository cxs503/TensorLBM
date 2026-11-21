"""W30 — DTRC SUBOFF anchor census guards (task m21512).

The task's EDIT_WHITELIST excludes ``tests/``, so this guard suite ships next to
the builder it guards; run it explicitly:

    python -m pytest benchmarks/anchors/test_dtrc_suboff_anchor_census.py -q

These tests do not merely "execute the builders": they re-derive the invariants
from the raw archives and assert that the shipped artifacts agree with them.

Covered invariants
------------------
1. the anchor register exists, carries >=3 points from >=2 transcription
   families, and every point's verbatim quote is really present at its locator
   and hashes to the recorded sha256;
2. every ``C_F``/``C_T`` value in the register is reproducible from the parsed
   ``(V, R_T)`` pair and the ITTC-1957 closed form to <= 1e-9;
3. evidence grades are honest: no A/B without a primary/hashed source document,
   every unavailable point marked WITHHELD with an explicit "what is missing";
4. the cross-family conflict is registered (not silently averaged) and its
   magnitude is reported relative to the 15% gate;
5. the census aligns the LBM artifacts point-by-point (11 anchors x 3 grid tiers
   for the verified case, 5/5 for the regime-matched case), and the 15% gate
   verdict is "not met" in every admissible branch — a FAIL may not be
   accidentally promoted to a PASS by a later edit;
6. both ladders have >=3 tiers and the tier admitted into the error table is
   declared;
7. every source file listed in the census still hashes to its recorded value
   (no post-hoc number editing).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from decimal import Decimal, getcontext
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ANCHOR = ROOT / "benchmarks/anchors/dtrc-suboff-1989.json"
CENSUS = ROOT / "benchmarks/anchors/evidence/suboff_w30/suboff_anchor_census.json"
BUILDER_A = ROOT / "benchmarks/anchors/dtrc_suboff_1989_anchor.py"
BUILDER_C = ROOT / "benchmarks/anchors/suboff_anchor_census.py"

# conventions re-declared here on purpose: the test must not import the code it
# is checking for the numbers it asserts on.
RHO = 1000.0
NU = 1.139e-06
KN = 0.514444
L_MAX = 4.356
WSA = 5.998


def _sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def _cf_decimal(re: float) -> float:
    getcontext().prec = 50
    r = Decimal(str(re))
    return float(Decimal("0.075") / (r.ln() / Decimal(10).ln() - Decimal(2)) ** 2)


@pytest.fixture(scope="module")
def anchor() -> dict:
    assert ANCHOR.is_file(), "run benchmarks/anchors/dtrc_suboff_1989_anchor.py"
    return json.loads(ANCHOR.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def census() -> dict:
    assert CENSUS.is_file(), "run benchmarks/anchors/suboff_anchor_census.py"
    return json.loads(CENSUS.read_text(encoding="utf-8"))


def test_builder_self_checks_pass():
    """Both builders' own --check mode must be green (rebuilds nothing)."""
    for script in (BUILDER_A, BUILDER_C):
        out = subprocess.run(
            [sys.executable, str(script), "--check"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert out.returncode == 0, f"{script.name} --check failed:\n{out.stdout}\n{out.stderr}"
        assert json.loads(out.stdout)["ok"] is True


def test_anchor_has_at_least_three_points_in_two_families(anchor):
    pts = anchor["anchor_points"]
    assert len(pts) >= 3
    assert len({p["family"] for p in pts}) >= 2
    assert len({p["id"] for p in pts}) == len(pts)


def test_every_quote_is_verbatim_at_its_locator(anchor):
    for p in anchor["anchor_points"]:
        rel = p["provenance"]["locator"]["path"]
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert p["provenance"]["verbatim_quote"] in text, p["id"]
        assert _sha(rel) == p["provenance"]["sha256"], p["id"]


def test_cf_and_ct_recompute_from_parsed_pairs(anchor):
    """Independent re-derivation from (V, R_T) — the values may not be hand-typed."""
    for p in anchor["anchor_points"]:
        v = p["speed_ms"]
        re = v * L_MAX / NU
        assert abs(re / p["Re"] - 1.0) < 1e-9, p["id"]
        assert abs(_cf_decimal(p["Re"]) - p["C_F"]["value"]) <= 1e-9, p["id"]
        ct = p["resistance_n"] / (0.5 * RHO * v * v * WSA)
        assert abs(ct - p["C_T"]) <= 1e-12, p["id"]
        # the recompute block must agree with the registers it claims to check
        assert p["recompute"]["residual_vs_registered_C_F_abs"] <= 1e-9, p["id"]
        # speed-unit bookkeeping: one of the two representations must be the raw one
        assert abs(v / p["speed_kn"] - KN) < 1e-6 or abs(p["speed_kn"] - v) < 1e-9


def test_ittc_residual_gate_is_satisfied(anchor):
    crit = {c["id"]: c for c in anchor["criteria"]}
    c2 = crit["C2_ittc57_recompute_residual"]
    assert c2["pass"] is True
    assert c2["max_abs_residual"] <= 1e-9
    residuals = [p["recompute"]["residual_vs_registered_C_F_abs"] for p in anchor["anchor_points"]]
    assert max(residuals) == pytest.approx(c2["max_abs_residual"], abs=1e-30)


def test_grades_are_honest(anchor):
    assert anchor["acquisition_status"] == "WITHHELD"
    assert anchor["status"] == "TRANSCRIBED"
    for p in anchor["anchor_points"]:
        assert p["evidence_grade"] in ("A", "B", "C", "D")
        if p["evidence_grade"] in ("A", "B"):
            pytest.fail(f"{p['id']} claims grade {p['evidence_grade']} without a primary document")
        assert p["acquisition_status"] == "WITHHELD"
        assert p["what_is_missing"], p["id"]
    for doc in anchor["primary_documents"]:
        assert doc["acquisition_status"] == "WITHHELD"
        assert doc["what_is_missing"]


def test_cross_family_conflict_is_disclosed_not_averaged(anchor):
    cf = anchor["cross_family_conflict"]
    assert cf["ratio_mean"] > 1.10
    assert cf["ratio_sd"] < 0.01  # a *constant* offset, not noise
    assert abs(cf["ratio_mean"] - 1.0) * 100.0 > 15.0
    assert cf["gate_15pct_verdict"] == "CONFLICT_EXCEEDS_GATE"
    assert len(cf["pairs"]) == 5
    assert all(p["same_speed_within_0p2pct"] for p in cf["pairs"])
    assert cf["hypotheses_ranked"], "conflict must come with ranked hypotheses"


def test_census_covers_every_anchor_point_with_every_tier(census):
    n_pts = census["anchor"]["n_points"]
    et = census["error_table"]
    assert et["branch_verified_laminar_Re1000"]["n_rows"] == n_pts * 3
    assert et["branch_pending_highre_wallmodel"]["n_rows"] == 5
    ids = {p["id"] for p in census["anchor"]["points"]}
    for r in et["branch_verified_laminar_Re1000"]["rows"]:
        assert r["anchor_id"] in ids
        reg = next(p for p in census["anchor"]["points"] if p["id"] == r["anchor_id"])
        assert r["C_T_anchor"] == pytest.approx(reg["C_T"], rel=0, abs=0)
        expect = (r["C_D_lbm"] / reg["C_T"] - 1.0) * 100.0
        assert r["dev_raw_pct"] == pytest.approx(expect, rel=1e-12)
    for r in et["branch_pending_highre_wallmodel"]["rows"]:
        assert r["anchor_id"] in ids
        assert r["regime_match"] is True
        assert r["dev_pct"] == pytest.approx((r["C_T_lbm"] / r["C_T_anchor"] - 1.0) * 100.0)


def test_gate_verdicts_cannot_be_silently_promoted(census):
    et = census["error_table"]
    assert et["gate_pct"] == 15.0
    assert et["branch_verified_laminar_Re1000"]["n_pass_raw"] == 0
    assert et["branch_pending_highre_wallmodel"]["n_pass"] == 0
    assert et["branch_verified_laminar_Re1000"]["admissible_for_the_anchor"] is False
    assert et["branch_pending_highre_wallmodel"]["admissible_for_the_anchor"] is True
    q3 = next(c for c in census["criteria"] if c["id"] == "Q3_lbm_vs_anchor")
    assert q3["verdict"] == "不达标"
    assert q3["root_cause_hypotheses"] and q3["next_experiments"]
    devs = [r["dev_pct"] for r in et["branch_pending_highre_wallmodel"]["rows"]]
    assert min(devs) > 1000.0 and max(devs) > 2000.0


def test_renormalised_column_is_flagged_as_non_evidence(census):
    br = census["error_table"]["branch_verified_laminar_Re1000"]
    assert "WEAK" in br["renormalisation_discriminating_power"]
    assert all(r["regime_match"] is False for r in br["rows"])


def test_sensitivity_has_three_tiers_and_declares_the_admitted_one(census):
    sens = census["resolution_and_tau_sensitivity"]
    assert sens["grid_ladder"]["n_tiers"] >= 3
    assert sens["tau_ladder"]["n_tiers"] >= 3
    assert sens["grid_ladder"]["monotone_in_L"] is True
    assert sens["tau_ladder"]["monotone_in_tau"] is True
    adm = sens["admitted_into_error_table"]
    assert adm["grid_tier"] == "L128"
    assert adm["Cd_total"] == pytest.approx(
        max(t["Cd_total_primary"] for t in sens["grid_ladder"]["tiers"]), rel=0, abs=0
    )
    assert "REYNOLDS" in sens["tau_ladder"]["what_changes"].upper()


def test_every_source_hash_still_matches(census):
    manifest = json.loads(
        (ROOT / "docs/architecture/migrations/suboff-w30-census-2026-11-21.json").read_text(
            encoding="utf-8"
        )
    )
    recovered = {
        item["old_path"]: item for item in manifest["recovered_inputs"]
        if item["source_commit"] == census["git_commit"]
    }
    for rel, digest in census["sources"].items():
        source = recovered[rel]["new_path"] if rel in recovered else rel
        assert _sha(source) == digest, f"post-hoc edit of {source}"
    for rel, item in recovered.items():
        assert census["sources"][rel] == item["sha256"]


def test_geometry_gate_passed_before_drag(census):
    g = census["anchor"]["geometry_gate"]
    assert g["pass"] is True
    assert abs(g["Cb_dev_pct"]) <= g["Cb_band_pct"]
    assert abs(g["L_over_D_dev_pct"]) <= g["L_over_D_band_pct"]
    assert abs(g["WSA_dev_pct"]) <= g["WSA_band_pct"]


def test_claim_boundary_refuses_promotion(census):
    cb = census["claim_boundary"]
    assert cb["what_it_does_not_support"]
    assert any("15%" in s for s in cb["what_it_does_not_support"])
    assert census["verdict"]["overall"].startswith("PARTIAL")
    assert next(c for c in census["criteria"] if c["id"] == "Q1_anchor_points_sha256")["verdict"] == "达标"
    assert next(c for c in census["criteria"] if c["id"] == "Q2_ittc57_recompute")["verdict"] == "达标"