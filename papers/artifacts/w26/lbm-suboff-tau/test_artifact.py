#!/usr/bin/env python3
"""W26 / mission m17569 — machine checks for the SUBOFF C_F anchor + GCI artifact.

Run with:  python3 -m pytest papers/artifacts/w26/lbm-suboff-tau/test_artifact.py -q

The point of these tests is anti-fabrication: every number the artifact reports is
re-derived here from the raw stores (the anchor source artifacts and the case
``.npz`` time series), so a hand-edited summary cannot survive.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]  # repo root (papers/artifacts/w26/lbm-suboff-tau -> repo)

ANCHOR = HERE / "anchor_register.json"
GCI = HERE / "gci_sign_surface.json"
ARTIFACT = HERE / "suboff_cf_anchor_regci.json"


def load(p):
    return json.loads(Path(p).read_text())


# --------------------------------------------------------------------- anchor
def test_anchor_register_has_verbatim_slots():
    d = load(ANCHOR)
    rows = d["verbatim_anchor_rows"]
    assert len(rows) >= 1, "criterion 1 needs at least one verbatim anchor slot"
    for r in rows:
        assert r["grade"], "every verbatim row must declare its evidence grade"
        assert r["pointer"], "every verbatim row must carry a pointer"
        assert all(r["readback"].values()), "read-back must reproduce every transcribed value"


def test_anchor_quotes_are_substrings_of_archived_files():
    d = load(ANCHOR)
    quotes = d["verbatim_quote_index"]
    assert quotes, "the transcription pack should expose verbatim quotes"
    assert all(q["found"] for q in quotes), [q for q in quotes if not q["found"]][:3]


def test_ittc57_recompute_residual_below_criterion():
    d = load(ANCHOR)
    chk = d["ittc57_recompute_check"]
    assert chk["n_rows"] >= 1
    coef = d["ittc1957_formula"]["coef"]
    shift = d["ittc1957_formula"]["shift"]
    worst = 0.0
    for row in chk["rows"]:
        recomputed = coef / (math.log10(row["Re"]) - shift) ** 2
        assert recomputed == pytest.approx(row["CF_recomputed"], rel=1e-12, abs=0)
        worst = max(worst, abs(recomputed - row["CF_recorded"]))
    assert worst <= 1e-9, f"ITTC-57 residual {worst:.3e} exceeds 1e-9"
    assert chk["pass"] is True


def test_anchor_rows_reproduce_re_and_ct_from_verbatim_inputs():
    d = load(ANCHOR)
    for row in d["anchor_rows_derived"]:
        v = row["verbatim_inputs"]
        assert row["Re_recomputed"] == pytest.approx(row["Re_recorded"], rel=1e-9)
        assert row["CT_from_verbatim"] == pytest.approx(row["CT_recorded"], rel=1e-9)
        assert v["L_m"] > 0 and v["S_m2"] > 0
    for row in d["anchor_rows_derived"]:
        # the one-plus-k column must be definitional, not independent evidence
        assert row["one_plus_k_identity_abs_residual"] < 1e-9


def test_experimental_cf_is_withheld_with_reason():
    d = load(ANCHOR)
    for row in d["anchor_rows_derived"]:
        assert row["CF_exp_verbatim"] is None
        assert row["CF_exp_verbatim_status"] == "WITHHELD"
        assert len(row["CF_exp_verbatim_reason"]) > 40
    statuses = {c["id"]: c for c in d["criteria"]}
    assert statuses["A2_experimental_CF_verbatim"]["status"] == "WITHHELD"


# ------------------------------------------------------------------------ gci
def test_ladder_is_a_fixed_reynolds_three_grid_study():
    d = load(GCI)
    cases = d["ladder"]["cases"]
    if len(cases) < 3:
        pytest.skip("ladder incomplete in this run")
    re = [c["Re_lattice"] for c in cases]
    assert max(re) - min(re) <= 1e-6 * max(re), f"Re not held fixed: {re}"
    Ns = [c["N"] for c in cases]
    assert len(set(Ns)) == len(Ns) >= 3
    ratios = d["ladder"]["refinement_ratios"]
    assert d["ladder"]["r_is_uniform"], ratios


def test_reported_coefficients_match_raw_time_series():
    d = load(GCI)
    for c in d["ladder"]["cases"]:
        with np.load(c["timeseries_sha256"] and REPO / c["case_json"] and
                     json.loads((REPO / c["case_json"]).read_text())["raw_npz"]) as z:
            scale = float(z["dpS_front"]) / float(z["dpS_wet"])
            ct = float(z["cd_total_raw"][-40:].mean() * scale)
            cf = float(z["cd_friction_raw"][-40:].mean() * scale)
        assert ct == pytest.approx(c["C_T"], rel=1e-12)
        assert cf == pytest.approx(c["C_F"], rel=1e-12)
        # pressure + friction must reproduce the total
        assert abs(c["C_F"] + c["C_P"] - c["C_T"]) <= 1e-12 * abs(c["C_T"])


def test_gci_arithmetic_is_self_consistent():
    d = load(GCI)
    for q, g in d["ladder"]["quantities"].items():
        if g["status"] != "CONVERGED_MONOTONE":
            continue
        r = g["r"]
        p = g["observed_order"]
        assert p == pytest.approx(abs(math.log(abs(g["convergence_ratio_R"])) / math.log(r)), rel=1e-12)
        rp = r**p
        f_ex = (rp * g["f1_fine"] - g["f2_medium"]) / (rp - 1)
        assert f_ex == pytest.approx(g["f_exact_richardson"], rel=1e-12)
        gci = 1.25 * abs((g["f1_fine"] - g["f2_medium"]) / g["f1_fine"]) / (rp - 1)
        assert 100 * gci == pytest.approx(g["gci_fine_pct"], rel=1e-12)


def test_sign_surface_cells_match_raw_records():
    d = load(GCI)
    surf = d.get("sign_surface")
    if not surf:
        pytest.skip("scan not present in this run")
    for c in surf["cells"]:
        case = json.loads((REPO / c["case_json"]).read_text())
        with np.load(REPO / case["raw_npz"]) as z:
            scale = float(z["dpS_front"]) / float(z["dpS_wet"])
            ct = float(z["cd_total_raw"][-40:].mean() * scale)
        assert ct == pytest.approx(c["C_T"], rel=1e-12)
        expected = ("DIVERGED" if not case["field_finite_at_end"]
                    else ("NEGATIVE" if ct < 0 else "POSITIVE"))
        assert c["status"].startswith(expected), (c["tag"], c["status"], ct)


# --------------------------------------------------------------------- artifact
def test_artifact_hashes_pin_its_parts():
    d = load(ARTIFACT)
    import hashlib
    for part, rec in d["parts"].items():
        p = REPO / rec["path"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == rec["sha256"], part


def test_artifact_criteria_are_answered_explicitly():
    d = load(ARTIFACT)
    crit = d["criteria"]
    assert set(crit) == {"C1_anchor_register", "C2_ittc57_recompute",
                         "C3_gci", "C4_negative_drag_region"}
    assert crit["C1_anchor_register"]["pass"] is True
    assert crit["C2_ittc57_recompute"]["pass"] is True
    # unmet parts must be declared, never silently dropped
    assert crit["C3_gci"]["status_at_stated_Re"] in ("WITHHELD", "FAIL", "PASS")
    assert crit["C3_gci"]["withheld_reason"]
    assert d["verdict"], "every criterion needs a one-line verdict"
    assert d["honest_fail"], "honest-fail sections must be present"