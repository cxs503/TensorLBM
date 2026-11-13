"""Regression lock for XVAL-LBM-DAM-BREAK-SLOT-MUSCL2-LOCAL-HLL-SSPRK2 (m17725).

Two things are asserted:

1. the frozen artifact ``xval/dam_break_slot_muscl2_local_hll_ssprk2/artifacts/result.json`` is *reproducible*
   from the arm files (``analyze.build()`` must regenerate every criterion to
   1e-9), so the published numbers cannot drift from the raw arm profiles;
2. the XVAL criteria hold: cross-implementation agreement between the in-house
   MUSCL2+local-HLL+SSPRK2 solver and Clawpack 5.14 stays below the owner 15%
   standard (and below 3% once the initial dry floor is matched), while the
   in-house scheme keeps its accuracy against the Ritter (1892) analytic anchor.

Clawpack itself is not needed here -- only the stored arm profiles.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "xval" / "dam_break_slot_muscl2_local_hll_ssprk2" / "artifacts"
RESULT = RUN_DIR / "result.json"
CASE_DIR = REPO_ROOT / "xval" / "dam_break_slot_muscl2_local_hll_ssprk2"

pytestmark = pytest.mark.skipif(
    not RESULT.exists() or not list(RUN_DIR.glob("arm_*.json")),
    reason="XVAL artifacts absent (run run_arms.py + analyze.py first)",
)


def _stored() -> dict:
    return json.loads(RESULT.read_text())


def test_artifact_reproducible_from_arm_files():
    sys.path.insert(0, str(CASE_DIR))
    import analyze  # noqa: E402

    fresh = analyze.build()
    old = _stored()
    assert fresh["overall"] == old["overall"]
    for k, v in old["criteria"].items():
        assert fresh["criteria"][k]["value"] == pytest.approx(v["value"], rel=0, abs=1e-9), k
        assert fresh["criteria"][k]["pass"] == v["pass"], k
    for new_row, old_row in zip(fresh["xval_table"], old["xval_table"]):
        assert new_row["nx"] == old_row["nx"]
        for key in ("A_vs_B_l2_pct", "Am_vs_B_l2_pct", "A_vs_C_l2_pct", "profile_A"):
            assert new_row[key] == pytest.approx(old_row[key], rel=0, abs=1e-9), (new_row["nx"], key)


def test_all_predeclared_criteria_pass():
    r = _stored()
    assert r["overall"] == "PASS"
    for name, c in r["criteria"].items():
        assert c["pass"] is True, f"{name}={c['value']} >= {c['threshold']}"
        assert c["value"] < c["threshold"]


def test_scheme_level_agreement_under_3pct():
    """Same initial dry floor => the two independent codes agree to <3%."""
    r = _stored()
    judged = [t for t in r["xval_table"] if t["nx"] in (200, 400)]
    assert judged
    for t in judged:
        assert t["Am_vs_B_l2_pct"] < 3.0, t
        assert t["A_vs_B_l2_pct"] < 15.0, t
        assert t["A_vs_C_l2_pct"] < 3.0, t


def test_inhouse_accuracy_vs_analytic_anchor():
    r = _stored()
    for t in r["xval_table"]:
        assert t["profile_A"] < 5.0, t          # frozen matrix gate
        assert abs(t["front_A"]) < 1.0, t


def test_mass_is_conserved_everywhere():
    """All *finite* arms conserve mass to <1%; the NaN arms are the documented
    OSS probe at dry floor 1e-3 (see test_oss_dry_floor_gap_is_documented)."""
    r = _stored()
    finite = [row for row in r["arms"] if row["finite"]]
    assert finite
    for row in finite:
        assert row["mass_drift_pct"] < 1.0, row
    for row in r["arms"]:
        if not row["finite"]:
            assert row["arm"] == "clawpack_classic_hlle" and row["dry_floor"] <= 1.0e-3, row


def test_oss_dry_floor_gap_is_documented():
    """The OSS 2nd-order instability on a thin dry layer stays on record."""
    r = _stored()
    probe = r["stability_probe"]
    assert probe, "stability probe missing"
    for nx, p in probe.items():
        assert p["nan_cells"] > 0 and not p["finite"], (nx, p)
    for nx, p in r["first_order_control"].items():
        assert p["nan_cells"] == 0, (nx, p)
    # the in-house arm is finite at a true dry bed (h = 0)
    a_arms = [x for x in r["arms"] if x["arm"] == "tensorlbm_muscl2_local_hll_ssprk2"]
    assert a_arms and all(x["finite"] and x["nan_cells"] == 0 for x in a_arms)


def test_ic_matched_attribution_is_consistent():
    """|A - A_matched| ~ |A - B| proves the raw gap is the IC dry floor."""
    r = _stored()
    for t in r["xval_table"]:
        if t["nx"] not in (200, 400):
            continue
        assert t["A_vs_Am_l2_pct"] == pytest.approx(t["A_vs_B_l2_pct"], abs=1.5), t
        assert t["Am_vs_B_l2_pct"] < t["A_vs_B_l2_pct"], t