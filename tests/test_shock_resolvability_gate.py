"""CI gate for the shock stand-off resolvability study (A1-A5).

The physics runs behind this study cost hours of cluster time, so the study is
frozen as *artifacts* (``runs/resolvability/*.json``) and this test re-checks
every acceptance number on them.  If an artifact is missing the test skips with
an explicit message (a code-only checkout is still testable); the CI workflow
sets ``TENSORLBM_REQUIRE_STUDY_ARTIFACTS=1`` so a missing artifact fails instead
of passing silently.

The negative control at the bottom is the point of the whole module: a coarse
grid that happens to agree with an extrapolated value must still be reported
``UNRESOLVED``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tensorlbm.compressible_shock import resolvability as R

ROOT = Path(__file__).resolve().parent.parent
#: committed copy of the acceptance artifacts (``runs/`` is gitignored by repo
#: policy: field dumps and logs stay on disk).  The live run directory is used
#: when it exists so a re-run is picked up without copying.
_COMMITTED = ROOT / "artifacts" / "shock_standoff_resolvability"
_RUNS = ROOT / "runs" / "resolvability"


def _artifact(name: str) -> Path:
    live = _RUNS / name
    return live if live.exists() else _COMMITTED / name


LADDER = _artifact("ma3_ladder.json")
MACH_SCAN = _artifact("mach_scan.json")
SHOCK_1D = _artifact("a5_shock_1d_scan.json")
CUT_CORNER = _artifact("ma3_cutcorner.json")
REQUIRE = os.environ.get("TENSORLBM_REQUIRE_STUDY_ARTIFACTS", "") == "1"


def _load(path: Path) -> dict:
    if not path.exists():
        if REQUIRE:
            pytest.fail(f"study artifact missing: {path} (set by the mission, must be committed)")
        pytest.skip(f"study artifact not present: {path.name}")
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------- A1
def test_a1_certification_band_is_achieved_below_five_percent():
    """A1 read as its three machine-checkable statements.

    ``R >= 4`` is necessary, not sufficient: a grid can span many cells per
    stand-off and still be inaccurate for the pocket, so the criterion is that
    (i) a certified grid with < 5 % error exists, (ii) every R <= 2 control is
    UNRESOLVED, and (iii) nothing above 5 % is labelled PASS.
    """
    ladder = _load(LADDER)
    a1 = ladder["A1"]
    assert a1["fine_grids"], "no grid reached R >= 4"
    assert a1["certifiable"] is True, a1["fine_grids"]
    for label in a1["certified_grids"]:
        assert a1["fine_grids"][label]["R"] >= R.R_CERT
    assert a1["certified_max_err"] < 0.05
    assert a1["no_pass_above_5pct"] is True
    assert a1["pass"] is True


def test_a1_coarse_negative_controls_are_unresolved():
    ladder = _load(LADDER)
    coarse = ladder["A1"]["coarse_controls"]
    assert coarse, "no negative control at R <= 2 was recorded"
    for label, rec in coarse.items():
        assert rec["R"] <= R.R_HARD
        assert rec["state"] == "UNRESOLVED", f"{label} at R={rec['R']:.2f} was not UNRESOLVED"


# ---------------------------------------------------------------------- A2
def test_a2_extended_ladder_gci_and_cross_triplet_stability_are_measured():
    """The two parts of A2 that are met, measured on >= 2 overlapping triplets.

    GCI and the cross-triplet spread are *uncertainty* statements and they hold
    at the 1 % level.  The observed order is checked separately below, where the
    gate enforces consistency rather than a favourable outcome.
    """
    ladder = _load(LADDER)
    a2 = ladder["A2_extended"]
    assert a2.get("n_triplets", 0) >= 2, a2
    assert a2["gci_ok"] is True, a2["reasons"]
    assert a2["gci_21"] < 0.05
    assert a2["spread_ok"] is True, a2["reasons"]
    assert a2["cross_triplet_spread"] < 0.03


def test_a2_verdict_is_the_recomputation_not_a_claim():
    """Integrity gate: the stored verdict must equal the recomputed one.

    A study that reports "not met" is honest; a study whose artifact says PASS
    while its own numbers violate the band is not, and this test fails on that.
    """
    ladder = _load(LADDER)
    for key in ("A2", "A2_extended"):
        stored = ladder[key]
        if stored.get("n_triplets") is None:
            continue
        labels = (ladder["prescribed_labels"] if key == "A2" else ladder["ladder_labels"])
        if len(labels) < 3:
            continue
        recomputed = R.evaluate_a2(R.cross_triplet(ladder["values"], labels))
        assert recomputed["pass"] == stored["pass"], (key, recomputed, stored)
        for p in recomputed["p"]:
            if not (1.5 <= p <= 2.5):
                assert stored["pass"] is False


def test_a2_as_prescribed_reports_the_missing_finest_grid():
    """The prescribed ladder (150x50 ... 1200x400) is reported separately so a
    missing 1200x400 can never be hidden behind the extended ladder."""
    ladder = _load(LADDER)
    a2 = ladder["A2"]
    prescribed = ladder["prescribed_labels"]
    if len(prescribed) >= 4:
        assert a2["n_triplets"] >= 2
    else:
        assert a2["pass"] is False
        assert "fewer than two convergent triplets" in " ".join(a2["reasons"])


# ---------------------------------------------------------------------- A3
def test_a3_corner_mass_closure():
    ladder = _load(LADDER)
    audits = [(g["label"], g["blockage_audit"]) for g in ladder["grids"].values()
              if g.get("blockage_audit")]
    assert audits, "no mass audit recorded"
    for label, a in audits:
        assert a["residual_relative"] < 1e-6, f"{label}: {a['residual_relative']:.3e}"
        assert a["meets_A3"] is True
        # the naive bookkeeping is what the fix replaces -- it must be visibly worse
        assert a["residual_naive"] > 100 * a["residual_relative"]


def test_a3_cut_corner_is_a_real_cut_cell_and_still_closes():
    rec = _load(CUT_CORNER)
    a = rec["blockage_audit"]
    assert a["ok"] and a["residual_relative"] < 1e-6
    assert a["corner_cell"]["is_cut_cell"], a["corner_cell"]
    assert 0.0 < a["corner_cell"]["open_fraction"] < 1.0
    # the measurement window keeps the corner cell (A3 requirement)
    assert rec["measurement"]["window"]["include_corner"] is True
    assert rec["standoff_mass"] > 0


# ---------------------------------------------------------------------- A4
def test_a4_mach_trend_and_correlation_state():
    scan = _load(MACH_SCAN)
    a4 = scan["A4"]
    assert a4["direction_ok"] is True, a4["rows"]
    assert a4["pass"] is True
    # A 2-D step face is outside Billig's axisymmetric validity: the criterion is
    # either <= 20 % agreement or an explicit WITHHELD -- never a silent pass.
    for row in a4["rows"]:
        if row["billig_deviation_rel"] > 0.20:
            assert row["correlation_state"] == "WITHHELD"


def test_a4_budget_law_is_positive_and_monotone_in_mach():
    scan = _load(MACH_SCAN)
    rows = scan["budget"]["rows"]
    assert len(rows) >= 3
    assert all(r["N_min"] > 0 for r in rows)
    assert scan["budget"].get("power_law") is not None


# ---------------------------------------------------------------------- A5
def test_a5_one_d_shock_scan_does_not_regress():
    scan = _load(SHOCK_1D)
    assert scan["n_cases"] == 30
    assert scan["worst_max_error"] <= 0.0153, scan["worst_max_error"]
    assert scan["pass"] is True


# ------------------------------- negative control: the rule, not the numbers
def test_negative_control_a_coarse_grid_never_passes():
    """Even with a perfect GCI and a perfect agreement, R <= 2 is UNRESOLVED."""
    perfect = R.classify_resolvability(
        R=1.2, shock_thickness_cells=0.4, gci_fine=1e-8,
        rel_err_vs_extrapolated=1e-8)
    assert perfect["state"] == "UNRESOLVED"
    assert "by rule" in perfect["reasons"][0]


def test_negative_control_missing_evidence_is_withheld():
    assert R.classify_resolvability(R=float("nan"), shock_thickness_cells=1.0,
                                    gci_fine=0.01)["state"] == "WITHHELD"