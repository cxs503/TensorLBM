"""Regression guard for XVAL-LBM-SHAN-CHEN-MCMP-VELOCITY-MIXING.

The cross-validation verdict lives in
``xval/XVAL-LBM-SHAN-CHEN-MCMP-VELOCITY-MIXING/artifacts/analysis.json`` (written by
``analyze.py`` from ``result.json``).  Re-running the full matrix takes ~10
minutes, so this test does not re-simulate: it checks that the *recorded*
verdict still satisfies the predeclared decision structure, i.e. that the four
load-bearing claims of ``REPORT.md`` cannot rot silently (someone editing a
threshold, a table or the JSON will break it).

The claims guarded:
1. the shipped convention matches the closed-form two-layer Poiseuille flux to
   < 3 % (in-house strict bound);
2. the OSS "force inside the mixture momentum" reading deviates by > 15 %
   (owner criterion) from the shipped convention - i.e. it must NOT be ported;
3. the deliberate defect control exceeds 15 % (the metric has teeth);
4. the harness runs the shipped solver (`LIB-ENTRY` == `GOT-MIX` for `u_eq="mixture"`).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

XVAL = Path(__file__).resolve().parents[2] / "xval" / "XVAL-LBM-SHAN-CHEN-MCMP-VELOCITY-MIXING"
ART = XVAL / "artifacts"


@pytest.fixture(scope="module")
def analysis() -> dict:
    path = ART / "analysis.json"
    if not path.exists():
        pytest.skip(f"{path} not generated (run run_xval.py + analyze.py)")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def predeclared() -> dict:
    return json.loads((XVAL / "predeclared.json").read_text())


def test_predeclared_thresholds_are_the_owner_criterion(predeclared: dict) -> None:
    tol = predeclared["metrics"]["thresholds"]["owner_accept"]
    assert "15%" in tol
    assert predeclared["metrics"]["thresholds"]["in_house_strict_poiseuille"].endswith("3%")
    # the design must have been frozen before the runs, and record its deviations
    assert predeclared["frozen_at"].startswith("before any arm was executed")


def _csv_rows():
    import csv

    with (ART / "table.csv").open() as fh:
        return list(csv.DictReader(fh))


def test_shipped_mixture_matches_analytic_poiseuille() -> None:
    """Claim 1: the shipped 'mixture' convention reproduces the closed-form
    two-layer Poiseuille band flux to < 3 % at the primary configuration."""
    rows = [r for r in _csv_rows()
            if r["workload"] == "W2" and r["block"] == "W2_poiseuille_primary"
            and r["arm"] == "GOT-MIX" and r["reference"] not in ("", "GOT-MIX")]
    assert rows, "no GOT-MIX analytic row in table.csv"
    err = float(rows[0]["err_or_dev"])  # percent
    assert err <= 3.0, f"analytic error {err:.2f}% exceeds the in-house 3% bound"


def test_oss_reading_must_not_be_ported(analysis: dict) -> None:
    lbm4 = [v for v in analysis["verdicts"] if v["arm"] == "LBM4" and "dev_Q" in v]
    assert lbm4, "LBM4 arm missing from the recorded verdicts"
    finite = [v for v in lbm4 if isinstance(v["dev_Q"], float) and math.isfinite(v["dev_Q"])]
    assert finite, "no finite LBM4 deviation recorded"
    assert all(v["dev_Q"] > 0.15 for v in finite), finite
    assert all(v["verdict"] == "FAIL" for v in finite), finite
    # the remaining OSS rows must be recorded as diverged, not as a small error
    assert all(v["verdict"] in ("FAIL", "NO-DATA") for v in lbm4), lbm4


def test_defect_control_has_teeth(analysis: dict) -> None:
    assert analysis["control_teeth"] is True
    ctrl = [v for v in analysis["verdicts"] if v["arm"] == "CTRL-NORM"]
    assert ctrl and all(v["verdict"] == "FAIL" for v in ctrl)


def test_harness_control_runs_the_shipped_solver(analysis: dict) -> None:
    ctl = analysis["library_control"]
    assert ctl["verdict"] == "HARNESS-OK"
    assert ctl["rel_diff_Q"] < 1e-6
    assert ctl["profile_max_abs_diff"] < 1e-6


def test_cross_validation_artifacts_are_complete() -> None:
    for name in ("result.json", "table.csv", "tables.md"):
        assert (ART / name).stat().st_size > 200, name
    assert len(list((XVAL / "figures").glob("*.png"))) >= 4
    for name in ("velocity_mixing_arms.py", "run_xval.py", "analyze.py", "REPORT.md"):
        assert (XVAL / name).exists(), name