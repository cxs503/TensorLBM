"""Regression tests for the round-3 harness defects of
XVAL-LBM-COROTATIONAL-2NODE-BEAM-NEWMARK-NEWTON (case D3).

Defect F7 (silent censorship).  Round 2 of the cross-validation appended a KPI
row to ``table.csv`` only when the run reported ``converged``.  A predeclared
point that diverged therefore disappeared from the table instead of being
counted as a failure: the merged ``summary.json`` read
``D3_NONLINEAR_ENERGY: n=3, n_fail=0`` for **four** predeclared points, so a
divergence looked like a clean pass.  These tests pin the corrected behaviour:
a failed point is always visible (``status="FAILED"``, ``pass15=False``), and
``assemble()`` refuses a table that is missing a predeclared point.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
XVAL = REPO / "xval" / "corotational_beam_newmark"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(XVAL))

run_xval = pytest.importorskip("run_xval")


@pytest.fixture(autouse=True)
def _clear_rows():
    run_xval.rows.clear()
    yield
    run_xval.rows.clear()


def test_record_fail_is_a_visible_failure():
    run_xval.record_fail("D3_NONLINEAR_ENERGY", "TL", "lambda=5.0,dt=T1/200,n=16",
                         "energy_drift_pct", "FAILED: diverged", "%")
    row = run_xval.rows[-1]
    assert row["status"] == "FAILED"
    assert row["pass15"] is False
    assert not (row["got"] == row["got"])  # NaN: no fabricated number
    assert set(row) == set(run_xval.rows[0]) or len(run_xval.rows) == 1


def test_predeclared_points_cover_every_arm_and_step():
    pts = run_xval.predeclared_points()
    arms = {p[1] for p in pts}
    steps = {p[2] for p in pts}
    assert arms == {"TL", "LIFE"}
    assert len(pts) == len(arms) * len(steps) == 4


def test_assemble_rejects_missing_predeclared_points(tmp_path, monkeypatch):
    """A stage file that silently dropped the D3 points must raise, not pass."""
    raw = tmp_path / "runs" / "raw"
    raw.mkdir(parents=True)
    payload = {"cases": {"S1": {}, "S2": {}, "S3": {}, "D1": {}, "D2": {}, "D3": {}},
               "rows": [], "wall_clock_s": 0.0}
    (raw / "stage_stat.json").write_text(json.dumps(payload))
    monkeypatch.setattr(run_xval, "HERE", tmp_path)
    with pytest.raises(RuntimeError, match="silent censorship"):
        run_xval.assemble()


def test_underresolved_d3_point_is_reported_not_dropped():
    """The predeclared D3 step (dt = T1/200) is ~19 x the shortest discrete
    period at n = 16 and the Newton solve leaves its basin there; the point must
    come back as an explicit FAILED row, never as an absent row."""
    cases = run_xval.case_d3(n_elem=8, periods=(1,), fracs=(200,))
    d3_rows = [r for r in run_xval.rows if r["case"] == "D3_NONLINEAR_ENERGY"]
    assert len(d3_rows) == 2, "one row per arm for the single predeclared step"
    statuses = {r["arm"]: r["status"] for r in d3_rows}
    for arm, d in cases.items():
        ladder = d["ladder"]["200"]
        if ladder["converged"]:
            assert statuses[arm] == "PASS"
            assert ladder["max_abs_q"] < run_xval.L * 1e3
        else:
            assert statuses[arm] == "FAILED"
            assert any("FAILED" in r["note"] for r in d3_rows if r["arm"] == arm)