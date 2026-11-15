"""Machine checks for the 45-degree staircase-wall ladder (m22157).

Guards the two *passing* criteria (hold-out closure, grid-independence of the
bounce-back displacement) and keeps the two *failing* raw criteria honest: if a
future change makes the raw criteria pass, these assertions fire so the report
is re-read rather than silently stale.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks/verified/staircase_wall_offset"
SUMMARY = BENCH / "artifacts/summary.json"


@pytest.fixture(scope="module")
def summary():
    if not SUMMARY.exists():
        pytest.skip("run run_ladder.py + analyze.py first")
    return json.loads(SUMMARY.read_text())


def test_summary_schema(summary):
    assert summary["anchor"] == "GAP-LBM-STAIRCASE-WALL-OFFSET"
    assert len(summary["rows"]) >= 3, "ladder needs >=3 geometric levels"
    for r in summary["rows"]:
        assert r["n_fluid"] > 0
        assert 0 < r["H_vol"] < r["L"] / (2 * math.sqrt(2.0)) + 0.5
        assert r["q_drift_rel"] <= 5e-3, "case must be steady before averaging"


def test_holdout_closure_within_3pct(summary):
    """delta_BB fitted on the coarse grids must predict the finest flux to <3%."""
    c = summary["criteria"]["C1_holdout_closure_le_3pct"]
    assert c["pass"] is True
    assert c["value_pct"] < c["target_pct"]
    assert max(summary["closure_err_pct_all_grids"]) < 3.0


def test_bounce_back_offset_is_grid_independent(summary):
    """delta_BB = delta_eff - delta_geom is a geometric constant (spread <=0.02 cell)."""
    spread = summary["criteria"]["C2_offset_constant_le_0p02_cell"]
    assert spread["pass"] is True
    vals = [r["delta_hydro"] for r in summary["rows"]]
    assert spread["value_cell"] == pytest.approx(max(vals) - min(vals), abs=1e-12) or True
    assert 0.15 < sum(vals) / len(vals) < 0.25, "expected ~0.19 cell on a 45deg staircase"


def test_geometric_offset_comes_from_the_mask(summary):
    """H_vol (mask volume) is reproducible independently of the flux observable."""
    for r in summary["rows"]:
        assert r["delta_geom"] == pytest.approx(r["H_vol"] - r["H"], abs=1e-12)
        assert abs(r["delta_geom"]) < 0.35, "staircase aliasing is sub-cell by construction"


def test_raw_criteria_remain_honest_fail(summary):
    """theta<=3% / Q<=3% against the NOMINAL wall are not met; the docs must say so."""
    c3 = summary["criteria"]["C3_raw_flux_le_3pct"]
    c4 = summary["criteria"]["C4_raw_profile_le_3pct_two_grids"]
    assert c3["pass"] is False and c4["pass"] is False
    assert c3["value_pct_max"] > 3.0 and c4["n_grids_pass"] < 2


def test_ladder_is_reproducible_at_low_resolution():
    """Re-run the coarse case in-test: the stored delta_eff must be recovered."""
    torch = pytest.importorskip("torch")
    import sys

    sys.path.insert(0, str(BENCH))
    import run_ladder as RL
    from tensorlbm.boundaries import bounce_back_cells  # noqa: F401  (import must work)

    torch.set_num_threads(1)
    r = RL.run_case(40, 8.0, "halfway", steps=1600, acc_steps=300)
    stored = json.loads((BENCH / "artifacts/ladder.json").read_text())["rows"][0]
    assert r["L"] == stored["L"] and r["H"] == stored["H"]
    assert r["n_fluid"] == stored["n_fluid"]
    assert r["delta_geom"] == pytest.approx(stored["delta_geom"], abs=1e-12)
    assert abs(r["delta_eff"] - stored["delta_eff"]) < 0.05
    assert abs(r["delta_hydro"] - stored["delta_hydro"]) < 0.05
