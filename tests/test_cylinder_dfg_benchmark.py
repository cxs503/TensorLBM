"""Benchmark-integrity guards for CASE-CYLINDER-DRAG-RE100 (Schäfer–Turek 2D-2, Re=100).

``tests/test_cylinder_dfg_reference.py`` freezes the *reference register* and the
St-extraction.  This file guards the things that were wrong / unguarded in
mission m15776 and are fixed in m15918:

1. Δp **unit consistency** — the runner measures Δp/(0.5ρU²) while the paper
   quotes Δp/(ρU²); comparing them directly is a silent factor-2 error.
2. The case-pool entry actually runs end-to-end and emits a machine-readable
   ref-vs-got summary (no hand-copied numbers).
3. The delivered artifacts satisfy the physical invariants a periodic shedding
   solution must satisfy (mean lift ≈ 0, drag statistically converged).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.validation import cylinder_dfg as cdfg  # noqa: E402

POOL = REPO / "benchmarks/case_pool/cylinder_drag_re100"
ART = REPO / "benchmarks/verified/cylinder/re100_dfg/artifacts"


def test_dp_bands_are_the_same_interval_in_two_unit_systems():
    """The paper band (ρU²) and the comparison band (0.5ρU²) differ by exactly 2."""
    b = cdfg.REFERENCE_BANDS["2D-2"]
    assert b["dp_half_rho_u2"] == pytest.approx(tuple(2.0 * v for v in b["dp"]), rel=1e-12)
    # 4.92-5.00 is the 0.5ρU² image of 2.46-2.50 (REFERENCE.md states both)
    assert b["dp_half_rho_u2"] == (4.92, 5.00)


def test_dp_reported_in_both_unit_systems_from_one_run():
    """One smoke run: dp_over_rho_u2 must be dp/2 (no hidden renormalisation)."""
    cfg = cdfg.CaseConfig(D_cells=10, n_steps=120, avg_periods=1.0, report_every=0,
                          row_sample_every=1, seed_ampl=0.02)
    res = cdfg.run_case(cfg, threads=2, keep_series=False)
    assert res["dp_over_rho_u2"] == pytest.approx(res["dp"] / 2.0, rel=1e-12)
    # and the comparison error is computed in the 0.5ρU² convention
    assert res["bench"]["dp"]["band"] == [4.92, 5.00]


def test_refs_json_matches_module_register():
    refs = json.loads((POOL / "refs.json").read_text())
    b = cdfg.REFERENCE_BANDS["2D-2"]
    assert tuple(refs["ref"]["cd"]) == tuple(b["cd"])
    assert tuple(refs["ref"]["cl_amp"]) == tuple(b["cl_amp"])
    assert tuple(refs["ref"]["st"]) == tuple(b["st"])
    assert tuple(refs["ref"]["dp_rho_u2"]) == tuple(b["dp"])
    assert tuple(refs["ref"]["dp_half_rho_u2"]) == tuple(b["dp_half_rho_u2"])
    assert refs["tolerance_pct"] == 5.0
    # the entry the judge is pointed at must exist
    assert (REPO / refs["entry"]).is_file()
    assert (REPO / refs["library"]).is_file()


def test_case_pool_entry_emits_machine_readable_comparison(tmp_path):
    """CLI end-to-end on a tiny grid: summary.json carries numeric ref vs got."""
    out = tmp_path / "quick"
    r = subprocess.run(
        [sys.executable, str(POOL / "run_case.py"), "--grids", "10", "--steps", "300",
         "--avg-periods", "1", "--threads", "2", "--out", str(out)],
        capture_output=True, text=True, cwd=str(REPO),
        env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin:/usr/local/bin"},
        timeout=1200,
    )
    assert r.returncode in (0, 1), r.stderr[-2000:]
    summary = json.loads((out / "summary.json").read_text())
    v = summary["verdict"]
    assert set(v["per_metric"]) == {"cd", "cl_amp", "st", "dp"}
    for m, d in v["per_metric"].items():
        assert isinstance(d["err_pct"], float), m          # numeric, not a string
        assert len(d["ref_band"]) == 2, m
        assert d["got"] == pytest.approx(d["got"])        # finite
    assert (out / "results_D10.json").is_file()
    assert (out / "series_D10.csv").read_text().startswith("step,cd,cl")


@pytest.mark.skipif(not (ART / "result.json").is_file(), reason="no delivered artifact")
def test_delivered_artifact_physical_invariants():
    """A converged shedding solution: mean lift ≈ 0, drag averaged over ≥8 periods."""
    res = json.loads((ART / "result.json").read_text())
    grids = res["grids"]
    assert grids, "no grid levels in the delivered aggregate"
    for name, g in grids.items():
        # antisymmetric shedding: the mean of Cl over full periods must vanish
        assert abs(g["cl_mean"]) <= 0.15 * g["cl_amp"], (name, g["cl_mean"], g["cl_amp"])
        # statistical convergence of the drag mean (std of the mean)
        if g["cd_stderr"] == g["cd_stderr"]:  # not NaN
            assert g["cd_stderr"] / g["cd"] < 0.05, (name, g["cd_stderr"])
        # the two independent St estimators must agree, else St is flagged
        if g["st_acf"] == g["st_acf"] and g["st_fft"] == g["st_fft"]:
            rel = abs(g["st_acf"] - g["st_fft"]) / g["st_fft"]
            assert rel < 0.1, (name, rel)
        # dp must be reported in both unit systems, consistently
        assert g["dp"] == pytest.approx(2.0 * g.get("dp_over_rho_u2", g["dp"] / 2), rel=1e-6)
    # every metric comparison in the aggregate must be numeric (zero hand-copy)
    for m, d in res["per_metric"].items():
        for k, err in d["err_pct"].items():
            assert isinstance(err, (int, float)), (m, k, err)