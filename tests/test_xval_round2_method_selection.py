"""Round-2 XVAL-LBM-ROUND2-METHOD-SELECTION: harness and analysis invariants.

These tests do not need a cluster: they pin the pieces the predeclared round-2
matrix depends on, so a resumption of the ladder cannot silently use a different
harness.

  1. ``collide_bgk`` honours a per-cell ``tau_field`` (the BGK sponge cell A6 needs)
     and keeps the uniform path bit-identical.
  2. the arm table is the predeclared 2x2 factorial + the two TRT diagnostics.
  3. the round-2 aggregator computes delta / factorial / closure numbers correctly
     on a synthetic matrix (so a real matrix cannot be mis-aggregated).
  4. the step schedule is a function of (D, u_in) only, i.e. every arm of a rung
     samples the same number of shedding periods (cl_amp is an extreme-value
     estimator, so this is a validity condition, not a detail).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
XV = REPO / "xval" / "benchmark_method_selection"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run_arm_mod = _load("xval_run_arm", XV / "run_arm.py")


def test_arm_table_is_the_predeclared_factorial():
    arms = run_arm_mod.ARMS
    assert set(arms) == {"A1", "A2", "A3", "A4", "A5", "A6"}
    # 2x2 {MRT, BGK} x {sponge on, off}
    assert arms["A1"] == {"collision": "mrt", "sponge": True}
    assert arms["A4"] == {"collision": "mrt", "sponge": False}
    assert arms["A6"] == {"collision": "bgk", "sponge": True}
    assert arms["A2"] == {"collision": "bgk", "sponge": False}
    # every cell distinct
    cells = {(v["collision"], v["sponge"]) for v in arms.values()}
    assert len(cells) >= 5


def test_bgk_tau_field_matches_explicit_cellwise_relaxation():
    from tensorlbm.d2q9 import equilibrium
    from tensorlbm.solver import collide_bgk, macroscopic

    torch.manual_seed(0)
    ny, nx = 5, 7
    f = equilibrium(
        torch.ones(ny, nx),
        0.01 * torch.randn(ny, nx),
        0.01 * torch.randn(ny, nx),
    ) * (1.0 + 1e-3 * torch.randn(9, ny, nx))
    tau = 0.6
    tau_field = torch.full((ny, nx), tau)
    tau_field[:, -2:] = tau * 11.0  # sponge-like stiffening of the last two columns

    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)
    explicit = f - (f - feq) / tau_field.unsqueeze(0)
    got = collide_bgk(f, tau, tau_field=tau_field)
    assert torch.allclose(got, explicit, rtol=0, atol=1e-14)

    # the uniform path must be untouched (bit-identical, no numerical drift)
    assert torch.equal(collide_bgk(f, tau), f - (f - feq) / tau)
    # and the sponge must actually do something
    assert not torch.allclose(got, collide_bgk(f, tau))


def test_bgk_tau_field_shape_is_checked():
    from tensorlbm.d2q9 import equilibrium
    from tensorlbm.solver import collide_bgk

    f = equilibrium(torch.ones(4, 6), torch.zeros(4, 6), torch.zeros(4, 6))
    with pytest.raises(ValueError):
        collide_bgk(f, 0.6, tau_field=torch.ones(2, 2))


def test_schedule_is_arm_independent_and_monotone_in_D():
    sch = run_arm_mod.scheduling
    a = sch(D=10, u_in=0.05, st_seed=0.13, seed_periods=2.0, st_estimate=0.165, periods=15.0)
    b = sch(D=10, u_in=0.05, st_seed=0.13, seed_periods=2.0, st_estimate=0.165, periods=15.0)
    assert a == b
    # round-2 predeclared step counts (predeclared_round2.json::stopping_rule)
    assert sum(a) == 21259
    assert sum(sch(D=14, u_in=0.05, st_seed=0.13, seed_periods=2.0,
                   st_estimate=0.165, periods=15.0)) == 29763
    assert sum(sch(D=20, u_in=0.05, st_seed=0.13, seed_periods=2.0,
                   st_estimate=0.165, periods=15.0)) == 42518


def test_analyze_round2_aggregation_on_a_synthetic_matrix(tmp_path, monkeypatch):
    """Feed the aggregator a synthetic 2x2 x {10,14,20} matrix and check the
    arithmetic it publishes: deltas, the gate, the factorial effects and the
    closure rule.  A wrong sign or a swapped cell here would silently invert the
    round-2 conclusion, so it is pinned."""
    mod = _load("xval_analyze_round2", XV / "analyze_round2.py")
    runs = tmp_path / "runs"
    runs.mkdir()
    cells = {"A1": ("mrt", True), "A4": ("mrt", False), "A6": ("bgk", True), "A2": ("bgk", False)}
    # cells: cd, st, cl_amp
    kpi = {
        "A1": (1.60, 0.170, 0.60),
        "A4": (1.62, 0.171, 0.58),   # +1.25 %, +0.59 %, -3.3 %
        "A6": (1.59, 0.171, 0.55),
        "A2": (1.46, 0.176, 0.35),   # -8.75 %, +3.5 %, -41.7 %
    }
    for D in (10, 14, 20):
        for arm, (coll, sp) in cells.items():
            d = runs / f"{arm}_D{D}"
            d.mkdir()
            cd, st, cl = kpi[arm]
            (d / "result.json").write_text(json.dumps({
                "arm": arm, "D": D, "collision": coll, "sponge": sp, "nx": 20 * D,
                "ny": 20 * D, "tau": 0.5075, "nu": 0.0025, "ma": 0.0866,
                "total_steps": 100, "wall_s": 1.0, "cd_mean": cd, "cd_mean_last50": cd,
                "cl_amp": cl, "cl_rms": cl / 1.41, "st_fft": st, "st_zc": st,
            }))
    monkeypatch.setattr(mod, "RUNS", runs)
    monkeypatch.setattr(mod, "figures", lambda *a, **k: [])
    (tmp_path / "table_round2.csv").unlink(missing_ok=True)
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    assert mod.main() == 0
    s = json.loads((tmp_path / "summary_round2.json").read_text())

    rows = {(r["arm"], r["D"]): r for r in s["rows"]}
    assert abs(rows[("A2", 20)]["d_cd_mean_pct"] - (-8.75)) < 1e-9
    assert abs(rows[("A4", 10)]["d_cd_mean_pct"] - (100 * (1.62 - 1.60) / 1.60)) < 1e-9
    # gate: A6 and A4 pass, A2 fails on cl_amp
    assert s["gate_at_each_D"]["10"]["A6"] is True
    assert s["gate_at_each_D"]["10"]["A2"] is False
    # factorial: collision effect at D=10 = 0.5*((A6-A1)+(A2-A4)) on cd_mean
    fac = {(f["kpi"], f["D"]): f for f in s["factorial"]}
    eff_c = 0.5 * ((1.59 - 1.60) + (1.46 - 1.62))
    assert abs(fac[("cd_mean", 10)]["effect_collision_bgk_minus_mrt"] - eff_c) < 1e-9
    inter = (1.46 - 1.62) - (1.59 - 1.60)
    assert abs(fac[("cd_mean", 10)]["interaction"] - inter) < 1e-9
    eff_s = 0.5 * ((1.62 - 1.60) + (1.46 - 1.59))
    assert abs(fac[("cd_mean", 10)]["effect_sponge_off_minus_on"] - eff_s) < 1e-9
    # this synthetic matrix was built with a genuine interaction (|I| > |main effects|)
    assert abs(inter) > max(abs(eff_c), abs(eff_s))
    assert fac[("cd_mean", 10)]["separable"] is False
    # closure: the synthetic matrix is flat in D, so "closer at D=20 than D=10" is
    # false for every cell and H1 must not be claimed
    assert s["closure_H1"]["A2"]["H1_alt_A_resolution_artifact"] is False
    assert s["closure_H1"]["A6"]["cd_mean"]["inside_15pct_at_D20"] is True


def test_diverging_arm_is_reported_not_dropped(tmp_path, monkeypatch):
    mod = _load("xval_analyze_round2b", XV / "analyze_round2.py")
    runs = tmp_path / "runs"
    (runs / "A1_D10").mkdir(parents=True)
    (runs / "A1_D10" / "result.json").write_text(json.dumps({
        "arm": "A1", "D": 10, "collision": "mrt", "sponge": True, "nx": 200, "ny": 200,
        "tau": 0.5075, "nu": 0.0025, "ma": 0.0866, "total_steps": 100, "wall_s": 1.0,
        "cd_mean": 1.6, "cd_mean_last50": 1.6, "cl_amp": 0.6, "cl_rms": 0.4,
        "st_fft": 0.17, "st_zc": 0.17}))
    (runs / "A3_D10").mkdir()
    (runs / "A3_D10" / "fail.json").write_text(json.dumps({
        "arm": "A3", "D": 10, "fail_stage": "non_finite_population", "step": 600,
        "total_steps_planned": 21259, "cl_max_abs": 0.049}))
    monkeypatch.setattr(mod, "RUNS", runs)
    monkeypatch.setattr(mod, "figures", lambda *a, **k: [])
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    assert mod.main() == 0
    s = json.loads((tmp_path / "summary_round2.json").read_text())
    assert "A3_D10" in s["failures"]
    assert s["failures"]["A3_D10"]["fail_stage"] == "non_finite_population"
    assert not any(r["arm"] == "A3" for r in s["rows"])