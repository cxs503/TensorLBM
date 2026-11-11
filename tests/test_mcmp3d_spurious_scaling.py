"""Tests for the 3-D MCMP spurious-current scaling bench (gap PP-3-3DMCMP).

Two layers, both cheap (<2 s, no solver in the loop):

* **protocol algebra** — the self-similarity rules (``R = n/8``, ``W = n/24``,
  hence ``R/W = 3`` at every grid) and the case matrix of the pre-registered
  protocol, including the two falsification controls;
* **fit / verdict machinery** — a synthetic ``results.json`` with a *known*
  exponent is pushed through :mod:`benchmarks.mcmp3d_spurious_scaling.fit_report`
  and the machine gate must recover the exponent, its uncertainty, both
  forcings and all controls.  This is what makes the numbers of the real run
  trustworthy: the same code path is validated against ground truth.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH = REPO_ROOT / "benchmarks" / "mcmp3d_spurious_scaling"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"mcmp3d_{name}", BENCH / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


run_mod = _load("run")
fit_mod = _load("fit_report")


# ---------------------------------------------------------------------------
# Protocol algebra
# ---------------------------------------------------------------------------


def test_grid_ladder_rules():
    """R = n/8 (box ratio 0.125) and a constant seed width over the ladder."""
    for n in run_mod.GRIDS:
        assert run_mod.radius_of(n) == pytest.approx(n / 8.0)
        assert run_mod.radius_of(n) / n == pytest.approx(0.125)
        assert run_mod.width_of(n) == pytest.approx(run_mod.WIDTH_INIT)
    assert run_mod.width_of(run_mod.GRIDS[0]) == pytest.approx(
        run_mod.width_of(run_mod.GRIDS[-1]))


def test_protocol_meets_acceptance_floor():
    """The frozen protocol must itself satisfy the gap's coverage floor."""
    assert len(run_mod.GRIDS) >= 3
    assert len(run_mod.G_LEVELS) >= 3
    assert len(set(run_mod.G_LEVELS)) == len(run_mod.G_LEVELS)
    assert set(run_mod.FORCINGS) == {"kupershtokh", "velocity_shift"}


def test_case_matrix_and_controls():
    cases = run_mod.build_cases([32, 48], [-4.7, -5.0], ["kupershtokh", "velocity_shift"])
    main = [c for c in cases if c["kind"] == "main"]
    assert len(main) == 2 * 2 * 2
    assert len({c["id"] for c in cases}) == len(cases)
    kinds = {c["kind"] for c in cases}
    assert kinds == {"main", "control_zero", "exploratory_psi2"}
    # only the exploratory arm switches the force form; it never gates the verdict
    ex = [c for c in cases if c["kind"] == "exploratory_psi2"]
    assert all(c["form"] == "psi2" for c in ex)
    assert all(c["G"] == run_mod.G_CONTROL_WB for c in ex)
    assert all(c["form"] == run_mod.FORM for c in main)


# ---------------------------------------------------------------------------
# Fit / verdict machinery against synthetic ground truth
# ---------------------------------------------------------------------------


def test_ols_recovers_slope():
    x = [0.1, 0.5, 1.0, 1.5, 2.0]
    y = [3.0 * math.exp(2.0 * xi) for xi in x]  # ln y = ln3 + 2 x
    fit = fit_mod.ols(x, [math.log(v) for v in y])
    assert fit["b"] == pytest.approx(2.0, abs=1e-9)
    assert fit["sb"] == pytest.approx(0.0, abs=1e-9)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-12)


def test_pooled_slope_uses_grids_as_replication():
    groups = []
    for i, off in enumerate((0.0, 0.7, -1.3)):
        x = [math.log(v) for v in (0.5, 1.0, 2.0)]
        y = [off + 1.5 * xi for xi in x]
        groups.append((x, y))
    pooled = fit_mod.pooled_slope(groups)
    assert pooled["b"] == pytest.approx(1.5, abs=1e-9)
    assert pooled["sb"] == pytest.approx(0.0, abs=1e-9)


def _synthetic_results(exponent: float = 2.0, scatter: float = 0.0) -> dict:
    """Fabricate a results.json whose true exponent is known."""
    a = {f: 1e-2 for f in run_mod.FORCINGS}
    cases, coex = [], {}
    for G in run_mod.G_LEVELS:
        rho_v = 0.15 * (1.0 + 0.05 * (G + 5.0))
        rho_l = rho_v + (1.0 + 0.35 * (-G - 4.0))       # monotonically rising Δρ
        coex[str(G)] = {"rho_l": rho_l, "rho_v": rho_v, "p_sat": 0.0,
                        "density_ratio": rho_l / rho_v, "converged": True}
    coex[str(run_mod.G_CONTROL_ZERO)] = {"rho_l": 0.678, "rho_v": 0.676, "p_sat": 0.0,
                                         "density_ratio": 1.003, "converged": True}
    k = 0
    for n in run_mod.GRIDS:
        for G in run_mod.G_LEVELS:
            for f in run_mod.FORCINGS:
                drho = coex[str(G)]["rho_l"] - coex[str(G)]["rho_v"]
                u = a[f] * drho ** exponent * n ** -1.0
                u *= (1.0 + scatter * ((k % 3) - 1))
                k += 1
                cases.append({
                    "id": f"main_n{n}_G{G}_{f}", "kind": "main", "grid": n,
                    "G_eff": G, "forcing": f, "form": "difference", "tau": 1.0,
                    "R": n / 8.0, "W": n / 24.0,
                    "rho_l_maxwell": coex[str(G)]["rho_l"],
                    "rho_v_maxwell": coex[str(G)]["rho_v"],
                    "drho_maxwell": drho, "drho_measured": drho * 0.995,
                    "ratio_maxwell": coex[str(G)]["density_ratio"],
                    "u_max": u, "u_max_iface": u, "u_max_core": u * 0.5,
                    "dp": 0.1, "sigma": 0.1 * n / 16.0, "R_eq": n / 8.0,
                    "mass_drift_rel": 1e-12, "rho_min_tail": 0.1, "rho_max_tail": 1.0,
                    "steps": 2000, "converged": True, "finite": True,
                    "u_max_spread_rel": 1e-3, "wall_s": 1.0, "history": [],
                })
    for n in run_mod.GRIDS:
        for f in run_mod.FORCINGS:
            ref = [c for c in cases if c["grid"] == n and c["forcing"] == f
                   and c["G_eff"] == run_mod.G_CONTROL_WB][0]
            cases.append({**ref, "id": f"expl_psi2_n{n}_G{run_mod.G_CONTROL_WB}",
                          "kind": "exploratory_psi2", "form": "psi2",
                          "u_max": ref["u_max"] * 10.0})
            zc = coex[str(run_mod.G_CONTROL_ZERO)]
            cases.append({**ref, "id": f"ctrlZ_n{n}_G{run_mod.G_CONTROL_ZERO}",
                          "kind": "control_zero", "G_eff": run_mod.G_CONTROL_ZERO,
                          "rho_l_maxwell": zc["rho_l"], "rho_v_maxwell": zc["rho_v"],
                          "drho_maxwell": zc["rho_l"] - zc["rho_v"],
                          "u_max": ref["u_max"] * 1e-6,
                          "u_max_spread_rel": 1e-3})
    return {
        "meta": {"gap": "GAP-PP-3-3DMCMP-SPURIOUS-SCALING", "protocol": "synthetic",
                 "grids": run_mod.GRIDS, "g_levels": run_mod.G_LEVELS,
                 "forcings": run_mod.FORCINGS, "form": run_mod.FORM, "tau": 1.0,
                 "steps_cap": 2000, "check_every": 200, "steady_tol": 1e-3,
                 "steady_window": 3, "radius_rule": "R = n/8", "width_rule": "W = n/24",
                 "quick": False, "engine": "synthetic", "wall_s": 0.0,
                 "host": "test", "platform": "test", "machine": "test",
                 "torch": "test", "tensorlbm": "test", "git_commit": "synthetic",
                 "git_branch": "test", "command": "synthetic"},
        "coexistence": coex,
        "cases": cases,
    }


@pytest.mark.parametrize("exponent", [1.0, 2.0, 3.0])
def test_pipeline_recovers_known_exponent(tmp_path, exponent):
    payload = _synthetic_results(exponent=exponent)
    rf = tmp_path / "results.json"
    rf.write_text(json.dumps(payload))
    rc = fit_mod.main(["--results", str(rf), "--out", str(tmp_path / "out")])
    assert rc == 0
    verdict = json.loads((tmp_path / "out" / "verdict.json").read_text())
    hf = verdict["headline_forcing"]
    assert verdict["headline_exponent"] == pytest.approx(exponent, rel=2e-3), hf
    assert verdict["headline_rel_unc"] < 0.20
    assert verdict["C1_table_coverage"]["pass"] is True
    assert verdict["C2_exponent_uncertainty"]["pass"] is True
    assert verdict["C3_force_format_crosscheck"]["pass"] is True
    assert verdict["controls"]["zero_interface_pass"] is True
    assert verdict["controls"]["plateau_pass"] is True
    assert verdict["controls"]["mass_pass"] is True
    assert verdict["controls"]["psi2_exploratory"]["rejected"] is True
    assert verdict["all_pass"] is True
    for name in ("tables.md", "REPORT.md"):
        assert (tmp_path / "out" / name).exists()
    assert len(list((tmp_path / "out" / "figures").glob("*.png"))) >= 3


def test_pipeline_flags_an_offset_law(tmp_path):
    """A law with a constant offset must fail control Z (no vanishing intercept)."""
    payload = _synthetic_results(exponent=2.0)
    for c in payload["cases"]:
        if c["kind"] == "control_zero":
            c["u_max"] = 1.0  # interface-free run still shows a huge u_max
    rf = tmp_path / "results.json"
    rf.write_text(json.dumps(payload))
    assert fit_mod.main(["--results", str(rf), "--out", str(tmp_path / "out2")]) == 0
    verdict = json.loads((tmp_path / "out2" / "verdict.json").read_text())
    assert verdict["controls"]["zero_interface_pass"] is False
    assert verdict["all_pass"] is False


def test_pipeline_flags_a_transient_run(tmp_path):
    """A case whose u_max has not plateaued (P control) must fail the gate."""
    payload = _synthetic_results(exponent=2.0)
    payload["cases"][0]["u_max_spread_rel"] = 0.5
    rf = tmp_path / "results.json"
    rf.write_text(json.dumps(payload))
    assert fit_mod.main(["--results", str(rf), "--out", str(tmp_path / "out3")]) == 0
    verdict = json.loads((tmp_path / "out3" / "verdict.json").read_text())
    assert verdict["controls"]["plateau_pass"] is False
    assert verdict["all_pass"] is False


def test_pipeline_flags_a_mass_leak(tmp_path):
    """Mass drift above round-off (M control) must fail the gate."""
    payload = _synthetic_results(exponent=2.0)
    payload["cases"][0]["mass_drift_rel"] = 1e-3
    rf = tmp_path / "results.json"
    rf.write_text(json.dumps(payload))
    assert fit_mod.main(["--results", str(rf), "--out", str(tmp_path / "out4")]) == 0
    verdict = json.loads((tmp_path / "out4" / "verdict.json").read_text())
    assert verdict["controls"]["mass_pass"] is False
    assert verdict["all_pass"] is False


def test_psi2_arm_never_gates_the_verdict(tmp_path):
    """The exploratory psi2 arm may be arbitrarily bad without failing the gate."""
    payload = _synthetic_results(exponent=2.0)
    for c in payload["cases"]:
        if c["kind"] == "exploratory_psi2":
            c["u_max"] = 1e6
    rf = tmp_path / "results.json"
    rf.write_text(json.dumps(payload))
    assert fit_mod.main(["--results", str(rf), "--out", str(tmp_path / "out5")]) == 0
    verdict = json.loads((tmp_path / "out5" / "verdict.json").read_text())
    assert verdict["all_pass"] is True