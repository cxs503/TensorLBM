"""Tests for the XVAL-LBM-TAYLORGREEN cross-validation surface.

The reference physics is analytic (2D Taylor-Green is an exact Navier-Stokes
solution), so most tests compare the *machinery* against closed form rather than
against golden numbers.  One end-to-end test runs the in-house solver and
asserts the measured decay rate against theory; the palabos leg is exercised
when the reference binary has been built (skipped otherwise, never faked).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm.xval import (  # noqa: E402
    DEFAULT_SPEC,
    TG2DSpec,
    boosted_spec,
    exact_K,
    exact_phase_rate,
    exact_velocity,
    field_l2_error_pct,
    find_bug_suspects,
    fit_exp_rate,
    mode_phase_deg,
    peak_position_fraction,
    run_campaign,
    run_tensorlbm,
)
from tensorlbm.xval import metrics as M  # noqa: E402
from tensorlbm.xval.runners import palabos_binary  # noqa: E402


# --------------------------------------------------------------------- spec
def test_case_spec_derived_quantities_are_consistent():
    spec = DEFAULT_SPEC
    # Re = U0 L / nu and tau = 1/2 + 3 nu must be invertible from the spec
    assert math.isclose(spec.nu, spec.u0 * spec.L / spec.re, rel_tol=0, abs_tol=0)
    assert math.isclose(spec.tau, 0.5 + 3.0 * spec.nu, rel_tol=0, abs_tol=1e-15)
    assert math.isclose(spec.k, 2.0 * math.pi / spec.n, rel_tol=0, abs_tol=1e-15)
    # the energy decay rate is exactly twice the velocity decay rate
    assert math.isclose(spec.gamma_k_theory, 2.0 * spec.gamma_vel_theory, rel_tol=1e-15)
    assert spec.mach < 0.1  # documented low-Mach regime of the case


def test_case_spec_roundtrip_and_palabos_argv():
    spec = TG2DSpec(n=64, re=50.0, u0=0.04, steps=800, sample_every=50, field_steps=(200, 400))
    payload = spec.to_dict()
    assert TG2DSpec.from_dict(payload) == spec
    argv = spec.palabos_argv("/tmp/out")
    assert argv[0] == "/tmp/out"
    assert argv[1] == "64" and argv[4] == "800" and argv[6] == "200,400"
    json.dumps(payload)  # must be JSON serialisable


def test_boosted_spec_keeps_geometry_and_sets_boost():
    base = TG2DSpec(n=32, steps=100)
    boosted = boosted_spec(base)
    assert boosted.boost == base.u0
    assert boosted.n == base.n and boosted.re == base.re
    assert boosted.is_boosted and not base.is_boosted
    assert boosted.to_dict()["nondim"]["boost_x"] == boosted.u0


# ---------------------------------------------------------------- analytic
def test_exact_energy_fit_recovers_closed_form_rate():
    spec = TG2DSpec(n=64, steps=2000, sample_every=100)
    steps = np.arange(0, spec.steps + 1, spec.sample_every)
    k_exact = [exact_K(spec, s) for s in steps]
    fit = fit_exp_rate(steps, k_exact)
    assert math.isclose(fit["gamma"], spec.gamma_k_theory, rel_tol=1e-9)
    assert fit["r2"] > 1.0 - 1e-12


def test_exact_field_decays_and_mode_phase_is_stationary():
    spec = TG2DSpec(n=32, steps=400, sample_every=200, field_steps=(200, 400))
    phases = []
    amps = []
    for step in (0, 200, 400):
        ux, uy = exact_velocity(spec, step)
        phases.append(mode_phase_deg(spec, ux))
        amps.append(float(np.sqrt((ux**2 + uy**2).max())))
    assert max(abs(p - phases[0]) for p in phases) < 1e-9
    assert amps[0] > amps[1] > amps[2]


def test_boosted_exact_phase_rate_matches_translation():
    spec = boosted_spec(TG2DSpec(n=32, steps=400, sample_every=200))
    steps = [0, 200, 400]
    phases = [mode_phase_deg(spec, exact_velocity(spec, s)[0]) for s in steps]
    rate = M.phase_rate(spec, steps, phases)
    assert math.isclose(rate["phase_rate_rad_per_step"], exact_phase_rate(spec), rel_tol=1e-6)
    assert rate["r2"] > 1.0 - 1e-9


# ----------------------------------------------------------------- metrics
def test_field_error_and_helper_primitives():
    spec = TG2DSpec(n=16, steps=10)
    ux, uy = exact_velocity(spec, 0)
    assert field_l2_error_pct(ux, uy, ux, uy) == 0.0
    assert field_l2_error_pct(0 * ux, 0 * uy, ux, uy) == pytest.approx(100.0)
    assert M.relative_error_pct(1.05, 1.0) == pytest.approx(5.0)
    assert M.cross_difference_pct(1.0, [1.1]) == pytest.approx(2 * 0.1 / 2.1 * 100)
    assert M.periodic_distance_fraction(0.02, 0.98) == pytest.approx(0.04)
    px, py = peak_position_fraction(spec, ux, uy)
    assert 0.0 < px < 1.0 and 0.0 < py < 1.0
    with pytest.raises(ValueError):
        fit_exp_rate([0, 1], [1.0, 0.5])
    with pytest.raises(ValueError):
        M.relative_error_pct(1.0, 0.0)


def test_table_and_criteria_flag_the_deviating_tool():
    refs = {"gamma_K": 1.0e-3, "field_l2_pct": None}
    good = {"gamma_K": 1.002e-3, "field_l2_pct": 0.4}
    bad = {"gamma_K": 1.2e-3, "field_l2_pct": 0.5}
    per_tool = {"tensorlbm": good, "palabos": bad}
    table = M.build_xval_table(refs, per_tool)
    assert {r["metric"] for r in table} == {"gamma_K", "field_l2_pct"}
    # a zero-reference metric reports the absolute value, not a division blow-up
    assert [r for r in table if r["metric"] == "field_l2_pct" and r["tool"] == "palabos"][0][
        "vs_ref_err"
    ] == pytest.approx(0.5)
    criteria = M.evaluate_criteria(
        table, [{"metric": "gamma_K", "kind": "cross_tool", "tol_pct": 5.0}]
    )
    assert criteria[0]["PASS"] is False  # 20 % apart -> cross-tool fails
    suspects = find_bug_suspects(per_tool, refs, table, {"gamma_K": 5.0}, {"gamma_K"})
    assert suspects and suspects[0]["suspected_side"] == "palabos"
    assert suspects[0]["locate"]


def test_clean_tools_produce_no_suspects():
    refs = {"gamma_K": 1.0e-3}
    per_tool = {"tensorlbm": {"gamma_K": 1.0005e-3}, "palabos": {"gamma_K": 1.0010e-3}}
    table = M.build_xval_table(refs, per_tool)
    assert find_bug_suspects(per_tool, refs, table, {"gamma_K": 5.0}, {"gamma_K"}) == []


# --------------------------------------------------------------- execution
def test_tensorlbm_runner_reproduces_the_analytic_decay_rate(tmp_path):
    spec = TG2DSpec(n=32, re=100.0, u0=0.05, steps=600, sample_every=50, field_steps=(300, 600))
    res = run_tensorlbm(spec, tmp_path, dtype="float64", threads=2)
    assert res["available"] is True
    hist = res["history"]
    assert set(hist) == {
        "step",
        "kinetic_energy",
        "umax",
        "phase_deg",
        "peak_x_frac",
        "peak_y_frac",
    }
    fit = fit_exp_rate(hist["step"], hist["kinetic_energy"])
    assert abs(fit["gamma"] / spec.gamma_k_theory - 1.0) < 0.05  # N=32 is coarse on purpose
    assert fit["r2"] > 0.999
    assert (tmp_path / "k_history_tensorlbm.csv").is_file()
    assert (tmp_path / "field_tensorlbm_300.csv").is_file()
    assert res["field_l2_pct"]  # field error computed at the reference steps


@pytest.mark.skipif(palabos_binary() is None, reason="palabos reference binary not built")
def test_palabos_runner_matches_the_analytic_decay_rate(tmp_path):
    spec = TG2DSpec(n=32, re=100.0, u0=0.05, steps=600, sample_every=50, field_steps=(300,))
    res = __import__("tensorlbm.xval.runners", fromlist=["run_palabos"]).run_palabos(spec, tmp_path)
    assert res["available"] is True
    fit = fit_exp_rate(res["history"]["step"], res["history"]["kinetic_energy"])
    assert abs(fit["gamma"] / spec.gamma_k_theory - 1.0) < 0.05


def test_campaign_writes_artifacts_and_records_frictions(tmp_path):
    spec = TG2DSpec(n=24, re=100.0, u0=0.05, steps=200, sample_every=50, field_steps=(100,))
    payload = run_campaign(spec, tmp_path, tools=("tensorlbm", "openfoam"), dtype="float64", threads=2)
    assert payload["overall"] in {"PASS", "DEGRADED_PASS"}
    assert payload["case_spec"]["ref_id"] == "XVAL-TAYLORGREEN"
    assert payload["xval_table"], "xval_table must not be empty"
    for row in payload["xval_table"]:
        assert {"tool", "metric", "got", "vs_ref_err", "cross_diff"} <= set(row)
    assert (tmp_path / "case_spec.json").is_file()
    assert (tmp_path / "result.json").is_file()
    assert (tmp_path / "xval_table.csv").is_file()
    tools = payload["tools"]
    if not tools["openfoam"]["available"]:
        assert any(f["tool"] == "openfoam" for f in payload["frictions"])
        assert any(c["item"] == "three_way_cross_check" for c in payload["criteria"])


def test_cli_spec_mapping_stays_in_sync_with_the_module():
    from tensorlbm.xval.cli import build_parser, spec_from_args

    args = build_parser().parse_args(
        ["--out", "/tmp/x", "--n", "48", "--steps", "400", "--field-steps", "100,200", "--boost"]
    )
    spec = spec_from_args(args)
    assert spec.n == 48 and spec.steps == 400 and spec.field_steps == (100, 200)
    assert spec.is_boosted
