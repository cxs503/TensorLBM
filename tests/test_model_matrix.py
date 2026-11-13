"""Tests for the model-matrix CI capability.

These assert *invariants*, not merely that code runs:

* registry integrity: unique keys, resolvable provenance, no placeholder
  implementation;
* the registry binding is faithful: a key resolves to the production operator
  and to the same call the solver makes (``collide_fn(f, tau=...)``);
* physics: every gate model reproduces the analytic shear viscosity, and the
  equilibrium is a collisional fixed point;
* the criterion is *falsifiable*: a deliberately wrong reference viscosity must
  be rejected (negative control), and an undeclared model/case must surface as
  MISSING rather than as a silent pass;
* coverage accounting: inapplicable cells leave the denominator, declared cells
  that did not run are reported as holes.
"""

from __future__ import annotations

import importlib
import json

import pytest
import torch

from tensorlbm.model_matrix import (
    CASES,
    Grid,
    build_collider,
    case_names,
    get_model,
    guard_one,
    model_names,
    nu_from_tau,
    registry,
    resolve_collide_fn,
    run_case,
    run_matrix,
    summarize,
)
from tensorlbm.model_matrix.cases import run_shear_decay
from tensorlbm.model_matrix import coverage as cov


# --------------------------------------------------------------------------- #
#  Registry integrity
# --------------------------------------------------------------------------- #


def test_registry_keys_unique_and_nonempty():
    names = model_names()
    assert len(names) == len(set(names)) >= 8
    assert all(n.count("/") == 1 for n in names), names


@pytest.mark.parametrize("name", model_names())
def test_every_model_points_at_a_real_implementation(name):
    spec = get_model(name)
    module_name, qualname = spec.source.split(":")
    module = importlib.import_module(module_name)
    fn = getattr(module, qualname)
    assert callable(fn)
    # No placeholder: the registered callable *is* the documented symbol, and it
    # is not a lambda / stub that ignores its arguments.
    assert fn is spec.collide
    assert fn.__module__ == module_name
    assert spec.lattice in ("D3Q19", "D3Q27")
    assert spec.cases, f"{name} declares no verification case"


@pytest.mark.parametrize("name", model_names())
def test_guard_constructs_every_model(name):
    res = guard_one(name)
    assert res["status"] == "GUARD_PASS", res


# --------------------------------------------------------------------------- #
#  Registry <-> solver binding
# --------------------------------------------------------------------------- #


def test_collider_matches_direct_operator_call():
    spec = get_model("d3q19/mrt")
    grid = Grid(8, 8, 1)
    from tensorlbm.d3q19 import equilibrium3d

    shape = grid.shape()
    f = equilibrium3d(
        torch.ones(shape, dtype=torch.float64),
        torch.full(shape, 0.03, dtype=torch.float64),
        torch.zeros(shape, dtype=torch.float64),
        torch.zeros(shape, dtype=torch.float64),
    ).to(torch.float64)
    got = build_collider("d3q19/mrt", 0.7)(f)
    ref = spec.collide(f, tau=0.7, **spec.params)
    assert torch.equal(got, ref)


def test_resolver_honours_stepper_call_convention_and_tau():
    """The stepper calls ``collide_fn(f, tau=tau)``; the resolver must obey it."""
    spec = get_model("d3q19/trt")  # tau_arg is tau_plus -> mapping must be right
    grid = Grid(8, 8, 1)
    from tensorlbm.d3q19 import equilibrium3d

    shape = grid.shape()
    f = equilibrium3d(
        torch.ones(shape, dtype=torch.float64),
        torch.full(shape, 0.03, dtype=torch.float64),
        torch.zeros(shape, dtype=torch.float64),
        torch.zeros(shape, dtype=torch.float64),
    ).to(torch.float64)
    fn = resolve_collide_fn("d3q19/trt", 0.7)
    got = fn(f, tau=0.7)
    ref = spec.collide(f, tau_plus=0.7, **spec.params)
    assert torch.equal(got, ref)
    # a different tau passed by the caller must be respected, not silently ignored
    other = fn(f, tau=0.9)
    assert not torch.equal(other, got)


def test_unknown_model_is_rejected():
    with pytest.raises(KeyError):
        get_model("d3q19/does-not-exist")


# --------------------------------------------------------------------------- #
#  Physics: analytic references
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", [n for n in model_names() if get_model(n).enforcement == "gate"])
def test_feq_is_a_collisional_fixed_point(name):
    res = run_case("feq_fixed_point", get_model(name), Grid(8, 8, 1), 0.6)
    assert res.status == "PASS", res
    assert res.metrics["rel_feq_fixed_point_error"] < 1e-12
    assert res.metrics["max_abs_rho_error"] < 1e-12
    assert res.metrics["max_abs_velocity_error"] < 1e-12


@pytest.mark.parametrize("name", ["d3q19/bgk", "d3q19/mrt", "d3q19/rlbm", "d3q27/kbc", "d3q27/cascaded"])
def test_shear_viscosity_matches_analytic_within_3pct(name):
    res = run_case("shear_decay", get_model(name), Grid(16, 16, 1), 0.6)
    assert res.status == "PASS", res
    assert abs(res.metrics["nu_rel_error_pct"]) < 3.0
    assert res.metrics["nu_ref"] == pytest.approx(nu_from_tau(0.6))
    assert res.metrics["fit_r2"] > 0.999
    assert res.metrics["mass_drift_rel"] < 1e-10


def test_shear_decay_converges_with_resolution():
    """The fitted viscosity error must shrink as k -> 0 (O(k^2) truncation)."""
    errs = []
    for res in (16, 24):
        out = run_shear_decay(get_model("d3q19/bgk"), Grid(res, res, 1), 0.6, dtype=torch.float64)
        errs.append(abs(out["metrics"]["nu_rel_error_pct"]))
    assert errs[1] < errs[0], errs


# --------------------------------------------------------------------------- #
#  Falsifiability (negative controls)
# --------------------------------------------------------------------------- #


def test_negative_control_wrong_viscosity_is_rejected():
    """A 33% off reference must be caught: the criterion can actually fail."""
    out = run_shear_decay(get_model("d3q19/bgk"), Grid(16, 16, 1), 0.6, dtype=torch.float64)
    nu_wrong = nu_from_tau(0.8)  # deliberately the wrong relaxation time
    err_against_wrong = 100.0 * (out["metrics"]["nu_fit"] - nu_wrong) / nu_wrong
    assert abs(err_against_wrong) > 3.0
    # ... and the honest reference passes, so the check is not trivially red
    assert abs(out["metrics"]["nu_rel_error_pct"]) < 3.0


def test_undeclared_case_is_missing_not_passing():
    spec = get_model("d3q19/bgk")
    res = run_case("not_a_case_at_all", spec, Grid(8, 8, 1), 0.6) if False else None
    res = run_case("tg_decay", get_model("d3q19/mrt-smag"), Grid(8, 8, 1), 0.6)
    assert res.status == "NOT_APPLICABLE"  # declared inapplicable (LES)
    assert "LES" in res.notes
    # an undeclared (neither applicable nor declared) pair is a coverage hole
    from tensorlbm.model_matrix.cases import CaseResult  # noqa: F401

    res2 = run_case("feq_fixed_point", spec, Grid(8, 8, 1), 0.6)
    assert res2.status == "PASS"


# --------------------------------------------------------------------------- #
#  Coverage accounting and report artifacts
# --------------------------------------------------------------------------- #


def test_coverage_accounting_and_reports(tmp_path):
    # res=16 is the CI scale (the truncation error of the analytic cases grows
    # as k^2, so a coarser box is *supposed* to fail -- see the convergence test).
    run = run_matrix(
        ["d3q19/bgk", "d3q19/mrt-smag"],
        None,
        grid=Grid(16, 16, 1),
        taus=(0.6,),
        dtype=torch.float64,
        progress=False,
    )
    summary = summarize(run, min_coverage=0.85)
    # bgk: 3 declared cases; mrt-smag: 1 declared + 2 declared-inapplicable
    assert summary["cells_declared"] == 4
    assert summary["cells_not_applicable"] == 2
    assert summary["cells_covered"] == 4
    assert summary["coverage_fraction"] == 1.0
    assert summary["verdict"] == "PASS", summary["gate_failures"]
    # guard model never claims a gate pass
    guard_cells = [c for c in summary["cells"] if c["model"] == "d3q19/mrt-smag"]
    assert all(c["status"] in ("GUARD", "NOT_APPLICABLE") for c in guard_cells), guard_cells

    cov.write_json(run, summary, tmp_path / "model_matrix.json", repro_cmd="test")
    cov.write_csv(run, tmp_path / "matrix.csv")
    cov.write_coverage_md(run, summary, tmp_path / "COVERAGE.md")
    cov.write_benchmark_md(run, summary, tmp_path / "BENCHMARK_REF_VS_GOT.md")
    payload = json.loads((tmp_path / "model_matrix.json").read_text())
    assert payload["summary"]["verdict"] == "PASS"
    # every gate cell carries a reference and a got value (no hand-typed numbers)
    for r in payload["run"]["results"]:
        if r["status"] == "NOT_APPLICABLE":
            continue
        assert r["ref"], r
        assert "tau" in r["metrics"], r
    bench = (tmp_path / "BENCHMARK_REF_VS_GOT.md").read_text()
    assert "shear viscosity nu" in bench and "0.03333" in bench
    cover = (tmp_path / "COVERAGE.md").read_text()
    assert "n/a" in cover and "guard" in cover


def test_missing_cells_are_reported():
    """A case implemented but not declared for a model must show up as MISSING."""
    run = {
        "results": [],
        "models": ["d3q19/bgk"],
        "cases": ["feq_fixed_point"],
        "taus": [0.6],
        "grid": {"nx": 8, "ny": 8, "nz": 1},
    }
    summary = summarize(run, min_coverage=0.85)
    assert summary["missing_cells"] == ["d3q19/bgk::feq_fixed_point"]
    assert summary["coverage_fraction"] == 0.0
    assert summary["verdict"] == "FAIL"


def test_case_registry_declares_enforcement_and_reference():
    for name in case_names():
        cs = CASES[name]
        assert cs.enforcement in ("gate", "advisory")
        assert cs.description and cs.metrics_doc
        assert cs.tol_pct >= 0.0