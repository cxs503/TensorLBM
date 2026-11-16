"""Model-matrix round 2: coverage extension + gate detection power.

Two invariants are asserted here, and both are *falsifiable*:

1. Every registry entry's bulk (trace) eigenvalue is declared, and the declared
   value is the one the operator actually uses -- tested by measuring the sound
   attenuation and inverting it (``s_b_implied`` vs ``s_b``).
2. The gate is not vacuous: a shear-channel defect at 5 % must go red, and a
   bulk-only defect must go red *only* on the acoustic case (which is exactly
   why that case was added; with the pre-R2 shear-only matrix the same defect
   is invisible).
"""

from __future__ import annotations

import importlib
import math

import pytest
import torch

from tensorlbm.model_matrix import model_names, get_model
from tensorlbm.model_matrix import mutation as mut
from tensorlbm.model_matrix.cases import CASES, Grid, run_case, run_shear_decay
from tensorlbm.model_matrix.registry import bulk_viscosity, nu_from_tau

SMALL = Grid(12, 12, 1)
GATE_MODELS = [n for n in model_names() if get_model(n).enforcement == "gate"]
LES_MODELS = [n for n in model_names() if get_model(n).enforcement == "guard"]


def _resolve_source(source: str):
    mod, qual = source.split(":")
    return getattr(importlib.import_module(mod), qual)


def test_registry_covers_the_second_order_moment_family():
    names = model_names()
    for expected in (
        "d3q19/cumulant",
        "d3q27/cumulant-geier",
        "d3q27/cumulant-legacy",
        "d3q19/cascaded",
        "d3q27/cascaded-impl2",
    ):
        assert expected in names


@pytest.mark.parametrize("name", model_names())
def test_every_entry_points_at_a_real_production_operator(name):
    spec = get_model(name)
    assert callable(_resolve_source(spec.source))


@pytest.mark.parametrize("name", GATE_MODELS)
def test_gate_models_declare_the_bulk_channel(name):
    spec = get_model(name)
    assert "acoustic_decay" in spec.cases
    nu_b, s_b = bulk_viscosity(spec, 0.6)
    assert s_b > 0.0
    assert math.isfinite(nu_b)
    # zero-tau degeneracy guard: a declared eigenvalue must be a usable rate
    assert s_b != 0.0


@pytest.mark.parametrize("name", LES_MODELS)
def test_les_models_declare_acoustic_inapplicable_with_a_reason(name):
    spec = get_model(name)
    assert "acoustic_decay" in spec.inapplicable_reason
    assert "acoustic_decay" not in spec.cases


@pytest.mark.parametrize("name", ["d3q19/bgk", "d3q19/rlbm", "d3q19/trt", "d3q27/kbc"])
def test_single_rate_operators_keep_the_bgk_bulk_relation(name):
    spec = get_model(name)
    assert spec.bulk_param is None
    nu_b, _s_b = bulk_viscosity(spec, 0.6)
    assert nu_b == pytest.approx(2.0 * nu_from_tau(0.6) / 3.0)


@pytest.mark.parametrize("name", ["d3q19/bgk", "d3q19/mrt", "d3q19/cascaded", "d3q27/cascaded"])
def test_acoustic_case_measures_the_declared_bulk_eigenvalue(name):
    """The reference is a formula; the operator must actually use that rate."""
    spec = get_model(name)
    out = run_case("acoustic_decay", spec, SMALL, 0.6)
    assert out.status == "PASS", (name, out.metrics, out.notes)
    assert abs(out.metrics["acoustic_rel_error_pct"]) < 3.0
    # the measured eigenvalue, inverted from the attenuation, must match the
    # declared one to within the case tolerance (measured: <= 1 % off).
    assert out.metrics["s_b_implied"] == pytest.approx(out.metrics["s_b"], rel=0.05)


def test_acoustic_case_reference_is_consistent_with_shear_channel():
    """S = 4nu/3 + nu_b must reduce to the textbook 2nu for BGK (nu_b = 2nu/3)."""
    spec = get_model("d3q19/bgk")
    nu = nu_from_tau(0.6)
    nu_b, _ = bulk_viscosity(spec, 0.6)
    assert 4.0 * nu / 3.0 + nu_b == pytest.approx(2.0 * nu)


def test_negative_control_wrong_bulk_reference_fails():
    """Falsifiability: feed the case a *wrong* bulk reference and it must fail."""
    out = run_case("acoustic_decay", get_model("d3q19/mrt"), SMALL, 0.6)
    S_meas = out.metrics["S_meas"]
    # pretend the operator were single-rate (s_b = 1/tau) -> wrong by >3 %
    nu = nu_from_tau(0.6)
    wrong_S = 4.0 * nu / 3.0 + 2.0 * nu / 3.0
    assert abs(100.0 * (S_meas - wrong_S) / wrong_S) > 3.0


def test_mutation_nu_scale_is_detected():
    """A 5 % shear-viscosity defect must turn the gate red."""
    rec = mut.probe(
        ["d3q19/bgk"], kinds=["nu_scale"], sweep={"nu_scale": (0.05,)}, grid=SMALL
    )
    rows = [r for r in rec["records"] if r["kind"] == "nu_scale"]
    assert rows and rows[0]["detected"] is True
    assert "shear_decay" in rows[0]["detected_by"]


def test_mutation_bulk_only_is_invisible_to_shear_but_caught_by_acoustic():
    """The R2 justification, asserted as a test.

    Corrupting only the trace eigenvalue leaves the shear-wave viscosity
    untouched, so ``shear_decay`` (the pre-R2 gate) stays green; the acoustic
    case catches it.
    """
    study = mut.probe(
        ["d3q19/cascaded"], kinds=["bulk_only"], sweep={"bulk_only": (0.5,)}, grid=SMALL
    )
    row = next(r for r in study["records"] if r["kind"] == "bulk_only")
    assert row["statuses"]["shear_decay"] == "PASS", row
    assert row["statuses"]["acoustic_decay"] == "FAIL", row
    assert row["detected_by"] == ["acoustic_decay"]


def test_mutation_single_rate_model_has_no_bulk_only_defect():
    """Documented limitation: single-rate operators cannot express that defect."""
    with pytest.raises(ValueError):
        mut.make_mutant("d3q19/bgk", "bulk_only", 0.5, tau=0.6)


def test_shear_gate_still_reproduces_analytic_viscosity_after_extension():
    """Regression: adding cases must not perturb the pre-existing shear gate."""
    out = run_shear_decay(get_model("d3q19/bgk"), Grid(16, 16, 1), 0.6)
    assert out["status"] == "PASS"
    assert out["metrics"]["fit_r2"] > 0.999
    assert abs(out["metrics"]["nu_rel_error_pct"]) < 3.0


def test_case_registry_lists_acoustic_as_gated():
    assert CASES["acoustic_decay"].enforcement == "gate"
    assert CASES["acoustic_decay"].tol_pct == 3.0