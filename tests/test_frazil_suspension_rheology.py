"""Tests for :mod:`tensorlbm.frazil_suspension_rheology` (H2-FRAZIL-RHEOLOGY).

The criteria that matter are asserted as invariants, not merely executed:

* the Krieger-Dougherty implementation must have Einstein's intrinsic
  viscosity as its first Taylor coefficient (a broken exponent or a wrong
  ``phi_max`` placement changes this number);
* the model chain must reproduce Einstein in the dilute window and agree with
  an **independent** literature correlation (Thomas 1965) at mid
  concentration;
* the power law must recover its own exponent under log-log fitting;
* an unrun check must be reported *undefined*, never passed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorlbm import frazil_rheology as root
from tensorlbm import frazil_suspension_rheology as rheo
from tensorlbm.frazil_suspension_rheology import FrazilSuspensionRheologyError

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "h2_frazil_rheology" / "campaign.json"


# ---------------------------------------------------------------------------
# Reference closures
# ---------------------------------------------------------------------------


def test_einstein_first_order_matches_analytic_values():
    assert rheo.einstein_relative_viscosity(0.0) == 1.0
    assert rheo.einstein_relative_viscosity(0.01) == pytest.approx(1.025, rel=0, abs=1e-15)
    assert rheo.einstein_relative_viscosity(0.05) == pytest.approx(1.125, rel=0, abs=1e-15)


def test_einstein_second_order_adds_batchelor_term():
    phi = 0.05
    assert rheo.einstein_relative_viscosity(phi, order=2) == pytest.approx(
        1.0 + 2.5 * phi + rheo.BATCHELOR_K2 * phi * phi, rel=0, abs=1e-15
    )


def test_einstein_excludes_non_dilute_and_negative():
    with pytest.raises(FrazilSuspensionRheologyError):
        rheo.einstein_relative_viscosity(0.5)
    with pytest.raises(FrazilSuspensionRheologyError):
        rheo.einstein_relative_viscosity(-0.01)
    with pytest.raises(FrazilSuspensionRheologyError):
        rheo.einstein_relative_viscosity(0.01, order=3)


def test_thomas_correlation_values():
    # 1 + 2.5*0.2 + 10.05*0.04 + 0.00273*exp(3.32)
    expected = 1.0 + 0.5 + 0.402 + 0.00273 * math.exp(3.32)
    assert rheo.thomas_relative_viscosity(0.2) == pytest.approx(expected, rel=1e-12)


def test_thomas_refuses_outside_its_validity():
    with pytest.raises(FrazilSuspensionRheologyError):
        rheo.thomas_relative_viscosity(0.55)


def test_krieger_dougherty_is_the_root_closure():
    for phi in (0.0, 0.05, 0.2, 0.45):
        assert rheo.krieger_dougherty_relative_viscosity(phi) == pytest.approx(
            float(root.relative_viscosity_krieger_dougherty(phi, 0.60, 2.5)), rel=0, abs=1e-15
        )


def test_krieger_dougherty_taylor_coefficient_is_einstein():
    """The O(phi) coefficient of KD must be [eta] = 2.5, not just 'some curve'."""
    eps = 1e-7
    slope = (rheo.krieger_dougherty_relative_viscosity(eps) - 1.0) / eps
    assert slope == pytest.approx(rheo.EINSTEIN_INTRINSIC, rel=1e-6)


def test_krieger_dougherty_diverges_at_packed_limit_and_is_monotone():
    vals = [rheo.krieger_dougherty_relative_viscosity(p) for p in (0.0, 0.2, 0.4, 0.55)]
    assert vals == sorted(vals)
    assert vals[-1] > 40.0
    with pytest.raises(FrazilSuspensionRheologyError):
        rheo.krieger_dougherty_relative_viscosity(0.60)


# ---------------------------------------------------------------------------
# Model chain
# ---------------------------------------------------------------------------


def test_effective_viscosity_at_reference_shear_is_consistency_index():
    phi = 0.2
    out = rheo.effective_viscosity(phi, 1.0, mu_water=1e-3)
    K = rheo.consistency_index(phi, mu_water=1e-3)
    assert float(np.asarray(out["mu_app"]).reshape(-1)[0]) == pytest.approx(K, rel=1e-14)
    assert float(np.asarray(out["mu_relative"]).reshape(-1)[0]) == pytest.approx(
        rheo.krieger_dougherty_relative_viscosity(phi), rel=1e-14
    )


def test_effective_viscosity_broadcasts_and_keeps_shape():
    phi = np.array([0.0, 0.1, 0.2])
    g = np.array([1e-2, 1.0, 1e2])
    out = rheo.effective_viscosity(phi, g, mu_water=1e-3)
    assert np.asarray(out["mu_app"]).shape == (3,)
    # x-homogeneous: a scalar shear rate must broadcast against an array of phi
    out2 = rheo.effective_viscosity(phi, 1.0, mu_water=1e-3)
    assert np.asarray(out2["mu_app"]).shape == (3,)


def test_effective_viscosity_power_law_exponent():
    phi = 0.3
    g = np.array([1e-1, 1.0, 1e1])
    mu = np.asarray(rheo.effective_viscosity(phi, g, mu_water=1.0)["mu_app"])
    slope = np.polyfit(np.log(g), np.log(mu), 1)[0]
    assert slope == pytest.approx(rheo.flow_index(phi) - 1.0, rel=0, abs=1e-12)


def test_effective_viscosity_guards():
    with pytest.raises(FrazilSuspensionRheologyError):
        rheo.effective_viscosity(0.6, 1.0)          # at the packed limit
    with pytest.raises(FrazilSuspensionRheologyError):
        rheo.effective_viscosity(-1e-3, 1.0)        # negative fraction
    with pytest.raises(FrazilSuspensionRheologyError):
        rheo.effective_viscosity(0.1, -1.0)         # negative shear rate
    with pytest.raises(FrazilSuspensionRheologyError):
        rheo.effective_viscosity(0.1, 1.0, mu_water=0.0)
    with pytest.raises(FrazilSuspensionRheologyError):
        rheo.effective_viscosity(0.1, 1.0, mu_water=float("nan"))


def test_flow_index_endpoints_and_monotonicity():
    assert rheo.flow_index(0.0) == pytest.approx(1.0, rel=0, abs=1e-15)
    assert rheo.flow_index(0.5999999) == pytest.approx(0.45, rel=1e-4)
    ns = [rheo.flow_index(p) for p in np.linspace(0.0, 0.55, 20)]
    assert all(b <= a for a, b in zip(ns, ns[1:]))
    assert all(0.45 - 1e-12 <= n <= 1.0 for n in ns)


def test_consistency_index_scales_with_water_viscosity():
    a = rheo.consistency_index(0.2, mu_water=1e-3)
    b = rheo.consistency_index(0.2, mu_water=2e-3)
    assert b == pytest.approx(2.0 * a, rel=1e-15)


# ---------------------------------------------------------------------------
# Criterion c1 / c2
# ---------------------------------------------------------------------------


def test_dilute_limit_benchmark_passes_einstein():
    out = rheo.dilute_limit_benchmark()
    assert out["n_defined"] == out["n_pass"] == len(out["rows"])
    assert out["max_rel_error"] < rheo.DILUTE_TOLERANCE
    assert out["passed"] is True
    assert out["rows"][0]["rel_error"] == pytest.approx(0.0, abs=1e-15)


def test_dilute_limit_benchmark_refuses_non_dilute():
    with pytest.raises(FrazilSuspensionRheologyError):
        rheo.dilute_limit_benchmark(phis=(0.05, 0.2))


def test_mid_concentration_benchmark_passes_kd_and_independent_thomas():
    out = rheo.mid_concentration_benchmark()
    assert out["passed"] is True
    assert out["n_defined"] == out["n_pass"] == len(out["rows"])
    # code-path check against the closure the chain is built on: round-off
    assert out["max_rel_error_kd"] < 1e-12
    # independent literature correlation: within the predeclared 10 %
    assert 0.0 < out["max_rel_error_thomas"] < rheo.MID_TOLERANCE


def test_mid_concentration_benchmark_rejects_out_of_window():
    with pytest.raises(FrazilSuspensionRheologyError):
        rheo.mid_concentration_benchmark(phis=(0.05, 0.2))


# ---------------------------------------------------------------------------
# Criterion c3 (analytic part)
# ---------------------------------------------------------------------------


def test_root_chain_benchmark_all_checks_pass():
    out = rheo.root_chain_benchmark(mu_water=1e-3)
    assert out["n_pass"] == out["n_defined"] == len(out["checks"])
    assert out["passed"] is True
    names = {c["name"] for c in out["checks"]}
    assert "n(phi=0)=1" in names
    assert "mu_app(gdot=1)=K" in names
    assert "dln(mu_app)/dln(gdot)=n-1" in names


def test_root_chain_benchmark_guards_all_raise():
    out = rheo.root_chain_benchmark()
    assert out["guards"] and all(out["guards"].values())


# ---------------------------------------------------------------------------
# Criterion c4 (shear-rate scan)
# ---------------------------------------------------------------------------


def test_shear_rate_sweep_recovers_power_law_and_is_newtonian_at_zero_phi():
    out = rheo.shear_rate_sweep((0.0, 0.1, 0.3), mu_water=1e-3)
    assert out["max_slope_error"] < 1e-9
    assert abs(out["newtonian_slope"]) < 1e-9
    for curve in out["curves"]:
        assert curve["fit_r2"] == pytest.approx(1.0, abs=1e-12)
        if curve["phi"] > 0:
            assert curve["shear_thinning_ratio"] > 1.0
            assert curve["fitted_slope"] == pytest.approx(curve["expected_slope"], abs=1e-12)


def test_shear_rate_sweep_guards():
    with pytest.raises(FrazilSuspensionRheologyError):
        rheo.shear_rate_sweep((0.1,), (0.0, 1.0))
    with pytest.raises(FrazilSuspensionRheologyError):
        rheo.shear_rate_sweep((0.7,))
    with pytest.raises(FrazilSuspensionRheologyError):
        rheo.shear_rate_sweep((0.1,), (1.0,))


# ---------------------------------------------------------------------------
# Acceptance semantics
# ---------------------------------------------------------------------------


def _fast_acceptance_inputs(*, run_channels=False, phis_channel=(0.1,)):
    return dict(
        dilute=rheo.dilute_limit_benchmark(),
        mid=rheo.mid_concentration_benchmark(),
        root_chain=rheo.root_chain_benchmark(),
        channels=rheo.slurry_channel_benchmark(phis_channel, run=run_channels),
        sweep=rheo.shear_rate_sweep((0.0, 0.1, 0.3)),
    )


def test_unrun_channel_is_undefined_not_passed():
    acc = rheo.evaluate_acceptance(**_fast_acceptance_inputs(run_channels=False))
    assert acc["criteria"]["c3_w9_1215_root"]["passed"] is None
    assert "c3_w9_1215_root" in acc["undefined"]
    assert acc["n_defined"] == len(acc["criteria"]) - 1


def test_failing_criterion_turns_the_verdict_to_fail():
    inputs = _fast_acceptance_inputs()
    bad = json.loads(json.dumps(inputs["dilute"]))
    bad["passed"] = False
    bad["max_rel_error"] = 0.5
    acc = rheo.evaluate_acceptance(**{**inputs, "dilute": bad})
    assert acc["criteria"]["c1_einstein_dilute"]["passed"] is False
    assert acc["verdict"] == "FAIL"
    assert acc["n_failed"] == 1


@pytest.mark.parametrize("phi", [0.20])
def test_lbm_slurry_channel_matches_exact_solution(phi):
    """H=12 in the test (seconds); the artifact run uses H=16.

    H=8 is deliberately *not* used: at that resolution the power-law profile is
    under-resolved and the L2 error is 2.3 %, i.e. above the root's 2 % node
    criterion -- a resolution effect, reported in docs/H2_FRAZIL_RHEOLOGY.md
    rather than hidden by a loose tolerance.
    """
    out = rheo.slurry_channel_benchmark(
        (phi,), H=12, nx=2, mu_water=1e-3, wall_viscosity_target=2e-2,
        max_steps=6000, tol=1e-3, check_every=200, return_profiles=True,
    )
    row = out["rows"][0]
    assert row["ran"] is True
    assert out["n_defined"] == 1
    assert row["l2_relative"] < rheo.CHANNEL_TOLERANCE
    # the solver must receive the same (n, K) pair the closure produced
    assert row["flow_index"] == pytest.approx(rheo.flow_index(phi), rel=0, abs=1e-15)
    assert row["relative_viscosity"] == pytest.approx(
        rheo.krieger_dougherty_relative_viscosity(phi), rel=0, abs=1e-12
    )
    assert row["route_consistent"] is True
    assert len(row["profile"]) == len(row["u_exact"])


# ---------------------------------------------------------------------------
# Artifact consistency (skipped when the run artifacts are not in the tree)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ARTIFACTS.exists(), reason="run artifacts not present in this checkout")
def test_committed_artifact_numbers_are_reproducible():
    report = json.loads(ARTIFACTS.read_text())
    assert report["schema_version"] == rheo.SCHEMA_VERSION
    acc = report["acceptance"]
    assert acc["n_defined"] >= 3
    for row in report["dilute_limit"]["rows"]:
        got = rheo.relative_viscosity(row["phi"])
        assert got == pytest.approx(row["got_model"], rel=1e-12)
    for row in report["mid_concentration"]["rows"]:
        assert rheo.relative_viscosity(row["phi"]) == pytest.approx(row["got_model"], rel=1e-12)
        assert rheo.thomas_relative_viscosity(row["phi"]) == pytest.approx(
            row["ref_thomas_1965"], rel=1e-12
        )
    for curve in report["shear_rate_sweep"]["curves"]:
        assert rheo.flow_index(curve["phi"]) == pytest.approx(curve["n"], rel=1e-12)
    # the declared verdict must be the one the numbers imply
    implied = all(c["passed"] is not False for c in acc["criteria"].values())
    assert bool(acc["all_passed"]) is bool(implied)
    if acc["all_passed"]:
        assert acc["verdict"] == "PASS"
