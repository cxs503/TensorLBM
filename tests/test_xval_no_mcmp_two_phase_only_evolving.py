"""Tests for the round-3 (Allen-Cahn-evolved) arm of XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY.

The invariants asserted here are the ones the round-3 report claims:

* the indicator/phase-field maps are exact inverses and the viscosity mixing
  agrees with round 1's implementation (the two rounds really do differ only in
  whether the indicator is advanced);
* the interface diagnostics recover a known ``tanh`` geometry (so a "the
  interface did not move" claim cannot come from a blind diagnostic);
* the Allen-Cahn *advection* term reproduces the exact shear-translation
  solution, is exactly conservative in flux form, and converges at second order;
* the coupled evolving arm holds the owner band and does not migrate the
  interface, and the frozen control is reproduced by the same code path;
* the round-3 gates read their thresholds from the predeclared contract - an
  artificially tightened threshold flips the verdict, so no threshold can be
  hidden in the driver.

Nothing here re-runs the full matrix: the arms are deliberately small and the
full matrix lives in ``outputs/xval_no_mcmp_evolving_round3/results_round3.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch.set_num_threads(1)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "examples"))

from tensorlbm.verification.two_phase_only_evolving import (  # noqa: E402
    _two_layer_analytic,
    ac_parameters,
    ac_shear_translation,
    indicator_from_phase_field,
    interface_diagnostics,
    mix_viscosity,
    phase_field_from_indicator,
    run_two_phase_only_evolving,
)

PREDECLARED = REPO_ROOT / "docs" / "XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY-ROUND3-predeclared.json"


# ---------------------------------------------------------------------------
# Maps, parameters, diagnostics
# ---------------------------------------------------------------------------


def test_indicator_and_phase_field_are_inverse_maps():
    C = torch.linspace(0.0, 1.0, 11, dtype=torch.float64)
    phi = phase_field_from_indicator(C)
    assert torch.allclose(phi, 2.0 * C - 1.0)
    assert torch.allclose(indicator_from_phase_field(phi), C)
    # the round-1 tanh indicator maps onto the AC-equilibrium phase field
    y = torch.arange(41, dtype=torch.float64)
    C_tanh = 0.5 * (1.0 - torch.tanh((y - 20.0) / 2.0))
    assert torch.allclose(indicator_from_phase_field(phase_field_from_indicator(C_tanh)), C_tanh)


def test_ac_parameters_recover_the_requested_interface_width():
    for width in (1.0, 2.0, 3.5):
        p = ac_parameters(width)
        assert p.W == pytest.approx(width)
        assert np.sqrt(2.0 * p.kappa / p.beta) == pytest.approx(width, rel=1e-12)


def test_interface_diagnostics_recovers_a_known_tanh():
    width, y_int, ny = 2.0, 20.5, 42
    y = np.arange(ny, dtype=np.float64)
    C = 0.5 * (1.0 - np.tanh((y - y_int) / width))
    diag = interface_diagnostics(C, y_int, first_row=0)
    assert diag["centre"] == pytest.approx(y_int, abs=1e-9)
    assert diag["centre_shift"] == pytest.approx(0.0, abs=1e-9)
    # 10-90 width of a tanh of width w is 2 * atanh(0.8) * w; the crossing is
    # linearly interpolated on a unit grid, which costs ~3.5% at w = 2 cells
    assert diag["width_10_90"] == pytest.approx(2.0 * np.arctanh(0.8) * width, rel=5e-2)
    assert diag["plateau_below"] == pytest.approx(1.0, abs=1e-3)
    assert diag["plateau_above"] == pytest.approx(0.0, abs=1e-3)

    shifted = interface_diagnostics(np.roll(C, 2), y_int, first_row=0)
    assert shifted["centre_shift"] > 1.5  # the diagnostic does see a migration


def test_viscosity_mixing_agrees_with_the_round1_implementation():
    round1 = _two_layer_analytic()
    C = torch.linspace(0.0, 1.0, 17, dtype=torch.float64).view(1, 17, 1)
    C2 = C.view(17).unsqueeze(1).expand(17, 3).contiguous()
    for rule in ("linear", "geometric", "harmonic"):
        mine = mix_viscosity(C2, 0.5, 0.125, rule)
        theirs = round1.mix_viscosity(C2, 0.5, 0.125, rule)
        assert torch.allclose(mine, theirs, rtol=0, atol=1e-15)


# ---------------------------------------------------------------------------
# Allen-Cahn advection term against the exact shear-translation solution
# ---------------------------------------------------------------------------


def test_ac_advection_matches_the_exact_shear_translation_solution():
    res = ac_shear_translation(24, 48, steps=100, dt=1.0)
    assert res["finite"]
    assert res["l2_rel"] < 5e-2, res["l2_rel"]
    assert res["phi_range"][0] >= -1.0 and res["phi_range"][1] <= 1.0
    # the runtime ordering is declared, not guessed
    assert res["max_advective_cfl"] < 1.0


def test_ac_advection_is_conservative_in_flux_form():
    res = ac_shear_translation(24, 48, steps=100, dt=1.0)
    assert res["phase_budget_rel"] < 1e-10, res["phase_budget_rel"]


def test_ac_advection_converges_at_second_order():
    coarse = ac_shear_translation(24, 48, steps=100, dt=1.0)
    fine = ac_shear_translation(48, 96, steps=200, dt=0.5)
    order = np.log2(coarse["l2_rel"] / fine["l2_rel"]) / np.log2(2.0)
    assert order > 1.5, f"observed order {order:.2f} (coarse {coarse['l2_rel']:.3e}, fine {fine['l2_rel']:.3e})"


# ---------------------------------------------------------------------------
# Coupled arm: evolving indicator vs the frozen control, same code path
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def coupled_pair():
    kwargs = dict(G=4e-6, nu_lo=0.5, nu_hi=0.25, width=2.0, rule="harmonic",
                  max_steps=4000, stop_tol=1e-9, check_every=500)
    frozen = run_two_phase_only_evolving(32, evolve=False, **kwargs)
    evolving = run_two_phase_only_evolving(32, evolve=True, **kwargs)
    return frozen, evolving


def test_coupled_arms_hold_the_owner_band(coupled_pair):
    frozen, evolving = coupled_pair
    for arm in (frozen, evolving):
        assert arm["finite"]
        assert arm["l2_rel"] < 0.15, arm["l2_rel"]
        assert arm["max_rel_peak"] < 0.15
        assert set(arm["analytic_checks"]) == {"wall_lo", "wall_hi", "u_jump", "stress_jump"}
        assert max(arm["analytic_checks"].values()) < 1e-12


def test_evolving_interface_does_not_migrate(coupled_pair):
    _, evolving = coupled_pair
    final = evolving["interface"]["final"]
    assert final["centre_shift"] is not None
    assert final["centre_shift"] < 1.0, final["centre_shift"]
    assert abs(evolving["interface"]["width_growth_rel"]) < 0.5
    assert evolving["phase_field_budget_rel"] < 1e-3
    assert evolving["uy_spurious_rel"] < 1e-2


def test_unfreezing_the_interface_costs_little(coupled_pair):
    frozen, evolving = coupled_pair
    assert evolving["l2_rel"] - frozen["l2_rel"] < 0.05
    # and the frozen arm is the round-1 model, not a different one
    assert frozen["config"]["evolve"] is False
    assert evolving["config"]["evolve"] is True


def test_width_mismatch_relaxes_towards_the_equilibrium(coupled_pair):
    """An initial layering thinner than W must thicken, not stay put."""
    _, evolving = coupled_pair
    mismatch = run_two_phase_only_evolving(
        32, G=4e-6, nu_lo=0.5, nu_hi=0.25, width=2.0, width_init=0.75,
        rule="harmonic", evolve=True, max_steps=4000, stop_tol=1e-9, check_every=500,
    )
    base_width = evolving["interface"]["initial"]["width_10_90"]
    mismatch_width0 = mismatch["interface"]["initial"]["width_10_90"]
    assert mismatch_width0 < base_width  # thinner layering was actually initialised
    assert mismatch["interface"]["width_growth_rel"] > 0.05  # and the AC term thickens it


# ---------------------------------------------------------------------------
# Gates read the predeclared contract, not the code
# ---------------------------------------------------------------------------


def _fake_results():
    arm = {
        "arm": "M2PE-test",
        "l2_rel": 0.01,
        "max_rel_peak": 0.01,
        "uy_spurious_rel": 1e-4,
        "phase_field_budget_rel": 1e-6,
        "config": {"evolve": True, "ny": 32, "nu_ratio": 2.0},
        "interface": {"final": {"centre_shift": 1e-4}, "width_growth_rel": 0.0},
    }
    frozen = json.loads(json.dumps(arm))
    frozen["arm"] = "M2P-test-frozen"
    frozen["config"]["evolve"] = False
    shear = {
        "levels": [{"l2_rel": 1e-4, "config": {"ny": 32, "nx": 64, "dt": 0.5}}],
        "observed_orders": {"pair": 2.0},
    }
    return [frozen, arm], shear


def test_gate_thresholds_come_from_the_contract():
    import xval_no_mcmp_two_phase_only_evolving as driver

    contract = json.loads(PREDECLARED.read_text())
    thr = contract["predeclared_thresholds"]
    matrix, shear = _fake_results()
    ok = driver.evaluate(matrix, shear, thr)
    assert ok["all_pass"], ok["failed_checks"]

    tightened = dict(thr)
    tightened["E1_l2_rel"] = 1e-6
    tightened["E6_observed_order_min"] = 99.0
    bad = driver.evaluate(matrix, shear, tightened)
    assert not bad["all_pass"]
    assert set(bad["failed_checks"]) == {"E1_l2_rel", "E6b_observed_order"}


def test_predeclared_contract_is_frozen_and_names_round1_artifacts():
    contract = json.loads(PREDECLARED.read_text())
    assert contract["status"] == "predeclared"
    assert contract["written_before_any_measurement"] is True
    assert "predeclared.json" in contract["round1_artifacts_stay_byte_frozen"]
    # every threshold the driver reads must exist in the contract
    for key in ("E1_l2_rel", "E3_interface_centre_shift_cells", "E6_observed_order_min",
                "E7_degradation_vs_frozen_l2_abs"):
        assert key in contract["predeclared_thresholds"]