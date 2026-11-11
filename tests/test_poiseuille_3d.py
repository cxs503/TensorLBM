"""CASE-POISEUILLE-3D: 3D square-duct Poiseuille resolution ladder (D3Q27 MRT).

Acceptance (machine-checkable, thresholds imported from the module so the test
cannot drift from the artifact):

* ``velocity_profile_error <= 0.02``   -- centreline profile vs Boussinesq series
* ``convergence_order     >= 1.5``     -- observed order over 4 refinements
* ``mass_conservation_error <= 0.01``  -- |sum(rho) - N| / N

The suite is deliberately built so that it can *fail*: it re-runs the ladder
rather than trusting the committed JSON, it asserts the error decreases
monotonically under refinement, it asserts the steady-state residual is
machine-zero (a non-converged rung may not masquerade as discretisation
error), and it carries two negative controls (a mis-specified viscosity
reference and the circular-pipe parabola) that must be *rejected* by the same
2 % gate.  Extra steps beyond the ladder's step count must not change the
error, which pins the step count as "converged", not "convenient".
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from tensorlbm.duct_poiseuille_validation import duct_profile_exact, run_duct_case
from tensorlbm.poiseuille_3d_resolution import (
    MASS_TOLERANCE,
    ORDER_TARGET,
    PROFILE_TOLERANCE,
    observed_order,
    parabola_profile,
    run_ladder,
)

ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "poiseuille_3d_resolution"
    / "results.json"
)


@pytest.fixture(autouse=True)
def _single_thread() -> None:
    """Tiny 3D tensors: torch's thread dispatch would dominate wall clock."""
    saved = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_num_threads(saved)


@pytest.fixture(scope="module")
def ladder() -> dict:
    """The full 4-rung ladder, computed once per session."""
    saved = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    try:
        return run_ladder()
    finally:
        torch.set_num_threads(saved)


@pytest.fixture(scope="module")
def recheck() -> dict:
    """L=8 re-run with 1.5x the ladder step count, full profile kept."""
    saved = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    try:
        return run_duct_case(length=10, steps=3000, nx=4, tau=1.0, u_nominal=0.05)
    finally:
        torch.set_num_threads(saved)


# --------------------------------------------------------------------------- #
# 1. the three acceptance metrics
# --------------------------------------------------------------------------- #


def test_metric_velocity_profile_error_within_tolerance(ladder: dict) -> None:
    """Fine-rung centreline profile vs the closed-form square-duct series."""
    metric = ladder["metrics"]["velocity_profile_error"]
    assert ladder["metrics"]["velocity_profile_error_length"] == 12
    assert metric <= PROFILE_TOLERANCE, f"velocity_profile_error={metric:.5f} > 2%"
    # the mean error must be consistent with the max (no single-point blow-up
    # hiding a globally bad profile)
    assert ladder["rungs"][-1]["velocity_profile_error_mean"] <= metric


def test_metric_convergence_order_at_least_1_5(ladder: dict) -> None:
    """Observed order over L in {6,8,10,12} must beat the 1.5 floor."""
    order = ladder["metrics"]["convergence_order"]
    assert order >= ORDER_TARGET, f"convergence_order={order:.3f} < 1.5"
    # the floor must hold pair-by-pair too, not only in the global fit
    assert ladder["order"]["order_min_adjacent"] >= ORDER_TARGET, ladder["order"]
    # a genuine power law: log-log fit must be essentially straight
    assert ladder["order"]["fit_r2"] > 0.999, ladder["order"]


def test_metric_mass_conservation_within_tolerance(ladder: dict) -> None:
    """Body force on fluid cells only => total mass is stationary."""
    worst = ladder["metrics"]["mass_conservation_error"]
    assert worst <= MASS_TOLERANCE, f"mass_conservation_error={worst:.3e} > 1%"
    # measured level is ~1e-12; the 1 % gate is a formality, not the binding
    # constraint -- record that rather than leaving the reader to assume 1 %.
    assert worst < 1e-9, f"mass drift unexpectedly large: {worst:.3e}"
    for rung in ladder["rungs"]:
        assert rung["mass_conservation_error"] <= MASS_TOLERANCE


def test_error_decreases_monotonically_with_refinement(ladder: dict) -> None:
    errors = ladder["metrics"]["velocity_profile_error_ladder"]
    lengths = [r["length"] for r in ladder["rungs"]]
    assert lengths == [6, 8, 10, 12]
    assert all(errors[i + 1] < errors[i] for i in range(len(errors) - 1)), errors
    assert ladder["acceptance"]["monotone_error_decrease"] is True


def test_every_rung_is_at_steady_state(ladder: dict) -> None:
    """Otherwise a rung's 'error' is transient contamination, not discretisation."""
    for rung in ladder["rungs"]:
        assert rung["steady_rel_change_tail"] < 1e-9, rung
        assert math.isfinite(rung["u_max_num"])
        assert rung["u_max_num"] > 0.0


# --------------------------------------------------------------------------- #
# 2. independent integral + literature anchor
# --------------------------------------------------------------------------- #


def test_poiseuille_number_matches_shah_london(ladder: dict) -> None:
    """Integral functional f*Re = 56.91 (Shah & London 1978) at 3 % tolerance."""
    rel = ladder["metrics"]["poiseuille_number_rel_error_vs_shah_london"]
    assert rel <= 0.03, f"f*Re rel error vs 56.91 = {rel:.4f}"
    assert ladder["acceptance"]["poiseuille_number_vs_literature"] is True


def test_mean_velocity_integral_observable(ladder: dict) -> None:
    """Cross-section mean velocity (integral observable) vs the exact mean.

    This is the quantity the resistance / Poiseuille number is built from and
    it does not reuse the pointwise max metric, so it is an independent check.
    It is *harder* than the centreline max norm (near-wall cells carry the
    half-way bounce-back wall-position error and the 2D average weights them
    more heavily), hence a shallower margin -- recorded, not hidden.
    """
    errors = ladder["metrics"]["mean_velocity_error_ladder"]
    assert all(errors[i + 1] < errors[i] for i in range(len(errors) - 1)), errors
    assert ladder["metrics"]["mean_velocity_rel_error"] <= PROFILE_TOLERANCE
    assert ladder["metrics"]["mean_velocity_order"] >= ORDER_TARGET, ladder["metrics"]
    for rung in ladder["rungs"]:
        assert rung["u_mean_num"] > 0.0 and rung["u_mean_exact"] > 0.0


# --------------------------------------------------------------------------- #
# 3. discriminating power: the gate must be able to say "no"
# --------------------------------------------------------------------------- #


def test_parabola_negative_control_is_rejected(ladder: dict) -> None:
    """The circular-pipe parabola is not the square-duct answer.

    If a peak-matched parabola passed the same 2 % gate, that gate would be
    insensitive to profile shape and would therefore be worthless as evidence.
    """
    assert ladder["acceptance"]["parabola_negative_control_rejected"] is True
    assert ladder["metrics"]["parabola_error"] > PROFILE_TOLERANCE


def test_wrong_viscosity_reference_is_rejected(recheck: dict) -> None:
    """A 25 % viscosity mis-specification must fail the 2 % gate."""
    length = recheck["length"]
    wrong = duct_profile_exact(length, recheck["body_force"], recheck["nu"] * 1.25)[0]
    u_num = np.asarray(recheck["profile_num_midz"], dtype=np.float64)
    err = float(np.abs(u_num - wrong[len(wrong) // 2]).max() / wrong.max())
    assert err > PROFILE_TOLERANCE, f"wrong-viscosity error {err:.4f} not rejected"
    # ... while the correct reference at the same resolution passes
    assert recheck["rel_error_max"] <= PROFILE_TOLERANCE


def test_extra_steps_do_not_change_the_error(ladder: dict, recheck: dict) -> None:
    """Step count is converged, not merely convenient (1.5x steps, L=10)."""
    recorded = ladder["rungs"][2]["velocity_profile_error"]
    assert ladder["rungs"][2]["length"] == 10
    rel_drift = abs(recheck["rel_error_max"] - recorded) / recorded
    assert rel_drift < 0.01, f"error drifts {rel_drift:.4f} with 2x steps"


# --------------------------------------------------------------------------- #
# 4. the order estimator itself, and the reference shape
# --------------------------------------------------------------------------- #


def test_observed_order_recovers_exact_power_law() -> None:
    lengths = [6, 8, 10, 12]
    for p_true in (1.0, 1.5, 2.0, 3.0):
        errors = [1.7 * L ** (-p_true) for L in lengths]
        res = observed_order(errors, lengths)
        assert abs(res["order_fit"] - p_true) < 1e-9, res
        for p_adj in res["orders_adjacent"]:
            assert abs(p_adj - p_true) < 1e-9, res
        assert res["fit_r2"] > 1.0 - 1e-12


def test_observed_order_rejects_degenerate_input() -> None:
    with pytest.raises(ValueError):
        observed_order([0.01], [8])
    with pytest.raises(ValueError):
        observed_order([0.01, 0.0], [8, 10])


def test_parabola_profile_matches_wall_and_centre_values() -> None:
    length = 10
    para = parabola_profile(length, 1.0)
    assert para.shape == (length,)
    # same sampling points the reference series uses (cell centres, half-way walls)
    coord = np.arange(1, length + 1, dtype=float) - 0.5 * (length + 1.0)
    a = 0.5 * length
    assert np.allclose(para, 1.0 - (coord / a) ** 2)
    assert para[-1] == pytest.approx(para[0])  # symmetry about the centre
    assert para.max() < 1.0  # cell centres never sit exactly at the peak
    # the parabola differs from the square-duct series by a few percent and
    # that is exactly why it works as a negative control
    exact_2d = duct_profile_exact(length, 8.0 * (1.0 / 6.0) * 0.05 / length**2, 1.0 / 6.0)[0]
    exact = exact_2d[len(exact_2d) // 2]  # mid-z row == the centreline the ladder uses
    rel = float(np.max(np.abs(para * exact.max() - exact)) / exact.max())
    assert PROFILE_TOLERANCE < rel < 0.2, rel


# --------------------------------------------------------------------------- #
# 5. provenance: the committed artifact must match a live recomputation
# --------------------------------------------------------------------------- #


def test_artifact_matches_recomputation(ladder: dict) -> None:
    if not ARTIFACT.exists():  # artifact is a product, not a prerequisite
        pytest.skip(f"artifact not generated: {ARTIFACT}")
    payload = json.loads(ARTIFACT.read_text())
    assert payload["artifact_id"] == "poiseuille-3d-resolution-r1"
    assert payload["all_accepted"] is True
    assert payload["ladder"] == ladder["ladder"]
    for stored, live in zip(payload["rungs"], ladder["rungs"]):
        assert stored["length"] == live["length"]
        assert stored["velocity_profile_error"] == pytest.approx(
            live["velocity_profile_error"], rel=1e-9, abs=1e-12
        )
        assert stored["mass_conservation_error"] == pytest.approx(
            live["mass_conservation_error"], rel=1e-6, abs=1e-15
        )
    assert payload["metrics"]["convergence_order"] == pytest.approx(
        ladder["metrics"]["convergence_order"], rel=1e-9
    )