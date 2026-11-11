"""Tests for the D3Q27 square-duct Poiseuille anchor (physical-accuracy evidence).

The point of these tests is *not* "the code ran": they bind the published
artifact to a re-executable measurement and to closed-form references, and
they include controls that must fail if the acceptance loses discriminating
power (a wrong-viscosity reference must be rejected; a quarter-length run must
not be better than the converged one).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from tensorlbm.duct_poiseuille_validation import (
    LITERATURE_POISEUILLE_NUMBER,
    LITERATURE_RESISTANCE,
    PROFILE_TOLERANCE,
    _canonical_sha256,
    check_series,
    control_mrt_bgk_limit,
    control_wrong_viscosity,
    duct_profile_exact,
    duct_series,
    run_duct_case,
)

ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "duct_poiseuille_validation" / "results.json"


@pytest.fixture(autouse=True)
def _single_thread() -> None:
    """Tiny 3D tensors: thread dispatch dominates the wall clock."""
    saved = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_num_threads(saved)


# --------------------------------------------------------------------------- #
# 1. the reference must be trustworthy before it can judge the solver
# --------------------------------------------------------------------------- #


def test_series_satisfies_no_slip_symmetry_and_poisson() -> None:
    ref = check_series(10)
    assert ref["wall_value_max"] < 1e-10, ref["wall_value_max"]
    assert ref["symmetry_max_abs_diff"] < 1e-6, ref["symmetry_max_abs_diff"]
    # analytically differentiated Laplacian vs -g/nu (no finite differences)
    assert ref["laplacian_rel_error"] < 1e-2, ref["laplacian_rel_error"]


def test_series_resistance_approaches_literature_value() -> None:
    """Integral functional of the closed form vs Shah & London f*Re = 56.91."""
    errors = []
    for length in (8, 10, 14):
        u, _, _ = duct_profile_exact(length, 8.0 * (1 / 6.0) * 0.05 / length**2, 1 / 6.0, n_terms=401)
        resistance = 8.0 * (1 / 6.0) * 0.05 / length**2 * length**2 / ((1 / 6.0) * float(u.mean()))
        errors.append(abs(resistance - LITERATURE_RESISTANCE) / LITERATURE_RESISTANCE)
    assert max(errors) < 0.03, errors


def test_series_laplacian_converges_with_term_count() -> None:
    """More terms must not make the self-check worse (guards a sign/系数 slip)."""
    coarse = check_series(10, n_terms=101)["laplacian_rel_error"]
    fine = check_series(10, n_terms=401)["laplacian_rel_error"]
    assert fine <= coarse * 1.5, (coarse, fine)


# --------------------------------------------------------------------------- #
# 2. the D3Q27 MRT composition against the closed form
# --------------------------------------------------------------------------- #


def test_duct_case_matches_closed_form_and_is_second_order() -> None:
    coarse = run_duct_case(length=8, steps=2000, collect_profile=True)
    fine = run_duct_case(length=10, steps=3000, collect_profile=True)
    # Ecosystem standard: 3 % is acceptable for a physics anchor.
    assert coarse["rel_error_max"] < 0.03, coarse["rel_error_max"]
    assert fine["rel_error_max"] < 0.03, fine["rel_error_max"]
    # Second-order spatial convergence (allow slack for the two rungs only).
    order = math.log(coarse["rel_error_max"] / fine["rel_error_max"]) / math.log(10 / 8)
    assert 1.5 < order < 3.0, order
    # A fully developed duct has no cross-flow and conserves mass.
    assert max(fine["max_uy"], fine["max_uz"]) / fine["u_max_num"] < 5e-2
    assert fine["mass_drift"] < 1e-8
    # Integral literature anchor.
    assert abs(fine["poiseuille_number_num"] - LITERATURE_POISEUILLE_NUMBER) / LITERATURE_POISEUILLE_NUMBER < 0.03


def test_profile_is_symmetric_about_the_duct_centre() -> None:
    case = run_duct_case(length=8, steps=1500, collect_profile=True)
    profile = np.asarray(case["profile_num_midz"], dtype=np.float64)
    assert profile.shape == (8,)
    assert np.max(np.abs(profile - profile[::-1])) / profile.max() < 1e-8


# --------------------------------------------------------------------------- #
# 3. controls: the acceptance must be able to fail
# --------------------------------------------------------------------------- #


def test_wrong_viscosity_reference_is_rejected() -> None:
    control = control_wrong_viscosity(length=8, steps=1200)
    assert control["discriminates"], control
    assert control["rel_error"] > PROFILE_TOLERANCE


def test_mrt_reduces_to_bgk_in_the_all_rates_limit() -> None:
    control = control_mrt_bgk_limit(length=6, steps=300)
    assert control["rel_diff"] < 1e-10, control
    # both collision families land on the same physics anchor
    assert abs(control["rel_error_mrt"] - control["rel_error_bgk"]) < 1e-8


# --------------------------------------------------------------------------- #
# 4. the published artifact must be self-consistent and still passing
# --------------------------------------------------------------------------- #


def test_artifact_exists_and_its_checks_pass() -> None:
    assert ARTIFACT.exists(), f"missing artifact: run python -m tensorlbm.duct_poiseuille_validation"
    data = json.loads(ARTIFACT.read_text())
    assert data["all_checks_passed"] is True, [c for c in data["checks"] if not c["passed"]]
    assert len(data["checks"]) >= 9, "vacuous check list"
    for name in (
        "series_wall_no_slip",
        "series_laplacian_matches_body_force",
        "duct_profile_error_finest",
        "duct_poiseuille_number_literature",
        "control_mrt_equals_bgk_in_limit",
    ):
        assert any(c["name"] == name for c in data["checks"]), name


def test_artifact_sha256_matches_its_own_content() -> None:
    data = json.loads(ARTIFACT.read_text())
    payload = {k: v for k, v in data.items() if k != "artifact_sha256"}
    assert data["artifact_sha256"] == _canonical_sha256(payload)


def test_artifact_ladder_is_monotone_and_passes_the_profile_gate() -> None:
    data = json.loads(ARTIFACT.read_text())
    errors = data["ladder_errors"]
    assert len(errors) >= 3
    assert all(b < a for a, b in zip(errors, errors[1:])), errors
    assert errors[-1] < PROFILE_TOLERANCE, errors[-1]
    assert all(1.5 < order < 2.5 for order in data["observed_orders"]), data["observed_orders"]


def test_artifact_records_withholding_and_forcing_convention_disclosure() -> None:
    data = json.loads(ARTIFACT.read_text())
    assert "WITHHELD" in data["general_capability_matrix_effect"]
    controls = {c["control"]: c for c in data["controls"]}
    # the forcing convention on solid cells must be disclosed, not hidden
    assert "body_force_on_solid_cells" in controls
    assert {"rel_error_fluid_only", "rel_error_force_everywhere"} <= set(controls["body_force_on_solid_cells"])