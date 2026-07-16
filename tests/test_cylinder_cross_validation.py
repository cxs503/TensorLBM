"""Tests for D2Q9 cylinder cross-validation (collision × turbulence matrix).

TDD: these tests define the expected interface and behaviour of the
cross-validation runner *before* the implementation is complete.

All combinations are diagnostic-only (status="diagnostic_only",
physical_validation=False).  The grid is deliberately small (100×50,
200 steps) so the full 4×4 matrix runs in a few minutes on CPU.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorlbm.cylinder_cross_validation import (
    D2Q9_COLLISION_FAMILIES,
    D2Q9_TURBULENCE_MODELS,
    run_single_combination,
    run_cross_validation_matrix,
)


# ---------------------------------------------------------------------------
# Matrix dimension sanity
# ---------------------------------------------------------------------------

def test_collision_families_match_d2q9() -> None:
    """D2Q9 has exactly BGK, MRT, TRT, RLBM (no CM/CUMULANT/KBC)."""
    assert set(D2Q9_COLLISION_FAMILIES) == {"BGK", "MRT", "TRT", "RLBM"}


def test_turbulence_models_match_spec() -> None:
    assert set(D2Q9_TURBULENCE_MODELS) == {"none", "Smagorinsky", "WALE", "Vreman"}


def test_matrix_is_4x4() -> None:
    assert len(D2Q9_COLLISION_FAMILIES) == 4
    assert len(D2Q9_TURBULENCE_MODELS) == 4
    assert len(D2Q9_COLLISION_FAMILIES) * len(D2Q9_TURBULENCE_MODELS) == 16


# ---------------------------------------------------------------------------
# Single-combination interface
# ---------------------------------------------------------------------------

def test_single_combination_returns_dict_with_required_fields() -> None:
    """Each result must carry the machine-readable schema fields."""
    result = run_single_combination(
        collision_family="BGK",
        turbulence_model="none",
        re=100,
        nx=100,
        ny=50,
        steps=200,
    )
    required = {
        "collision_family", "turbulence_model", "Cd",
        "finite", "steps_completed",
        "status", "physical_validation",
    }
    assert required.issubset(result.keys()), f"missing keys: {required - set(result.keys())}"


def test_single_combination_bgk_none_finite() -> None:
    """BGK + none at Re=100 should produce a finite Cd."""
    result = run_single_combination("BGK", "none", re=100, nx=100, ny=50, steps=200)
    assert result["finite"] is True
    assert math.isfinite(result["Cd"])
    assert result["steps_completed"] == 200
    assert result["status"] == "diagnostic_only"
    assert result["physical_validation"] is False


def test_single_combination_status_diagnostic() -> None:
    result = run_single_combination("BGK", "none", re=100, nx=100, ny=50, steps=50)
    assert result["status"] == "diagnostic_only"
    assert result["physical_validation"] is False


# ---------------------------------------------------------------------------
# Full matrix
# ---------------------------------------------------------------------------

def test_full_matrix_has_16_entries(tmp_path: Path) -> None:
    artifact_path = tmp_path / "matrix.json"
    results = run_cross_validation_matrix(
        re=100, nx=100, ny=50, steps=50,
        artifact_path=str(artifact_path),
    )
    assert len(results) == 16
    # Every combination present
    seen = {(r["collision_family"], r["turbulence_model"]) for r in results}
    expected = {
        (cf, tm)
        for cf in D2Q9_COLLISION_FAMILIES
        for tm in D2Q9_TURBULENCE_MODELS
    }
    assert seen == expected


def test_artifact_is_valid_json(tmp_path: Path) -> None:
    artifact_path = tmp_path / "matrix.json"
    run_cross_validation_matrix(
        re=100, nx=100, ny=50, steps=50,
        artifact_path=str(artifact_path),
    )
    data = json.loads(artifact_path.read_text())
    assert "matrix" in data
    assert len(data["matrix"]) == 16
    for entry in data["matrix"]:
        assert "collision_family" in entry
        assert "turbulence_model" in entry
        assert "Cd" in entry
        assert "finite" in entry
        assert "steps_completed" in entry


@pytest.mark.parametrize("cf", ["BGK", "MRT", "TRT", "RLBM"])
@pytest.mark.parametrize("tm", ["none", "Smagorinsky", "WALE", "Vreman"])
def test_each_combination_finite_or_handled(cf: str, tm: str) -> None:
    """Every combination must either produce a finite Cd or report finite=False."""
    result = run_single_combination(cf, tm, re=100, nx=100, ny=50, steps=50)
    if result["finite"]:
        assert math.isfinite(result["Cd"]), f"{cf}×{tm}: Cd not finite"
        assert result["steps_completed"] == 50
    else:
        # If it crashed/NaN, steps_completed may be < 50
        assert result["steps_completed"] <= 50
