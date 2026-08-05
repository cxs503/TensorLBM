"""Tests for high-Re cylinder flow turbulence model comparison.

status=diagnostic_only

Tests the effect of SGS turbulence models (none/Smagorinsky/WALE/Vreman/
DynSmag) on drag coefficient (Cd) and Strouhal number at Re=1000 and Re=5000
using D2Q9 BGK and MRT collision operators.

These tests verify:
    - The collision dispatch table covers all 7 D2Q9 combinations.
    - A single run returns the required machine-readable fields.
    - At Re=1000, baseline and SGS runs produce finite Cd.
    - The full matrix produces 14 results with correct schema.
    - A JSON artifact is written to disk.
    - The target-grid (200×100, 500-step) matrix runs end-to-end.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

# Reduce thread count to avoid overhead on small grids
torch.set_num_threads(4)

from tensorlbm.high_re_turbulence_test import (  # noqa: E402
    COLLISION_DISPATCH,
    run_high_re_cylinder,
    run_high_re_turbulence_matrix,
)

# Fast unit-test grid
SMALL_NX, SMALL_NY, SMALL_STEPS = 100, 50, 200
# Target diagnostic grid
TARGET_NX, TARGET_NY, TARGET_STEPS = 200, 100, 500


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

def test_dispatch_table_completeness() -> None:
    """All 7 D2Q9 collision+turbulence combinations are registered."""
    expected = {
        ("bgk", "none"),
        ("bgk", "smagorinsky"),
        ("bgk", "wale"),
        ("bgk", "vreman"),
        ("bgk", "dynsmag"),
        ("mrt", "none"),
        ("mrt", "smagorinsky"),
    }
    assert set(COLLISION_DISPATCH.keys()) == expected


def test_dispatch_table_excludes_unavailable_mrt_sgs() -> None:
    """WALE/Vreman/DynSmag have no D2Q9 MRT variant."""
    for turb in ("wale", "vreman", "dynsmag"):
        assert ("mrt", turb) not in COLLISION_DISPATCH


# ---------------------------------------------------------------------------
# Single-run contract
# ---------------------------------------------------------------------------

def test_single_run_returns_required_fields() -> None:
    """A single run returns a dict with all required machine-readable fields."""
    result = run_high_re_cylinder(
        re=1000,
        collision="bgk",
        turbulence_model="none",
        nx=SMALL_NX,
        ny=SMALL_NY,
        steps=SMALL_STEPS,
    )
    required = {"Re", "collision", "turbulence_model", "Cd", "Strouhal", "finite"}
    assert set(result.keys()) >= required


def test_single_run_field_types() -> None:
    """Field types match the machine-readable schema."""
    result = run_high_re_cylinder(
        re=1000,
        collision="bgk",
        turbulence_model="none",
        nx=SMALL_NX,
        ny=SMALL_NY,
        steps=SMALL_STEPS,
    )
    assert isinstance(result["Re"], int)
    assert isinstance(result["collision"], str)
    assert isinstance(result["turbulence_model"], str)
    assert isinstance(result["Cd"], float)
    assert isinstance(result["Strouhal"], float)
    assert isinstance(result["finite"], bool)


def test_unsupported_combination_raises() -> None:
    """Unsupported (collision, turbulence) raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported"):
        run_high_re_cylinder(
            re=1000,
            collision="mrt",
            turbulence_model="wale",
            nx=SMALL_NX,
            ny=SMALL_NY,
            steps=SMALL_STEPS,
        )


# ---------------------------------------------------------------------------
# Re=1000 finiteness (small grid)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "collision, turbulence_model",
    sorted(COLLISION_DISPATCH.keys()),
)
def test_re1000_all_combinations_finite(
    collision: str,
    turbulence_model: str,
) -> None:
    """At Re=1000 every registered combination should produce finite Cd."""
    result = run_high_re_cylinder(
        re=1000,
        collision=collision,
        turbulence_model=turbulence_model,
        nx=SMALL_NX,
        ny=SMALL_NY,
        steps=SMALL_STEPS,
    )
    assert result["finite"], (
        f"Re=1000 {collision}/{turbulence_model}: "
        f"Cd={result['Cd']}, St={result['Strouhal']}"
    )


# ---------------------------------------------------------------------------
# Matrix run + artifact
# ---------------------------------------------------------------------------

def test_matrix_small_grid_result_count() -> None:
    """Matrix run on small grid produces exactly 14 results."""
    results = run_high_re_turbulence_matrix(
        nx=SMALL_NX,
        ny=SMALL_NY,
        steps=SMALL_STEPS,
    )
    assert len(results) == 14


def test_matrix_small_grid_schema() -> None:
    """Every result in the matrix has the required schema."""
    results = run_high_re_turbulence_matrix(
        nx=SMALL_NX,
        ny=SMALL_NY,
        steps=SMALL_STEPS,
    )
    for r in results:
        assert r["Re"] in (1000, 5000)
        assert r["collision"] in ("bgk", "mrt")
        assert r["turbulence_model"] in (
            "none", "smagorinsky", "wale", "vreman", "dynsmag",
        )
        assert isinstance(r["Cd"], float)
        assert isinstance(r["Strouhal"], float)
        assert isinstance(r["finite"], bool)


def test_matrix_json_artifact(tmp_path: Path) -> None:
    """Matrix run writes a valid JSON artifact to disk."""
    out = tmp_path / "high_re_turbulence.json"
    results = run_high_re_turbulence_matrix(
        nx=SMALL_NX,
        ny=SMALL_NY,
        steps=SMALL_STEPS,
        output_path=out,
    )
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert len(loaded) == 14
    assert loaded == results


def test_matrix_re1000_results_finite() -> None:
    """At Re=1000, all 7 combinations should produce finite results."""
    results = run_high_re_turbulence_matrix(
        nx=SMALL_NX,
        ny=SMALL_NY,
        steps=SMALL_STEPS,
    )
    re1000 = [r for r in results if r["Re"] == 1000]
    assert len(re1000) == 7
    for r in re1000:
        assert r["finite"], (
            f"Re=1000 {r['collision']}/{r['turbulence_model']}: "
            f"Cd={r['Cd']}, St={r['Strouhal']}"
        )


# ---------------------------------------------------------------------------
# Target-grid diagnostic (200×100, 500 steps)
# ---------------------------------------------------------------------------

def test_target_grid_matrix_end_to_end(tmp_path: Path) -> None:
    """Full diagnostic matrix at target grid (200×100, 500 steps).

    status=diagnostic_only — verifies that the matrix runs end-to-end and
    produces a machine-readable artifact.  At least the Re=1000 results
    should be finite.
    """
    out = tmp_path / "high_re_turbulence_target.json"
    results = run_high_re_turbulence_matrix(
        nx=TARGET_NX,
        ny=TARGET_NY,
        steps=TARGET_STEPS,
        output_path=out,
    )
    assert len(results) == 14
    assert out.exists()

    re1000 = [r for r in results if r["Re"] == 1000]
    finite_1000 = [r for r in re1000 if r["finite"]]
    assert len(finite_1000) >= 5, (
        f"Only {len(finite_1000)}/7 Re=1000 runs were finite"
    )

    # At least one Re=5000 result should be finite (SGS should help)
    re5000 = [r for r in results if r["Re"] == 5000]
    finite_5000 = [r for r in re5000 if r["finite"]]
    assert len(finite_5000) >= 1, "No Re=5000 runs were finite"
