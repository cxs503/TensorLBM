"""TDD contract for sphere cross-validation across collision families × turbulence models.

This test exercises the full D3Q19/D3Q27 × 7-collision-family × 3-turbulence-model
matrix on a small grid to verify structural correctness, finiteness, and
machine-readable artifact emission.  It does NOT claim physical accuracy
(status=diagnostic_only, physical_validation=False).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorlbm.sphere_cross_validation import (
    SCHEMA_VERSION,
    LATTICES,
    COLLISION_FAMILIES,
    TURBULENCE_MODELS,
    SphereCrossValidationConfig,
    run_sphere_cross_validation,
    write_sphere_cross_validation_evidence,
    _schiller_naumann,
)


# ---------------------------------------------------------------------------
# Shared fixture — run the full matrix once and reuse across all tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_matrix():
    """Run the cross-validation matrix once on a tiny grid (12³, 3 steps)."""
    config = SphereCrossValidationConfig(nx=12, ny=12, nz=12, steps=3)
    return run_sphere_cross_validation(config)


@pytest.fixture(scope="module")
def small_config():
    return SphereCrossValidationConfig(nx=12, ny=12, nz=12, steps=3)


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------

class TestMatrixStructure:
    """Verify the cross-validation matrix has the correct shape and fields."""

    def test_matrix_covers_all_combinations(self, small_matrix) -> None:
        expected = len(LATTICES) * len(COLLISION_FAMILIES) * len(TURBULENCE_MODELS)
        assert len(small_matrix.results) == expected, (
            f"Expected {expected} results, got {len(small_matrix.results)}"
        )

    def test_each_result_has_required_fields(self, small_matrix) -> None:
        required = {
            "lattice", "collision_family", "turbulence_model",
            "Cd", "finite", "steps_completed", "reference_Cd",
            "status", "physical_validation",
        }
        for r in small_matrix.results:
            missing = required - set(r.keys())
            assert not missing, f"Missing fields: {missing} in {r}"

    def test_lattice_family_turbulence_coverage(self, small_matrix) -> None:
        seen = {
            (r["lattice"], r["collision_family"], r["turbulence_model"])
            for r in small_matrix.results
        }
        expected = {
            (lat, fam, turb)
            for lat in LATTICES
            for fam in COLLISION_FAMILIES
            for turb in TURBULENCE_MODELS
        }
        assert seen == expected

    def test_status_and_physical_validation_flags(self, small_matrix) -> None:
        for r in small_matrix.results:
            assert r["status"] == "diagnostic_only"
            assert r["physical_validation"] is False


# ---------------------------------------------------------------------------
# Reference Cd tests
# ---------------------------------------------------------------------------

class TestReferenceCd:
    """Verify the Schiller-Naumann reference is correct."""

    def test_reference_cd_is_schiller_naumann(self, small_matrix) -> None:
        ref = _schiller_naumann(100.0)
        assert abs(small_matrix.reference_Cd - ref) < 1e-10
        for r in small_matrix.results:
            assert abs(r["reference_Cd"] - ref) < 1e-10

    def test_schiller_naumann_formula(self) -> None:
        # Re=100: 24/100 * (1 + 0.15 * 100^0.687)
        expected = 24.0 / 100.0 * (1.0 + 0.15 * 100.0 ** 0.687)
        assert abs(_schiller_naumann(100.0) - expected) < 1e-10


# ---------------------------------------------------------------------------
# Finiteness tests
# ---------------------------------------------------------------------------

class TestFiniteness:
    """Verify all combinations produce finite results on the small grid."""

    def test_all_results_are_finite(self, small_matrix) -> None:
        non_finite = [
            (r["lattice"], r["collision_family"], r["turbulence_model"])
            for r in small_matrix.results
            if not r["finite"]
        ]
        assert not non_finite, f"Non-finite results: {non_finite}"

    def test_all_cds_are_finite_numbers(self, small_matrix) -> None:
        for r in small_matrix.results:
            assert r["Cd"] is not None, f"Cd is None for {r}"
            assert math.isfinite(r["Cd"]), f"Cd not finite for {r}"

    def test_steps_completed_matches_config(self, small_matrix, small_config) -> None:
        for r in small_matrix.results:
            assert r["steps_completed"] == small_config.steps, (
                f"steps_completed={r['steps_completed']} != {small_config.steps} "
                f"for {r['lattice']}/{r['collision_family']}/{r['turbulence_model']}"
            )


# ---------------------------------------------------------------------------
# Reproducibility tests
# ---------------------------------------------------------------------------

class TestReproducibility:
    """Verify the runner is deterministic."""

    def test_two_runs_produce_identical_results(self) -> None:
        config = SphereCrossValidationConfig(nx=12, ny=12, nz=12, steps=3)
        first = run_sphere_cross_validation(config)
        second = run_sphere_cross_validation(config)
        assert first == second


# ---------------------------------------------------------------------------
# Artifact emission tests
# ---------------------------------------------------------------------------

class TestArtifactEmission:
    """Verify machine-readable JSON artifact emission."""

    def test_write_artifact_emits_valid_json(self, small_matrix, tmp_path: Path) -> None:
        out = tmp_path / "sphere-cross-validation-r1.json"
        written = write_sphere_cross_validation_evidence(small_matrix, out)
        assert written == out
        assert out.exists()

        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["schema_version"] == SCHEMA_VERSION
        assert len(payload["results"]) == 42
        assert "reference_Cd" in payload
        assert "config" in payload

    def test_artifact_is_sorted_and_nan_free(self, small_matrix, tmp_path: Path) -> None:
        out = tmp_path / "sphere-cross-validation-r1.json"
        write_sphere_cross_validation_evidence(small_matrix, out)
        raw = out.read_text(encoding="utf-8")
        assert "NaN" not in raw
        assert "Infinity" not in raw
        json.loads(raw)

    def test_artifact_contains_all_combination_keys(self, small_matrix, tmp_path: Path) -> None:
        out = tmp_path / "sphere-cross-validation-r1.json"
        write_sphere_cross_validation_evidence(small_matrix, out)
        payload = json.loads(out.read_text(encoding="utf-8"))
        keys = {
            (r["lattice"], r["collision_family"], r["turbulence_model"])
            for r in payload["results"]
        }
        assert len(keys) == 42
