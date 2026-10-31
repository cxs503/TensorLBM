"""Pytest test suite for ITTC Ice Pool Benchmarking Framework.

Tests cover:
1. Ship definitions validation
2. Ice resistance model correctness
3. Benchmark runner acceptance criteria
4. Report generation
5. Reproducibility checks
"""

import json
import math
import os
import sys
import tempfile

import pytest

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorlbm.ittc_ice_benchmark import (
    SHIP_DATABASE,
    ShipDefinition,
    ITTCIceResistanceModel,
    IceProperties,
    IceResistanceResult,
    ITTCBenchmarkRunner,
    BenchmarkResult,
    ITTCReportGenerator,
    compute_ice_resistance,
    generate_comparison_report,
    list_ship_types,
    run_full_benchmark,
)
from tensorlbm.ittc_ice_benchmark.ship_definitions import get_ship_by_key, get_ship_by_type


# ── Ship Definitions Tests ─────────────────────────────────────────────────────

class TestShipDefinitions:
    """Tests for ship database and definitions."""

    def test_database_has_at_least_3_ships(self):
        """Verify database contains at least 3 ship types."""
        assert len(SHIP_DATABASE) >= 3

    def test_required_ship_types_present(self):
        """Verify container, bulk, and research ship types exist."""
        types = list_ship_types()
        assert "container" in types
        assert "bulk" in types
        assert "research" in types

    def test_ship_dimensions_positive(self):
        """Verify all ship dimensions are positive."""
        for key, ship in SHIP_DATABASE.items():
            assert ship.L_pp > 0, f"{key}: L_pp must be positive"
            assert ship.B > 0, f"{key}: B must be positive"
            assert ship.T > 0, f"{key}: T must be positive"
            assert ship.Cb > 0 and ship.Cb < 1, f"{key}: Cb must be in (0,1)"
            assert ship.Cwp > 0 and ship.Cwp < 1, f"{key}: Cwp must be in (0,1)"

    def test_ship_geometry_consistency(self):
        """Verify geometric consistency (displacement ~ L*B*T*Cb)."""
        for key, ship in SHIP_DATABASE.items():
            approx_disp = ship.L_pp * ship.B * ship.T * ship.Cb
            # Allow 30% tolerance for hull form effects
            ratio = ship.displacement / approx_disp
            assert 0.7 < ratio < 1.3, (
                f"{key}: displacement/approx ratio = {ratio:.2f}"
            )

    def test_get_ship_by_type(self):
        """Verify get_ship_by_type returns correct ship."""
        ship = get_ship_by_type("container")
        assert ship.ship_type == "container"
        assert ship.L_pp > 0

    def test_get_ship_by_key(self):
        """Verify get_ship_by_key returns correct ship."""
        ship = get_ship_by_key("container_vessel")
        assert ship.name == "Arctic Container 4400"

    def test_invalid_ship_type_raises(self):
        """Verify invalid ship type raises KeyError."""
        with pytest.raises(KeyError):
            get_ship_by_type("nonexistent")

    def test_ship_derived_properties(self):
        """Verify derived properties are computed correctly."""
        ship = SHIP_DATABASE["container_vessel"]
        assert ship.length_beam_ratio == ship.L_pp / ship.B
        assert ship.breadth_draught_ratio == ship.B / ship.T


# ── Ice Properties Tests ───────────────────────────────────────────────────────

class TestIceProperties:
    """Tests for ice property definitions."""

    def test_default_properties(self):
        """Verify default ice properties are valid."""
        ice = IceProperties()
        assert ice.thickness_m > 0
        assert ice.flexural_strength_Pa > 0
        assert ice.density_kg_m3 > 0

    def test_custom_properties(self):
        """Verify custom ice properties are accepted."""
        ice = IceProperties(thickness_m=1.0, flexural_strength_Pa=600e3)
        assert ice.thickness_m == 1.0
        assert ice.flexural_strength_Pa == 600e3

    def test_invalid_thickness_raises(self):
        """Verify negative thickness raises ValueError."""
        with pytest.raises(ValueError):
            IceProperties(thickness_m=-0.5)

    def test_invalid_friction_raises(self):
        """Verify friction coefficient out of range raises ValueError."""
        with pytest.raises(ValueError):
            IceProperties(friction_coefficient=1.5)


# ── Ice Resistance Model Tests ─────────────────────────────────────────────────

class TestIceResistanceModel:
    """Tests for ice resistance prediction model."""

    def test_open_water_resistance_positive(self):
        """Verify open water resistance is positive."""
        model = ITTCIceResistanceModel()
        ship = SHIP_DATABASE["container_vessel"]
        R_ow = model.compute_open_water_resistance(ship, 5.0)
        assert R_ow > 0

    def test_ice_breaking_resistance_positive(self):
        """Verify ice breaking resistance is positive."""
        model = ITTCIceResistanceModel()
        ship = SHIP_DATABASE["container_vessel"]
        R_break = model.compute_ice_breaking_resistance(ship, 5.0)
        assert R_break > 0

    def test_total_resistance_positive(self):
        """Verify total ice resistance is positive."""
        model = ITTCIceResistanceModel()
        ship = SHIP_DATABASE["container_vessel"]
        result = model.compute_resistance(ship, 5.0, "container_vessel")
        assert result.total_resistance_N > 0

    def test_resistance_increases_with_thickness(self):
        """Verify ice resistance increases with ice thickness."""
        ship = SHIP_DATABASE["container_vessel"]
        R_thin = compute_ice_resistance(ship, 5.0, ice_thickness_m=0.4)
        R_thick = compute_ice_resistance(ship, 5.0, ice_thickness_m=1.2)
        assert R_thick.total_resistance_N > R_thin.total_resistance_N

    def test_resistance_increases_with_speed(self):
        """Verify ice resistance increases with speed."""
        ship = SHIP_DATABASE["container_vessel"]
        R_slow = compute_ice_resistance(ship, 2.0)
        R_fast = compute_ice_resistance(ship, 6.0)
        assert R_fast.total_resistance_N > R_slow.total_resistance_N

    def test_result_has_all_components(self):
        """Verify result contains all resistance components."""
        ship = SHIP_DATABASE["container_vessel"]
        result = compute_ice_resistance(ship, 5.0)
        assert result.breaking_resistance_N >= 0
        assert result.clearance_resistance_N >= 0
        assert result.open_water_resistance_N >= 0
        assert result.friction_resistance_N >= 0

    def test_relative_error_computation(self):
        """Verify relative error computation."""
        ship = SHIP_DATABASE["container_vessel"]
        result = compute_ice_resistance(ship, 5.0)
        error = result.relative_error(3000.0)
        assert error >= 0
        expected = abs(result.total_resistance_kN - 3000.0) / 3000.0
        assert abs(error - expected) < 1e-10


# ── Benchmark Runner Tests ─────────────────────────────────────────────────────

class TestBenchmarkRunner:
    """Tests for the ITTC benchmark runner."""

    def test_runner_initializes_with_defaults(self):
        """Verify runner initializes with default configuration."""
        runner = ITTCBenchmarkRunner()
        assert len(runner.ship_keys) >= 3

    def test_single_benchmark_returns_result(self):
        """Verify single benchmark returns valid result."""
        runner = ITTCBenchmarkRunner()
        result = runner.run_single_benchmark("container_vessel")
        assert isinstance(result, BenchmarkResult)
        assert result.predicted_resistance_kN > 0
        assert result.reference_resistance_kN > 0
        assert result.relative_error >= 0

    def test_full_benchmark_generates_report(self):
        """Verify full benchmark generates complete report."""
        runner = ITTCBenchmarkRunner()
        report = runner.run_full_benchmark(n_reproducibility_runs=3)
        assert report.ship_types_validated >= 0
        assert report.auto_report_generated is True
        assert report.reproducibility_across_runs >= 0
        assert len(report.results) >= 3

    def test_reproducibility_is_deterministic(self):
        """Verify reproducibility CV is very small (deterministic model)."""
        runner = ITTCBenchmarkRunner()
        cv = runner.check_reproducibility(n_runs=5)
        # The model is deterministic, so CV should be essentially 0
        assert cv < 0.02

    def test_acceptance_criteria_ship_types(self):
        """Verify ship_types_validated >= 3 criterion."""
        runner = ITTCBenchmarkRunner()
        report = runner.run_full_benchmark(n_reproducibility_runs=3)
        assert report.ship_types_validated >= 3

    def test_acceptance_criteria_reproducibility(self):
        """Verify reproducibility < 0.02 criterion."""
        runner = ITTCBenchmarkRunner()
        report = runner.run_full_benchmark(n_reproducibility_runs=3)
        assert report.reproducibility_across_runs < 0.02

    def test_acceptance_criteria_auto_report(self):
        """Verify auto_report_generated == True criterion."""
        runner = ITTCBenchmarkRunner()
        report = runner.run_full_benchmark(n_reproducibility_runs=3)
        assert report.auto_report_generated is True

    def test_all_criteria_pass(self):
        """Verify all acceptance criteria pass."""
        runner = ITTCBenchmarkRunner()
        report = runner.run_full_benchmark(n_reproducibility_runs=3)
        assert report.all_criteria_pass is True

    def test_export_json(self):
        """Verify JSON export works."""
        runner = ITTCBenchmarkRunner()
        report = runner.run_full_benchmark(n_reproducibility_runs=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_report.json")
            runner.export_report_json(report, path)
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert "acceptance_criteria" in data
            assert "results" in data


# ── Report Generator Tests ─────────────────────────────────────────────────────

class TestReportGenerator:
    """Tests for report generation."""

    def test_json_report_generation(self):
        """Verify JSON report is generated correctly."""
        report = run_full_benchmark(n_reproducibility_runs=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_comparison_report(report, tmpdir, formats=["json"])
            assert "json" in paths
            assert os.path.exists(paths["json"])
            with open(paths["json"]) as f:
                data = json.load(f)
            assert "report_metadata" in data
            assert "acceptance_criteria" in data

    def test_markdown_report_generation(self):
        """Verify Markdown report is generated correctly."""
        report = run_full_benchmark(n_reproducibility_runs=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_comparison_report(report, tmpdir, formats=["markdown"])
            assert "markdown" in paths
            assert os.path.exists(paths["markdown"])
            with open(paths["markdown"]) as f:
                content = f.read()
            assert "ITTC Ice Pool" in content
            assert "Acceptance Criteria" in content

    def test_both_formats_generation(self):
        """Verify both JSON and Markdown reports are generated."""
        report = run_full_benchmark(n_reproducibility_runs=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_comparison_report(report, tmpdir)
            assert "json" in paths
            assert "markdown" in paths
            assert os.path.exists(paths["json"])
            assert os.path.exists(paths["markdown"])


# ── Integration Tests ──────────────────────────────────────────────────────────

class TestIntegration:
    """Integration tests for the full benchmarking pipeline."""

    def test_full_pipeline(self):
        """Verify full pipeline from ship definition to report."""
        # Run benchmark
        report = run_full_benchmark(
            ship_keys=["container_vessel", "bulk_carrier", "research_vessel"],
            ice_thickness_m=0.8,
        )

        # Verify acceptance criteria
        assert report.ship_types_validated >= 3
        assert report.auto_report_generated is True
        assert report.reproducibility_across_runs < 0.02
        assert report.all_criteria_pass is True

    def test_four_ship_types(self):
        """Verify benchmark works with all 4 ship types."""
        report = run_full_benchmark(
            ship_keys=list(SHIP_DATABASE.keys()),
            ice_thickness_m=0.8,
        )
        assert len(report.results) == 4
        assert report.ship_types_validated >= 3

    def test_different_ice_thicknesses(self):
        """Verify benchmark works with different ice thicknesses."""
        for thickness in [0.4, 0.8, 1.2]:
            report = run_full_benchmark(ice_thickness_m=thickness)
            assert report.auto_report_generated is True
            assert len(report.results) >= 3
