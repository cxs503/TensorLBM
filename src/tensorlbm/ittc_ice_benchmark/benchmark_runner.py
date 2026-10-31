"""ITTC Ice Pool Benchmark Runner.

Executes the benchmarking procedure across multiple ship types,
computes error metrics against ITTC reference data, and checks
reproducibility across multiple runs.

The runner follows the ITTC 7.5-04-01-01.1 procedure:
1. Define ship geometry and ice conditions
2. Compute ice resistance using semi-empirical models
3. Compare against reference model test data
4. Assess reproducibility (coefficient of variation < 2%)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from .ice_resistance_model import (
    ITTCIceResistanceModel,
    IceProperties,
    IceResistanceResult,
    compute_ice_resistance,
)
from .ship_definitions import SHIP_DATABASE, ShipDefinition, get_ship_by_key


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run for one ship type.

    Attributes:
        ship_key: Database key for the ship
        ship_type: Ship category (container/bulk/research/tanker)
        predicted_resistance_kN: Predicted total ice resistance [kN]
        reference_resistance_kN: ITTC reference resistance [kN]
        relative_error: |predicted - reference| / reference [-]
        error_within_15pct: Whether error is within 15% threshold
        components: Breakdown of resistance components [N]
        run_timestamp: ISO timestamp of the run
        run_hash: Hash for reproducibility tracking
    """

    ship_key: str
    ship_type: str
    predicted_resistance_kN: float
    reference_resistance_kN: float
    relative_error: float
    error_within_15pct: bool
    components: Dict[str, float]
    run_timestamp: str
    run_hash: str


@dataclass
class FullBenchmarkReport:
    """Complete benchmark report across all ship types.

    Attributes:
        ship_types_validated: Number of ship types validated
        results: Per-ship benchmark results
        auto_report_generated: Whether report was auto-generated
        reproducibility_across_runs: Max coefficient of variation across runs
        all_criteria_pass: Whether all acceptance criteria are met
        ice_properties: Ice properties used
        run_metadata: Additional run metadata
    """

    ship_types_validated: int
    results: List[Dict]
    auto_report_generated: bool
    reproducibility_across_runs: float
    all_criteria_pass: bool
    ice_properties: Dict[str, float]
    run_metadata: Dict


class ITTCBenchmarkRunner:
    """Runner for ITTC ice pool benchmarking campaign.

    Executes the benchmark procedure for multiple ship types,
    computes error metrics, and assesses reproducibility.
    """

    # Acceptance thresholds per ITTC recommendations
    MAX_RELATIVE_ERROR = 0.15  # 15% max error vs reference
    MAX_CV_REPRODUCIBILITY = 0.02  # 2% max coefficient of variation
    MIN_SHIP_TYPES = 3  # Minimum ship types to validate

    def __init__(
        self,
        ice_properties: Optional[IceProperties] = None,
        ship_keys: Optional[List[str]] = None,
    ):
        """Initialize benchmark runner.

        Args:
            ice_properties: Ice properties for all runs. If None, uses defaults.
            ship_keys: List of ship database keys to benchmark.
                       If None, uses first 3+ ships from database.
        """
        self.ice_properties = ice_properties or IceProperties()
        self.model = ITTCIceResistanceModel(self.ice_properties)

        if ship_keys is None:
            # Default: container, bulk, research (minimum 3)
            self.ship_keys = ["container_vessel", "bulk_carrier", "research_vessel"]
        else:
            self.ship_keys = ship_keys

    def run_single_benchmark(self, ship_key: str) -> BenchmarkResult:
        """Run benchmark for a single ship type.

        Uses the ship's reference ice thickness for the calculation.

        Args:
            ship_key: Database key for the ship

        Returns:
            BenchmarkResult with error metrics
        """
        ship = get_ship_by_key(ship_key)

        # Use ship's reference ice thickness
        ice_props = IceProperties(thickness_m=ship.reference_ice_thickness)
        model = ITTCIceResistanceModel(ice_props)

        # Compute resistance at reference speed and ice thickness
        result = model.compute_resistance(
            ship, ship.reference_speed, ship_key
        )

        predicted_kN = result.total_resistance_kN
        reference_kN = ship.ITTC_reference_resistance
        rel_error = abs(predicted_kN - reference_kN) / reference_kN

        # Generate run hash for reproducibility tracking
        run_data = f"{ship_key}_{predicted_kN}_{time.time_ns()}"
        run_hash = hashlib.sha256(run_data.encode()).hexdigest()[:16]

        return BenchmarkResult(
            ship_key=ship_key,
            ship_type=ship.ship_type,
            predicted_resistance_kN=predicted_kN,
            reference_resistance_kN=reference_kN,
            relative_error=rel_error,
            error_within_15pct=rel_error <= self.MAX_RELATIVE_ERROR,
            components={
                "breaking_N": result.breaking_resistance_N,
                "clearance_N": result.clearance_resistance_N,
                "open_water_N": result.open_water_resistance_N,
                "friction_N": result.friction_resistance_N,
                "total_N": result.total_resistance_N,
            },
            run_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            run_hash=run_hash,
        )

    def check_reproducibility(self, n_runs: int = 5) -> float:
        """Check reproducibility across multiple runs.

        Runs the full benchmark n_runs times and computes the
        coefficient of variation (CV) for each ship type.

        Args:
            n_runs: Number of runs for reproducibility check

        Returns:
            Maximum coefficient of variation across all ships
        """
        all_results: Dict[str, List[float]] = {key: [] for key in self.ship_keys}

        for _ in range(n_runs):
            for ship_key in self.ship_keys:
                result = self.run_single_benchmark(ship_key)
                all_results[ship_key].append(result.predicted_resistance_kN)

        # Compute CV for each ship
        max_cv = 0.0
        for ship_key, values in all_results.items():
            mean_val = sum(values) / len(values)
            if mean_val > 0:
                variance = sum((v - mean_val) ** 2 for v in values) / len(values)
                std_val = math.sqrt(variance)
                cv = std_val / mean_val
                max_cv = max(max_cv, cv)

        return max_cv

    def run_full_benchmark(self, n_reproducibility_runs: int = 5) -> FullBenchmarkReport:
        """Run the complete ITTC benchmarking campaign.

        Executes benchmarks for all configured ship types,
        checks reproducibility, and generates the full report.

        Args:
            n_reproducibility_runs: Number of runs for reproducibility check

        Returns:
            FullBenchmarkReport with all results and acceptance criteria
        """
        # Run benchmarks for each ship type
        results = []
        for ship_key in self.ship_keys:
            result = self.run_single_benchmark(ship_key)
            results.append(asdict(result))

        # Check reproducibility
        reproducibility_cv = self.check_reproducibility(n_reproducibility_runs)

        # Count validated ship types (error within 15%)
        ship_types_validated = sum(
            1 for r in results if r["error_within_15pct"]
        )

        # Determine if all criteria pass
        all_criteria_pass = (
            ship_types_validated >= self.MIN_SHIP_TYPES
            and reproducibility_cv < self.MAX_CV_REPRODUCIBILITY
        )

        # Build report
        report = FullBenchmarkReport(
            ship_types_validated=ship_types_validated,
            results=results,
            auto_report_generated=True,
            reproducibility_across_runs=reproducibility_cv,
            all_criteria_pass=all_criteria_pass,
            ice_properties=asdict(self.ice_properties),
            run_metadata={
                "n_ships": len(self.ship_keys),
                "n_reproducibility_runs": n_reproducibility_runs,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model": "ITTC_Lindqvist_Riska_combined",
                "framework_version": "1.0.0",
            },
        )

        return report

    def export_report_json(self, report: FullBenchmarkReport, output_path: str) -> str:
        """Export benchmark report to JSON file.

        Args:
            report: Full benchmark report
            output_path: Path for output JSON file

        Returns:
            Path to the written file
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        report_dict = asdict(report)
        report_dict["acceptance_criteria"] = {
            "ship_types_validated": {
                "value": report.ship_types_validated,
                "threshold": f">= {self.MIN_SHIP_TYPES}",
                "pass": report.ship_types_validated >= self.MIN_SHIP_TYPES,
            },
            "auto_report_generated": {
                "value": report.auto_report_generated,
                "threshold": "== True",
                "pass": report.auto_report_generated,
            },
            "reproducibility_across_runs": {
                "value": report.reproducibility_across_runs,
                "threshold": f"< {self.MAX_CV_REPRODUCIBILITY}",
                "pass": report.reproducibility_across_runs < self.MAX_CV_REPRODUCIBILITY,
            },
        }

        with open(output_path, "w") as f:
            json.dump(report_dict, f, indent=2)

        return output_path


def run_full_benchmark(
    ship_keys: Optional[List[str]] = None,
    ice_thickness_m: float = 0.8,
    output_path: Optional[str] = None,
    n_reproducibility_runs: int = 5,
) -> FullBenchmarkReport:
    """Convenience function to run the full ITTC benchmark.

    Args:
        ship_keys: Ship database keys to benchmark
        ice_thickness_m: Ice thickness [m] (used as default, ships may override)
        output_path: Optional path for JSON report output
        n_reproducibility_runs: Number of runs for reproducibility check

    Returns:
        FullBenchmarkReport
    """
    ice_props = IceProperties(thickness_m=ice_thickness_m)
    runner = ITTCBenchmarkRunner(ice_props, ship_keys)
    report = runner.run_full_benchmark(n_reproducibility_runs)

    if output_path:
        runner.export_report_json(report, output_path)

    return report
