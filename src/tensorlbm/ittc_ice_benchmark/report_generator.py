"""ITTC Ice Pool Benchmark Report Generator.

Generates automated comparison reports in JSON and Markdown formats
following ITTC documentation standards. The report includes:
- Ship geometry and parameters
- Ice resistance predictions vs reference data
- Error analysis and acceptance criteria
- Reproducibility assessment
- Audit trail with timestamps and hashes
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict
from typing import Dict, List, Optional

from .benchmark_runner import FullBenchmarkReport, ITTCBenchmarkRunner
from .ice_resistance_model import IceProperties
from .ship_definitions import SHIP_DATABASE


class ITTCReportGenerator:
    """Generator for ITTC ice pool benchmark comparison reports.

    Produces structured reports in JSON and Markdown formats
    with full audit trail and acceptance criteria assessment.
    """

    def __init__(self, report: FullBenchmarkReport):
        """Initialize report generator.

        Args:
            report: Full benchmark report to format
        """
        self.report = report

    def to_json(self, output_path: str) -> str:
        """Export report as JSON.

        Args:
            output_path: Path for output file

        Returns:
            Path to written file
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        report_data = {
            "report_metadata": {
                "title": "ITTC Ice Pool Test Benchmarking Report",
                "standard": "ITTC 7.5-04-01-01.1",
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "framework_version": "1.0.0",
                "mission_id": 718,
            },
            "acceptance_criteria": {
                "ship_types_validated": {
                    "value": self.report.ship_types_validated,
                    "threshold": ">= 3",
                    "unit": "count",
                    "pass": self.report.ship_types_validated >= 3,
                },
                "auto_report_generated": {
                    "value": self.report.auto_report_generated,
                    "threshold": "== True",
                    "unit": "bool",
                    "pass": self.report.auto_report_generated,
                },
                "reproducibility_across_runs": {
                    "value": self.report.reproducibility_across_runs,
                    "threshold": "< 0.02",
                    "unit": "ratio",
                    "pass": self.report.reproducibility_across_runs < 0.02,
                },
            },
            "ice_properties": self.report.ice_properties,
            "run_metadata": self.report.run_metadata,
            "ship_results": self.report.results,
            "summary": {
                "total_ships_tested": len(self.report.results),
                "ships_within_tolerance": self.report.ship_types_validated,
                "max_relative_error": max(
                    r["relative_error"] for r in self.report.results
                ) if self.report.results else 0.0,
                "mean_relative_error": sum(
                    r["relative_error"] for r in self.report.results
                ) / len(self.report.results) if self.report.results else 0.0,
                "all_criteria_pass": self.report.all_criteria_pass,
            },
        }

        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2)

        return output_path

    def to_markdown(self, output_path: str) -> str:
        """Export report as Markdown.

        Args:
            output_path: Path for output file

        Returns:
            Path to written file
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        lines = []
        lines.append("# ITTC Ice Pool Test Benchmarking Report")
        lines.append("")
        lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
        lines.append(f"**Standard:** ITTC 7.5-04-01-01.1")
        lines.append(f"**Mission ID:** 718")
        lines.append("")

        # Acceptance criteria
        lines.append("## Acceptance Criteria")
        lines.append("")
        lines.append("| Criterion | Value | Threshold | Status |")
        lines.append("|-----------|-------|-----------|--------|")
        lines.append(
            f"| Ship types validated | {self.report.ship_types_validated} | >= 3 | "
            f"{'✅ PASS' if self.report.ship_types_validated >= 3 else '❌ FAIL'} |"
        )
        lines.append(
            f"| Auto report generated | {self.report.auto_report_generated} | True | "
            f"{'✅ PASS' if self.report.auto_report_generated else '❌ FAIL'} |"
        )
        lines.append(
            f"| Reproducibility (CV) | {self.report.reproducibility_across_runs:.4f} | < 0.02 | "
            f"{'✅ PASS' if self.report.reproducibility_across_runs < 0.02 else '❌ FAIL'} |"
        )
        lines.append("")

        # Ice properties
        lines.append("## Ice Properties")
        lines.append("")
        lines.append("| Property | Value | Unit |")
        lines.append("|----------|-------|------|")
        ice = self.report.ice_properties
        lines.append(f"| Thickness | {ice['thickness_m']:.2f} | m |")
        lines.append(f"| Flexural strength | {ice['flexural_strength_Pa']/1e3:.0f} | kPa |")
        lines.append(f"| Crushing strength | {ice['crushing_strength_Pa']/1e6:.1f} | MPa |")
        lines.append(f"| Density | {ice['density_kg_m3']:.0f} | kg/m³ |")
        lines.append(f"| Friction coefficient | {ice['friction_coefficient']:.3f} | - |")
        lines.append("")

        # Ship results
        lines.append("## Ship Results")
        lines.append("")

        for result in self.report.results:
            ship_key = result["ship_key"]
            ship = SHIP_DATABASE.get(ship_key)

            lines.append(f"### {result['ship_type'].replace('_', ' ').title()} ({ship.name if ship else ship_key})")
            lines.append("")

            # Ship geometry
            if ship:
                lines.append("**Ship Geometry:**")
                lines.append("")
                lines.append(f"- L_pp = {ship.L_pp:.1f} m, B = {ship.B:.1f} m, T = {ship.T:.1f} m")
                lines.append(f"- Cb = {ship.Cb:.3f}, Cwp = {ship.Cwp:.3f}")
                lines.append(f"- Bow angle = {ship.bow_angle:.1f}°, Stem angle = {ship.stem_angle:.1f}°")
                lines.append(f"- Ice class: {ship.ice_class}")
                lines.append("")

            # Resistance comparison
            lines.append("**Ice Resistance Comparison:**")
            lines.append("")
            lines.append("| Quantity | Predicted | Reference | Error |")
            lines.append("|----------|-----------|-----------|-------|")
            lines.append(
                f"| Total resistance | {result['predicted_resistance_kN']:.1f} kN | "
                f"{result['reference_resistance_kN']:.1f} kN | "
                f"{result['relative_error']*100:.1f}% |"
            )
            lines.append("")

            # Resistance breakdown
            comp = result["components"]
            lines.append("**Resistance Breakdown:**")
            lines.append("")
            lines.append(f"- Ice breaking: {comp['breaking_N']/1000:.1f} kN")
            lines.append(f"- Ice clearance: {comp['clearance_N']/1000:.1f} kN")
            lines.append(f"- Open water: {comp['open_water_N']/1000:.1f} kN")
            lines.append(f"- Ice friction: {comp['friction_N']/1000:.1f} kN")
            lines.append("")

            status = "✅ PASS" if result["error_within_15pct"] else "❌ FAIL"
            lines.append(f"**Status:** {status} (error {'within' if result['error_within_15pct'] else 'exceeds'} 15% tolerance)")
            lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")
        errors = [r["relative_error"] for r in self.report.results]
        lines.append(f"- **Ships tested:** {len(self.report.results)}")
        lines.append(f"- **Ships within tolerance:** {self.report.ship_types_validated}")
        lines.append(f"- **Mean relative error:** {sum(errors)/len(errors)*100:.1f}%")
        lines.append(f"- **Max relative error:** {max(errors)*100:.1f}%")
        lines.append(f"- **Reproducibility (CV):** {self.report.reproducibility_across_runs:.4f}")
        lines.append("")

        overall = "✅ ALL CRITERIA PASS" if self.report.all_criteria_pass else "❌ SOME CRITERIA FAIL"
        lines.append(f"**Overall Assessment:** {overall}")
        lines.append("")

        # Audit trail
        lines.append("## Audit Trail")
        lines.append("")
        lines.append(f"- Framework version: {self.report.run_metadata.get('framework_version', 'unknown')}")
        lines.append(f"- Model: {self.report.run_metadata.get('model', 'unknown')}")
        lines.append(f"- Reproducibility runs: {self.report.run_metadata.get('n_reproducibility_runs', 'unknown')}")
        for r in self.report.results:
            lines.append(f"- {r['ship_key']}: hash={r['run_hash']}, timestamp={r['run_timestamp']}")
        lines.append("")

        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        return output_path


def generate_comparison_report(
    report: FullBenchmarkReport,
    output_dir: str,
    formats: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Generate comparison report in multiple formats.

    Args:
        report: Full benchmark report
        output_dir: Directory for output files
        formats: List of formats ('json', 'markdown'). Default: both.

    Returns:
        Dict mapping format name to output file path
    """
    if formats is None:
        formats = ["json", "markdown"]

    generator = ITTCReportGenerator(report)
    output_paths = {}

    os.makedirs(output_dir, exist_ok=True)

    if "json" in formats:
        json_path = os.path.join(output_dir, "ittc_ice_benchmark_report.json")
        output_paths["json"] = generator.to_json(json_path)

    if "markdown" in formats:
        md_path = os.path.join(output_dir, "ittc_ice_benchmark_report.md")
        output_paths["markdown"] = generator.to_markdown(md_path)

    return output_paths
