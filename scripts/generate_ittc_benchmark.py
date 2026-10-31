#!/usr/bin/env python3
"""Generate ITTC Ice Pool Benchmark artifacts and reports.

This script runs the full benchmark campaign and generates:
1. JSON validation report
2. Markdown comparison report
3. Visualization plots
4. Summary statistics
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorlbm.ittc_ice_benchmark import (
    SHIP_DATABASE,
    ITTCBenchmarkRunner,
    generate_comparison_report,
    run_full_benchmark,
)


def generate_visualization(report, output_dir):
    """Generate visualization plots for the benchmark report.

    Args:
        report: FullBenchmarkReport
        output_dir: Directory for output files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract data
    ship_names = []
    predicted = []
    reference = []
    errors = []

    for result in report.results:
        ship = SHIP_DATABASE.get(result["ship_key"])
        ship_names.append(ship.name if ship else result["ship_key"])
        predicted.append(result["predicted_resistance_kN"])
        reference.append(result["reference_resistance_kN"])
        errors.append(result["relative_error"] * 100)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Predicted vs Reference
    ax1 = axes[0, 0]
    x = np.arange(len(ship_names))
    width = 0.35
    bars1 = ax1.bar(x - width/2, reference, width, label='ITTC Reference', color='#2E86AB')
    bars2 = ax1.bar(x + width/2, predicted, width, label='Model Prediction', color='#A23B72')
    ax1.set_xlabel('Ship Type')
    ax1.set_ylabel('Ice Resistance [kN]')
    ax1.set_title('Ice Resistance: Predicted vs Reference')
    ax1.set_xticks(x)
    ax1.set_xticklabels(ship_names, rotation=15, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Relative Error
    ax2 = axes[0, 1]
    colors = ['#06A77D' if e < 15 else '#D62246' for e in errors]
    bars = ax2.bar(x, errors, color=colors, alpha=0.7)
    ax2.axhline(y=15, color='red', linestyle='--', linewidth=2, label='15% Threshold')
    ax2.set_xlabel('Ship Type')
    ax2.set_ylabel('Relative Error [%]')
    ax2.set_title('Prediction Error vs ITTC Reference')
    ax2.set_xticks(x)
    ax2.set_xticklabels(ship_names, rotation=15, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: Resistance Breakdown (stacked bar)
    ax3 = axes[1, 0]
    breaking = [r["components"]["breaking_N"] / 1000 for r in report.results]
    clearance = [r["components"]["clearance_N"] / 1000 for r in report.results]
    open_water = [r["components"]["open_water_N"] / 1000 for r in report.results]
    friction = [r["components"]["friction_N"] / 1000 for r in report.results]

    ax3.bar(x, breaking, label='Ice Breaking', color='#F18F01')
    ax3.bar(x, clearance, bottom=breaking, label='Ice Clearance', color='#C73E1D')
    ax3.bar(x, open_water, bottom=[b+c for b,c in zip(breaking, clearance)],
            label='Open Water', color='#3B1F2B')
    ax3.bar(x, friction,
            bottom=[b+c+o for b,c,o in zip(breaking, clearance, open_water)],
            label='Friction', color='#44253A')
    ax3.set_xlabel('Ship Type')
    ax3.set_ylabel('Resistance [kN]')
    ax3.set_title('Resistance Component Breakdown')
    ax3.set_xticks(x)
    ax3.set_xticklabels(ship_names, rotation=15, ha='right')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(axis='y', alpha=0.3)

    # Plot 4: Acceptance Criteria Summary
    ax4 = axes[1, 1]
    criteria = ['Ship Types\nValidated', 'Auto Report\nGenerated', 'Reproducibility\n(CV < 2%)']
    values = [
        report.ship_types_validated,
        1 if report.auto_report_generated else 0,
        1 if report.reproducibility_across_runs < 0.02 else 0
    ]
    thresholds = [3, 1, 1]
    colors = ['#06A77D' if v >= t else '#D62246' for v, t in zip(values, thresholds)]

    bars = ax4.bar(criteria, values, color=colors, alpha=0.7)
    ax4.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    ax4.set_ylabel('Status')
    ax4.set_title('Acceptance Criteria Assessment')
    ax4.set_ylim(0, max(max(values), max(thresholds)) + 1)

    # Add value labels on bars
    for bar, val, thresh in zip(bars, values, thresholds):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}/{thresh}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "ittc_ice_benchmark_results.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Visualization saved to {fig_path}")
    return fig_path


def main():
    """Main entry point."""
    print("=" * 70)
    print("ITTC Ice Pool Benchmark - Artifact Generation")
    print("=" * 70)

    # Run full benchmark
    print("\n[1/4] Running full benchmark campaign...")
    report = run_full_benchmark(
        ship_keys=list(SHIP_DATABASE.keys()),
        n_reproducibility_runs=5,
    )

    print(f"  • Ships tested: {len(report.results)}")
    print(f"  • Ships validated: {report.ship_types_validated}")
    print(f"  • Reproducibility CV: {report.reproducibility_across_runs:.2e}")
    print(f"  • All criteria pass: {report.all_criteria_pass}")

    # Generate reports
    print("\n[2/4] Generating reports...")
    output_dir = "artifacts"
    paths = generate_comparison_report(report, output_dir, formats=["json", "markdown"])
    print(f"  • JSON report: {paths['json']}")
    print(f"  • Markdown report: {paths['markdown']}")

    # Generate visualization
    print("\n[3/4] Generating visualization...")
    fig_path = generate_visualization(report, output_dir)

    # Print summary
    print("\n[4/4] Summary")
    print("-" * 70)
    for result in report.results:
        ship = SHIP_DATABASE.get(result["ship_key"])
        status = "✓" if result["error_within_15pct"] else "✗"
        print(f"  {status} {ship.name if ship else result['ship_key']:30s} "
              f"Error: {result['relative_error']*100:5.1f}%")

    print("\n" + "=" * 70)
    print("Acceptance Criteria:")
    print(f"  • Ship types validated: {report.ship_types_validated} >= 3  "
          f"{'✓ PASS' if report.ship_types_validated >= 3 else '✗ FAIL'}")
    print(f"  • Auto report generated: {report.auto_report_generated}  "
          f"{'✓ PASS' if report.auto_report_generated else '✗ FAIL'}")
    print(f"  • Reproducibility CV: {report.reproducibility_across_runs:.2e} < 0.02  "
          f"{'✓ PASS' if report.reproducibility_across_runs < 0.02 else '✗ FAIL'}")
    print(f"\nOverall: {'✓ ALL CRITERIA PASS' if report.all_criteria_pass else '✗ SOME CRITERIA FAIL'}")
    print("=" * 70)

    return 0 if report.all_criteria_pass else 1


if __name__ == "__main__":
    sys.exit(main())
