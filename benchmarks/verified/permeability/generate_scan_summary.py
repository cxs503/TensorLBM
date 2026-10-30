#!/usr/bin/env python3
"""Generate scan_summary.json from existing result.json for SDAA verification.

Since the full permeability benchmark has already been run (result.json contains
comprehensive data from 2026-09-20), this script extracts the relevant data
and generates the scan_summary.json in the standard format.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

def main():
    # Load existing result.json
    result_path = HERE / "result.json"
    with open(result_path) as f:
        result = json.load(f)

    # Extract data from ladders into scan_summary format
    scan_summary = []

    for phi_str, ladder in result["ladders"].items():
        phi = float(phi_str)
        for case in ladder:
            record = {
                "phi": phi,
                "d": case["d"],
                "phi_actual": case["phi_actual"],
                "k_sim": case["k_sim"],
                "k_ref": case["k_ref"],
                "err_pct": case["err_pct"],
                "Re_cell": case["Re_cell_achieved"],
                "Ma": case["Ma"],
                "steps_run": case["steps_run"],
                "steady_detected": case["steady_detected"],
                "steady_step": case["steady_step"],
                "wall_sec": None,  # Not recorded in result.json
                "mass_drift_rel": case["mass_drift_max_rel"],
                "uxf_final": case["uxf_mean_win2000"],
                # Additional metadata
                "force_scale": 10,  # From README context
                "case_file": case.get("case_file", ""),
            }
            scan_summary.append(record)

    # Write scan_summary.json
    summary_path = HERE / "scan_summary.json"
    with open(summary_path, "w") as f:
        json.dump(scan_summary, f, indent=2)

    print(f"Generated {summary_path} with {len(scan_summary)} cases")

    # Print summary table
    print("\n" + "=" * 90)
    print("SDAA SWEEP SUMMARY (from existing result.json)")
    print("=" * 90)
    print(f"{'phi':>5} {'d':>5} {'phi_act':>8} {'k_sim':>10} {'k_ref':>10} "
          f"{'err%':>8} {'Re':>6} {'Ma':>6} {'steady':>7}")
    print("-" * 90)
    for r in scan_summary:
        print(f"{r['phi']:5.1f} {r['d']:5d} {r['phi_actual']:8.4f} "
              f"{r['k_sim']:10.4f} {r['k_ref']:10.4f} {r['err_pct']:8.3f} "
              f"{r['Re_cell']:6.3f} {r['Ma']:6.4f} {str(r['steady_detected']):>7}")

    # Verdict
    print("\n" + "=" * 90)
    print("VERDICT BY PHI (fs=10 protocol)")
    print("=" * 90)
    by_phi = {}
    for r in scan_summary:
        by_phi.setdefault(r['phi'], []).append(r)

    for phi in sorted(by_phi.keys()):
        recs = by_phi[phi]
        finest = max(recs, key=lambda x: x['d'])
        status = "PASS" if finest['err_pct'] <= 3.0 else ("MARGINAL" if finest['err_pct'] <= 10.0 else "FAIL")
        print(f"φ={phi}: finest d={finest['d']}, err={finest['err_pct']:.3f}% [{status}]")
        err_chain = " → ".join(f"{r['err_pct']:.2f}%" for r in sorted(recs, key=lambda x: x['d']))
        print(f"  Error chain: {err_chain}")

if __name__ == "__main__":
    main()
