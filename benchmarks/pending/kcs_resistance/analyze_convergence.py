#!/usr/bin/env python3
"""Analyze KCS resistance convergence from result.json and force log.

Reads the result.json produced by run.py and generates:
  1. Convergence plot (Cd_total vs step fraction)
  2. Three-window drift analysis
  3. Summary table for the report
"""

import json
import sys
import math
from pathlib import Path

import numpy as np


def analyze_result(result_path: Path):
    with open(result_path) as f:
        r = json.load(f)

    cf_ref = r["Cf_ref_Blasius"]
    ss = r["steady_state"]

    print("=" * 72)
    print("KCS Resistance Convergence Analysis (m460)")
    print("=" * 72)
    print(f"Device: {r['device']}")
    print(f"Resolution: L={r['L_cells']} cells/Lpp")
    print(f"Domain: {r['domain_lu']}  ({r['n_cells']:,} cells)")
    print(f"Steps: {r['n_steps']}  |  Elapsed: {r['elapsed_s']:.0f}s  |  {r['ms_per_step']:.1f} ms/step")
    print(f"Re = {r['Re']:.0f}  |  u_lb = {r['u_lb']}  |  tau = {r['tau']}")
    print(f"Variant: {r['variant']}  |  Collision: {r['collision']}")
    print()

    print("--- Smooth Wetted Area (quadrature-fixed) ---")
    area = r["smooth_area_lu"]
    for k, v in area.items():
        print(f"  {k}: {v:.4f} lu^2" if not math.isnan(v) else f"  {k}: NaN (BUG!)")
    print()

    print("--- Steady-State Averages (skip first {:.0%} as transient) ---".format(
        ss["steady_start_frac"]
    ))
    print(f"  Steady window: samples [{ss['steady_start_sample']}..{ss['steady_end_sample']}] "
          f"({ss['window_length_samples']} samples)")
    print(f"  Cd_friction  = {ss['Cd_friction']:.6f}")
    print(f"  Cd_pressure  = {ss['Cd_pressure']:.6f}")
    print(f"  Cd_total     = {ss['Cd_total']:.6f}")
    print(f"  Cf_ref (Blasius) = {cf_ref:.6f}")
    print()

    print("--- Error vs Blasius Cf_ref ---")
    print(f"  Cf  (friction) : {ss['err_pct_Cf_vs_Blasius']:+.2f}%  "
          f"{'✓ <3%' if abs(ss['err_pct_Cf_vs_Blasius']) < 3 else '✗ ≥3%'}")
    print(f"  Cp  (pressure) : {ss['err_pct_Cp_vs_Blasius']:+.2f}%  "
          f"(no Blasius ref; reported as % of Cf_ref)")
    print(f"  Cd  (total)    : {ss['err_pct_Cd_total_vs_Blasius']:+.2f}%  "
          f"{'✓ <3%' if abs(ss['err_pct_Cd_total_vs_Blasius']) < 3 else '✗ ≥3%'}")
    print()

    print("--- Three-Window Convergence ---")
    wins = ss["window_Cd_totals"]
    for i, w in enumerate(wins):
        print(f"  Window {i+1}: Cd_total = {w:.6f}")
    print(f"  Max pairwise drift: {ss['max_drift_pct']:.2f}%")
    print(f"  Convergence tol:    {ss['convergence_tol_pct']:.1f}%")
    print(f"  Converged:          {ss['converged']}")
    print()

    print("--- Convergence Sequence (full-log windows) ---")
    for k, v in r["convergence_windows"].items():
        err = (v - cf_ref) / cf_ref * 100 if not math.isnan(v) else float("nan")
        print(f"  {k:>4s}: Cd_total = {v:.6f}  (err = {err:+.2f}%)")
    print()

    print("--- Legacy (full-log last-1000 average, for m439 comparison) ---")
    print(f"  Cd_total_legacy = {r['Cd_total_legacy']:.6f}  "
          f"(err = {r['error_pct_total_legacy_vs_Blasius']:+.2f}%)")
    print()

    print("--- Final-Field Checks (snapshot at last step) ---")
    fc = r["final_field_checks"]
    p0 = r["p0_method"]
    if p0 in fc:
        row = fc[p0]
        print(f"  p0={p0}: Cd_p={row['cd_p']:.6f}")
        for formula in ("standard", "faces", "mix50", "2nd_order", "central", "lagrange"):
            key = f"cd_f_{formula}"
            if key in row:
                err = (row[key] - cf_ref) / cf_ref * 100
                print(f"    Cf({formula}) = {row[key]:.6f}  (err = {err:+.2f}%)")
    print()

    print("--- Verdict ---")
    print(f"  {r['verdict']}")
    print()

    # Write summary JSON for artifacts
    summary = {
        "device": r["device"],
        "resolution": r["L_cells"],
        "n_steps": r["n_steps"],
        "elapsed_s": r["elapsed_s"],
        "ms_per_step": r["ms_per_step"],
        "Re": r["Re"],
        "Cf_ref_Blasius": cf_ref,
        "steady_state": ss,
        "convergence_sequence": r["convergence_windows"],
        "verdict": r["verdict"],
        "finite": r["finite"],
        "diverged": r["diverged"],
    }
    return summary


if __name__ == "__main__":
    result_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("result.json")
    summary = analyze_result(result_path)
    out_path = result_path.parent / "convergence_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {out_path}")
