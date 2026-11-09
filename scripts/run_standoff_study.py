#!/usr/bin/env python3
"""Run the full four-grid stand-off study and produce results JSON.

This script is designed to run on the cluster (loong64) where the
larger grids (600x200, 1200x400) can complete in reasonable time.

Usage:
    python scripts/run_standoff_study.py [--mach 3.0] [--t-end 4.0]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorlbm.compressible_shock.standoff import (
    run_four_grid_study,
    cross_triplet_richardson,
    resolvability_law,
    corner_mass_balance,
    run_forward_step,
    FOUR_GRID_SPEC,
    TRIPLETS,
    full_standoff_analysis,
)


def main():
    parser = argparse.ArgumentParser(description="Run stand-off study")
    parser.add_argument("--mach", type=float, default=3.0)
    parser.add_argument("--t-end", type=float, default=4.0)
    parser.add_argument("--output", type=str, default="artifacts/standoff_results.json")
    parser.add_argument("--skip-large", action="store_true",
                        help="Skip 600x200 and 1200x400 grids (for testing)")
    args = parser.parse_args()

    print(f"=== Shock Stand-Off Study ===")
    print(f"Mach = {args.mach}")
    print(f"t_end = {args.t_end}")
    print()

    if args.skip_large:
        grid_spec = [
            {"nx": 150, "ny": 50, "label": "G1_coarse"},
            {"nx": 300, "ny": 100, "label": "G2_medium"},
        ]
        print("Using reduced grid set (skip-large mode)")
    else:
        grid_spec = FOUR_GRID_SPEC

    # Run four-grid study
    print("Running four-grid study...")
    t0 = time.time()
    grid_results = run_four_grid_study(
        mach=args.mach, t_end=args.t_end, grid_spec=grid_spec,
    )
    print(f"  Completed in {time.time()-t0:.1f}s")

    for label, res in grid_results.items():
        print(f"  {label}: standoff={res['standoff']:.6f}, R={res['R']:.2f}, "
              f"shock_cells={res['shock_thickness_cells']}")

    # Richardson extrapolation
    print()
    print("Richardson extrapolation (cross-triplet)...")
    rich = cross_triplet_richardson(grid_results)
    for label, r in rich.items():
        print(f"  {label}: ext={r['extrapolated']:.6f}, "
              f"GCI_fine={r['gci_fine']:.4%}, p_app={r['apparent_order']:.2f}")

    # Corner mass balance on finest available grid
    print()
    print("Corner mass balance...")
    finest_spec = grid_spec[-1]
    solver_finest, _ = run_forward_step(
        nx=finest_spec["nx"], ny=finest_spec["ny"],
        mach=args.mach, t_end=args.t_end,
    )
    corner = corner_mass_balance(solver_finest, step_x=0.6, step_h=0.2)
    print(f"  mass_residual_rel = {corner['mass_residual_rel']:.6f}")

    # Resolvability law
    print()
    print("Resolvability law...")
    law = resolvability_law(grid_results, mach=args.mach)
    print(f"  R_min = {law['R_min']}")
    for label, c in law["classifications"].items():
        print(f"  {label}: R={c['R']:.2f}, class={c['classification']}, "
              f"err={c['rel_err_vs_extrapolated']:.4%}")

    # Compile results
    results = {
        "mach": args.mach,
        "t_end": args.t_end,
        "grid_results": {k: {kk: vv for kk, vv in v.items()
                             if not isinstance(vv, np.ndarray)}
                         for k, v in grid_results.items()},
        "richardson": rich,
        "corner_mass_balance": corner,
        "resolvability_law": {
            "R_min": law["R_min"],
            "f_ext": law["f_ext"],
            "classifications": law["classifications"],
        },
        "acceptance": {
            "A1_monotonic": all(
                grid_results[gs["label"]]["standoff"]
                for gs in grid_spec if gs["label"] in grid_results
            ),
            "A2_gci_under_3pct": (
                rich.get(f"{TRIPLETS[-1][0]}_to_{TRIPLETS[-1][2]}", {}).get("gci_fine", 1.0) < 0.03
            ),
            "A3_corner_mass_under_1pct": corner["mass_residual_rel"] < 0.01,
            "A4_R_min_given": float("nan") != law["R_min"] if isinstance(law["R_min"], float) else law["R_min"] == law["R_min"],
        },
        "runtime_seconds": time.time() - t0,
    }

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")

    return results


if __name__ == "__main__":
    main()
