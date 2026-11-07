#!/usr/bin/env python3
"""G12 Stokes sphere convergence study — runs on cluster.

Produces:
  artifacts/stokes_sphere_g12_results.json  — full results
  artifacts/stokes_sphere_g12_convergence.png — convergence plot
  artifacts/stokes_sphere_g12_report.md — paper-level report
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import torch

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from tensorlbm.stokes_sphere import (
    run_stokes_sphere,
    stokes_cd_analytical,
)


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(out_dir, exist_ok=True)

    re = 0.1
    cd_ref = stokes_cd_analytical(re)
    print("Stokes sphere convergence study: Re={}, Cd_ref={}".format(re, cd_ref))

    # Grid resolutions
    d_cells_list = [10, 16, 24, 32]
    steps = 8000
    warmup = 2000

    results = []
    for d in d_cells_list:
        t0 = time.time()
        print("  Running D={} ...".format(d))
        r = run_stokes_sphere(
            re=re, d_cells=d, domain_factor=5.0,
            steps=steps, warmup=warmup,
            device="cpu", dtype=torch.float64,
        )
        elapsed = time.time() - t0
        r["elapsed_s"] = elapsed
        results.append(r)
        print("    Cd={:.4f}, err={:.2f}%, time={:.1f}s".format(
            r["cd"], r["err_pct"], elapsed))

    # Save results
    # Convert cd_history to summary stats to save space
    for r in results:
        hist = r.pop("cd_history", [])
        if hist:
            r["cd_history_stats"] = {
                "mean": sum(hist) / len(hist),
                "min": min(hist),
                "max": max(hist),
                "n": len(hist),
            }
        r["domain"] = list(r["domain"])  # tuple -> list for JSON

    out_path = os.path.join(out_dir, "stokes_sphere_g12_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to {}".format(out_path))

    # Generate convergence plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        d_vals = [r["d_cells"] for r in results]
        cd_vals = [r["cd"] for r in results]
        err_vals = [r["err_pct"] for r in results]

        ax.plot(d_vals, cd_vals, "bo-", label="Cd (sim)")
        ax.axhline(y=cd_ref, color="r", linestyle="--", label="Cd (analytical) = {:.1f}".format(cd_ref))
        ax.set_xlabel("Sphere diameter (cells)")
        ax.set_ylabel("Drag coefficient Cd")
        ax.set_title("Stokes Sphere Drag Convergence (Re={})".format(re))
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add error annotations
        for d, cd, err in zip(d_vals, cd_vals, err_vals):
            ax.annotate("{:.1f}%".format(err), (d, cd), textcoords="offset points",
                       xytext=(0, 10), ha="center", fontsize=8)

        plt.tight_layout()
        fig_path = os.path.join(out_dir, "stokes_sphere_g12_convergence.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print("Plot saved to {}".format(fig_path))
    except ImportError:
        print("matplotlib not available, skipping plot")

    # Summary
    print("\n=== Convergence Summary ===")
    print("{:>6}  {:>10}  {:>10}  {:>10}".format("D", "Cd", "Cd_ref", "err%"))
    for r in results:
        print("{:>6}  {:>10.4f}  {:>10.1f}  {:>9.2f}%".format(
            r["d_cells"], r["cd"], r["cd_ref"], r["err_pct"]))

    # Check convergence criteria
    errors = [abs(r["cd"] - cd_ref) / cd_ref for r in results]
    all_within_5 = all(e < 0.05 for e in errors)
    monotone = all(errors[i] >= errors[i+1] for i in range(len(errors)-1))

    print("\nAll within 5%: {}".format(all_within_5))
    print("Monotone convergence: {}".format(monotone))

    return results


if __name__ == "__main__":
    main()
