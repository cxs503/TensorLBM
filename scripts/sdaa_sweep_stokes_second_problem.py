#!/usr/bin/env python
"""Independent SDAA-verification sweep for Stokes second problem.

This script is a *diagnostic wrapper* — it does NOT modify any business code.
It imports the existing benchmark run.py and executes a subset of configurations
on the available device (CPU fallback since no SDAA/CUDA is present in this
environment), then programmatically extracts results and compares with the
analytical solution.

Usage:
    TRITON_INTERPRET=1 python scripts/sdaa_sweep_stokes_second_problem.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

# Ensure benchmarks/ is on path so compile_route + run.py are importable
REPO = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(REPO / "src"))

# Force CPU if no CUDA/SDAA available
import torch
if not torch.cuda.is_available():
    try:
        import torch_sdaa  # noqa: F401
    except Exception:
        pass

import numpy as np

# Import the benchmark runner directly from file
import importlib.util
spec = importlib.util.spec_from_file_location(
    "stokes2_run",
    BENCH_DIR / "verified" / "stokes_second_problem" / "run.py"
)
stokes2_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stokes2_mod)
run_case = stokes2_mod.run_case


def main():
    print("=" * 70)
    print("SDAA Sweep: Stokes Second Problem (Oscillating Plate)")
    print("=" * 70)
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Torch:  {torch.__version__}")
    print(f"TRITON_INTERPRET: {os.environ.get('TRITON_INTERPRET', 'not set')}")
    print()

    # Run a small test case: H=20 at kH=4 (much faster on CPU)
    # This is a reduced configuration for CPU testing
    configs = [
        {"H": 20, "kH": 4.0, "tau": 0.8, "U": 0.05, "nx": 8},
    ]

    results = []
    all_pass = True

    for cfg in configs:
        print(f"\n--- Running H={cfg['H']}, kH={cfg['kH']} ---")
        t0 = time.time()
        res = run_case(
            H=cfg["H"],
            kH=cfg["kH"],
            tau=cfg["tau"],
            U=cfg["U"],
            nx=cfg["nx"],
            out_path=None,
            seed=0,
            compile_mode="eager",  # eager to avoid compile issues on CPU
        )
        elapsed = time.time() - t0
        print(f"  Elapsed: {elapsed:.1f}s, total_steps={res['total_steps']}")
        print(f"  amp_rel_max:    {res['amp_rel_max_pct']:.4f}%")
        print(f"  phase_rel_max:  {res['phase_rel_max_pct']:.4f}%")
        print(f"  complex_rel_max:{res['complex_rel_max_pct']:.4f}%")
        print(f"  finite:         {res['finite']}")
        print(f"  mass_drift:     {res['mass_drift_pct']:.6f}%")

        tol = 3.0
        amp_ok = res["amp_rel_max_pct"] <= tol
        phase_ok = res["phase_rel_max_pct"] <= tol
        complex_ok = res["complex_rel_max_pct"] <= tol
        finite_ok = res["finite"]
        case_pass = amp_ok and phase_ok and complex_ok and finite_ok
        verdict = "PASS" if case_pass else "FAIL"
        print(f"  Verdict: {verdict}")
        if not case_pass:
            all_pass = False

        results.append({
            "config": cfg,
            "result": res,
            "verdict": verdict,
            "amp_ok": amp_ok,
            "phase_ok": phase_ok,
            "complex_ok": complex_ok,
        })

    # ---- Summary table ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'H':>5} {'kH':>5} {'amp%':>8} {'phase%':>9} {'complex%':>10} {'verdict':>8}")
    for r in results:
        res = r["result"]
        print(f"{res['H']:>5} {res['kH']:>5.1f} {res['amp_rel_max_pct']:>8.4f} "
              f"{res['phase_rel_max_pct']:>9.4f} {res['complex_rel_max_pct']:>10.4f} "
              f"{r['verdict']:>8}")

    overall = "PASS" if all_pass else "FAIL"
    print(f"\nOverall: {overall}")

    # ---- Save results ----
    out_dir = REPO / "runs" / "sdaa_sweep_stokes2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "result.json"
    summary = {
        "sweep": "stokes_second_problem",
        "device": "cpu" if not torch.cuda.is_available() else "cuda",
        "torch_version": torch.__version__,
        "triton_interpret": os.environ.get("TRITON_INTERPRET", "not set"),
        "overall_verdict": overall,
        "tol_pct": 3.0,
        "cases": [],
    }
    for r in results:
        res = r["result"]
        summary["cases"].append({
            "H": res["H"],
            "kH": res["kH"],
            "amp_rel_max_pct": res["amp_rel_max_pct"],
            "phase_rel_max_pct": res["phase_rel_max_pct"],
            "complex_rel_max_pct": res["complex_rel_max_pct"],
            "phase_err_max_deg": res["phase_err_max_deg"],
            "finite": res["finite"],
            "mass_drift_pct": res["mass_drift_pct"],
            "total_steps": res["total_steps"],
            "elapsed_s": res["elapsed_s"],
            "verdict": r["verdict"],
        })
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to: {out_file}")

    # ---- Generate comparison plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for r in results:
            res = r["result"]
            H = res["H"]
            k = res["k"]
            U = res["U"]
            ny = res["ny"]
            y = np.array(res["y_profile"])
            A_num = np.array(res["A_num"])
            A_ana = np.array(res["A_ana"])
            phi_num = np.array(res["phi_num_deg"])
            phi_ana = np.array(res["phi_ana_deg"])

            ax = axes[0]
            ax.plot(y, A_num / U, "o-", label=f"H={H} num", markersize=3)
            ax.plot(y, A_ana / U, "--", label=f"H={H} ana", alpha=0.7)

            ax2 = axes[1]
            ax2.plot(y, phi_num, "o-", label=f"H={H} num", markersize=3)
            ax2.plot(y, phi_ana, "--", label=f"H={H} ana", alpha=0.7)

        axes[0].set_xlabel("y (depth from plate)")
        axes[0].set_ylabel("Amplitude / U")
        axes[0].set_title("Stokes 2nd Problem: Amplitude Profile")
        axes[0].legend(fontsize=8)
        axes[0].set_yscale("log")
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel("y (depth from plate)")
        axes[1].set_ylabel("Phase lag (deg)")
        axes[1].set_title("Stokes 2nd Problem: Phase Profile")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = out_dir / "stokes2_comparison.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Figure saved to: {fig_path}")
    except Exception as e:
        print(f"Plot generation failed: {e}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
