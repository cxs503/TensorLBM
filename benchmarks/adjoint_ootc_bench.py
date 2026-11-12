#!/usr/bin/env python3
"""Benchmark: OOTC adjoint gradient vs BPTT reference for D2Q9 BGK LBM.

Validates the out-of-core (checkpointed) adjoint gradient against the
full-storage BPTT reference, measuring:
1. Gradient accuracy (relative error vs 1e-5 gate)
2. Peak memory (vs 8 GiB gate for 128^2, T=10^4)
3. Objective value consistency
4. Scaling with checkpoint interval k

Usage:
    python benchmarks/adjoint_ootc_bench.py --quick          # 32^2, T=200
    python benchmarks/adjoint_ootc_bench.py --medium         # 64^2, T=2000
    python benchmarks/adjoint_ootc_bench.py                  # 128^2, T=10000
    python benchmarks/adjoint_ootc_bench.py --out artifacts/adjoint_ootc_bench
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tracemalloc

import numpy as np

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "TensorLBM", "src"))

from tensorlbm.adjoint.lbm2d import initial_pulse, step, macro
from tensorlbm.adjoint.bptt import bptt_gradient
from tensorlbm.adjoint.ootc import ootc_gradient


PROFILES = {
    "quick":  {"N": 32,  "T": 200,   "k_values": [20, 50, 100]},
    "medium": {"N": 64,  "T": 2000,  "k_values": [50, 100, 200]},
    "full":   {"N": 128, "T": 10000, "k_values": [100, 200, 500]},
}


def run_benchmark(N: int, T: int, k_values: list[int], out_dir: str | None = None):
    """Run the full benchmark suite."""
    tau = 1.0
    amp = 0.05
    sigma = float(N) / 8.0  # scale pulse width with grid

    print(f"[bench] N={N}, T={T}, tau={tau}, amp={amp}, sigma={sigma:.1f}")
    print(f"[bench] Grid memory per state: {9 * N * N * 8 / 1024**2:.2f} MiB")
    print(f"[bench] BPTT total trajectory: {9 * N * N * 8 * (T + 1) / 1024**3:.3f} GiB")
    print()

    f0 = initial_pulse(N, N, amp=amp, sigma=sigma)

    # Compute terminal reference (perturbed)
    t0 = time.perf_counter()
    f = f0.copy()
    for _ in range(T):
        f = step(f, tau)
    rho_t, ux_t, uy_t = macro(f)
    t_fwd = time.perf_counter() - t0
    ux_ref = ux_t * 0.9
    uy_ref = uy_t * 0.9
    print(f"[bench] Forward reference: {t_fwd:.2f}s")

    # ---- BPTT reference ----
    print(f"\n[bench] Running BPTT (full storage)...")
    tracemalloc.start()
    t0 = time.perf_counter()
    dJ_bptt, J_bptt, (ux_T, uy_T), states = bptt_gradient(tau, T, f0, ux_ref, uy_ref)
    t_bptt = time.perf_counter() - t0
    _, peak_bptt = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"[bench] BPTT: dJ/dtau = {dJ_bptt:.10e}, J = {J_bptt:.10e}")
    print(f"[bench] BPTT time: {t_bptt:.2f}s, peak memory: {peak_bptt / 1024**3:.3f} GiB")

    # ---- OOTC for each k ----
    results = []
    for k in k_values:
        print(f"\n[bench] Running OOTC (k={k})...")
        ckpt_path = None
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            ckpt_path = os.path.join(out_dir, f"ootc_k{k}.ckpt.npy")

        tracemalloc.start()
        t0 = time.perf_counter()
        dJ_ootc, J_ootc, (ux_o, uy_o), n_ckpt, ckpt_path_ret = ootc_gradient(
            tau, T, f0, ux_ref, uy_ref, k=k, ckpt_path=ckpt_path, verbose=False
        )
        t_ootc = time.perf_counter() - t0
        _, peak_ootc = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        rel_err = abs(dJ_bptt - dJ_ootc) / max(abs(dJ_bptt), 1e-30)
        J_err = abs(J_bptt - J_ootc)
        mem_ratio = peak_ootc / max(peak_bptt, 1)

        print(f"[bench] OOTC k={k}: dJ/dtau = {dJ_ootc:.10e}, J = {J_ootc:.10e}")
        print(f"[bench]   time: {t_ootc:.2f}s, peak mem: {peak_ootc / 1024**3:.3f} GiB")
        print(f"[bench]   gradient rel_err: {rel_err:.2e}, J_err: {J_err:.2e}")
        print(f"[bench]   mem ratio (OOTC/BPTT): {mem_ratio:.4f}")
        print(f"[bench]   checkpoints: {n_ckpt}, ckpt file: {ckpt_path_ret}")

        results.append({
            "k": k,
            "dJ_ootc": float(dJ_ootc),
            "J_ootc": float(J_ootc),
            "dJ_bptt": float(dJ_bptt),
            "J_bptt": float(J_bptt),
            "rel_err": float(rel_err),
            "J_err": float(J_err),
            "time_ootc_s": round(t_ootc, 2),
            "time_bptt_s": round(t_bptt, 2),
            "peak_ootc_gib": round(peak_ootc / 1024**3, 4),
            "peak_bptt_gib": round(peak_bptt / 1024**3, 4),
            "mem_ratio": round(mem_ratio, 4),
            "n_checkpoints": n_ckpt,
            "ckpt_path": ckpt_path_ret,
            "passes_1e5_gate": rel_err < 1e-5,
        })

    # ---- Summary ----
    summary = {
        "N": N,
        "T": T,
        "tau": tau,
        "amp": amp,
        "sigma": sigma,
        "dJ_bptt": float(dJ_bptt),
        "J_bptt": float(J_bptt),
        "peak_bptt_gib": round(peak_bptt / 1024**3, 4),
        "time_bptt_s": round(t_bptt, 2),
        "results": results,
        "all_pass_1e5_gate": all(r["passes_1e5_gate"] for r in results),
        "peak_ootc_under_8gib": all(r["peak_ootc_gib"] < 8.0 for r in results),
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "adjoint_ootc_bench.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[bench] Results written to {out_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Adjoint OOTC benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 32^2, T=200")
    parser.add_argument("--medium", action="store_true", help="Medium mode: 64^2, T=2000")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    if args.quick:
        profile = PROFILES["quick"]
    elif args.medium:
        profile = PROFILES["medium"]
    else:
        profile = PROFILES["full"]

    summary = run_benchmark(profile["N"], profile["T"], profile["k_values"], args.out)

    # Print pass/fail
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    print(f"Gradient accuracy (<1e-5): {'PASS' if summary['all_pass_1e5_gate'] else 'FAIL'}")
    print(f"Memory <8 GiB (OOTC):      {'PASS' if summary['peak_ootc_under_8gib'] else 'FAIL'}")
    for r in summary["results"]:
        status = "PASS" if r["passes_1e5_gate"] else "FAIL"
        print(f"  k={r['k']:4d}: rel_err={r['rel_err']:.2e} [{status}]  mem={r['peak_ootc_gib']:.3f} GiB")


if __name__ == "__main__":
    main()
