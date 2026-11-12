#!/usr/bin/env python3
"""Convergence study: OOTC gradient accuracy vs checkpoint interval k.

For a fixed problem (N=64, T=2000), sweep k from 1 to T and measure
the relative gradient error vs BPTT reference.  Also measures memory
scaling.
"""

from __future__ import annotations

import json
import os
import sys
import time
import tracemalloc

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "TensorLBM", "src"))

from tensorlbm.adjoint.lbm2d import initial_pulse, step, macro
from tensorlbm.adjoint.bptt import bptt_gradient
from tensorlbm.adjoint.ootc import ootc_gradient


def main():
    N, T, tau = 64, 2000, 1.0
    f0 = initial_pulse(N, N, amp=0.05, sigma=float(N) / 8.0)

    # Terminal reference
    f = f0.copy()
    for _ in range(T):
        f = step(f, tau)
    _, ux_t, uy_t = macro(f)
    ux_ref = ux_t * 0.9
    uy_ref = uy_t * 0.9

    # BPTT reference
    print("[conv] Running BPTT reference...")
    dJ_bptt, J_bptt, _, _ = bptt_gradient(tau, T, f0, ux_ref, uy_ref)
    print(f"[conv] BPTT: dJ/dtau = {dJ_bptt:.10e}")

    # Sweep k values
    k_values = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
    results = []

    for k in k_values:
        print(f"[conv] k={k}...", end=" ", flush=True)
        tracemalloc.start()
        t0 = time.perf_counter()
        dJ_ootc, J_ootc, _, _, _ = ootc_gradient(tau, T, f0, ux_ref, uy_ref, k=k)
        t_ootc = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        rel_err = abs(dJ_bptt - dJ_ootc) / max(abs(dJ_bptt), 1e-30)
        print(f"rel_err={rel_err:.2e}, mem={peak/1024**3:.3f} GiB, time={t_ootc:.1f}s")

        results.append({
            "k": k,
            "rel_err": float(rel_err),
            "peak_mem_gib": round(peak / 1024**3, 4),
            "time_s": round(t_ootc, 2),
            "dJ_ootc": float(dJ_ootc),
        })

    out_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts", "adjoint_ootc_bench")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "convergence_study.json")
    with open(out_path, "w") as f:
        json.dump({
            "N": N, "T": T, "tau": tau,
            "dJ_bptt": float(dJ_bptt),
            "J_bptt": float(J_bptt),
            "k_sweep": results,
        }, f, indent=2)
    print(f"\n[conv] Results written to {out_path}")


if __name__ == "__main__":
    main()
