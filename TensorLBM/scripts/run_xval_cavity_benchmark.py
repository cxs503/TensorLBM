#!/usr/bin/env python3
"""Full-resolution cross-validation benchmark: lid-driven cavity Re=100 at n=65.

This script runs the full-resolution lid-driven cavity benchmark and saves
reference data for regression testing.  Submitted to the cluster because
the torch-based solver is slow on the local machine.

Produces:
- Reference data .npz file with velocity profiles and error metrics
- Benchmark timing data
- Convergence data (n=33, n=65)
"""

import sys
import time
import json
import numpy as np

def main():
    from tensorlbm.verification.lid_driven_cavity.solver import CavityConfig, CavitySolver
    from tensorlbm.verification.lid_driven_cavity.reference_data import get_ghia_reference
    from tensorlbm.verification.lid_driven_cavity.verification import compute_errors
    from tensorlbm.verification.xval.bitwise_refdata import (
        compute_state_hash, ReferenceDataStore,
    )
    from tensorlbm.verification.xval.benchmark import BenchmarkConfig, run_benchmark

    results = {}

    # --- Run 1: n=33, Re=100, 5000 steps ---
    print("Running n=33, Re=100, 5000 steps...")
    cfg33 = CavityConfig(n=33, re=100, max_steps=5000, u_lid=0.1)
    solver33 = CavitySolver(cfg33)
    for _ in range(5000):
        solver33.step()
    y33, u33, x33, v33 = solver33.get_profiles()
    ref = get_ghia_reference(100)
    errors33 = compute_errors(y33, u33, x33, v33, ref)
    hash33 = compute_state_hash(solver33.f)
    results["n33"] = {
        "l2_u": errors33["l2_u"],
        "l2_v": errors33["l2_v"],
        "max_u": errors33["max_u"],
        "max_v": errors33["max_v"],
        "state_hash": hash33,
    }
    print(f"  n=33: l2_u={errors33['l2_u']:.4f}, l2_v={errors33['l2_v']:.4f}, hash={hash33[:16]}...")

    # --- Run 2: n=65, Re=100, 15000 steps ---
    print("Running n=65, Re=100, 15000 steps...")
    cfg65 = CavityConfig(n=65, re=100, max_steps=15000, u_lid=0.1)
    solver65 = CavitySolver(cfg65)
    for _ in range(15000):
        solver65.step()
    y65, u65, x65, v65 = solver65.get_profiles()
    errors65 = compute_errors(y65, u65, x65, v65, ref)
    hash65 = compute_state_hash(solver65.f)
    results["n65"] = {
        "l2_u": errors65["l2_u"],
        "l2_v": errors65["l2_v"],
        "max_u": errors65["max_u"],
        "max_v": errors65["max_v"],
        "state_hash": hash65,
    }
    print(f"  n=65: l2_u={errors65['l2_u']:.4f}, l2_v={errors65['l2_v']:.4f}, hash={hash65[:16]}...")

    # --- Bitwise reproducibility check ---
    print("Running bitwise reproducibility check (n=33, 500 steps)...")
    cfg_repro = CavityConfig(n=33, re=100, max_steps=500, u_lid=0.1)
    solver_r1 = CavitySolver(cfg_repro)
    for _ in range(500):
        solver_r1.step()
    solver_r2 = CavitySolver(cfg_repro)
    for _ in range(500):
        solver_r2.step()
    h1 = compute_state_hash(solver_r1.f)
    h2 = compute_state_hash(solver_r2.f)
    results["bitwise_reproducible"] = (h1 == h2)
    results["repro_hash1"] = h1
    results["repro_hash2"] = h2
    print(f"  Bitwise reproducible: {h1 == h2}")

    # --- Benchmark ---
    print("Running benchmark (n=33, 100 steps per rep)...")
    cfg_bench = CavityConfig(n=33, re=100, max_steps=100, u_lid=0.1)
    solver_bench = CavitySolver(cfg_bench)

    def bench_step():
        solver_bench.step()

    bench_config = BenchmarkConfig(n_warmup=2, n_repeats=5, label="lid_cavity_n33")
    bench_result = run_benchmark(bench_step, bench_config, n_steps=100, grid_size=33*33)
    results["benchmark"] = {
        "label": bench_result.label,
        "mean_s": bench_result.mean,
        "std_s": bench_result.std,
        "min_s": bench_result.min,
        "max_s": bench_result.max,
        "mlups": bench_result.mlups,
    }
    print(f"  Benchmark: {bench_result.summary()}")

    # --- Convergence ratio ---
    if errors33["l2_u"] > 0 and errors65["l2_u"] > 0:
        ratio = errors33["l2_u"] / errors65["l2_u"]
        results["convergence_ratio_l2_u"] = ratio
        print(f"  Convergence ratio (l2_u, n=33/n=65): {ratio:.2f}")

    # --- Save results ---
    output_path = "xval_cavity_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # --- Save reference data ---
    import pathlib
    refdata_dir = pathlib.Path("refdata_output")
    refdata_dir.mkdir(exist_ok=True)
    store = ReferenceDataStore(base_dir=refdata_dir)
    store.save(
        "lid_cavity_re100_n65",
        {"u_centerline": u65, "v_centerline": v65},
        {"l2_u": errors65["l2_u"], "l2_v": errors65["l2_v"]},
        {"l2_u": 0.15, "l2_v": 0.15},
    )
    print(f"Reference data saved to {refdata_dir}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    for key, val in results.items():
        if isinstance(val, dict):
            print(f"  {key}:")
            for k2, v2 in val.items():
                print(f"    {k2}: {v2}")
        else:
            print(f"  {key}: {val}")

    # Check acceptance
    all_pass = True
    if errors65["l2_u"] >= 0.15:
        print(f"  FAIL: l2_u={errors65['l2_u']:.4f} >= 15%")
        all_pass = False
    if errors65["l2_v"] >= 0.15:
        print(f"  FAIL: l2_v={errors65['l2_v']:.4f} >= 15%")
        all_pass = False
    if not results["bitwise_reproducible"]:
        print("  FAIL: not bitwise reproducible")
        all_pass = False

    if all_pass:
        print("\n  ALL ACCEPTANCE CRITERIA PASSED ✓")
    else:
        print("\n  SOME ACCEPTANCE CRITERIA FAILED ✗")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
