#!/usr/bin/env python3
"""P67-N01 全量基准测试脚本。

在集群上运行，生成论文级数据。
"""

import json
import sys
import time

sys.path.insert(0, "src")

import torch
from tensorlbm.heat_convection_benchmarks import (
    simulate_rayleigh_benard,
    simulate_forced_convection_cylinder,
    churchill_bernstein,
    de_vahl_davis_reference,
)


def run_rb_benchmark():
    """Rayleigh-Bénard 基准测试。"""
    print("=" * 60)
    print("Rayleigh-Bénard Natural Convection Benchmark")
    print("=" * 60)
    
    results = []
    
    # Ra sweep
    for ra in [1e3, 1e4, 1e5]:
        nx = 64
        print(f"\n--- Ra = {ra:.0e}, N = {nx} ---")
        
        t0 = time.time()
        result = simulate_rayleigh_benard(
            nx=nx, ny=nx, ra=ra, pr=0.71, tau=0.6,
            steps=80000, report_every=5000,
            enable_buoyancy=True,
        )
        elapsed = time.time() - t0
        
        ref = de_vahl_davis_reference(ra)
        
        print(f"  Nu = {result['nu']:.4f}")
        print(f"  Nu_bot = {result['nu_bot']:.4f}, Nu_top = {result['nu_top']:.4f}")
        print(f"  Energy imbalance = {result['energy_imbalance']:.6f}")
        print(f"  u_max = {result['u_max']:.6f}")
        print(f"  T range = [{result['T_min']:.4f}, {result['T_max']:.4f}]")
        print(f"  Elapsed = {elapsed:.1f}s")
        print(f"  de Vahl Davis ref (cavity) = {ref['nu_ref']:.3f}")
        
        results.append({
            "ra": ra,
            "nx": nx,
            "nu": result["nu"],
            "nu_bot": result["nu_bot"],
            "nu_top": result["nu_top"],
            "energy_imbalance": result["energy_imbalance"],
            "u_max": result["u_max"],
            "nu_ref_cavity": ref["nu_ref"],
            "elapsed_s": elapsed,
            "nu_history": result["nu_history"],
        })
    
    # Pure conduction (negative control)
    print(f"\n--- Negative control: pure conduction (no buoyancy) ---")
    result_nc = simulate_rayleigh_benard(
        nx=64, ny=64, ra=1e3, pr=0.71, tau=0.6,
        steps=5000, report_every=1000,
        enable_buoyancy=False,
    )
    print(f"  Nu = {result_nc['nu']:.4f} (expected: 1.0)")
    print(f"  Energy imbalance = {result_nc['energy_imbalance']:.6f}")
    
    results.append({
        "ra": 1e3,
        "nx": 64,
        "nu": result_nc["nu"],
        "enable_buoyancy": False,
        "energy_imbalance": result_nc["energy_imbalance"],
        "label": "pure_conduction",
    })
    
    return results


def run_cylinder_benchmark():
    """强制对流绕圆柱基准测试。"""
    print("\n" + "=" * 60)
    print("Forced Convection Around Cylinder Benchmark")
    print("=" * 60)
    
    results = []
    
    # Re sweep
    for re in [20, 40, 100, 200]:
        nx = 120
        ny = 60
        print(f"\n--- Re = {re}, Pr = 0.7, N = {nx}x{ny} ---")
        
        t0 = time.time()
        result = simulate_forced_convection_cylinder(
            nx=nx, ny=ny, re=re, pr=0.7, tau=0.55,
            u_in=0.1, steps=30000, report_every=2000,
        )
        elapsed = time.time() - t0
        
        nu_cb = churchill_bernstein(re, 0.7)
        rel_err = abs(result["nu_avg"] - nu_cb) / nu_cb * 100
        
        print(f"  Nu_avg = {result['nu_avg']:.4f}")
        print(f"  Nu_Churchill-Bernstein = {nu_cb:.4f}")
        print(f"  Relative error = {rel_err:.1f}%")
        print(f"  u_max = {result['u_max']:.6f}")
        print(f"  Elapsed = {elapsed:.1f}s")
        
        results.append({
            "re": re,
            "pr": 0.7,
            "nx": nx,
            "ny": ny,
            "nu_avg": result["nu_avg"],
            "nu_churchill_bernstein": nu_cb,
            "relative_error_pct": rel_err,
            "u_max": result["u_max"],
            "elapsed_s": elapsed,
            "nu_history": result["nu_history"],
        })
    
    return results


def main():
    print("P67-N01: LBM Heat Convection Verification")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"PyTorch: {torch.__version__}")
    print()
    
    rb_results = run_rb_benchmark()
    cyl_results = run_cylinder_benchmark()
    
    # Save results
    output = {
        "rayleigh_benard": rb_results,
        "forced_convection_cylinder": cyl_results,
    }
    
    with open("runs/p67n01_benchmark_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    print("\nRayleigh-Bénard:")
    for r in rb_results:
        if r.get("label") == "pure_conduction":
            print(f"  Pure conduction: Nu = {r['nu']:.4f} (expected 1.0)")
        else:
            print(f"  Ra={r['ra']:.0e}: Nu = {r['nu']:.4f}")
    
    print("\nForced convection cylinder:")
    for r in cyl_results:
        print(f"  Re={r['re']}: Nu = {r['nu_avg']:.4f} (CB: {r['nu_churchill_bernstein']:.4f}, err: {r['relative_error_pct']:.1f}%)")
    
    print("\nResults saved to runs/p67n01_benchmark_results.json")


if __name__ == "__main__":
    main()
