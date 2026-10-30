#!/usr/bin/env python3
"""SDAA single-card matmul benchmark: 1200x1200 fp32.

Protocol (r3-plan approved):
  - SDAA: 10 warmup + 100 timed iterations
  - CPU:  20 warmup + 20 timed iterations (run in separate process to avoid interference)
  - Record mean/std/min/max wall time per iteration
  - Machine criteria: CV <= 2%
  - Reference: ~15ms (SDAA), ~212ms (CPU)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

# Import torch_sdaa to register the sdaa backend
import torch_sdaa  # noqa: F401


def get_git_sha() -> str:
    """Get current git commit SHA."""
    try:
        repo_root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_env_fingerprint() -> dict:
    """Collect environment fingerprint."""
    return {
        "torch_version": torch.__version__,
        "torch_sdaa_version": torch_sdaa.__version__,
        "python_version": sys.version.split()[0],
        "device_name": torch.sdaa.get_device_name(0) if torch.sdaa.device_count() > 0 else "N/A",
        "device_count": torch.sdaa.device_count(),
        "sdaa_card_index": int(os.environ.get("SDAA_VISIBLE_DEVICES", "0")),
        "git_sha": get_git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def bench_matmul(
    device: torch.device,
    N: int = 1200,
    warmup: int = 10,
    n_iter: int = 100,
    dtype: torch.dtype = torch.float32,
    label: str = "",
) -> dict:
    """Benchmark NxN fp32 matmul on given device.
    
    Returns dict with mean/std/min/max in milliseconds, plus raw times.
    """
    print(f"[{device}] Allocating {N}x{N} {dtype} tensors...", flush=True)
    A = torch.randn(N, N, device=device, dtype=dtype)
    B = torch.randn(N, N, device=device, dtype=dtype)
    
    # Warmup
    print(f"[{device}] Warmup ({warmup} iters)...", flush=True)
    for _ in range(warmup):
        C = A @ B
    if device.type == "sdaa":
        torch.sdaa.synchronize()
    elif device.type == "cpu":
        pass  # CPU is synchronous
    
    # Timed iterations
    print(f"[{device}] Timing ({n_iter} iters)...", flush=True)
    times_ms = []
    for i in range(n_iter):
        if device.type == "sdaa":
            torch.sdaa.synchronize()
        t0 = time.perf_counter()
        C = A @ B
        if device.type == "sdaa":
            torch.sdaa.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
        
        if (i + 1) % 20 == 0 or (i + 1) <= 5:
            print(f"  iter {i+1}/{n_iter}: {times_ms[-1]:.3f} ms", flush=True)
    
    times_arr = np.array(times_ms)
    stats = {
        "mean_ms": float(np.mean(times_arr)),
        "std_ms": float(np.std(times_arr)),
        "min_ms": float(np.min(times_arr)),
        "max_ms": float(np.max(times_arr)),
        "cv_pct": float(np.std(times_arr) / np.mean(times_arr) * 100),
        "n_iter": n_iter,
        "warmup": warmup,
        "times_ms": times_ms,
    }
    
    print(f"[{device}] mean={stats['mean_ms']:.3f} ms, std={stats['std_ms']:.3f} ms, "
          f"CV={stats['cv_pct']:.2f}%", flush=True)
    
    return stats


def run_cpu_benchmark_separate() -> dict:
    """Run CPU benchmark in a separate process to avoid interference."""
    print("\n" + "=" * 70)
    print("CPU Benchmark (separate process)")
    print("=" * 70)
    
    # Create a temporary script for CPU-only benchmark
    cpu_script = """
import json
import time
import numpy as np
import torch

N = 1200
WARMUP = 20
N_ITER = 20

A = torch.randn(N, N, dtype=torch.float32)
B = torch.randn(N, N, dtype=torch.float32)

# Warmup
for _ in range(WARMUP):
    C = A @ B

# Timed
times = []
for i in range(N_ITER):
    t0 = time.perf_counter()
    C = A @ B
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000.0)
    if (i + 1) % 5 == 0 or (i + 1) <= 3:
        print(f"  iter {i+1}/{N_ITER}: {times[-1]:.3f} ms", flush=True)

times_arr = np.array(times)
result = {
    "mean_ms": float(np.mean(times_arr)),
    "std_ms": float(np.std(times_arr)),
    "min_ms": float(np.min(times_arr)),
    "max_ms": float(np.max(times_arr)),
    "cv_pct": float(np.std(times_arr) / np.mean(times_arr) * 100),
    "n_iter": N_ITER,
    "warmup": WARMUP,
    "times_ms": times,
}
print(f"CPU_RESULT:{json.dumps(result)}")
"""
    
    # Write temp script
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(cpu_script)
        temp_path = f.name
    
    try:
        # Run with taskset to pin to first NUMA node (cores 0-31)
        result = subprocess.run(
            ["python3", temp_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        
        # Parse result
        for line in result.stdout.split('\n'):
            if line.startswith('CPU_RESULT:'):
                cpu_data = json.loads(line[len('CPU_RESULT:'):])
                print(f"[cpu] mean={cpu_data['mean_ms']:.3f} ms, std={cpu_data['std_ms']:.3f} ms, "
                      f"CV={cpu_data['cv_pct']:.2f}%", flush=True)
                return cpu_data
        
        print("CPU benchmark failed to produce result")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return None
    finally:
        os.unlink(temp_path)


def main():
    N = 1200
    SDAA_ITERS = 100
    SDAA_WARMUP = 10
    
    print("=" * 70)
    print("SDAA Matmul Benchmark: 1200x1200 fp32")
    print("=" * 70)
    
    # Environment fingerprint
    env_fp = get_env_fingerprint()
    print("\nEnvironment Fingerprint:")
    for k, v in env_fp.items():
        print(f"  {k}: {v}")
    
    # CPU benchmark FIRST (in separate process to avoid interference)
    cpu_stats = run_cpu_benchmark_separate()
    if cpu_stats is None:
        print("ERROR: CPU benchmark failed")
        sys.exit(1)
    
    # SDAA benchmark
    print("\n" + "=" * 70)
    print("SDAA Benchmark")
    print("=" * 70)
    sdaa_device = torch.device("sdaa:0")
    torch.sdaa.set_device(0)
    sdaa_stats = bench_matmul(sdaa_device, N, SDAA_WARMUP, SDAA_ITERS)
    
    # Comparison with reference
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    
    ref_sdaa_ms = 15.0
    ref_cpu_ms = 212.0
    
    sdaa_diff_pct = (sdaa_stats["mean_ms"] - ref_sdaa_ms) / ref_sdaa_ms * 100
    cpu_diff_pct = (cpu_stats["mean_ms"] - ref_cpu_ms) / ref_cpu_ms * 100
    speedup = cpu_stats["mean_ms"] / sdaa_stats["mean_ms"]
    
    print(f"\nSDAA: {sdaa_stats['mean_ms']:.3f} ms (ref {ref_sdaa_ms} ms, diff {sdaa_diff_pct:+.2f}%)")
    print(f"CPU:  {cpu_stats['mean_ms']:.3f} ms (ref {ref_cpu_ms} ms, diff {cpu_diff_pct:+.2f}%)")
    print(f"Speedup (CPU/SDAA): {speedup:.2f}x")
    print(f"\nCV SDAA: {sdaa_stats['cv_pct']:.2f}% (target <=2%)")
    print(f"CV CPU:  {cpu_stats['cv_pct']:.2f}% (target <=2%)")
    
    # Machine criteria check
    sdaa_cv_pass = sdaa_stats["cv_pct"] <= 2.0
    sdaa_ref_pass = abs(sdaa_diff_pct) <= 3.0
    cpu_cv_pass = cpu_stats["cv_pct"] <= 2.0
    cpu_ref_pass = abs(cpu_diff_pct) <= 3.0
    
    print("\n" + "=" * 70)
    print("Machine Criteria")
    print("=" * 70)
    print(f"SDAA CV <= 2%:        {'PASS' if sdaa_cv_pass else 'FAIL'} ({sdaa_stats['cv_pct']:.2f}%)")
    print(f"SDAA vs ref < 3%:     {'PASS' if sdaa_ref_pass else 'FAIL'} ({sdaa_diff_pct:+.2f}%)")
    print(f"CPU CV <= 2%:         {'PASS' if cpu_cv_pass else 'FAIL'} ({cpu_stats['cv_pct']:.2f}%)")
    print(f"CPU vs ref < 3%:      {'PASS' if cpu_ref_pass else 'FAIL'} ({cpu_diff_pct:+.2f}%)")
    
    # Build output JSON
    output = {
        "benchmark": "sdaa_matmul_1200x1200_fp32",
        "environment": env_fp,
        "sdaa": {
            "mean_ms": sdaa_stats["mean_ms"],
            "std_ms": sdaa_stats["std_ms"],
            "min_ms": sdaa_stats["min_ms"],
            "max_ms": sdaa_stats["max_ms"],
            "cv_pct": sdaa_stats["cv_pct"],
            "n_iter": sdaa_stats["n_iter"],
            "warmup": sdaa_stats["warmup"],
            "times_ms": sdaa_stats["times_ms"],
            "reference_ms": ref_sdaa_ms,
            "diff_from_ref_pct": sdaa_diff_pct,
        },
        "cpu": {
            "mean_ms": cpu_stats["mean_ms"],
            "std_ms": cpu_stats["std_ms"],
            "min_ms": cpu_stats["min_ms"],
            "max_ms": cpu_stats["max_ms"],
            "cv_pct": cpu_stats["cv_pct"],
            "n_iter": cpu_stats["n_iter"],
            "warmup": cpu_stats["warmup"],
            "times_ms": cpu_stats["times_ms"],
            "reference_ms": ref_cpu_ms,
            "diff_from_ref_pct": cpu_diff_pct,
        },
        "comparison": {
            "speedup": speedup,
            "sdaa_cv_pass": sdaa_cv_pass,
            "sdaa_ref_pass": sdaa_ref_pass,
            "cpu_cv_pass": cpu_cv_pass,
            "cpu_ref_pass": cpu_ref_pass,
        },
    }
    
    # Write artifacts/bench.json
    artifacts_dir = Path(__file__).resolve().parent.parent / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    out_path = artifacts_dir / "bench.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {out_path}")
    
    # Generate comparison plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: SDAA times with mean line
        ax = axes[0, 0]
        sdaa_times = sdaa_stats["times_ms"]
        ax.plot(range(1, len(sdaa_times) + 1), sdaa_times, "o-", markersize=3, label="SDAA per-iter")
        ax.axhline(sdaa_stats["mean_ms"], color="r", linestyle="--", label=f"Mean {sdaa_stats['mean_ms']:.2f} ms")
        ax.axhline(ref_sdaa_ms, color="g", linestyle=":", label=f"Ref {ref_sdaa_ms} ms")
        ax.fill_between(
            range(1, len(sdaa_times) + 1),
            sdaa_stats["mean_ms"] - sdaa_stats["std_ms"],
            sdaa_stats["mean_ms"] + sdaa_stats["std_ms"],
            alpha=0.2,
            color="r",
            label=f"±1σ ({sdaa_stats['std_ms']:.2f} ms)",
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Time (ms)")
        ax.set_title("SDAA 1200x1200 Matmul (100 iters)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: CPU times with mean line
        ax = axes[0, 1]
        cpu_times = cpu_stats["times_ms"]
        ax.plot(range(1, len(cpu_times) + 1), cpu_times, "s-", markersize=4, label="CPU per-iter")
        ax.axhline(cpu_stats["mean_ms"], color="r", linestyle="--", label=f"Mean {cpu_stats['mean_ms']:.2f} ms")
        ax.axhline(ref_cpu_ms, color="g", linestyle=":", label=f"Ref {ref_cpu_ms} ms")
        ax.fill_between(
            range(1, len(cpu_times) + 1),
            cpu_stats["mean_ms"] - cpu_stats["std_ms"],
            cpu_stats["mean_ms"] + cpu_stats["std_ms"],
            alpha=0.2,
            color="r",
            label=f"±1σ ({cpu_stats['std_ms']:.2f} ms)",
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Time (ms)")
        ax.set_title("CPU 1200x1200 Matmul (20 iters)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Rolling mean comparison (every 10 iters for SDAA)
        ax = axes[1, 0]
        # SDAA: group by 10
        sdaa_groups = [sdaa_times[i:i+10] for i in range(0, len(sdaa_times), 10)]
        sdaa_means = [np.mean(g) for g in sdaa_groups]
        sdaa_stds = [np.std(g) for g in sdaa_groups]
        sdaa_x = [10 * (i + 1) for i in range(len(sdaa_means))]
        
        ax.errorbar(sdaa_x, sdaa_means, yerr=sdaa_stds, fmt="o-", capsize=5, label="SDAA (per 10 iters)")
        ax.axhline(ref_sdaa_ms, color="g", linestyle=":", label=f"Ref {ref_sdaa_ms} ms")
        ax.set_xlabel("Iteration (grouped)")
        ax.set_ylabel("Mean Time (ms)")
        ax.set_title("SDAA Rolling Mean (10-iter groups)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: CPU rolling mean (every 5 iters since only 20 total)
        ax = axes[1, 1]
        cpu_groups = [cpu_times[i:i+5] for i in range(0, len(cpu_times), 5)]
        cpu_means = [np.mean(g) for g in cpu_groups]
        cpu_stds = [np.std(g) for g in cpu_groups]
        cpu_x = [5 * (i + 1) for i in range(len(cpu_means))]
        
        ax.errorbar(cpu_x, cpu_means, yerr=cpu_stds, fmt="s-", capsize=5, label="CPU (per 5 iters)")
        ax.axhline(ref_cpu_ms, color="g", linestyle=":", label=f"Ref {ref_cpu_ms} ms")
        ax.set_xlabel("Iteration (grouped)")
        ax.set_ylabel("Mean Time (ms)")
        ax.set_title("CPU Rolling Mean (5-iter groups)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = artifacts_dir / "sdaa_matmul_bench.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Wrote {plot_path}")
        
    except ImportError:
        print("matplotlib not available, skipping plot generation")
    
    print("\n" + "=" * 70)
    print("Benchmark Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
