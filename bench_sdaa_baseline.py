#!/usr/bin/env python3
"""M2 · Single-Card SDAA Performance Baseline — cavity_3d_full (64³, Re=1000, TRT/D3Q19).

Runs the cavity_3d_full case 5 times on a single SDAA card (exclusive),
records wall-clock time, MLUPS, and peak device memory for each run.
Computes CV (coefficient of variation) on wall-clock time.

Output: artifacts/sdaa_perf_baseline.json
"""

import gc
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

# Ensure SDAA exclusive mode
os.environ["SDAA_VISIBLE_DEVICES"] = "0"

import torch
import torch_sdaa  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from tensorlbm.boundaries3d import zou_he_moving_lid_3d
from tensorlbm.d3q19 import OPPOSITE, equilibrium3d
from tensorlbm.solver3d import collide_trt3d, stream3d


def stationary_pre_bounce3d(f_pre, f, wall):
    """Pre-streaming half-way bounce-back (stationary walls)."""
    opp = OPPOSITE.to(f.device)
    return torch.where(wall.unsqueeze(0), f_pre[opp], f)


def run_single(nx=64, re=1000, u_lid=0.06, steps=200, device_str="sdaa:0", warmup=30):
    """Run cavity_3d_full once and return (wall_s, mlups, mem_gib)."""
    device = torch.device(device_str)
    nz = ny = nx
    tau = 3.0 * u_lid * nx / re + 0.5

    # Reset peak memory stats
    if device.type == "sdaa":
        torch.sdaa.reset_peak_memory_stats(device)
        torch.sdaa.synchronize(device)

    # Initialize fields
    rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=device)
    u0 = torch.zeros((nz, ny, nx), dtype=torch.float32, device=device)
    f = equilibrium3d(rho0, u0, u0, u0)

    # 5 stationary walls (top lid y=ny-1 handled by Zou-He)
    wall = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    wall[:, :, 0] = True    # x=0
    wall[:, :, -1] = True   # x=nx-1
    wall[:, 0, :] = True    # y=0 (bottom)
    wall[0, :, :] = True    # z=0
    wall[-1, :, :] = True   # z=nz-1

    # Warmup (not timed)
    for _ in range(warmup):
        f_pre = f
        f = collide_trt3d(f, tau)
        f = stationary_pre_bounce3d(f_pre, f, wall)
        f = stream3d(f)
        f = zou_he_moving_lid_3d(f, u_lid)

    if device.type == "sdaa":
        torch.sdaa.synchronize(device)
    else:
        torch.synchronize()

    # Timed run
    t0 = time.perf_counter()
    for _ in range(steps):
        f_pre = f
        f = collide_trt3d(f, tau)
        f = stationary_pre_bounce3d(f_pre, f, wall)
        f = stream3d(f)
        f = zou_he_moving_lid_3d(f, u_lid)

    if device.type == "sdaa":
        torch.sdaa.synchronize(device)
    else:
        torch.synchronize()
    wall_s = time.perf_counter() - t0

    # MLUPS: total lattice-site updates / wall time
    total_sites = nz * ny * nx
    total_updates = steps * total_sites
    mlups = total_updates / (wall_s * 1e6)

    # Peak memory (GiB)
    if device.type == "sdaa":
        mem_bytes = torch.sdaa.max_memory_allocated(device)
    else:
        mem_bytes = 0
    mem_gib = mem_bytes / (1024 ** 3)

    # Verify finite
    assert torch.isfinite(f).all().item(), "NaN/Inf detected in distribution function!"

    return wall_s, mlups, mem_gib


def main():
    print("=" * 60)
    print("M2 · SDAA Single-Card Performance Baseline")
    print("Case: cavity_3d_full (64³, Re=1000, TRT/D3Q19)")
    print("=" * 60, flush=True)

    # Check SDAA availability
    sdaa_ok = hasattr(torch, "sdaa") and torch.sdaa.is_available()

    if not sdaa_ok:
        print("HONEST-FAIL: SDAA backend not available!", flush=True)
        print("Falling back to CPU baseline...", flush=True)
        runs = []
        for i in range(5):
            print(f"\n--- CPU Run {i+1}/5 ---", flush=True)
            wall_s, mlups, mem_gib = run_single(steps=200, device_str="cpu", warmup=30)
            runs.append({"wall_s": round(wall_s, 4), "mlups": round(mlups, 4), "mem_gib": round(mem_gib, 4)})
            print(f"  wall_s={wall_s:.4f}, mlups={mlups:.4f}, mem_gib={mem_gib:.4f}", flush=True)
            if i < 4:
                time.sleep(2)  # Cooldown

        wall_times = [r["wall_s"] for r in runs]
        mean_w = statistics.mean(wall_times)
        std_w = statistics.stdev(wall_times) if len(wall_times) > 1 else 0.0
        cv_pct = 100.0 * std_w / mean_w if mean_w > 0 else 0.0
        median_w = statistics.median(wall_times)

        result = {
            "case": "cavity_3d_full",
            "device": "cpu",
            "device_note": "SDAA backend not available; CPU fallback",
            "nx": 64, "ny": 64, "nz": 64,
            "re": 1000, "u_lid": 0.06,
            "collision": "trt", "lattice": "D3Q19",
            "steps": 200,
            "runs": runs,
            "cv_pct": round(cv_pct, 4),
            "median_wall_s": round(median_w, 4),
            "mean_wall_s": round(mean_w, 4),
            "std_wall_s": round(std_w, 4),
        }
        out_path = Path("artifacts/sdaa_perf_baseline.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\nResult written to {out_path}")
        return

    # SDAA available
    device_props = torch.sdaa.get_device_properties(0)
    print(f"Device: SDAA:0 — {device_props.name}, {device_props.total_memory} MiB")
    print(f"PyTorch: {torch.__version__}")
    print(f"Steps per run: 200, Warmup: 30")
    print(f"Cooldown between runs: 2s")
    print(flush=True)

    runs = []
    oom_error = None
    for i in range(5):
        print(f"--- Run {i+1}/5 ---", flush=True)
        
        # Clear memory and garbage collect before each run
        gc.collect()
        if hasattr(torch, 'sdaa') and torch.sdaa.is_available():
            torch.sdaa.empty_cache()
        
        try:
            wall_s, mlups, mem_gib = run_single(steps=200, device_str="sdaa:0", warmup=30)
            runs.append({
                "wall_s": round(wall_s, 4),
                "mlups": round(mlups, 4),
                "mem_gib": round(mem_gib, 4),
            })
            print(f"  wall_s={wall_s:.4f}  MLUPS={mlups:.2f}  mem_gib={mem_gib:.4f}", flush=True)
        except Exception as e:
            err_str = str(e)
            if "out of memory" in err_str.lower() or "oom" in err_str.lower():
                print(f"  OOM ERROR: {e}", flush=True)
                oom_error = err_str
                break
            else:
                print(f"  ERROR: {e}", flush=True)
                traceback.print_exc()
                raise
        
        # Cooldown between runs (except after last run)
        if i < 4:
            print("  Cooling down 2s...", flush=True)
            time.sleep(2)

    # Handle OOM
    if oom_error:
        print(f"\nHONEST-FAIL: SDAA OOM — {oom_error}", flush=True)
        print("Falling back to CPU baseline...", flush=True)
        runs = []
        for i in range(5):
            print(f"--- CPU Fallback Run {i+1}/5 ---", flush=True)
            wall_s, mlups, mem_gib = run_single(steps=200, device_str="cpu", warmup=30)
            runs.append({"wall_s": round(wall_s, 4), "mlups": round(mlups, 4), "mem_gib": round(mem_gib, 4)})
            print(f"  wall_s={wall_s:.4f}, mlups={mlups:.4f}, mem_gib={mem_gib:.4f}", flush=True)
            if i < 4:
                time.sleep(2)
        device_label = "cpu"
        device_note = f"SDAA OOM ({oom_error}); CPU fallback"
        steps_used = 200
    else:
        device_label = "sdaa:0"
        device_note = f"SDAA:0 exclusive ({device_props.name})"
        steps_used = 200

    # Compute statistics
    wall_times = [r["wall_s"] for r in runs]
    mean_w = statistics.mean(wall_times)
    std_w = statistics.stdev(wall_times) if len(wall_times) > 1 else 0.0
    cv_pct = 100.0 * std_w / mean_w if mean_w > 0 else 0.0
    median_w = statistics.median(wall_times)

    mlups_list = [r["mlups"] for r in runs]
    mean_mlups = statistics.mean(mlups_list)

    result = {
        "case": "cavity_3d_full",
        "device": device_label,
        "device_note": device_note,
        "device_properties": {
            "name": device_props.name,
            "total_memory_mib": device_props.total_memory,
        },
        "nx": 64, "ny": 64, "nz": 64,
        "re": 1000, "u_lid": 0.06,
        "collision": "trt", "lattice": "D3Q19",
        "steps": steps_used,
        "warmup_steps": 30,
        "cooldown_s": 2,
        "runs": runs,
        "cv_pct": round(cv_pct, 4),
        "median_wall_s": round(median_w, 4),
        "mean_wall_s": round(mean_w, 4),
        "std_wall_s": round(std_w, 4),
        "mean_mlups": round(mean_mlups, 4),
        "cv_pass": cv_pct <= 2.0,
    }

    out_path = Path("artifacts/sdaa_perf_baseline.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Device: {device_label}")
    print(f"Runs: {len(runs)}")
    for i, r in enumerate(runs):
        print(f"  Run {i+1}: wall_s={r['wall_s']:.4f}  MLUPS={r['mlups']:.2f}  mem_gib={r['mem_gib']:.4f}")
    print(f"Median wall_s: {median_w:.4f}")
    print(f"Mean wall_s:   {mean_w:.4f}")
    print(f"Std wall_s:    {std_w:.4f}")
    print(f"CV:            {cv_pct:.4f}%")
    print(f"CV ≤ 2%:       {'PASS' if cv_pct <= 2.0 else 'FAIL'}")
    print(f"Mean MLUPS:    {mean_mlups:.2f}")
    print(f"\nOutput: {out_path}")


if __name__ == "__main__":
    main()
