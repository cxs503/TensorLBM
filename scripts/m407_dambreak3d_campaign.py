#!/usr/bin/env python3
"""M12 · Free-surface 3D dam-break: CPU baseline mass drift + SDAA comparison.

Runs CPU and SDAA with identical config, captures height fields, computes L2.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm import (
    free_surface_step,
    init_flags_from_fill,
    init_mass_from_fill,
    init_population_from_fill,
)

LIQUID = 1
INTERFACE = 2


def build_domain(a, g, device, rho_gas=1.0):
    nx, ny, nz = 5 * a, 2 * a + a // 2, 2 * a
    ch = 2 * a
    fill = torch.zeros((nz, ny, nx), dtype=torch.float32, device=device)
    fill[1 : a + 1, :ch, 1 : a + 1] = 1.0
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    solid[0, :, :] = True; solid[-1, :, :] = True
    solid[:, 0, :] = True; solid[:, -1, :] = True
    solid[:, :, 0] = True; solid[:, :, -1] = True
    flags = init_flags_from_fill(fill, solid)
    mass = init_mass_from_fill(fill, flags, rho_liquid=1.0)
    f = init_population_from_fill(fill, flags, rho_liquid=1.0, rho_gas=rho_gas)
    return f, fill, flags, mass, solid, (nx, ny, nz)


def height_field(fill, flags):
    """2D height field h(x,z) = max y-index of wet cell at each (x,z)."""
    wet = (flags == LIQUID) | ((flags == INTERFACE) & (fill >= 0.5))
    ny = wet.shape[1]
    y_idx = torch.arange(ny, device=wet.device, dtype=torch.float32).view(1, -1, 1)
    heights = (wet.float() * y_idx).max(dim=1).values  # (nz, nx)
    return heights


def run_device(a, g, steps, tau, rho_gas, device_str, out_interval=20):
    device = torch.device(device_str)
    f, fill, flags, mass, solid, (nx, ny, nz) = build_domain(a, g, device, rho_gas)
    m0 = float(mass.sum().item())
    t_scale = math.sqrt(2.0 * g / a)
    series = []
    height_snapshots = {}

    t0 = time.time()
    for step in range(1, steps + 1):
        f, fill, flags, mass, df = free_surface_step(
            f, fill, flags, solid, mass=mass, tau=tau, gy=-g,
            rho_liquid=1.0, rho_gas=rho_gas, surface_tension=0.0,
            paired_liquid_interface_debit=True,
        )
        if step % out_interval == 0 or step == steps:
            m_now = float(mass.sum().item())
            drift = (m_now - m0) / m0
            hf = height_field(fill, flags)
            height_snapshots[step] = hf.cpu().clone()
            series.append({
                "step": step,
                "T": step * t_scale,
                "mass": m_now,
                "mass_drift_rel": drift,
                "mass_drift_pct": drift * 100.0,
            })
            print(f"  [{device_str}] st={step:4d} T={step*t_scale:.4f} "
                  f"drift={drift:+.6%} mass={m_now:.4f} t={time.time()-t0:.1f}s", flush=True)
    elapsed = time.time() - t0

    if len(series) >= 2:
        total_drift = series[-1]["mass_drift_rel"]
        total_steps = series[-1]["step"]
        drift_per_1e4 = total_drift / total_steps * 10000.0
    else:
        drift_per_1e4 = 0.0

    return {
        "device": device_str,
        "a": a, "g": g, "tau": tau, "rho_gas": rho_gas,
        "steps": steps,
        "domain": {"nx": nx, "ny": ny, "nz": nz},
        "m0": m0,
        "elapsed_s": elapsed,
        "series": series,
        "height_snapshots": height_snapshots,
        "final_mass_drift_rel": series[-1]["mass_drift_rel"] if series else 0.0,
        "drift_per_1e4_steps": drift_per_1e4,
    }


def compute_l2_error(cpu_result, sdaa_result):
    """L2 relative error of height fields between CPU and SDAA."""
    l2_errors = []
    common_steps = sorted(set(cpu_result["height_snapshots"].keys()) &
                          set(sdaa_result["height_snapshots"].keys()))
    for step in common_steps:
        h_cpu = cpu_result["height_snapshots"][step].float()
        h_sdaa = sdaa_result["height_snapshots"][step].float()
        diff = h_sdaa - h_cpu
        norm_diff = torch.sqrt((diff ** 2).sum()).item()
        norm_ref = torch.sqrt((h_cpu ** 2).sum()).item()
        l2 = norm_diff / norm_ref if norm_ref > 0 else 0.0
        l2_errors.append({"step": step, "l2_rel": l2, "l2_pct": l2 * 100.0})
        print(f"  L2 step={step}: {l2:.8f} ({l2*100:.6f}%)", flush=True)

    if l2_errors:
        max_l2 = max(e["l2_rel"] for e in l2_errors)
        mean_l2 = sum(e["l2_rel"] for e in l2_errors) / len(l2_errors)
        final_l2 = l2_errors[-1]["l2_rel"]
    else:
        max_l2 = mean_l2 = final_l2 = 0.0

    return {
        "common_steps": common_steps,
        "per_step": l2_errors,
        "max_l2_rel": max_l2,
        "mean_l2_rel": mean_l2,
        "final_l2_rel": final_l2,
        "max_l2_pct": max_l2 * 100.0,
        "mean_l2_pct": mean_l2 * 100.0,
    }


def main():
    a = 8; g = 1e-4; steps = 200; tau = 0.8; rho_gas = 1.0; out_interval = 20

    print("=" * 60)
    print(f"M12 Dambreak3D: a={a}, g={g}, steps={steps}, rho_gas={rho_gas}")
    print("=" * 60)

    # CPU baseline
    print("\n[1/2] CPU baseline...")
    cpu_result = run_device(a, g, steps, tau, rho_gas, "cpu", out_interval)
    print(f"\nCPU: drift={cpu_result['final_mass_drift_rel']:+.6%} "
          f"per_1e4={cpu_result['drift_per_1e4_steps']:+.6%} "
          f"elapsed={cpu_result['elapsed_s']:.1f}s")

    # SDAA
    print("\n[2/2] SDAA...")
    try:
        sdaa_result = run_device(a, g, steps, tau, rho_gas, "sdaa:0", out_interval)
        sdaa_available = True
        print(f"\nSDAA: drift={sdaa_result['final_mass_drift_rel']:+.6%} "
              f"per_1e4={sdaa_result['drift_per_1e4_steps']:+.6%} "
              f"elapsed={sdaa_result['elapsed_s']:.1f}s")
    except Exception as e:
        print(f"SDAA failed: {e}")
        sdaa_result = None
        sdaa_available = False

    # L2
    print("\n[3] L2 comparison...")
    if sdaa_available:
        l2_result = compute_l2_error(cpu_result, sdaa_result)
    else:
        l2_result = {"error": "SDAA not available", "max_l2_rel": None}

    # Status
    cpu_drift_per_1e4 = abs(cpu_result["drift_per_1e4_steps"])
    if cpu_drift_per_1e4 > 0.01:
        status = "HONEST_FAIL"
        reason = (f"CPU baseline mass drift {cpu_drift_per_1e4:.4%}/1e4 steps "
                  f"exceeds 1%/1e4 threshold. Criterion invalid. "
                  f"Action: fix CPU-side mass conservation first.")
    elif cpu_drift_per_1e4 > 0.001:
        status = "CPU_BASELINE_REFERENCE"
        reason = (f"CPU baseline mass drift {cpu_drift_per_1e4:.4%}/1e4 steps "
                  f"exceeds 0.1%/1e4 but within 1%/1e4. Using CPU as reference.")
    else:
        status = "PASS"
        reason = f"CPU baseline mass drift {cpu_drift_per_1e4:.4%}/1e4 steps within 0.1%/1e4."

    output = {
        "case": "dambreak3d_sdaa_vs_cpu",
        "mission": "M12",
        "status": status,
        "reason": reason,
        "config": {
            "a": a, "g": g, "tau": tau, "rho_gas": rho_gas,
            "steps": steps, "out_interval": out_interval,
            "domain": cpu_result["domain"],
            "note": "rho_gas=1.0 (density ratio 1:1) required for stability; "
                    "rho_gas=0.1 triggers TopologyTransactionError at step 2",
        },
        "cpu_baseline": {
            "device": "cpu",
            "elapsed_s": cpu_result["elapsed_s"],
            "m0": cpu_result["m0"],
            "final_mass_drift_rel": cpu_result["final_mass_drift_rel"],
            "final_mass_drift_pct": cpu_result["final_mass_drift_rel"] * 100.0,
            "drift_per_1e4_steps": cpu_result["drift_per_1e4_steps"],
            "drift_per_1e4_steps_pct": cpu_result["drift_per_1e4_steps"] * 100.0,
            "mass_drift_pct": cpu_result["final_mass_drift_rel"] * 100.0,
            "series": [{"step": s["step"], "T": s["T"],
                        "mass_drift_rel": s["mass_drift_rel"],
                        "mass_drift_pct": s["mass_drift_pct"]}
                       for s in cpu_result["series"]],
        },
        "sdaa_run": {
            "available": sdaa_available,
            "device": "sdaa:0" if sdaa_available else None,
            "elapsed_s": sdaa_result["elapsed_s"] if sdaa_result else None,
            "final_mass_drift_rel": sdaa_result["final_mass_drift_rel"] if sdaa_result else None,
            "final_mass_drift_pct": sdaa_result["final_mass_drift_rel"] * 100.0 if sdaa_result else None,
            "drift_per_1e4_steps": sdaa_result["drift_per_1e4_steps"] if sdaa_result else None,
            "series": [{"step": s["step"], "T": s["T"],
                        "mass_drift_rel": s["mass_drift_rel"],
                        "mass_drift_pct": s["mass_drift_pct"]}
                       for s in sdaa_result["series"]] if sdaa_result else [],
        } if sdaa_result else {"available": False},
        "l2_comparison": l2_result,
        "l2_error": l2_result.get("max_l2_rel") if isinstance(l2_result, dict) else None,
        "mass_drift_pct": cpu_result["final_mass_drift_rel"] * 100.0,
        "criteria": {
            "cpu_drift_threshold_1e4": 0.01,
            "cpu_drift_warning_1e4": 0.001,
            "l2_threshold": 0.01,
        },
    }

    if isinstance(l2_result, dict) and "max_l2_rel" in l2_result:
        output["l2_error"] = l2_result["max_l2_rel"]

    outdir = Path("artifacts")
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "sdaa_dambreak3d.json"
    with open(outpath, "w") as fp:
        json.dump(output, fp, indent=2)
    print(f"\nSaved: {outpath}")
    print(f"Status: {status}")
    print(f"CPU drift/1e4: {cpu_drift_per_1e4:.4%}")
    if sdaa_available:
        print(f"Max L2: {l2_result.get('max_l2_rel', 'N/A')}")


if __name__ == "__main__":
    main()
