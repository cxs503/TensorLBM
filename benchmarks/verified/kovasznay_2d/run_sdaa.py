#!/usr/bin/env python3
"""SDAA verification run for B18: Kovasznay 2D.

Produces artifacts/sdaa_kovasznay_2d_v2.json with:
- Device info (torch.sdaa.get_device_name(0), /dev/tcaicard*)
- Per-gate criteria with thresholds and PASS/FAIL
- Wall clock
"""
import json
import os
import subprocess
import sys
import time
import glob
from pathlib import Path

# Ensure repo paths
repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo / "src"))
sys.path.insert(0, str(repo / "benchmarks"))

import torch
import numpy as np

# ---- Device info ----
device_info = {}
try:
    device_info["torch_sdaa_available"] = hasattr(torch, "sdaa")
    device_info["torch_sdaa_device_count"] = torch.sdaa.device_count() if hasattr(torch, "sdaa") else 0
    device_info["torch_sdaa_device_name_0"] = torch.sdaa.get_device_name(0) if hasattr(torch, "sdaa") and torch.sdaa.device_count() > 0 else None
except Exception as e:
    device_info["torch_sdaa_error"] = str(e)

dev_nodes = sorted(glob.glob("/dev/tcaicard*"))
device_info["dev_tcaicard_nodes"] = dev_nodes
device_info["dev_tcaicard_count"] = len(dev_nodes)

print(f"[SDAA] Device: {device_info.get('torch_sdaa_device_name_0')}")
print(f"[SDAA] /dev/tcaicard* nodes: {len(dev_nodes)}")

# ---- Run the benchmark on SDAA ----
# We use the existing run.py but with device=sdaa
# Import run_case from run.py
from run import run_case, kov_lambda

device_str = "sdaa"
grids = [32, 64, 128]
re_val = 40.0
u0_val = 0.03
xmax_val = 3.0
outlet_val = "zerograd"
min_steps = 20000
max_steps = 60000

out_dir = repo / "artifacts" / "kovasznay_2d_sdaa_v2"
out_dir.mkdir(parents=True, exist_ok=True)

results = []
t_total_start = time.time()

for ny in grids:
    print(f"\n=== Running ny={ny} on SDAA ===", flush=True)
    p = out_dir / f"case_ny{ny}.json"
    t0 = time.time()
    try:
        r = run_case(
            ny=ny,
            re=re_val,
            u0=u0_val,
            xmax=xmax_val,
            outlet=outlet_val,
            min_steps=min_steps,
            max_steps=max_steps,
            out_path=str(p),
            device=device_str,
            threads=48,
            compile_mode="eager",  # SDAA eager mode for stability
        )
        r["device"] = device_str
        r["wall_clock_s"] = round(time.time() - t0, 2)
        results.append(r)
        print(f"  ny={ny}: u_l2={r['u_l2_rel']:.6f} v_l2={r['v_l2_rel']:.6f} "
              f"u_mr={r['u_max_rel_pct']:.3f}% v_mr={r['v_max_rel_pct']:.3f}% "
              f"steps={r['n_steps']} steady={r['steady']} "
              f"wall={r['wall_clock_s']}s", flush=True)
    except Exception as e:
        print(f"  ny={ny}: FAILED with {e}", flush=True)
        results.append({"ny": ny, "error": str(e), "device": device_str})

t_total = time.time() - t_total_start

# ---- Build verdict ----
by_n = {r["ny"]: r for r in results if "error" not in r}
criteria = []
verdict = "FAIL"

if 64 in by_n and 128 in by_n:
    c64, c128 = by_n[64], by_n[128]
    
    # Criterion 1: u L2 rel err <= 3% for ny=64
    u_l2_64 = c64["u_l2_rel"] * 100
    criteria.append({
        "name": "u_L2_rel_ny64",
        "value_pct": round(u_l2_64, 4),
        "threshold_pct": 3.0,
        "pass": u_l2_64 <= 3.0
    })
    
    # Criterion 2: v L2 rel err <= 3% for ny=64
    v_l2_64 = c64["v_l2_rel"] * 100
    criteria.append({
        "name": "v_L2_rel_ny64",
        "value_pct": round(v_l2_64, 4),
        "threshold_pct": 3.0,
        "pass": v_l2_64 <= 3.0
    })
    
    # Criterion 3: u L2 rel err <= 3% for ny=128
    u_l2_128 = c128["u_l2_rel"] * 100
    criteria.append({
        "name": "u_L2_rel_ny128",
        "value_pct": round(u_l2_128, 4),
        "threshold_pct": 3.0,
        "pass": u_l2_128 <= 3.0
    })
    
    # Criterion 4: v L2 rel err <= 3% for ny=128
    v_l2_128 = c128["v_l2_rel"] * 100
    criteria.append({
        "name": "v_L2_rel_ny128",
        "value_pct": round(v_l2_128, 4),
        "threshold_pct": 3.0,
        "pass": v_l2_128 <= 3.0
    })
    
    # Criterion 5: u masked max rel err <= 3% for ny=64
    u_mr_64 = c64["u_max_rel_pct"]
    criteria.append({
        "name": "u_masked_max_rel_ny64",
        "value_pct": round(u_mr_64, 4),
        "threshold_pct": 3.0,
        "pass": u_mr_64 <= 3.0
    })
    
    # Criterion 6: v masked max rel err <= 3% for ny=64
    v_mr_64 = c64["v_max_rel_pct"]
    criteria.append({
        "name": "v_masked_max_rel_ny64",
        "value_pct": round(v_mr_64, 4),
        "threshold_pct": 3.0,
        "pass": v_mr_64 <= 3.0
    })
    
    # Criterion 7: u masked max rel err <= 3% for ny=128
    u_mr_128 = c128["u_max_rel_pct"]
    criteria.append({
        "name": "u_masked_max_rel_ny128",
        "value_pct": round(u_mr_128, 4),
        "threshold_pct": 3.0,
        "pass": u_mr_128 <= 3.0
    })
    
    # Criterion 8: v masked max rel err <= 3% for ny=128
    v_mr_128 = c128["v_max_rel_pct"]
    criteria.append({
        "name": "v_masked_max_rel_ny128",
        "value_pct": round(v_mr_128, 4),
        "threshold_pct": 3.0,
        "pass": v_mr_128 <= 3.0
    })
    
    # Criterion 9: Convergence (64->128 all decrease)
    conv = {
        "u_l2_decreases": c128["u_l2_rel"] < c64["u_l2_rel"],
        "v_l2_decreases": c128["v_l2_rel"] < c64["v_l2_rel"],
        "u_max_rel_decreases": c128["u_max_rel_pct"] < c64["u_max_rel_pct"],
        "v_max_rel_decreases": c128["v_max_rel_pct"] < c64["v_max_rel_pct"],
    }
    conv_ok = all(conv.values())
    criteria.append({
        "name": "convergence_64_to_128",
        "value": conv,
        "threshold": "all four metrics decrease",
        "pass": conv_ok
    })
    
    # Criterion 10: min steps ok
    min_steps_ok = all(r["n_steps"] >= min_steps for r in [c64, c128])
    criteria.append({
        "name": "min_steps_ok",
        "value": min_steps_ok,
        "threshold": f"n_steps >= {min_steps}",
        "pass": min_steps_ok
    })
    
    # Criterion 11: finite
    finite = all(r["finite"] for r in [c64, c128])
    criteria.append({
        "name": "finite",
        "value": finite,
        "threshold": "all fields finite",
        "pass": finite
    })
    
    all_pass = all(c["pass"] for c in criteria)
    verdict = "PASS" if all_pass else "FAIL"
else:
    verdict = "NOT_ENOUGH_GRIDS"

# ---- Build artifact ----
artifact = {
    "case": "B18_kovasznay_2d",
    "run_id": "sdaa_kovasznay_2d_v2",
    "device_info": device_info,
    "config": {
        "re": re_val,
        "u0": u0_val,
        "xmax": xmax_val,
        "outlet": outlet_val,
        "lambda": kov_lambda(re_val),
        "grids": grids,
        "min_steps": min_steps,
        "max_steps": max_steps,
        "device": device_str,
        "compile_mode": "eager",
    },
    "criteria": criteria,
    "verdict": verdict,
    "per_grid_results": results,
    "wall_clock_total_s": round(t_total, 2),
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
}

out_path = repo / "artifacts" / "sdaa_kovasznay_2d_v2.json"
out_path.write_text(json.dumps(artifact, indent=2))
print(f"\n=== Artifact written to {out_path} ===")
print(f"Verdict: {verdict}")
print(f"Total wall clock: {t_total:.1f}s")

# Print criteria summary
print("\n--- Criteria Summary ---")
for c in criteria:
    status = "PASS" if c["pass"] else "FAIL"
    if "value_pct" in c:
        print(f"  {c['name']}: {c['value_pct']:.4f}% (threshold {c['threshold_pct']}%) [{status}]")
    elif "value" in c and isinstance(c["value"], dict):
        print(f"  {c['name']}: {c['value']} [{status}]")
    else:
        print(f"  {c['name']}: {c['value']} [{status}]")
