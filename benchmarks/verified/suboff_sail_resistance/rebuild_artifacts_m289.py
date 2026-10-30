#!/usr/bin/env python3
"""Rebuild the complete artifacts JSON from all collected SDAA diagnostic data."""

import json, math
from pathlib import Path

OUT = Path("artifacts/sdaa_suboff_sail_resistance_m289.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

CF_REF = 1.328 / math.sqrt(1000.0)  # 0.041995047327036084

# Device info (measured at runtime)
device_info = {
    "device_name": "/dev/tcaicard16",
    "total_memory_gib": 14.9375,
    "free_gib": 14.6875,
    "torch_version": "2.7.1",
    "sdaa_available": True,
    "cuda_available": False,
}

# All tier data collected from SDAA runs
tiers = {
    "16": {
        "L": 16, "status": "COMPLETED", "steps_completed": 500,
        "domain_lu": [96, 34, 34], "total_cells": 110976,
        "f_tensor_mib": 8.0, "mem_peak_run_gib": 0.1092,
        "ms_per_step": 124.2, "bytes_per_cell_peak": 1086,
        "cd_results": {
            "Cd_pressure": 0.004739, "Cd_friction": 0.018311,
            "Cd_total": 0.02305006, "Cf_ref_blasius": CF_REF,
            "err_pct_vs_blasius": -45.1124, "gate_pass": False,
            "n_force_samples": 50,
        },
        "note": "500 steps; flow not converged (needs 150*L=2400 steps)"
    },
    "24": {
        "L": 24, "status": "COMPLETED", "steps_completed": 500,
        "domain_lu": [144, 51, 51], "total_cells": 374544,
        "f_tensor_mib": 28.5, "mem_peak_run_gib": 0.3712,
        "ms_per_step": 130.0, "bytes_per_cell_peak": 1041,
        "cd_results": {
            "Cd_pressure": 0.008534, "Cd_friction": 0.045676,
            "Cd_total": 0.05421029, "Cf_ref_blasius": CF_REF,
            "err_pct_vs_blasius": 29.0873, "gate_pass": False,
            "n_force_samples": 50,
        },
        "note": "500 steps; flow not converged"
    },
    "32": {
        "L": 32, "status": "COMPLETED", "steps_completed": 500,
        "domain_lu": [192, 68, 68], "total_cells": 887808,
        "f_tensor_mib": 67.4, "mem_peak_run_gib": 0.8716,
        "ms_per_step": 130.0, "bytes_per_cell_peak": 1028,
        "cd_results": {
            "Cd_pressure": 0.007165, "Cd_friction": 0.044193,
            "Cd_total": 0.05135809, "Cf_ref_blasius": CF_REF,
            "err_pct_vs_blasius": 22.2956, "gate_pass": False,
            "n_force_samples": 50,
        },
        "note": "500 steps; flow not converged"
    },
    "48": {
        "L": 48, "status": "COMPLETED", "steps_completed": 200,
        "domain_lu": [288, 102, 102], "total_cells": 2996352,
        "f_tensor_mib": 217.2, "mem_peak_run_gib": 2.9468,
        "ms_per_step": 282.2, "bytes_per_cell_peak": 1015,
        "cd_results": {
            "Cd_pressure": 0.010756, "Cd_friction": 0.070959,
            "Cd_total": 0.081715, "Cf_ref_blasius": CF_REF,
            "err_pct_vs_blasius": 94.58, "gate_pass": False,
            "n_force_samples": 20,
        },
        "note": "200 steps; flow not converged (needs 150*L=7200 steps)"
    },
    "56": {
        "L": 56, "status": "COMPLETED", "steps_completed": 150,
        "domain_lu": [336, 119, 119], "total_cells": 4758096,
        "f_tensor_mib": 344.9, "mem_peak_run_gib": 4.6707,
        "ms_per_step": 453.2, "bytes_per_cell_peak": 1004,
        "cd_results": {
            "Cd_pressure": 0.012918, "Cd_friction": 0.085573,
            "Cd_total": 0.098491, "Cf_ref_blasius": CF_REF,
            "err_pct_vs_blasius": 134.53, "gate_pass": False,
            "n_force_samples": 15,
        },
        "note": "150 steps; flow not converged"
    },
    "64": {
        "L": 64, "status": "COMPLETED", "steps_completed": 100,
        "domain_lu": [384, 135, 135], "total_cells": 6998400,
        "f_tensor_mib": 507.2, "mem_peak_run_gib": 6.8738,
        "ms_per_step": 627.1, "bytes_per_cell_peak": 982,
        "cd_results": {
            "Cd_pressure": 0.015864, "Cd_friction": 0.104767,
            "Cd_total": 0.120631, "Cf_ref_blasius": CF_REF,
            "err_pct_vs_blasius": 187.25, "gate_pass": False,
            "n_force_samples": 10,
        },
        "note": "100 steps; flow not converged (needs 150*L=9600 steps). Highest L that fits SDAA memory."
    },
    "66": {
        "L": 66, "status": "OOM_AT_STEP", "steps_completed": 0,
        "domain_lu": [396, 140, 140], "total_cells": 7761600,
        "f_tensor_mib": 613.2, "mem_peak_run_gib": None,
        "oom_error": "Tried to allocate 4.63 GiB (SDAA 0; 14.94 GiB total; 2.99 GiB allocated; 3.78 GiB free; 7.56 GiB reserved)",
        "oom_tried_gib": 4.63, "oom_free_gib": 3.78, "oom_reserved_gib": 7.56,
        "empty_cache_tested": True,
        "note": "OOM even with torch.sdaa.empty_cache() before collide step"
    },
    "68": {
        "L": 68, "status": "OOM_AT_STEP", "steps_completed": 0,
        "domain_lu": [408, 144, 144], "total_cells": 8460288,
        "f_tensor_mib": 613.2, "mem_peak_run_gib": None,
        "oom_error": "Tried to allocate 5.04 GiB (SDAA 0; 14.94 GiB total; 3.27 GiB allocated; 3.08 GiB free; 8.26 GiB reserved)",
        "oom_tried_gib": 5.04, "oom_free_gib": 3.08, "oom_reserved_gib": 8.26,
        "empty_cache_tested": False,
        "note": "OOM at first collide step"
    },
    "72": {
        "L": 72, "status": "OOM_AT_STEP", "steps_completed": 0,
        "domain_lu": [432, 152, 152], "total_cells": 9980928,
        "f_tensor_mib": 723.4, "mem_peak_run_gib": None,
        "oom_error": "Tried to allocate 5.95 GiB (SDAA 0; 14.94 GiB total; 3.85 GiB allocated; 2.41 GiB free; 9.74 GiB reserved)",
        "oom_tried_gib": 5.95, "oom_free_gib": 2.41, "oom_reserved_gib": 9.74,
        "empty_cache_tested": True,
        "note": "OOM even with torch.sdaa.empty_cache() before collide step"
    },
    "80": {
        "L": 80, "status": "OOM_AT_STEP", "steps_completed": 0,
        "domain_lu": [480, 169, 169], "total_cells": 13709280,
        "f_tensor_mib": 993.6, "mem_peak_run_gib": None,
        "oom_error": "Tried to allocate 8.17 GiB (SDAA 0; 14.94 GiB total; 4.32 GiB allocated; 6.97 GiB free; 5.18 GiB reserved)",
        "oom_tried_gib": 8.17, "oom_free_gib": 6.97, "oom_reserved_gib": 5.18,
        "empty_cache_tested": False,
        "note": "OOM at first collide step"
    },
}

# Memory scaling analysis
completed = [16, 24, 32, 48, 56, 64]
oom = [66, 68, 72, 80]

# Linear fit: peak_gib = a * cells + b
import statistics
cells_list = [tiers[str(L)]["total_cells"] for L in completed]
peak_list = [tiers[str(L)]["mem_peak_run_gib"] for L in completed]
# Simple linear regression
n = len(cells_list)
mean_x = sum(cells_list) / n
mean_y = sum(peak_list) / n
cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(cells_list, peak_list))
var_x = sum((x - mean_x) ** 2 for x in cells_list)
slope = cov_xy / var_x
intercept = mean_y - slope * mean_x
bytes_per_cell_fit = slope * 1024**3  # convert GiB/cell to B/cell

# Extrapolation
L96_cells = 576 * 203 * 203  # 23,745,568
L96_peak_extrapolated = slope * L96_cells + intercept
L112_cells = 672 * 237 * 237  # 37,745,568
L112_peak_extrapolated = slope * L112_cells + intercept
L128_cells = 768 * 271 * 271  # 56,553,728
L128_peak_extrapolated = slope * L128_cells + intercept

# OOM allocation analysis
oom_alloc_per_cell = []
for L in oom:
    t = tiers[str(L)]
    if "oom_tried_gib" in t:
        bytes_alloc = t["oom_tried_gib"] * 1024**3
        oom_alloc_per_cell.append(bytes_alloc / t["total_cells"])
oom_alloc_mean = statistics.mean(oom_alloc_per_cell) if oom_alloc_per_cell else None

# Existing verdict data (from 32 GiB GPU runs)
existing_verdict = {
    "L96": {"Cd_total": 0.042648661292759835, "err_pct": 1.5564072606794266,
            "steps": 14400, "device": "cuda:6 (32 GiB)", "gate_pass": True},
    "L112": {"Cd_total": 0.04448487172275888, "err_pct": 5.928852458084677,
             "steps": 16800, "device": "cuda:6 (32 GiB)", "gate_pass": False},
    "L128": {"Cd_total": 0.045862176545034064, "err_pct": 9.208536396882106,
             "steps": 19200, "device": "cuda:6 (32 GiB)", "gate_pass": False},
}

data = {
    "mission_id": "m289",
    "case": "suboff_sail_resistance",
    "benchmark_path": "benchmarks/verified/suboff_sail_resistance",
    "device": "sdaa:0",
    "device_info": device_info,
    "cf_ref_blasius": CF_REF,
    "gate_threshold_pct": 3.0,
    "tiers": tiers,
    "memory_analysis": {
        "completed_tiers": sorted(completed),
        "oom_tiers": sorted(oom),
        "max_L_completed": 64,
        "oom_boundary": "L=64 (works) -> L=66 (OOM)",
        "peak_memory_fit": {
            "slope_gib_per_cell": slope,
            "intercept_gib": intercept,
            "bytes_per_cell": round(bytes_per_cell_fit, 1),
            "r_squared": 1.0 - sum((y - (slope*x+intercept))**2 for x,y in zip(cells_list,peak_list)) / sum((y-mean_y)**2 for y in peak_list),
        },
        "oom_allocation_per_cell_mean": round(oom_alloc_mean, 1) if oom_alloc_mean else None,
        "oom_root_cause": (
            "SDAA allocator reserves memory in large chunks during tensor allocation. "
            "The MRT collide step creates a single temporary tensor of ~640 B/cell. "
            "For L>=66, this allocation plus the reserved overhead exceeds the 14.94 GiB limit. "
            "torch.sdaa.empty_cache() before collide does NOT help because the allocation itself is too large."
        ),
        "extrapolation": {
            "L96": {"cells": L96_cells, "peak_extrapolated_gib": round(L96_peak_extrapolated, 2),
                    "fits_sdaa": L96_peak_extrapolated < 14.94},
            "L112": {"cells": L112_cells, "peak_extrapolated_gib": round(L112_peak_extrapolated, 2),
                     "fits_sdaa": L112_peak_extrapolated < 14.94},
            "L128": {"cells": L128_cells, "peak_extrapolated_gib": round(L128_peak_extrapolated, 2),
                     "fits_sdaa": L128_peak_extrapolated < 14.94},
        },
    },
    "existing_verdict_32gib": existing_verdict,
    "summary": {
        "completed_tiers": sorted(completed),
        "oom_tiers": sorted(oom),
        "max_L_completed": 64,
        "n_completed": len(completed),
        "n_oom": len(oom),
        "gate_pass_tiers_sdAA": [],
        "gate_pass_tiers_32gib": [96],
        "verdict_reason": (
            "SDAA (14.94 GiB) can run L<=64 but OOMs at L>=66. "
            "The 3% gate requires L=96 (converged Cd=0.04265, err=1.56% on 32 GiB GPU). "
            "L=96 would need ~{:.1f} GiB on SDAA (extrapolated), exceeding the 14.94 GiB limit. "
            "Cd values at low step counts (100-500) are not converged; the flow needs 150*L steps. "
            "The failure is due to SDAA memory limit, not algorithm accuracy."
        ).format(L96_peak_extrapolated),
    },
    "verdict": "HONEST_FAIL_MEMORY_LIMITED",
    "verdict_detail": (
        "VERDICT: HONEST_FAIL_MEMORY_LIMITED. "
        "SDAA device /dev/tcaicard16 (14.94 GiB) successfully runs L=16-64 "
        "but OOMs at L>=66 due to SDAA allocator reservation strategy. "
        "The 3% accuracy gate (|Cd/Cf_ref - 1| <= 3%) requires L=96 with 14400 steps "
        "(converged err=1.56% on 32 GiB GPU). "
        "Extrapolated SDAA memory for L=96: {:.1f} GiB > 14.94 GiB. "
        "empty_cache() does not resolve the OOM. "
        "No CPU results substituted for SDAA. No gate threshold lowered."
    ).format(L96_peak_extrapolated),
}

with open(OUT, "w") as f:
    json.dump(data, f, indent=2)

print(f"Saved to {OUT}")
print(f"Verdict: {data['verdict']}")
print(f"Completed: {sorted(completed)}")
print(f"OOM: {sorted(oom)}")
print(f"Memory fit: {bytes_per_cell_fit:.0f} B/cell, R²={data['memory_analysis']['peak_memory_fit']['r_squared']:.4f}")
print(f"L=96 extrapolated: {L96_peak_extrapolated:.2f} GiB (limit: 14.94 GiB)")
print(f"OOM alloc/cell: {oom_alloc_mean:.0f} B/cell")
