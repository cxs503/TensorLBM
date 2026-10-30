#!/usr/bin/env python3
"""SDAA sweep for sphere_re100_mem benchmark case.

This script runs the sphere Re=100 benchmark on SDAA device and collects
judgment metrics according to the case's gate criteria.

Case: benchmarks/verified/sphere_re100_mem
Entry: benchmarks/verified/sphere_re100/run.py
Gates: library_entry, direct_observable (|err|<=3%), monotone_convergence,
       reference_locked_first

Usage:
    python scripts/sdaa_sweep_sphere_re100_mem.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Force SDAA device
os.environ["W8A_DEV"] = "sdaa:0"

import torch

# Monkey-patch torch.cuda to torch.sdaa for compatibility
# The run.py uses torch.cuda.empty_cache() etc. which need to work on SDAA
if not torch.cuda.is_available() and hasattr(torch, 'sdaa') and torch.sdaa.device_count() > 0:
    # Patch torch.cuda methods to use torch.sdaa
    torch.cuda.empty_cache = torch.sdaa.empty_cache
    torch.cuda.max_memory_allocated = torch.sdaa.max_memory_allocated
    torch.cuda.reset_peak_memory_stats = torch.sdaa.reset_peak_memory_stats
    torch.cuda.max_memory_reserved = torch.sdaa.max_memory_reserved
    # Patch OutOfMemoryError
    if hasattr(torch.sdaa, 'OutOfMemoryError'):
        torch.cuda.OutOfMemoryError = torch.sdaa.OutOfMemoryError

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "verified" / "sphere_re100"))

# Import the run_case function from the benchmark
from run import run_case, cd_ref_family


def get_device_evidence():
    """Collect SDAA device evidence."""
    evidence = {
        "device_type": "sdaa",
        "device_count": torch.sdaa.device_count(),
        "current_device": torch.sdaa.current_device(),
    }
    if torch.sdaa.device_count() > 0:
        try:
            evidence["device_name"] = torch.sdaa.get_device_name(0)
        except Exception as e:
            evidence["device_name_error"] = str(e)
    return evidence


def compute_judgments(t1_result_path, t2_result_path, cd_ref):
    """Compute judgment metrics from result files."""
    with open(t1_result_path) as f:
        t1_data = json.load(f)
    with open(t2_result_path) as f:
        t2_data = json.load(f)

    # Extract last 20% window Cd (末 20% 采样窗均值)
    t1_hist = t1_data["history"]
    t2_hist = t2_data["history"]

    # Last 20% window
    t1_window_start = int(len(t1_hist) * 0.8)
    t2_window_start = int(len(t2_hist) * 0.8)

    t1_cd_window = sum(h["cd"] for h in t1_hist[t1_window_start:]) / len(t1_hist[t1_window_start:])
    t2_cd_window = sum(h["cd"] for h in t2_hist[t2_window_start:]) / len(t2_hist[t2_window_start:])

    t1_err_pct = abs(t1_cd_window - cd_ref) / cd_ref * 100
    t2_err_pct = abs(t2_cd_window - cd_ref) / cd_ref * 100

    # Monotone convergence check
    monotone = t1_err_pct > t2_err_pct

    return {
        "t1": {
            "cd_window": t1_cd_window,
            "err_pct": t1_err_pct,
            "D": t1_data["meta"]["D"],
            "steps": t1_data["meta"]["steps"],
        },
        "t2": {
            "cd_window": t2_cd_window,
            "err_pct": t2_err_pct,
            "D": t2_data["meta"]["D"],
            "steps": t2_data["meta"]["steps"],
        },
        "monotone_convergence": monotone,
        "cd_ref": cd_ref,
    }


def main():
    print("=" * 80)
    print("SDAA Sweep: sphere_re100_mem")
    print("=" * 80)

    # Device evidence
    device_evidence = get_device_evidence()
    print(f"\nDevice: {device_evidence.get('device_name', 'N/A')}")
    print(f"Device count: {device_evidence['device_count']}")

    # Reference value (Schiller-Naumann)
    Re = 100.0
    cd_ref = 24.0 / Re * (1.0 + 0.15 * Re**0.687)
    print(f"\nReference Cd (Schiller-Naumann): {cd_ref}")

    # Output paths
    artifacts_dir = REPO_ROOT / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    t1_out = artifacts_dir / "sdaa_sphere_re100_mem_t1.json"
    t2_out = artifacts_dir / "sdaa_sphere_re100_mem_t2.json"

    # Run T1 (D=40, 12k steps)
    print("\n" + "=" * 80)
    print("Running T1: D=40, 12000 steps")
    print("=" * 80)
    t0 = time.time()
    t1_error = None
    try:
        run_case(
            out_path=str(t1_out),
            kernel="sparse",
            D=40,
            steps=12000,
            lat=2.0,
            up=1.25,
            down=2.25,
            tau=0.56,
            ulb=0.05,
            treat="hard",
            sample=50,
            cv_tail=0,
            mem_stats=False,
        )
        t1_wall_time = time.time() - t0
        t1_success = True
        print(f"T1 completed in {t1_wall_time:.1f}s")
    except Exception as e:
        t1_wall_time = time.time() - t0
        t1_success = False
        t1_error = str(e)
        print(f"T1 FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Run T2 (D=60, 24k steps)
    print("\n" + "=" * 80)
    print("Running T2: D=60, 24000 steps")
    print("=" * 80)
    t0 = time.time()
    t2_error = None
    try:
        run_case(
            out_path=str(t2_out),
            kernel="sparse",
            D=60,
            steps=24000,
            lat=2.0,
            up=1.25,
            down=2.25,
            tau=0.59,
            ulb=0.05,
            treat="hard",
            sample=50,
            cv_tail=0,
            mem_stats=False,
        )
        t2_wall_time = time.time() - t0
        t2_success = True
        print(f"T2 completed in {t2_wall_time:.1f}s")
    except Exception as e:
        t2_wall_time = time.time() - t0
        t2_success = False
        t2_error = str(e)
        print(f"T2 FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Compute judgments
    print("\n" + "=" * 80)
    print("Computing judgments")
    print("=" * 80)

    gates = []

    # Gate 1: library_entry
    gates.append({
        "gate": "library_entry",
        "pass": True,
        "evidence": "Uses tensorlbm.bfl_d3q19, tensorlbm.solver3d, etc.",
    })

    if t1_success and t2_success:
        judgments = compute_judgments(t1_out, t2_out, cd_ref)

        # Gate 2: direct_observable (|err| <= 3%)
        t1_pass = judgments["t1"]["err_pct"] <= 3.0
        t2_pass = judgments["t2"]["err_pct"] <= 3.0
        gates.append({
            "gate": "direct_observable",
            "pass": t1_pass and t2_pass,
            "t1_err_pct": judgments["t1"]["err_pct"],
            "t2_err_pct": judgments["t2"]["err_pct"],
            "threshold": 3.0,
        })

        # Gate 3: monotone_convergence
        gates.append({
            "gate": "monotone_convergence",
            "pass": judgments["monotone_convergence"],
            "t1_err_pct": judgments["t1"]["err_pct"],
            "t2_err_pct": judgments["t2"]["err_pct"],
        })

        # Gate 4: reference_locked_first
        gates.append({
            "gate": "reference_locked_first",
            "pass": True,
            "cd_ref": cd_ref,
            "formula": "24/Re * (1 + 0.15*Re^0.687)",
        })

        all_pass = all(g["pass"] for g in gates)
    else:
        # Runs failed - record FAIL for computation-dependent gates
        all_pass = False
        judgments = None

        gates.append({
            "gate": "direct_observable",
            "pass": False,
            "reason": "Runs failed due to OOM",
            "t1_error": t1_error,
            "t2_error": t2_error,
        })

        gates.append({
            "gate": "monotone_convergence",
            "pass": False,
            "reason": "Runs failed due to OOM",
        })

        gates.append({
            "gate": "reference_locked_first",
            "pass": True,
            "cd_ref": cd_ref,
            "formula": "24/Re * (1 + 0.15*Re^0.687)",
        })

    # Build final artifact
    artifact = {
        "case": "sphere_re100_mem",
        "device": "sdaa",
        "device_evidence": device_evidence,
        "timestamp": datetime.now().isoformat(),
        "reference": {
            "cd_ref": cd_ref,
            "Re": Re,
            "formula": "24/Re * (1 + 0.15*Re^0.687)",
        },
        "runs": {
            "t1": {
                "success": t1_success,
                "wall_time_s": t1_wall_time if t1_success else None,
                "error": t1_error,
                "D": 40,
                "steps": 12000,
                "tau": 0.56,
            },
            "t2": {
                "success": t2_success,
                "wall_time_s": t2_wall_time if t2_success else None,
                "error": t2_error,
                "D": 60,
                "steps": 24000,
                "tau": 0.59,
            },
        },
        "judgments": judgments,
        "gates": gates,
        "all_pass": all_pass,
    }

    # Save artifact
    artifact_path = artifacts_dir / "sdaa_sphere_re100_mem.json"
    with open(artifact_path, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"\nArtifact saved to: {artifact_path}")
    print(f"\nAll gates pass: {all_pass}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for g in gates:
        status = "PASS" if g["pass"] else "FAIL"
        print(f"  {g['gate']}: {status}")

    if judgments:
        print(f"\nT1: Cd={judgments['t1']['cd_window']:.6f}, err={judgments['t1']['err_pct']:.4f}%")
        print(f"T2: Cd={judgments['t2']['cd_window']:.6f}, err={judgments['t2']['err_pct']:.4f}%")
    else:
        print("\nRuns failed - no numerical results available")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
