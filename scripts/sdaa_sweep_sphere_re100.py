#!/usr/bin/env python3
"""SDAA sweep for sphere_re100 benchmark.

This script runs the sphere Re=100 case on SDAA device and produces:
1. artifacts/sdaa_sphere_re100.json - device evidence + gate judgments
2. docs/sdaa_sweep/sphere_re100.md - documentation

Budget discipline: minimal steps to get Cd + gate judgment, then finish.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

# Fixed paths
WORK_DIR = Path("/root/tp-agent/work/m75")
CASE_DIR = WORK_DIR / "benchmarks" / "verified" / "sphere_re100"
ARTIFACTS_DIR = WORK_DIR / "artifacts"
DOCS_DIR = WORK_DIR / "docs" / "sdaa_sweep"

def get_device_evidence():
    """Get SDAA device evidence."""
    if not hasattr(torch, 'sdaa') or not torch.sdaa.is_available():
        return {"available": False, "error": "SDAA not available"}
    
    try:
        device_name = torch.sdaa.get_device_name(0)
        device_count = torch.sdaa.device_count()
        return {
            "available": True,
            "device_name": device_name,
            "device_count": device_count,
            "torch_version": torch.__version__,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def run_sphere_case(D, steps, tau, device="sdaa:0"):
    """Run sphere case with given parameters."""
    print(f"\n{'='*60}")
    print(f"Running D={D}, steps={steps}, tau={tau}, device={device}")
    print(f"{'='*60}")
    
    env = os.environ.copy()
    env["W8A_DEV"] = device
    env["PYTHONPATH"] = f"/root/TensorLBM_feat2/src:/root/TensorLBM_feat2/benchmarks:{env.get('PYTHONPATH', '')}"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    out_file = f"sdaa_d{D}_s{steps}.json"
    
    # Command: run.py OUTPUT_FILE [options]
    cmd = [
        sys.executable, "run.py",
        out_file,
        "--D", str(D),
        "--steps", str(steps),
        "--lat", "3.0",
        "--up", "3.0",
        "--down", "4.0",
        "--tau", str(tau),
        "--sample", "50",
        "--cv_tail", "0",
        "--kernel", "sparse",
    ]
    
    t0 = time.time()
    result = subprocess.run(
        cmd,
        cwd=CASE_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=7200,  # 2 hour timeout
    )
    wall_time = time.time() - t0
    
    print(f"Wall time: {wall_time:.1f}s")
    print(f"Return code: {result.returncode}")
    if result.returncode != 0:
        print(f"STDERR (last 2000 chars):\n{result.stderr[-2000:]}")
    
    # Load result if successful
    out_path = CASE_DIR / out_file
    if out_path.exists():
        with open(out_path) as f:
            data = json.load(f)
        return data, wall_time, result.returncode == 0
    else:
        return None, wall_time, False


def compute_gate_judgment(data, cd_ref=1.0917311):
    """Compute gate judgment from run data."""
    if not data or "history" not in data:
        return {"pass": False, "reason": "no data"}
    
    hist = data["history"]
    if not hist:
        return {"pass": False, "reason": "empty history"}
    
    # Last 20% window mean Cd
    n = len(hist)
    w = max(1, int(n * 0.2))
    tail = [h["cd"] for h in hist[-w:]]
    prev = [h["cd"] for h in hist[-2*w:-w]] if n >= 2*w else [h["cd"] for h in hist[:w]]
    
    cd_mean = sum(tail) / len(tail)
    cd_prev_mean = sum(prev) / len(prev)
    drift_pct = (cd_mean - cd_prev_mean) / cd_prev_mean * 100.0
    
    err_pct = (cd_mean - cd_ref) / cd_ref * 100.0
    
    # Gate checks
    within_gate = abs(err_pct) <= 3.0
    steady = abs(drift_pct) < 0.3 and data["meta"]["steps"] >= 8000
    
    return {
        "cd_window": cd_mean,
        "cd_prev_window": cd_prev_mean,
        "drift_pct": drift_pct,
        "err_pct": err_pct,
        "within_gate": within_gate,
        "steady": steady,
        "pass": within_gate and steady,
        "steps": data["meta"]["steps"],
        "D": data["meta"]["D"],
    }


def main():
    print("SDAA Sweep: sphere_re100")
    print("="*60)
    
    # Device evidence
    device_ev = get_device_evidence()
    print(f"Device: {device_ev}")
    
    if not device_ev.get("available"):
        print("ERROR: SDAA not available")
        sys.exit(1)
    
    results = []
    
    # D=40, 8000 steps (minimum for steady state gate)
    tau_d40 = 0.56
    data_d40, wall_d40, ok_d40 = run_sphere_case(40, 8000, tau_d40)
    if ok_d40 and data_d40:
        judgment_d40 = compute_gate_judgment(data_d40)
        results.append({
            "tier": "T1_sdaa_d40_8k",
            "D": 40,
            "steps": 8000,
            "tau": tau_d40,
            "wall_time_s": wall_d40,
            "judgment": judgment_d40,
            "success": True,
        })
    else:
        results.append({
            "tier": "T1_sdaa_d40_8k",
            "D": 40,
            "steps": 8000,
            "tau": tau_d40,
            "wall_time_s": wall_d40,
            "judgment": {"pass": False, "reason": "run failed"},
            "success": False,
        })
    
    # D=60, 16000 steps (scaled from 24000 for speed, but still >= 8000)
    tau_d60 = 0.59
    data_d60, wall_d60, ok_d60 = run_sphere_case(60, 16000, tau_d60)
    if ok_d60 and data_d60:
        judgment_d60 = compute_gate_judgment(data_d60)
        results.append({
            "tier": "T2_sdaa_d60_16k",
            "D": 60,
            "steps": 16000,
            "tau": tau_d60,
            "wall_time_s": wall_d60,
            "judgment": judgment_d60,
            "success": True,
        })
    else:
        results.append({
            "tier": "T2_sdaa_d60_16k",
            "D": 60,
            "steps": 16000,
            "tau": tau_d60,
            "wall_time_s": wall_d60,
            "judgment": {"pass": False, "reason": "run failed"},
            "success": False,
        })
    
    # Overall judgment
    all_pass = all(r["judgment"].get("pass", False) for r in results if r["success"])
    monotone = False
    if len(results) >= 2 and all(r["success"] for r in results):
        err0 = abs(results[0]["judgment"].get("err_pct", 999))
        err1 = abs(results[1]["judgment"].get("err_pct", 999))
        monotone = err0 > err1
    
    overall_pass = all_pass and monotone
    
    # Write artifact
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "case": "sphere_re100",
        "device_evidence": device_ev,
        "runs": results,
        "overall": {
            "all_within_gate_and_steady": all_pass,
            "monotone_error_decrease": monotone,
            "pass": overall_pass,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    artifact_path = ARTIFACTS_DIR / "sdaa_sphere_re100.json"
    with open(artifact_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nArtifact written: {artifact_path}")
    
    # Write docs
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    doc_path = DOCS_DIR / "sphere_re100.md"
    
    with open(doc_path, "w") as f:
        f.write("# SDAA Sweep: sphere_re100\n\n")
        f.write("## Case\n\n")
        f.write("球体绕流 Re=100，BFL 插值壁，稀疏核。\n\n")
        f.write("## Device\n\n")
        f.write(f"- Available: {device_ev.get('available')}\n")
        f.write(f"- Device name: {device_ev.get('device_name', 'N/A')}\n")
        f.write(f"- Device count: {device_ev.get('device_count', 'N/A')}\n")
        f.write(f"- Torch version: {device_ev.get('torch_version', 'N/A')}\n\n")
        f.write("## Configuration\n\n")
        f.write("| Tier | D | Steps | τ | Domain (lat/up/down) |\n")
        f.write("|------|---|-------|---|----------------------|\n")
        f.write(f"| T1 | 40 | 8000 | {tau_d40} | 3.0/3.0/4.0 |\n")
        f.write(f"| T2 | 60 | 16000 | {tau_d60} | 3.0/3.0/4.0 |\n\n")
        f.write("## Results\n\n")
        f.write("| Tier | Cd (window) | err% | drift% | steady | gate | PASS |\n")
        f.write("|------|-------------|------|--------|--------|------|------|\n")
        for r in results:
            j = r["judgment"]
            if r["success"]:
                f.write(f"| {r['tier']} | {j.get('cd_window', 'N/A'):.6f} | ")
                f.write(f"{j.get('err_pct', 0):+.3f} | {j.get('drift_pct', 0):+.3f} | ")
                f.write(f"{j.get('steady', False)} | {j.get('within_gate', False)} | ")
                f.write(f"{j.get('pass', False)} |\n")
            else:
                f.write(f"| {r['tier']} | FAILED | - | - | - | - | False |\n")
        f.write(f"\n## Wall Clock\n\n")
        for r in results:
            f.write(f"- {r['tier']}: {r['wall_time_s']:.1f}s\n")
        f.write(f"\n## Overall Judgment\n\n")
        f.write(f"- All within gate & steady: {all_pass}\n")
        f.write(f"- Monotone error decrease: {monotone}\n")
        f.write(f"- **PASS: {overall_pass}**\n\n")
        if not overall_pass:
            f.write("## Failure Analysis\n\n")
            for r in results:
                if not r["judgment"].get("pass", False):
                    f.write(f"- {r['tier']}: {r['judgment'].get('reason', 'gate not met')}\n")
                    if "err_pct" in r["judgment"]:
                        f.write(f"  - err = {r['judgment']['err_pct']:+.3f}% (gate: ±3%)\n")
                    if "drift_pct" in r["judgment"]:
                        f.write(f"  - drift = {r['judgment']['drift_pct']:+.3f}% (gate: <0.3%)\n")
                    if "steps" in r["judgment"]:
                        f.write(f"  - steps = {r['judgment']['steps']} (min: 8000)\n")
    
    print(f"Docs written: {doc_path}")
    print(f"\nOverall PASS: {overall_pass}")
    
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
