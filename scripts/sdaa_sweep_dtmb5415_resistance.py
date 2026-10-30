#!/usr/bin/env python3
"""SDAA verification script for DTMB 5415 resistance benchmark - Quick version.

This script:
1. Verifies SDAA device availability
2. Runs a quick benchmark test on SDAA
3. Generates evidence file

Usage:
    PYTHONPATH=src python scripts/sdaa_sweep_dtmb5415_resistance.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

# Configuration
_CASE_DIR = Path("/root/tp-agent/work/m438/benchmarks/pending/dtmb5415_resistance")
_REPO_ROOT = _CASE_DIR.parents[2]
_ARTIFACTS_DIR = _REPO_ROOT / "artifacts"

# Add paths
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_CASE_DIR))


def check_sdaa_availability():
    """Check SDAA device availability and return device info."""
    print("=" * 60)
    print("STEP 1: SDAA Device Verification")
    print("=" * 60)
    
    device_info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "sdaa_available": False,
        "sdaa_device_count": 0,
        "sdaa_device_name": None,
        "sdaa_backend_version": None,
    }
    
    # Check SDAA
    if hasattr(torch, "sdaa"):
        device_info["sdaa_available"] = torch.sdaa.is_available()
        if device_info["sdaa_available"]:
            device_info["sdaa_device_count"] = torch.sdaa.device_count()
            device_info["sdaa_device_name"] = torch.sdaa.get_device_name(0)
            try:
                import torch_sdaa
                device_info["sdaa_backend_version"] = getattr(torch_sdaa, "__version__", "unknown")
            except ImportError:
                device_info["sdaa_backend_version"] = "torch_sdaa not importable"
    
    print(f"  PyTorch version: {device_info['torch_version']}")
    print(f"  CUDA available: {device_info['cuda_available']}")
    print(f"  SDAA available: {device_info['sdaa_available']}")
    if device_info["sdaa_available"]:
        print(f"  SDAA device count: {device_info['sdaa_device_count']}")
        print(f"  SDAA device name: {device_info['sdaa_device_name']}")
        print(f"  SDAA backend version: {device_info['sdaa_backend_version']}")
    
    # Verify device is usable
    if device_info["sdaa_available"] and device_info["sdaa_device_count"] > 0:
        try:
            test_tensor = torch.tensor([1.0, 2.0, 3.0], device="sdaa:0")
            result = test_tensor.sum().item()
            print(f"  SDAA tensor test: {result} (expected 6.0)")
            device_info["sdaa_tensor_test_passed"] = True
        except Exception as e:
            print(f"  SDAA tensor test FAILED: {e}")
            device_info["sdaa_tensor_test_passed"] = False
    else:
        device_info["sdaa_tensor_test_passed"] = False
    
    print()
    return device_info


def run_benchmark_quick(device: str = "sdaa:0", resolution: int = 48, steps: int = 500):
    """Run a quick benchmark test."""
    print("=" * 60)
    print(f"STEP 2: Quick Benchmark Test (device={device}, res={resolution}, steps={steps})")
    print("=" * 60)
    
    cmd = [
        sys.executable,
        str(_CASE_DIR / "run.py"),
        "--geometry", "stl",
        "--resolution", str(resolution),
        "--steps", str(steps),
        "--device", device,
        "--collision", "mrt",
        "--friction", "mix50",
    ]
    
    print(f"  Running: {' '.join(cmd)}")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{_REPO_ROOT / 'src'}:{_CASE_DIR}"
    
    start_time = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_CASE_DIR),
        timeout=300,  # 5 min timeout
    )
    elapsed = time.time() - start_time
    
    print(f"  Return code: {result.returncode}")
    print(f"  Elapsed time: {elapsed:.1f}s")
    
    # Show key output lines
    output_lines = result.stdout.splitlines()
    for line in output_lines[-20:]:
        if "===" in line or "Cd_" in line or "err" in line.lower() or "setup" in line:
            print(f"    {line}")
    
    if result.returncode != 0 and result.stderr:
        print("  STDERR:")
        for line in result.stderr.splitlines()[:10]:
            print(f"    {line}")
    
    print()
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "elapsed_s": elapsed,
    }


def parse_results():
    """Parse the result.json file from benchmark run."""
    print("=" * 60)
    print("STEP 3: Parsing Results")
    print("=" * 60)
    
    # Find the result directory
    result_dirs = sorted(
        _REPO_ROOT.glob("results_dtmb5415_L48_stl_mix50"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    if not result_dirs:
        result_dirs = sorted(
            _REPO_ROOT.glob("results_dtmb5415_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
    
    if not result_dirs:
        print("  ERROR: No result directory found")
        return None
    
    result_dir = result_dirs[0]
    result_file = result_dir / "result.json"
    
    if not result_file.exists():
        print(f"  ERROR: Result file not found: {result_file}")
        return None
    
    with open(result_file) as f:
        data = json.load(f)
    
    print(f"  Result dir: {result_dir.name}")
    print(f"  Case: {data.get('case')}")
    print(f"  Device: {data.get('device')}")
    print(f"  Resolution: {data.get('L_cells')}")
    print(f"  Steps: {data.get('n_steps')}")
    print(f"  Cd_total: {data.get('Cd_total'):.6f}")
    print(f"  Diverged: {data.get('diverged')}")
    
    # Reference values
    refs = data.get("references", {})
    errors = data.get("errors_pct", {})
    print()
    print("  Reference & Errors:")
    print(f"    Blasius Cf: {refs.get('blasius_Cf_Re1000'):.6f}, err: {errors.get('vs_blasius'):.2f}%")
    print(f"    ITTC57 Cf: {refs.get('ittc57_Cf'):.6f}, err: {errors.get('vs_ittc57'):.2f}%")
    print(f"    Exp CT: {refs.get('exp_CT_Fr0.28'):.6f}, err: {errors.get('vs_exp_CT'):.2f}%")
    
    # No-wave bound note
    bound = data.get("no_wave_friction_bound", {})
    print(f"    No-wave bound: {bound.get('value'):.6f}, err: {bound.get('err_vs_exp_CT_pct'):.2f}%")
    print(f"    Note: {bound.get('note', 'N/A')[:60]}...")
    
    print()
    return data


def validate_criteria(result_data: dict):
    """Validate results against acceptance criteria."""
    print("=" * 60)
    print("STEP 4: Criteria Validation")
    print("=" * 60)
    
    THRESHOLD = 3.0
    errors = result_data.get("errors_pct", {})
    bound = result_data.get("no_wave_friction_bound", {})
    
    validations = {}
    
    # Blasius comparison
    err_blasius = errors.get("vs_blasius", float("inf"))
    validations["vs_blasius"] = {
        "error_pct": err_blasius,
        "threshold_pct": THRESHOLD,
        "passed": abs(err_blasius) < THRESHOLD if err_blasius != float("inf") else False,
    }
    
    # ITTC57 comparison
    err_ittc = errors.get("vs_ittc57", float("inf"))
    validations["vs_ittc57"] = {
        "error_pct": err_ittc,
        "threshold_pct": THRESHOLD,
        "passed": abs(err_ittc) < THRESHOLD if err_ittc != float("inf") else False,
    }
    
    # Experimental CT comparison (expected to fail - single phase LBM)
    err_exp = errors.get("vs_exp_CT", float("inf"))
    validations["vs_exp_CT"] = {
        "error_pct": err_exp,
        "threshold_pct": THRESHOLD,
        "passed": abs(err_exp) < THRESHOLD if err_exp != float("inf") else False,
        "note": "Expected to fail - single-phase LBM cannot produce wave-making resistance",
    }
    
    # No-wave bound
    err_bound = bound.get("err_vs_exp_CT_pct", float("inf"))
    validations["no_wave_bound"] = {
        "error_pct": err_bound,
        "threshold_pct": THRESHOLD,
        "passed": abs(err_bound) < THRESHOLD if err_bound != float("inf") else False,
        "note": "Theoretical upper bound for single-phase LBM",
    }
    
    for key, val in validations.items():
        status = "PASS" if val["passed"] else "FAIL"
        print(f"  {key}: {val['error_pct']:.2f}% (threshold: ±{val['threshold_pct']}%) -> {status}")
        if "note" in val:
            print(f"    Note: {val['note']}")
    
    print()
    return validations


def generate_evidence(device_info, run_result, result_data, validations):
    """Generate evidence JSON file."""
    print("=" * 60)
    print("STEP 5: Generating Evidence File")
    print("=" * 60)
    
    evidence = {
        "benchmark": "dtmb5415_resistance",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device_verification": device_info,
        "run_info": {
            "returncode": run_result["returncode"],
            "elapsed_s": run_result["elapsed_s"],
        },
        "results": result_data,
        "validations": validations,
        "conclusion": {
            "device_works": device_info.get("sdaa_tensor_test_passed", False),
            "benchmark_runs": run_result["returncode"] == 0,
            "results_valid": result_data.get("diverged", True) == False,
            "criteria_passed": any(v["passed"] for v in validations.values()),
        },
    }
    
    # Write evidence file
    evidence_file = _ARTIFACTS_DIR / "sdaa_dtmb5415_resistance_evidence.json"
    _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(evidence_file, "w") as f:
        json.dump(evidence, f, indent=2)
    
    print(f"  Evidence written to: {evidence_file}")
    print()
    
    return evidence_file


def main():
    """Main verification workflow."""
    print()
    print("=" * 60)
    print("SDAA VERIFICATION: DTMB 5415 Resistance Benchmark")
    print("=" * 60)
    print()
    
    # Step 1: Device verification
    device_info = check_sdaa_availability()
    
    if not device_info["sdaa_tensor_test_passed"]:
        print("ERROR: SDAA device not available or not working")
        evidence = {
            "benchmark": "dtmb5415_resistance",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device_verification": device_info,
            "error": "SDAA device not available",
        }
        evidence_file = _ARTIFACTS_DIR / "sdaa_dtmb5415_resistance_evidence.json"
        _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(evidence_file, "w") as f:
            json.dump(evidence, f, indent=2)
        print(f"  Evidence (failure) written to: {evidence_file}")
        return 1
    
    # Step 2: Run benchmark (quick test with 500 steps)
    run_result = run_benchmark_quick(device="sdaa:0", resolution=48, steps=500)
    
    if run_result["returncode"] != 0:
        print("ERROR: Benchmark run failed")
        generate_evidence(device_info, run_result, {}, {"error": "Run failed"})
        return 1
    
    # Step 3: Parse results
    result_data = parse_results()
    
    if result_data is None:
        print("ERROR: Could not parse results")
        generate_evidence(device_info, run_result, {}, {"error": "Parse failed"})
        return 1
    
    # Step 4: Validate criteria
    validations = validate_criteria(result_data)
    
    # Step 5: Generate evidence
    evidence_file = generate_evidence(device_info, run_result, result_data, validations)
    
    # Summary
    print("=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"  Device: SDAA available and working")
    print(f"  Benchmark: Ran successfully")
    print(f"  Results: Parsed successfully")
    print(f"  Criteria: See validation results above")
    print(f"  Evidence: {evidence_file}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())