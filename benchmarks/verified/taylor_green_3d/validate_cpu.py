#!/usr/bin/env python3
"""CPU fallback validation for taylor_green_3d benchmark.

This script provides a quick validation path for hosts without CUDA or when
torch.compile/triton backend is unavailable (e.g., SDAA/Hygon with incomplete
teco_inductor, or CPU-only environments).

Usage:
    python validate_cpu.py  # Runs N=64, 200 steps, eager mode on CPU

The validation checks:
    - Code executes without errors on CPU
    - Energy decay rate error < 3% (relaxed from full benchmark due to fewer steps)
    - R² ≥ 0.999 (exponential decay confirmed)

This is a diagnostic tool, not a replacement for the full GPU benchmark.
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def validate_cpu() -> dict:
    """Run a quick CPU validation and return results."""
    cmd = [
        sys.executable,
        str(HERE / "run.py"),
        "--n", "64",
        "--re", "24",
        "--steps", "200",
        "--compile-mode", "eager",
        "--device", "cpu",
        "--threads", "4",
        "--out", "/tmp/tgv_cpu_validation",
    ]
    
    print(f"Running CPU validation: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"CPU validation failed with exit code {result.returncode}")
    
    # Load the result
    case_file = Path("/tmp/tgv_cpu_validation/case_N64.json")
    if not case_file.exists():
        raise RuntimeError(f"Expected output file not found: {case_file}")
    
    with open(case_file) as f:
        case = json.load(f)
    
    # Validation checks
    checks = {
        "code_runs": True,
        "err_e_within_3pct": abs(case["err_e_pct"]) <= 3.0,
        "r2_above_0999": case["r2"] >= 0.999,
        "mass_drift_small": abs(case["mass_drift_rel"]) < 1e-3,
    }
    
    validation = {
        "device": "cpu",
        "compile_mode": "eager",
        "n": case["n"],
        "steps": case["steps"],
        "err_e_pct": case["err_e_pct"],
        "r2": case["r2"],
        "mass_drift_rel": case["mass_drift_rel"],
        "wall_sec": case["wall_sec"],
        "checks": checks,
        "passed": all(checks.values()),
    }
    
    return validation


def main():
    """Run validation and print results."""
    print("=" * 70)
    print("Taylor-Green 3D CPU Fallback Validation")
    print("=" * 70)
    
    try:
        result = validate_cpu()
        
        print("\nValidation Results:")
        print(f"  Device: {result['device']}")
        print(f"  Compile mode: {result['compile_mode']}")
        print(f"  Grid: N={result['n']}³")
        print(f"  Steps: {result['steps']}")
        print(f"  Energy decay error: {result['err_e_pct']:+.4f}%")
        print(f"  R²: {result['r2']:.6f}")
        print(f"  Mass drift: {result['mass_drift_rel']:.2e}")
        print(f"  Wall time: {result['wall_sec']:.1f}s")
        
        print("\nChecks:")
        for check, passed in result["checks"].items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check}")
        
        print("\n" + "=" * 70)
        if result["passed"]:
            print("CPU VALIDATION: PASS")
            print("The benchmark code runs correctly on CPU with eager mode.")
            print("This provides a fallback path when CUDA/triton is unavailable.")
        else:
            print("CPU VALIDATION: FAIL")
            print("Some checks did not pass. See details above.")
        print("=" * 70)
        
        # Save validation result
        out_file = HERE / "cpu_validation.json"
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nValidation result saved to: {out_file}")
        
        return 0 if result["passed"] else 1
        
    except Exception as e:
        print(f"\nCPU VALIDATION: ERROR")
        print(f"Exception: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
