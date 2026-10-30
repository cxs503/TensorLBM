#!/usr/bin/env python3
"""SDAA sweep for thermal_cavity benchmark (de Vahl Davis natural convection).

This script runs the thermal_cavity case on SDAA device and validates against
pre-registered reference values from the NVIDIA-verified benchmark.

Criteria (from benchmarks/verified/thermal_cavity/README.md):
- |Nu err| <= 3% vs DVD 1983 reference
- |u_max err| <= 3% vs DVD 1983 reference  
- |v_max err| <= 3% vs DVD 1983 reference

Reference values (pre-registered):
  Ra=1e3: Nu=1.118, u_max=3.649, v_max=3.697
  Ra=1e4: Nu=2.243, u_max=16.178, v_max=19.617

Note: This sweep uses reduced step counts for feasibility (≤20 min wall clock).
Full convergence requires 37k-563k steps per case (see result.json).
SDAA timing: ~13-18 ms/step depending on grid size.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

# Add paths
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "verified" / "thermal_cavity"))

import driver

# Reference values (from result.json)
REFERENCES = {
    1e3: {"nu": 1.118, "u_max": 3.649, "v_max": 3.697},
    1e4: {"nu": 2.243, "u_max": 16.178, "v_max": 19.617},
}

# Reduced step counts for SDAA sweep (feasible within shell timeout)
# Timing: N=64 ~13ms/step, N=128 ~13ms/step, N=256 ~18ms/step
# 1000 steps per case × 6 cases ≈ 90s total
STEP_COUNTS = {
    1e3: {64: 1000, 128: 1000, 256: 800},
    1e4: {64: 1000, 128: 1000, 256: 800},
}

CASES = [(ra, n) for ra in (1e3, 1e4) for n in (64, 128, 256)]


def tag_of(ra: float, nx: int) -> str:
    return f"ra1e{int(round(__import__('math').log10(ra)))}_n{nx}"


def run_case_sdaa(
    ra: float,
    nx: int,
    device: torch.device,
    max_steps: int,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """Run a single case on SDAA and return observables."""
    tag = tag_of(ra, nx)
    t_hot, t_cold = 1.0, 0.0
    
    # Initialize fields
    f, g, wall, p = driver.make_fields(nx, ra, 0.71, 0.6, device, dtype=dtype)
    tau_T, g_beta = p["tau_T"], p["g_beta"]
    
    # Time the run
    t0 = time.time()
    
    # Run simulation
    for step in range(1, max_steps + 1):
        f, g = driver.advance(f, g, wall, 0.6, tau_T, g_beta, t_hot, t_cold)
        
        # Sync periodically to get accurate timing
        if step % 500 == 0:
            torch.sdaa.synchronize()
    
    # Final sync
    torch.sdaa.synchronize()
    elapsed = time.time() - t0
    
    # Extract observables
    obs = driver.observables(f, g, p, t_hot, t_cold)
    
    # Compare with reference
    ref = REFERENCES[ra]
    nu_err_pct = 100.0 * (obs["nu"] - ref["nu"]) / ref["nu"]
    u_max_err_pct = 100.0 * (obs["u_max_nd"] - ref["u_max"]) / ref["u_max"]
    v_max_err_pct = 100.0 * (obs["v_max_nd"] - ref["v_max"]) / ref["v_max"]
    
    result = {
        "tag": tag,
        "ra": ra,
        "nx": nx,
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "steps_run": max_steps,
        "elapsed_s": round(elapsed, 2),
        "nu": obs["nu"],
        "nu_ref": ref["nu"],
        "nu_err_pct": nu_err_pct,
        "u_max_nd": obs["u_max_nd"],
        "u_max_ref": ref["u_max"],
        "u_max_err_pct": u_max_err_pct,
        "v_max_nd": obs["v_max_nd"],
        "v_max_ref": ref["v_max"],
        "v_max_err_pct": v_max_err_pct,
        "gate_nu_le_3pct": abs(nu_err_pct) <= 3.0,
        "gate_u_le_3pct": abs(u_max_err_pct) <= 3.0,
        "gate_v_le_3pct": abs(v_max_err_pct) <= 3.0,
    }
    result["gate_case_pass"] = (
        result["gate_nu_le_3pct"] 
        and result["gate_u_le_3pct"] 
        and result["gate_v_le_3pct"]
    )
    
    return result


def main() -> None:
    """Main entry point."""
    # Device setup
    device = torch.device("sdaa")
    device_name = torch.sdaa.get_device_name(0)
    
    print(f"[sdaa_sweep] device={device} name={device_name}")
    print(f"[sdaa_sweep] torch={torch.__version__}")
    print(f"[sdaa_sweep] cases={[tag_of(r, n) for r, n in CASES]}")
    
    # Run all cases
    results = []
    total_t0 = time.time()
    
    for ra, nx in CASES:
        tag = tag_of(ra, nx)
        steps = STEP_COUNTS[ra][nx]
        print(f"\n[sdaa_sweep] Running {tag} (steps={steps})...")
        
        result = run_case_sdaa(ra, nx, device, steps)
        results.append(result)
        
        status = "PASS" if result["gate_case_pass"] else "FAIL"
        print(
            f"[sdaa_sweep] {tag}: Nu={result['nu']:.6f} (err={result['nu_err_pct']:+.2f}%) "
            f"u_max={result['u_max_nd']:.4f} (err={result['u_max_err_pct']:+.2f}%) "
            f"v_max={result['v_max_nd']:.4f} (err={result['v_max_err_pct']:+.2f}%) "
            f"[{status}] ({result['elapsed_s']:.1f}s)"
        )
    
    total_elapsed = time.time() - total_t0
    
    # Build output JSON
    output = {
        "benchmark": "thermal_cavity",
        "description": "de Vahl Davis natural convection square cavity (D2Q9+D2Q5, BGK, Pr=0.71)",
        "device_evidence": {
            "device_type": str(device),
            "device_name": device_name,
            "device_count": torch.sdaa.device_count(),
            "torch_version": torch.__version__,
        },
        "sweep_config": {
            "step_counts": STEP_COUNTS,
            "note": "Reduced steps for SDAA sweep feasibility. Full convergence requires 37k-563k steps.",
        },
        "references": REFERENCES,
        "criteria": {
            "nu_err_le_3pct": "|Nu error| <= 3% vs DVD 1983 reference",
            "u_max_err_le_3pct": "|u_max error| <= 3% vs DVD 1983 reference",
            "v_max_err_le_3pct": "|v_max error| <= 3% vs DVD 1983 reference",
        },
        "results": results,
        "summary": {
            "total_cases": len(results),
            "passed_cases": sum(1 for r in results if r["gate_case_pass"]),
            "failed_cases": sum(1 for r in results if not r["gate_case_pass"]),
            "total_elapsed_s": round(total_elapsed, 2),
        },
        "notes": [
            "SDAA sweep uses reduced step counts (800-1000) vs original (37k-563k).",
            "Steady state not reached; results show early development trend.",
            "Original verdict: PARTIAL (N>=128 pass, N=64 fails due to coarse-grid discretization).",
            "No triton dependency detected; no TRITON_INTERPRET workaround needed.",
            "SDAA timing: N=64 ~13ms/step, N=128 ~13ms/step, N=256 ~18ms/step.",
        ],
    }
    
    # Write output
    out_path = Path(__file__).resolve().parents[1] / "artifacts" / "sdaa_thermal_cavity.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n[sdaa_sweep] Results written to {out_path}")
    print(f"[sdaa_sweep] Total elapsed: {total_elapsed:.1f}s")
    print(f"[sdaa_sweep] Summary: {output['summary']['passed_cases']}/{output['summary']['total_cases']} cases passed")


if __name__ == "__main__":
    main()
