#!/usr/bin/env python3
"""SDAA sweep for permeability benchmark - FINAL VERSION.

This script runs the permeability benchmark on SDAA device with two test cases
to verify the simulation produces results consistent with CUDA within wall time.

Test cases:
- phi=0.3, d=26: expected err ~6.53% (CUDA)
- phi=0.5, d=20: expected err ~73.87% (CUDA)

Note: These are coarse-grid cases for quick verification. The full ladder
(phi=0.3: [26,52,104], phi=0.5: [20,40,80,160,320]) requires ~40 minutes
on CUDA and would take similar time on SDAA.
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import numpy as np

from tensorlbm.boundaries import bounce_back_cells
from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.solver import collide_bgk, stream
from tensorlbm.turbulent_channel import _apply_body_force_2d

TAU = 1.0
NU = (TAU - 0.5) / 3.0
RE_TARGET = 0.05

SANGANI_TABLE = [
    (0.05, 15.56), (0.10, 24.83), (0.20, 51.53), (0.30, 102.90),
    (0.40, 217.89), (0.50, 532.55), (0.60, 1.763e3), (0.70, 1.352e4),
    (0.75, 1.263e5),
]
F_TABLE = {phi: f for phi, f in SANGANI_TABLE}

# Quick verification cases
TEST_CASES = [(0.3, 26), (0.5, 20)]
STEPS_PER_CASE = 10000


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cylinder_array_mask(d: int, phi: float, device: torch.device) -> torch.Tensor:
    r_real = d * math.sqrt(phi / math.pi)
    idx = np.arange(d, dtype=np.float64)
    dx = np.abs(idx - d / 2.0)
    dx = np.minimum(dx, d - dx)
    dy = dx.copy()
    dist2 = dx[None, :] ** 2 + dy[:, None] ** 2
    mask = dist2 <= r_real * r_real
    return torch.as_tensor(mask, dtype=torch.bool, device=device)


def run_case(phi: float, d: int, device: torch.device, steps: int) -> dict:
    f_table = F_TABLE[phi]
    r_real = d * math.sqrt(phi / math.pi)
    k_lu = d * d / f_table
    a_body = RE_TARGET * NU * NU / (k_lu * r_real) * 10.0

    solid = cylinder_array_mask(d, phi, device)
    fluid = ~solid
    n_solid = int(solid.sum().item())
    phi_actual = n_solid / (d * d)

    rho0 = torch.ones(d, d, dtype=torch.float32, device=device)
    u0 = torch.zeros_like(rho0)
    f = equilibrium(rho0, u0, u0, device=device)
    del rho0, u0

    solid_b = solid.unsqueeze(0)

    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        f = collide_bgk(f, TAU)
        f = stream(f)
        f = torch.where(solid_b, f, _apply_body_force_2d(f, a_body))
        f = bounce_back_cells(f, solid)

    torch.sdaa.synchronize()
    wall = time.perf_counter() - t0

    _, ux, uy = macroscopic(f)
    uxf = float(ux[fluid].mean().item())
    umax = float(ux[fluid].abs().max().item())

    k_sim = NU * uxf / a_body
    k_ref = d * d / ((1.0 - phi) * f_table)
    err_pct = abs(k_sim / k_ref - 1.0) * 100.0

    u_d_meas = uxf * (1.0 - phi_actual)
    re_achieved = u_d_meas * r_real / NU
    ma = math.sqrt(3.0) * umax

    return {
        "phi": phi, "d": d, "steps": steps,
        "phi_actual": phi_actual, "k_sim": k_sim, "k_ref": k_ref,
        "err_pct": err_pct, "Re_cell": re_achieved, "Ma": ma,
        "wall_seconds": wall, "device": str(device),
    }


def main():
    print("=" * 70)
    print("SDAA Sweep: Permeability Benchmark")
    print("=" * 70)

    device = torch.device("sdaa:0")
    device_name = torch.sdaa.get_device_name(0)
    print(f"\nDevice: {device}")
    print(f"Device name: {device_name}")
    print(f"Torch version: {torch.__version__}")

    all_cases = []
    for phi, d in TEST_CASES:
        print(f"\nRunning phi={phi}, d={d}, steps={STEPS_PER_CASE}...")
        case_result = run_case(phi, d, device, STEPS_PER_CASE)
        all_cases.append(case_result)
        print(f"  err={case_result['err_pct']:.4f}%, wall={case_result['wall_seconds']:.1f}s")

    print(f"\n{'='*70}")
    print("Results vs CUDA Reference")
    print(f"{'='*70}")

    cuda_reference = {(0.3, 26): 6.53, (0.5, 20): 73.87}
    gates = []
    
    for case in all_cases:
        phi, d = case["phi"], case["d"]
        err_sdaa = case["err_pct"]
        err_cuda = cuda_reference.get((phi, d))
        
        if err_cuda is not None:
            deviation = abs(err_sdaa - err_cuda)
            # Pass if within 1% of CUDA result
            gate_pass = deviation < 1.0
            print(f"phi={phi}, d={d}:")
            print(f"  SDAA err: {err_sdaa:.4f}%")
            print(f"  CUDA err: {err_cuda:.4f}%")
            print(f"  Deviation: {deviation:.4f}%")
            print(f"  PASS (<1% deviation): {gate_pass}")
        else:
            gate_pass = True
            print(f"phi={phi}, d={d}: No CUDA reference, SKIP")
        
        gates.append({
            "phi": phi, "d": d,
            "err_sdaa_pct": err_sdaa,
            "err_cuda_pct": err_cuda,
            "deviation_pct": deviation if err_cuda else None,
            "gate_pass": gate_pass,
        })

    overall_pass = all(g["gate_pass"] for g in gates)
    print(f"\n{'='*70}")
    print(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    print(f"{'='*70}")

    output = {
        "benchmark": "permeability_square_array_2d",
        "created_utc": utc_now(),
        "device_evidence": {
            "device": str(device),
            "device_name": device_name,
            "torch_version": torch.__version__,
        },
        "settings": {
            "tau": TAU, "nu": NU, "Re_target": RE_TARGET,
            "force_scale": 10.0, "steps_per_case": STEPS_PER_CASE,
            "test_cases": TEST_CASES,
        },
        "cases": all_cases,
        "gates": gates,
        "overall_pass": overall_pass,
        "notes": {
            "quick_verification": "Two coarse-grid cases for quick verification",
            "criterion": "SDAA results within 1% of CUDA reference",
            "triton_bypass": "No triton dependency in core kernels; TRITON_INTERPRET not needed",
            "compile_mode": "eager (SDAA teco_inductor limitations prevent torch.compile)",
            "full_ladder_note": "Full ladder requires ~40 min; this sweep uses reduced cases for wall time",
        },
    }

    artifacts_dir = REPO_ROOT / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    output_path = artifacts_dir / "sdaa_permeability.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nResults written to: {output_path}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
