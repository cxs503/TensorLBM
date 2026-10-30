#!/usr/bin/env python3
"""SDAA sweep for poiseuille_two_phase_cg benchmark.

This script runs the two-layer Poiseuille color-gradient benchmark on SDAA
device and records device evidence, gate values, and per-arm results.

Due to SDAA performance characteristics (~20ms/step), we use reduced step
counts to fit within wall-clock constraints. The script can be called with
--arms to run specific arms.

Usage:
    # Run all arms with reduced steps (for quick validation)
    python scripts/sdaa_sweep_poiseuille_two_phase_cg.py

    # Run specific arm
    python scripts/sdaa_sweep_poiseuille_two_phase_cg.py --arms eq_ny64 --steps-override 20000

Output:
    artifacts/sdaa_poiseuille_two_phase_cg.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

# Setup paths
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
CASE_DIR = REPO_ROOT / "benchmarks" / "verified" / "poiseuille_two_phase_cg"

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(CASE_DIR))

import torch

# Import from case directory
from cg_reference import (
    first_blue_row,
    single_phase_profile_wallplanes,
    two_layer_profile_wallplanes,
)

import tensorlbm
from tensorlbm.boundaries import bounce_back_cells
from tensorlbm.d2q9 import C, equilibrium
from tensorlbm.solver import stream


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _load_engine():
    """Load color_gradient_two_tau_step function."""
    md5s = {"cg_reference.py": _md5(CASE_DIR / "cg_reference.py")}
    try:
        from tensorlbm.color_gradient2d import color_gradient_two_tau_step
        eng = Path(sys.modules["tensorlbm.color_gradient2d"].__file__ or "")
        md5s["color_gradient2d.py"] = _md5(eng)
        return color_gradient_two_tau_step, "repo tensorlbm.color_gradient2d", md5s
    except ImportError:
        vendored = CASE_DIR / "color_gradient2d.py"
        spec = importlib.util.spec_from_file_location("tensorlbm.color_gradient2d", vendored)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules["tensorlbm.color_gradient2d"] = mod
        spec.loader.exec_module(mod)
        md5s["color_gradient2d.py"] = _md5(vendored)
        return mod.color_gradient_two_tau_step, "vendored color_gradient2d.py", md5s


# Constants
DT = torch.float64
NX = 8
A = 0.04
GATE_RELL2 = 0.03
GATE_MASS_DRIFT = 1e-10

# Frozen arms from the benchmark
FROZEN_ARMS = {
    "eq_ny64": {
        "name": "eq_ny64",
        "ny": 64,
        "tau_r": 1.0,
        "tau_b": 1.0,
        "gx": 5e-6,
        "beta": 0.9,
        "steps": 60000,
    },
    "eq_ny128": {
        "name": "eq_ny128",
        "ny": 128,
        "tau_r": 1.0,
        "tau_b": 1.0,
        "gx": 5e-6 / 4,
        "beta": 0.9,
        "steps": 200000,
    },
    "g10_ny64": {
        "name": "g10_ny64",
        "ny": 64,
        "tau_r": 2.5,
        "tau_b": 0.7,
        "gx": 1.5e-7,
        "beta": 0.9,
        "steps": 80000,
    },
    "g10_ny128": {
        "name": "g10_ny128",
        "ny": 128,
        "tau_r": 2.5,
        "tau_b": 0.7,
        "gx": 1.5e-7 / 4,
        "beta": 0.9,
        "steps": 250000,
    },
}

# Reduced steps for SDAA (feasible within wall-clock constraints)
# Based on convergence analysis: ~20ms/step on SDAA
SDAA_REDUCED_STEPS = {
    "eq_ny64": 20000,    # ~400s, should reach ~1% relL2
    "eq_ny128": 5000,    # ~100s, partial convergence
    "g10_ny64": 25000,   # ~500s, should reach ~1% relL2
    "g10_ny128": 5000,   # ~100s, partial convergence
}


def run_arm(arm: dict, step_fn, dev: torch.device) -> dict:
    """Run a single arm of the benchmark."""
    ny, gx = arm["ny"], arm["gx"]
    half = ny // 2
    j = torch.arange(ny, dtype=DT, device=dev).view(ny, 1)
    frac_r = 0.5 * (1.0 - torch.tanh((j - half) / 2.0)).expand(ny, NX).contiguous()
    z = torch.zeros((ny, NX), dtype=DT, device=dev)
    f_r = equilibrium(1.0 * frac_r, z, z)
    f_b = equilibrium(1.0 * (1.0 - frac_r), z, z)
    m_r0, m_b0 = float(f_r.sum()), float(f_b.sum())
    wall = torch.zeros((ny, NX), dtype=torch.bool, device=dev)
    wall[0, :] = True
    wall[-1, :] = True

    c9 = C.to(DT).to(dev)
    umax_hist = []
    t0 = time.perf_counter()
    for step in range(1, arm["steps"] + 1):
        f_r, f_b = step_fn(
            f_r,
            f_b,
            tau_r=arm["tau_r"],
            tau_b=arm["tau_b"],
            A=A,
            beta=arm["beta"],
            gx=gx,
            solid_mask=wall,
        )
        f_r = bounce_back_cells(stream(f_r), wall)
        f_b = bounce_back_cells(stream(f_b), wall)
        if not bool(torch.isfinite(f_r).all() and torch.isfinite(f_b).all()):
            return {**arm, "stable": False, "nan_step": step}
        if step % max(1, arm["steps"] // 20) == 0:
            f = f_r + f_b
            ux = (f * c9[:, 0].view(9, 1, 1)).sum(dim=0) / f.sum(dim=0)
            umax_hist.append((step, float(ux[:, NX // 2].max())))

    f = f_r + f_b
    rho = f.sum(dim=0)
    ux = (f * c9[:, 0].view(9, 1, 1)).sum(dim=0) / rho
    prof = ux[:, NX // 2]
    phi = (f_r.sum(0) - f_b.sum(0)) / rho
    phi_col = phi[:, NX // 2]

    if arm["tau_r"] == arm["tau_b"]:
        ana = single_phase_profile_wallplanes(ny, gx, (arm["tau_r"] - 0.5) / 3.0)
        jb = None
    else:
        try:
            jb = first_blue_row(phi_col.cpu())
            nu_lo = (arm["tau_r"] - 0.5) / 3.0
            nu_hi = (arm["tau_b"] - 0.5) / 3.0
            ana = two_layer_profile_wallplanes(ny, jb, gx, nu_lo, nu_hi, nu_lo / nu_hi)
        except RuntimeError:
            # Interface not yet formed
            jb = None
            ana = single_phase_profile_wallplanes(ny, gx, (arm["tau_r"] - 0.5) / 3.0)
    ana = ana.to(dev)
    rel = float(((prof[1:-1] - ana).norm() / ana.norm()))
    umax_ratio = float(prof.max() / ana.max())
    tail = umax_hist[-len(umax_hist) // 5 :] if len(umax_hist) >= 5 else umax_hist
    frozen = abs(tail[-1][1] - tail[0][1]) / abs(tail[-1][1]) if tail else 0.0
    return {
        **{k: arm[k] for k in ("name", "ny", "tau_r", "tau_b", "gx", "beta", "steps")},
        "stable": True,
        "j_first_blue": jb,
        "relL2": rel,
        "u_max_ratio": umax_ratio,
        "u_max_hist": umax_hist,
        "u_max_frozen_rel_tail20": frozen,
        "mass_drift_r_per_step": abs(float(f_r.sum()) - m_r0) / arm["steps"],
        "mass_drift_b_per_step": abs(float(f_b.sum()) - m_b0) / arm["steps"],
        "wall_s": time.perf_counter() - t0,
    }


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SDAA sweep for poiseuille_two_phase_cg")
    parser.add_argument("--arms", default="eq_ny64,g10_ny64",
                       help="Comma-separated arm names")
    parser.add_argument("--steps-override", type=int, default=None,
                       help="Override step count for all arms")
    parser.add_argument("--use-reduced-steps", action="store_true",
                       help="Use SDAA-friendly reduced step counts")
    args = parser.parse_args()

    print("=" * 70)
    print("SDAA Sweep: poiseuille_two_phase_cg")
    print("=" * 70)

    # Device evidence
    device_evidence = {}
    try:
        device_evidence["torch_version"] = torch.__version__
        device_evidence["sdaa_available"] = torch.sdaa.is_available()
        device_evidence["sdaa_device_count"] = torch.sdaa.device_count()
        device_evidence["sdaa_device_name"] = torch.sdaa.get_device_name(0)
        dev = torch.device("sdaa")
        print(f"Device: {device_evidence['sdaa_device_name']}")
    except Exception as e:
        print(f"ERROR: SDAA not available: {e}")
        device_evidence["error"] = str(e)
        sys.exit(1)

    # Load engine
    step_fn, origin, md5s = _load_engine()
    print(f"Engine: {origin}")
    print(f"Module MD5s: {md5s}")

    # Select arms
    arm_names = [s.strip() for s in args.arms.split(",") if s.strip()]
    arms_to_run = []
    for name in arm_names:
        if name not in FROZEN_ARMS:
            print(f"ERROR: Unknown arm {name}")
            sys.exit(1)
        arm = FROZEN_ARMS[name].copy()
        if args.steps_override is not None:
            arm["steps"] = args.steps_override
        elif args.use_reduced_steps and name in SDAA_REDUCED_STEPS:
            arm["steps"] = SDAA_REDUCED_STEPS[name]
        arms_to_run.append(arm)

    # Run arms
    arms_results = []
    total_start = time.perf_counter()
    for arm in arms_to_run:
        print(f"\nRunning arm: {arm['name']} (ny={arm['ny']}, steps={arm['steps']})")
        result = run_arm(arm, step_fn, dev)
        arms_results.append(result)
        if result["stable"]:
            print(f"  relL2 = {result['relL2']:.6f} ({result['relL2']*100:.3f}%)")
            print(f"  mass_drift_r = {result['mass_drift_r_per_step']:.3e}")
            print(f"  mass_drift_b = {result['mass_drift_b_per_step']:.3e}")
            print(f"  wall_time = {result['wall_s']:.2f}s")
        else:
            print(f"  UNSTABLE at step {result.get('nan_step')}")
    total_wall = time.perf_counter() - total_start

    # Compute gates
    stable_arms = [a for a in arms_results if a.get("relL2") is not None]
    if stable_arms:
        rel2_max = max(a["relL2"] for a in stable_arms)
        drift_max = max(
            max(a["mass_drift_r_per_step"], a["mass_drift_b_per_step"])
            for a in stable_arms
        )
    else:
        rel2_max = float("inf")
        drift_max = float("inf")
    all_stable = all(a["stable"] for a in arms_results)

    # Per-arm gate checks
    arm_gates = []
    for a in arms_results:
        arm_gate = {
            "name": a["name"],
            "relL2": a.get("relL2"),
            "relL2_pass": a.get("relL2", float("inf")) <= GATE_RELL2,
            "mass_drift_r": a.get("mass_drift_r_per_step"),
            "mass_drift_b": a.get("mass_drift_b_per_step"),
            "mass_drift_pass": (
                max(a.get("mass_drift_r_per_step", 0), a.get("mass_drift_b_per_step", 0))
                <= GATE_MASS_DRIFT
            ),
            "stable": a.get("stable", False),
        }
        arm_gates.append(arm_gate)

    # Overall verdict
    overall_pass = (
        rel2_max <= GATE_RELL2
        and drift_max <= GATE_MASS_DRIFT
        and all_stable
    )

    # Build output
    output = {
        "case": "poiseuille_two_phase_cg",
        "device_evidence": device_evidence,
        "config": {
            "nx": NX,
            "A": A,
            "dtype": "float64",
            "device": str(dev),
            "note": "Reduced steps for SDAA feasibility" if args.use_reduced_steps or args.steps_override else "Full benchmark steps",
        },
        "engine_origin": origin,
        "module_md5": md5s,
        "gates": {
            "gate_rell2": GATE_RELL2,
            "rell2_max": rel2_max,
            "gate_mass_drift": GATE_MASS_DRIFT,
            "mass_drift_max": drift_max,
            "all_stable": all_stable,
            "pass": overall_pass,
        },
        "arm_gates": arm_gates,
        "arms": arms_results,
        "total_wall_s": total_wall,
    }

    # Save artifacts
    artifacts_dir = REPO_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts_dir / "sdaa_poiseuille_two_phase_cg.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n{'=' * 70}")
    print(f"Saved: {out_path}")
    print(f"Total wall time: {total_wall:.2f}s")
    print(f"VERDICT: {'PASS' if overall_pass else 'FAIL'}")
    print(f"  relL2_max = {rel2_max:.6f} (gate: {GATE_RELL2})")
    print(f"  mass_drift_max = {drift_max:.3e} (gate: {GATE_MASS_DRIFT})")
    print(f"  all_stable = {all_stable}")
    print("=" * 70)


if __name__ == "__main__":
    main()
