#!/usr/bin/env python3
"""M-S1·P2 verification script: sphere_re100 D40 on SDAA vs CPU baseline.

Criteria:
  ① Cd vs CPU baseline ≤ 1%
  ② Cd vs reference (Schiller-Naumann) ≤ 3%
  ③ Cross-card bit-exact sha256 (honest-FAIL: only 1 SDAA device available)

Outputs: artifacts/m_s1.json
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tensorlbm.general_sim import (
    CollisionModel,
    ForceMethod,
    GeneralSimConfig,
    GeneralSimEngine,
    GeometryConfig,
    GeometrySource,
    LatticeModel,
    OutputConfig,
    OutputFormat,
    PhysicsConfig,
    SolverConfig,
    WallTreatment,
)


def schiller_naumann_cd(re: float) -> float:
    return 24.0 / re * (1.0 + 0.15 * re**0.687)


def clift_gauvin_cd(re: float) -> float:
    return 24.0 / re * (1.0 + 0.1315 * re ** (0.82 - 0.05 * math.log10(re)))


def sha256_diagnostics(forces_log: list) -> str:
    data = json.dumps(forces_log, sort_keys=True).encode()
    return hashlib.sha256(data).hexdigest()


def run_sphere_re100(device, n_steps, collision="BGK",
                     domain_padding=(2.0, 2.0, 2.0, 2.0, 2.0, 3.0),
                     force_sample_interval=10):
    collision_map = {
        "BGK": CollisionModel.BGK,
        "MRT": CollisionModel.MRT,
        "AUTO": CollisionModel.AUTO,
    }
    config = GeneralSimConfig(
        name=f"sphere_re100_d40_{device.replace(':', '')}",
        geometry=GeometryConfig(
            source=GeometrySource.PARAMETRIC_SPHERE,
            sphere_radius=0.5,
            sphere_center=(0.0, 0.0, 0.0),
        ),
        physics=PhysicsConfig(
            density=1000.0, viscosity=1.0e-6,
            inlet_velocity=1.0e-4, reference_length=1.0,
        ),
        solver=SolverConfig(
            lattice=LatticeModel.D3Q19,
            collision=collision_map.get(collision, CollisionModel.BGK),
            resolution=40,
            domain_padding=domain_padding,
            max_steps=n_steps,
            warmup_steps=None,
            snapshot_interval=50000,
            force_sample_interval=force_sample_interval,
            device=device,
            wall_treatment=WallTreatment.AUTO,
            force_method=ForceMethod.PRESSURE_FRICTION,
            pressure_extrap="none",
            p0_method="near_wall",
            friction_formula="standard",
            mass_correction=True,
            mass_correction_interval=200,
        ),
        output=OutputConfig(
            directory=f"/tmp/sphere_re100_{device.replace(':', '')}",
            formats=[], save_macroscopic=False, save_forces=True,
        ),
    )
    engine = GeneralSimEngine(config)
    t0 = time.time()
    setup_info = engine.setup()
    t_setup = time.time() - t0
    t0 = time.time()
    run_info = engine.run(steps=n_steps)
    t_run = time.time() - t0
    res = engine.results()
    cd = res["Cd_Cl"]
    forces = engine.forces_log
    if forces:
        n_window = max(1, len(forces) // 5)
        window = forces[-n_window:]
        valid = [e["cd_total"] for e in window if e.get("cd_total") is not None]
        cd_window = sum(valid) / max(1, len(valid)) if valid else None
    else:
        cd_window = None
        n_window = 0
    return {
        "device": device, "collision": collision,
        "domain_lu": list(setup_info["domain_lu"]),
        "total_cells": setup_info["total_cells"],
        "tau": setup_info["tau"], "u_lb": setup_info["u_lb"],
        "Re": setup_info["Re"],
        "steps_completed": run_info["steps"],
        "diverged": run_info["diverged"],
        "t_setup_s": round(t_setup, 2),
        "t_run_s": round(t_run, 2),
        "ms_per_step": round(t_run / max(run_info["steps"], 1) * 1000, 1),
        "Cd_pressure": cd.get("Cd_pressure"),
        "Cd_friction": cd.get("Cd_friction"),
        "Cd_total": cd.get("Cd_total"),
        "Cd_window_mean": cd_window,
        "n_window_samples": n_window,
        "forces_log": forces,
        "diag_sha256": sha256_diagnostics(forces),
    }


def main():
    print("=" * 70)
    print("M-S1·P2: sphere_re100 D40 SDAA verification")
    print("=" * 70)

    sdaa_available = hasattr(torch, "sdaa") and torch.sdaa.is_available()
    sdaa_count = torch.sdaa.device_count() if sdaa_available else 0
    print(f"SDAA: available={sdaa_available}, count={sdaa_count}")
    if sdaa_available:
        print(f"  Device 0: {torch.sdaa.get_device_name(0)}")
        props = torch.sdaa.get_device_properties(0)
        print(f"  Memory: {props.total_memory}")

    cd_ref_sn = schiller_naumann_cd(100.0)
    cd_ref_cg = clift_gauvin_cd(100.0)
    print(f"\nReference Cd @ Re=100:")
    print(f"  Schiller-Naumann: {cd_ref_sn:.6f}")
    print(f"  Clift-Gauvin:     {cd_ref_cg:.6f}")

    # CPU baseline from verified archive
    cpu_baseline_cd = 1.1219804015117896
    print(f"\nCPU baseline (verified archive, D40 big domain, MRT+BFL, 12000 steps):")
    print(f"  Cd = {cpu_baseline_cd:.6f} (err vs SN: 2.77%)")

    # SDAA run - 100 steps fits in ~40s at 366ms/step
    n_steps = 100
    print(f"\n[SDAA Run] D40, BGK, {n_steps} steps on sdaa:0 ...")
    sdaa_result = run_sphere_re100(
        device="sdaa:0", n_steps=n_steps, collision="BGK",
        domain_padding=(2.0, 2.0, 2.0, 2.0, 2.0, 3.0),
        force_sample_interval=5,
    )
    print(f"  Setup: {sdaa_result['t_setup_s']:.1f}s")
    print(f"  Run: {sdaa_result['t_run_s']:.1f}s ({sdaa_result['ms_per_step']:.1f} ms/step)")
    print(f"  Steps: {sdaa_result['steps_completed']}, diverged: {sdaa_result['diverged']}")
    print(f"  Domain: {sdaa_result['domain_lu']}, cells: {sdaa_result['total_cells']}")
    print(f"  tau={sdaa_result['tau']}, u_lb={sdaa_result['u_lb']}")
    print(f"  Cd_total (final): {sdaa_result['Cd_total']}")
    print(f"  Cd_window_mean: {sdaa_result['Cd_window_mean']}")
    print(f"  Diag sha256: {sdaa_result['diag_sha256'][:16]}...")
    if sdaa_result['forces_log']:
        print(f"  Force samples: {len(sdaa_result['forces_log'])}")
        for e in sdaa_result['forces_log'][-5:]:
            print(f"    step={e['step']}, cd={e['cd_total']:.6f}")

    # Criteria evaluation
    cd_sdaa = sdaa_result["Cd_window_mean"]

    # ① Cd vs CPU baseline ≤ 1%
    if cd_sdaa is not None:
        err_vs_cpu = abs(cd_sdaa - cpu_baseline_cd) / cpu_baseline_cd * 100
        c1_pass = err_vs_cpu <= 1.0
    else:
        err_vs_cpu = None
        c1_pass = False

    # ② Cd vs reference ≤ 3%
    if cd_sdaa is not None:
        err_vs_ref = abs(cd_sdaa - cd_ref_sn) / cd_ref_sn * 100
        c2_pass = err_vs_ref <= 3.0
    else:
        err_vs_ref = None
        c2_pass = False

    # ③ Cross-card bit-exact
    if sdaa_count >= 2:
        c3_pass = False
        c3_note = "multi_card_not_tested"
    else:
        c3_pass = False
        c3_note = f"HONEST-FAIL: only {sdaa_count} SDAA device(s)"

    print(f"\n{'='*50}")
    if err_vs_cpu is not None:
        print(f"Criterion ① Cd vs CPU ≤1%: {'PASS' if c1_pass else 'FAIL'} (err={err_vs_cpu:.2f}%)")
    else:
        print(f"Criterion ①: FAIL (no data)")
    if err_vs_ref is not None:
        print(f"Criterion ② Cd vs ref ≤3%: {'PASS' if c2_pass else 'FAIL'} (err={err_vs_ref:.2f}%)")
    else:
        print(f"Criterion ②: FAIL (no data)")
    print(f"Criterion ③ Cross-card:    FAIL ({c3_note})")

    overall = c1_pass and c2_pass and c3_pass
    print(f"Overall: {'PASS' if overall else 'FAIL'}")

    # Build artifact
    artifact = {
        "mission": "M-S1·P2",
        "case": "sphere_re100_D40",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reference": {
            "cd_ref_sn": cd_ref_sn,
            "cd_ref_cg": cd_ref_cg,
            "source_primary": "Schiller-Naumann 24/Re*(1+0.15*Re^0.687)",
            "source_secondary": "Clift-Gauvin",
            "Re": 100.0,
        },
        "cpu_baseline": {
            "source": "benchmarks/verified/sphere_re100/result.json",
            "grid": "D40", "domain_lu": [280, 280, 320],
            "collision": "MRT", "force_method": "BFL_momentum_ledger",
            "steps": 12000, "Cd_total": cpu_baseline_cd,
            "err_pct_vs_SN": 2.77,
        },
        "sdaa_run": {k: v for k, v in sdaa_result.items() if k != "forces_log"},
        "criteria": {
            "c1_cd_vs_cpu_1pct": {
                "pass": c1_pass, "cpu_cd": cpu_baseline_cd,
                "sdaa_cd": cd_sdaa, "err_pct": err_vs_cpu, "threshold_pct": 1.0,
            },
            "c2_cd_vs_ref_3pct": {
                "pass": c2_pass, "ref_cd": cd_ref_sn,
                "sdaa_cd": cd_sdaa, "err_pct": err_vs_ref, "threshold_pct": 3.0,
            },
            "c3_cross_card_bitexact": {
                "pass": c3_pass, "note": c3_note,
                "sdaa_count": sdaa_count,
                "sdaa0_sha256": sdaa_result["diag_sha256"],
            },
        },
        "overall_pass": overall,
        "device_evidence": {
            "sdaa_available": sdaa_available,
            "sdaa_device_count": sdaa_count,
            "sdaa_device_name": torch.sdaa.get_device_name(0) if sdaa_available else None,
            "sdaa_total_memory": str(torch.sdaa.get_device_properties(0).total_memory) if sdaa_available else None,
            "torch_version": torch.__version__,
        },
    }

    artifact_path = Path(__file__).resolve().parent.parent / "artifacts" / "m_s1.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with open(artifact_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nArtifact: {artifact_path}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
