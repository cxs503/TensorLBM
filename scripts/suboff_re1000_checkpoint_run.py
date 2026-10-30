#!/usr/bin/env python3
"""Checkpoint-enabled runner for suboff_re1000 benchmark (fixed).

Runs the GeneralSimEngine step-by-step with periodic checkpoint saves.
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "benchmarks"))

import numpy as np
import torch
from compile_route import route_step, compile_status_of, compile_mode_from_args, add_compile_mode_arg
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.drag_pressure import (
    drag_friction_integration,
    drag_pressure_integration,
    suboff_smooth_q,
)
from tensorlbm.general_sim import (
    CollisionModel, ForceMethod, GeneralSimConfig, GeneralSimEngine,
    GeometryConfig, GeometrySource, LatticeModel, OutputConfig,
    PhysicsConfig, SolverConfig, WallTreatment,
)
from tensorlbm.lbm_step_correct import lbm_step_correct

SUBOFF_LENGTH_M = 4.356
SUBOFF_RADIUS_M = 0.254
U_PHYS = 1.0e-3


def wetted_dpS(u_lb, R_lb, L_lb):
    return 0.5 * u_lb**2 * math.pi * (2.0 * R_lb) * L_lb

def frontal_dpS(u_lb, R_lb):
    return 0.5 * u_lb**2 * math.pi * R_lb**2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=48)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--device", default="sdaa:0")
    ap.add_argument("--collision", default="mrt")
    ap.add_argument("--friction", default="mix50")
    ap.add_argument("--p0", default="near_wall")
    ap.add_argument("--checkpoint-interval", type=int, default=500)
    ap.add_argument("--wall-budget", type=float, default=900)
    ap.add_argument("--out", default=None)
    add_compile_mode_arg(ap)
    args = ap.parse_args()
    compile_mode = compile_mode_from_args(args)

    L = args.resolution
    collision = CollisionModel.MRT
    viscosity = U_PHYS * SUBOFF_LENGTH_M / 1000.0

    out_dir = Path(args.out or f"runs/suboff_re1000_ckpt_L{L}")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = GeneralSimConfig(
        name=f"suboff_re1000_ckpt_L{L}",
        geometry=GeometryConfig(
            source=GeometrySource.PARAMETRIC_SUBOFF,
            suboff_length=SUBOFF_LENGTH_M,
            suboff_radius=SUBOFF_RADIUS_M,
        ),
        physics=PhysicsConfig(
            density=1000.0, viscosity=viscosity,
            inlet_velocity=U_PHYS, reference_length=SUBOFF_LENGTH_M,
        ),
        solver=SolverConfig(
            lattice=LatticeModel.D3Q19, collision=collision,
            resolution=L,
            domain_padding=(1.0, 4.0, 1.0, 1.0, 1.0, 1.0),
            max_steps=args.steps,
            warmup_steps=None,
            snapshot_interval=10_000_000,
            force_sample_interval=10,
            device=args.device,
            wall_treatment=WallTreatment.AUTO,
            force_method=ForceMethod.PRESSURE_FRICTION,
            pressure_extrap="none",
            p0_method=args.p0,
            friction_formula=args.friction,
            mass_correction=True,
            mass_correction_interval=200,
            smagorinsky_cs=0.05,
        ),
        output=OutputConfig(
            directory=str(out_dir), formats=[],
            save_macroscopic=False, save_forces=True,
        ),
    )

    print(f"=== SUBOFF Re=1000 L={L} steps={args.steps} device={args.device} "
          f"ckpt_every={args.checkpoint_interval} budget={args.wall_budget}s ===", flush=True)

    engine = GeneralSimEngine(config)
    setup_info = engine.setup()
    print(f"setup: Re={setup_info['Re']} tau={setup_info['tau']} "
          f"u_lb={setup_info['u_lb']} nu_lb={setup_info['nu_lb']} "
          f"domain={setup_info['domain_lu']} "
          f"solid={setup_info['obstacle_cells']} "
          f"near_wall={setup_info['near_wall_cells']} "
          f"total={setup_info['total_cells']}", flush=True)

    tau = engine.uc.tau
    nu_lb = engine.uc.nu_lb
    u_in = engine.uc.u_lb
    collide_fn, collide_kwargs = engine._get_collide_fn()
    far_field_fn = functools.partial(far_field_bc_3d, bc_config=engine._build_bc_config())
    solid = engine.solid if engine.solid is not None else torch.zeros_like(engine.f[0], dtype=torch.bool)

    try:
        from tensorlbm.solver3d import correct_mass3d
        correct_mass_fn = correct_mass3d
        target_mass = engine._initial_mass
    except ImportError:
        correct_mass_fn = None
        target_mass = None

    dpS = engine._compute_dpS()  # frontal area dpS
    sol = engine.config.solver

    # Wetted-area rescale factor
    R_lb = SUBOFF_RADIUS_M / (SUBOFF_LENGTH_M / L)
    dpS_wet = wetted_dpS(u_in, R_lb, float(L))
    dpS_front = frontal_dpS(u_in, R_lb)
    rescale = dpS_front / dpS_wet  # frontal->wetted

    Cf_ref = 1.328 / math.sqrt(1000.0)

    def _step(f):
        return lbm_step_correct(f, collide_fn, tau, solid, u_in, far_field_fn, **collide_kwargs)

    step_fn = route_step(_step, compile_mode, name=f"suboff_ckpt[L{L}]")

    t0 = time.time()
    checkpoints = []

    for step in range(1, args.steps + 1):
        engine.f = step_fn(engine.f)
        engine.step_count += 1

        if correct_mass_fn is not None and target_mass is not None and step % sol.mass_correction_interval == 0:
            engine.f = correct_mass_fn(engine.f, target_mass)

        if step % sol.force_sample_interval == 0:
            engine._sample_forces(dpS, nu_lb)

        # Checkpoint
        if step % args.checkpoint_interval == 0:
            elapsed = time.time() - t0
            log = engine.forces_log
            ckpt = {
                "step": step,
                "elapsed_s": round(elapsed, 1),
                "ms_per_step": round(elapsed / step * 1000, 1),
                "force_samples": len(log),
            }
            if len(log) >= 10:
                n_win = min(100, len(log))
                win = log[-n_win:]
                cd_p_wet = float(np.mean([e["cd_pressure"] for e in win])) * rescale
                cd_f_wet = float(np.mean([e["cd_friction"] for e in win])) * rescale
                cd_tot_wet = cd_p_wet + cd_f_wet
                ckpt["Cd_p"] = round(cd_p_wet, 8)
                ckpt["Cd_f"] = round(cd_f_wet, 8)
                ckpt["Cd_tot"] = round(cd_tot_wet, 8)
                ckpt["err_pct"] = round((cd_tot_wet - Cf_ref) / Cf_ref * 100, 4)
            checkpoints.append(ckpt)
            print(f"[ckpt] step={step} elapsed={elapsed:.1f}s "
                  f"ms/step={elapsed/step*1000:.1f} "
                  f"samples={len(log)} "
                  f"Cd_tot={ckpt.get('Cd_tot', 'N/A')} "
                  f"err={ckpt.get('err_pct', 'N/A')}%", flush=True)

            ckpt_path = out_dir / f"checkpoint_{step}.json"
            with open(ckpt_path, 'w') as f:
                json.dump(ckpt, f, indent=2)

        # Wall clock budget check
        elapsed = time.time() - t0
        if elapsed > args.wall_budget:
            print(f"[BUDGET] wall budget {args.wall_budget}s exceeded at step {step} "
                  f"(elapsed {elapsed:.1f}s) — stopping with checkpoints", flush=True)
            break

        # Divergence guard
        if step % 100 == 0 and not torch.isfinite(engine.f).all():
            print(f"[DIVERGED] NaN/Inf at step {step}", flush=True)
            break

    elapsed = time.time() - t0
    final_step = engine.step_count
    compile_status = compile_status_of(step_fn)

    log = engine.forces_log
    result = {
        "case": "suboff_re1000",
        "runner": "scripts/suboff_re1000_checkpoint_run.py",
        "config": {
            "L": L, "steps_requested": args.steps, "steps_completed": final_step,
            "device": args.device, "collision": args.collision,
            "friction": args.friction, "compile_mode": str(compile_mode),
            "wall_budget_s": args.wall_budget,
        },
        "setup": {
            "Re": setup_info["Re"], "tau": setup_info["tau"],
            "u_lb": setup_info["u_lb"], "nu_lb": setup_info["nu_lb"],
            "domain_lu": setup_info["domain_lu"],
            "solid_cells": setup_info["obstacle_cells"],
            "near_wall_cells": setup_info["near_wall_cells"],
            "total_cells": setup_info["total_cells"],
        },
        "reference": {"Cf_ref": Cf_ref, "name": "Blasius 1.328/sqrt(Re)"},
        "normalization": {
            "dpS_frontal": dpS_front,
            "dpS_wetted": dpS_wet,
            "rescale_frontal_to_wetted": rescale,
        },
        "timing": {
            "elapsed_s": round(elapsed, 1),
            "ms_per_step": round(elapsed / max(final_step, 1) * 1000, 1),
            "steps_completed": final_step,
        },
        "checkpoints": checkpoints,
        "compile_status": compile_status,
        "force_samples": len(log),
        "finite": torch.isfinite(engine.f).all().item(),
    }

    # Window-averaged Cd_tot (wetted area)
    if len(log) >= 10:
        n_win = min(500, len(log))
        win = log[-n_win:]
        cd_p_wet = float(np.mean([e["cd_pressure"] for e in win])) * rescale
        cd_f_wet = float(np.mean([e["cd_friction"] for e in win])) * rescale
        cd_tot_wet = cd_p_wet + cd_f_wet
        result["Cd_p"] = cd_p_wet
        result["Cd_f"] = cd_f_wet
        result["Cd_tot"] = cd_tot_wet
        result["err_pct"] = (cd_tot_wet - Cf_ref) / Cf_ref * 100
        result["pass"] = abs(result["err_pct"]) < 3.0
        result["window_samples"] = n_win

    # Convergence windows (25/50/75/100%)
    if len(log) >= 40:
        windows = {}
        for frac in [0.25, 0.50, 0.75, 1.0]:
            end = int(len(log) * frac)
            if end < 10:
                continue
            sub = log[max(0, end-100):end]
            cd_w = float(np.mean([e["cd_total"] for e in sub])) * rescale
            windows[f"{int(frac*100)}%"] = round(cd_w, 6)
        result["convergence_windows"] = windows

    result_path = out_dir / "result.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    # Save forces log
    if log:
        forces_path = out_dir / "forces.csv"
        with open(forces_path, 'w') as fh:
            keys = list(log[0].keys())
            fh.write(",".join(keys) + "\n")
            for entry in log:
                fh.write(",".join(str(entry.get(k, "")) for k in keys) + "\n")

    print(f"\n=== RESULT ===", flush=True)
    print(json.dumps(result, indent=2, default=str), flush=True)
    print(f"\nResult saved to {result_path}", flush=True)


if __name__ == "__main__":
    main()
