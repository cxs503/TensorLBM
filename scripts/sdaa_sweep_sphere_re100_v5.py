#!/usr/bin/env python3
"""SDAA sweep for sphere_re100 benchmark — v5 diagnostic + repair bypass.

This script addresses the SDAA agent failures (m191/m186/m179/m41: "未完成：模型未调用 finish")
by implementing:
1. Independent driver script (no agent framework dependency)
2. Mid-run checkpoint saves (survives crashes/timeouts)
3. Feasible SDAA run parameters (D=20, 3000 steps ≈ 3.7 hours)
4. Comprehensive result JSON with SDAA performance metrics

Physical problem: Uniform flow past sphere, Re = U*D/ν = 100
Reference: Schiller-Naumann Cd = 24/Re*(1+0.15*Re^0.687) = 1.0917311
Method: D3Q19 + MRT + BFL interpolated bounce-back + far-field BC
Force: BFL per-link momentum ledger (laboratory frame)

SDAA constraints:
- Device: /dev/tcaicard1, 16 GB
- D=20 performance: ~4.4 s/step → 3000 steps ≈ 3.7 hours
- D=40 would be ~150 hours (infeasible)
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

import torch

# Add TensorLBM paths
TP_AGENT_WORK = os.environ.get("TP_AGENT_WORK", "/root/tp-agent/work/m198")
sys.path.insert(0, os.path.join(TP_AGENT_WORK, "src"))
sys.path.insert(0, os.path.join(TP_AGENT_WORK, "benchmarks"))

from tensorlbm.bfl_d3q19 import bouzidi_bounce_back_d3q19
from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d
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
from tensorlbm.interpolated_bc import compute_q_sphere
from tensorlbm.solver3d import (
    collide_mrt3d_low_memory,
    correct_mass3d,
    stream3d_roll,
)

DEV = "sdaa:0"
CHECKPOINT_INTERVAL = 500  # Save every 500 steps


def build_engine_config(D, lat, up, down):
    pad = (up, down, lat, lat, lat, lat)
    return GeneralSimConfig(
        name=f"sdaa_sphere_D{D}",
        geometry=GeometryConfig(
            source=GeometrySource.PARAMETRIC_SPHERE,
            sphere_radius=0.5,
            sphere_center=(0.0, 0.0, 0.0),
        ),
        physics=PhysicsConfig(
            density=1000.0,
            viscosity=1.0e-6,
            inlet_velocity=1.0e-4,
            reference_length=1.0,
        ),
        solver=SolverConfig(
            lattice=LatticeModel.D3Q19,
            collision=CollisionModel.MRT,
            resolution=D,
            domain_padding=pad,
            max_steps=10,
            snapshot_interval=10**9,
            force_sample_interval=50,
            device=DEV,
            wall_treatment=WallTreatment.BOUNCE_BACK,
            mass_correction=True,
            mass_correction_interval=200,
            force_method=ForceMethod.MOMENTUM_EXCHANGE,
            mem_variant="wet_node",
        ),
        output=OutputConfig(
            directory="out",
            formats=[OutputFormat.NPY],
            save_macroscopic=False,
            save_forces=True,
        ),
    )


def cd_ref_family(re: float) -> float:
    """Schiller-Naumann correlation."""
    return 24.0 / re * (1.0 + 0.15 * re**0.687)


def save_checkpoint(out_path, meta, hist, step):
    """Save intermediate result (survives crashes)."""
    checkpoint = {
        "meta": meta,
        "history": hist,
        "checkpoint_step": step,
        "timestamp": time.time(),
    }
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as fh:
        json.dump(checkpoint, fh, indent=2)
    os.replace(tmp_path, out_path)
    print(f"[checkpoint] step {step} saved to {out_path}", flush=True)


def run_sphere_sdaa(out_path, D, steps, lat, up, down, tau, sample):
    """Run sphere_re100 on SDAA with checkpoint strategy."""
    print(f"=" * 60, flush=True)
    print(f"SDAA sphere_re100 sweep: D={D}, steps={steps}", flush=True)
    print(f"Parameters: lat={lat}, up={up}, down={down}, tau={tau}", flush=True)
    print(f"=" * 60, flush=True)

    # Build engine
    cfg = build_engine_config(D, lat, up, down)
    engine = GeneralSimEngine(cfg)
    engine.setup()

    # Lattice units
    nu_lb = (tau - 0.5) / 3.0
    ulb = 100.0 * nu_lb / D  # Re = u_lb * D / nu_lb = 100
    re_eff = ulb * D / nu_lb
    R_lb = cfg.geometry.sphere_radius / (cfg.physics.reference_length / cfg.solver.resolution)
    dpS = 0.5 * ulb**2 * math.pi * R_lb**2

    print(f"Lattice: nu_lb={nu_lb:.6f}, u_lb={ulb:.6f}, Re_eff={re_eff:.2f}", flush=True)
    print(f"Domain: {engine.solid.shape}, cells={engine.solid.numel()/1e6:.2f}M", flush=True)
    print(f"R_lb={R_lb:.2f}, dpS={dpS:.6e}", flush=True)

    # BFL setup
    solid = engine.solid
    nz, ny, nx = solid.shape
    dx = cfg.physics.reference_length / cfg.solver.resolution
    cx_lb = (cfg.geometry.sphere_center[0] - engine.domain_phys[0]) / dx
    cy_lb = (cfg.geometry.sphere_center[1] - engine.domain_phys[2]) / dx
    cz_lb = (cfg.geometry.sphere_center[2] - engine.domain_phys[4]) / dx
    dev = torch.device(DEV)

    bfl_mask, bfl_q = compute_q_sphere(nx, ny, nz, cx_lb, cy_lb, cz_lb, R_lb, dev)
    n_links = int(bfl_mask[1:].sum().item())
    print(f"BFL links: {n_links}", flush=True)

    # Simulation loop
    bc_config = {"far_field_faces": ["y-", "y+", "z-", "z+"], "periodic_faces": []}
    far_field_fn = functools.partial(far_field_bc_3d, bc_config=bc_config)
    collide = lambda f: collide_mrt3d_low_memory(f, tau=tau)
    stream_fn = stream3d_roll

    f = engine.f.clone()
    engine.f = None  # memory trim
    target_mass = float(f.sum().item())
    initial_mass = target_mass
    sm = solid.unsqueeze(0).expand_as(f)

    hist = []
    finite = True
    diverged_at = None
    sample_set = set(range(sample, steps + 1, sample))

    t0 = time.time()
    step_times = []

    for step in range(1, steps + 1):
        step_t0 = time.time()

        # Collision
        f_pre = f.clone()
        f = collide(f)
        for q in range(f.shape[0]):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        del f_pre

        # Bounce-back
        f = bounce_back_cells_3d(f, solid)

        # Stream
        f_pre_stream = f.clone()
        f = stream_fn(f)

        # Far-field BC
        f = far_field_fn(f, ulb)

        # BFL bounce-back + force
        f, force3 = bouzidi_bounce_back_d3q19(
            f, f_pre_stream, bfl_mask, bfl_q, return_force=True
        )

        # Sample
        if step in sample_set:
            cd = float(force3[0].item()) / dpS
            cl = float(force3[1].item()) / dpS
            cs = float(force3[2].item()) / dpS
            entry = {
                "step": step,
                "cd": cd,
                "cl": cl,
                "cs": cs,
                "mass": float(f.sum().item()),
            }
            hist.append(entry)

        del f_pre_stream

        # Mass correction
        if step % 200 == 0:
            f = correct_mass3d(f, target_mass)

        # Divergence check
        if step % 500 == 0 and not bool(torch.isfinite(f).all()):
            finite = False
            diverged_at = step
            break

        # Step timing
        step_dt = time.time() - step_t0
        step_times.append(step_dt)

        # Progress
        if step % 100 == 0:
            el = time.time() - t0
            avg_step = el / step
            eta = avg_step * (steps - step)
            print(
                f"step {step}/{steps} ({el:.0f}s, {avg_step*1000:.0f} ms/step, ETA {eta/3600:.1f}h)",
                flush=True,
            )

        # Checkpoint
        if step % CHECKPOINT_INTERVAL == 0:
            meta = build_meta(
                D, lat, up, down, tau, ulb, nu_lb, re_eff, steps if finite else diverged_at,
                finite, nz, ny, nx, dpS, R_lb, n_links, initial_mass, f, step_times, t0
            )
            save_checkpoint(out_path, meta, hist, step)

    # Final metadata
    wall_s = time.time() - t0
    meta = build_meta(
        D, lat, up, down, tau, ulb, nu_lb, re_eff, steps if finite else diverged_at,
        finite, nz, ny, nx, dpS, R_lb, n_links, initial_mass, f, step_times, t0
    )
    meta["wall_s"] = wall_s
    meta["status"] = "completed" if finite else "diverged"

    # Final save
    result = {"meta": meta, "history": hist}
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\n[final] written {out_path}", flush=True)

    # Summary
    if hist:
        tail = [h["cd"] for h in hist[-8:]]
        cd_mean = sum(tail) / len(tail)
        cd_ref = cd_ref_family(re_eff)
        err_pct = (cd_mean - cd_ref) / cd_ref * 100
        print(f"\n[summary] last-8 Cd mean: {cd_mean:.6f}", flush=True)
        print(f"[summary] Cd_ref (Schiller-Naumann): {cd_ref:.6f}", flush=True)
        print(f"[summary] Error: {err_pct:+.3f}%", flush=True)
        print(f"[summary] Wall time: {wall_s:.1f}s ({wall_s/3600:.2f}h)", flush=True)
        print(f"[summary] Avg step: {sum(step_times)/len(step_times)*1000:.0f} ms", flush=True)

    return result


def build_meta(D, lat, up, down, tau, ulb, nu_lb, re_eff, steps, finite, nz, ny, nx, dpS, R_lb, n_links, initial_mass, f, step_times, t0):
    """Build metadata dict."""
    return {
        "device": DEV,
        "kernel": "dense",
        "route": "bfl",
        "collision": "mrt",
        "D": D,
        "lat": lat,
        "up": up,
        "down": down,
        "tau": tau,
        "u_lb": ulb,
        "nu_lb": nu_lb,
        "re_eff": re_eff,
        "steps": steps,
        "diverged": not finite,
        "domain_lu": [nz, ny, nx],
        "dpS": dpS,
        "R_lb": R_lb,
        "blockage": math.pi * R_lb**2 / (ny * nz),
        "cd_ref_at_re_eff": cd_ref_family(re_eff),
        "mass_drift_ppm": (float(f.sum().item()) - initial_mass) / initial_mass * 1e6 if finite else None,
        "bfl_links": n_links,
        "stream": "roll",
        "avg_step_ms": sum(step_times) / len(step_times) * 1000 if step_times else None,
        "elapsed_s": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser(description="SDAA sphere_re100 sweep v5")
    ap.add_argument("--out", default="/tmp/sdaa_sphere_re100_v5.json", help="Output JSON path")
    ap.add_argument("--D", type=int, default=20, help="Resolution (D=20 feasible on SDAA)")
    ap.add_argument("--steps", type=int, default=3000, help="Number of steps (3000 ≈ 3.7h on SDAA)")
    ap.add_argument("--lat", type=float, default=3.0, help="Lateral domain (D units)")
    ap.add_argument("--up", type=float, default=3.0, help="Upstream domain (D units)")
    ap.add_argument("--down", type=float, default=4.0, help="Downstream domain (D units)")
    ap.add_argument("--tau", type=float, default=0.54, help="Relaxation time")
    ap.add_argument("--sample", type=int, default=50, help="Sample interval")
    args = ap.parse_args()

    result = run_sphere_sdaa(
        args.out, args.D, args.steps, args.lat, args.up, args.down, args.tau, args.sample
    )

    # Exit code
    if result["meta"].get("diverged"):
        print("[FAIL] Simulation diverged", flush=True)
        sys.exit(1)
    else:
        print("[OK] Simulation completed", flush=True)
        sys.exit(0)


if __name__ == "__main__":
    main()
