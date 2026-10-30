#!/usr/bin/env python3
"""Checkpoint/restart wrapper for backward_step benchmark.

This script runs the simulation in segments, saving checkpoints between segments
to allow accumulation of steps beyond the timeout limit.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmarks.pending.backward_step.run import (
    REF_X1_H,
    build_solid,
    inlet_profile,
    measure_profile,
    _rows_near_walls,
    _x1_from_shear,
    _x2x3_from_shear,
)
from tensorlbm.boundaries import make_sponge_strength, zou_he_inlet_velocity, zou_he_outlet_pressure
from tensorlbm.d2q9 import OPPOSITE, equilibrium, macroscopic
from tensorlbm.solver import collide_mrt, stream


def run_with_checkpoint(
    m: int,
    re: float,
    u_in: float,
    total_steps: int,
    segment_steps: int,
    checkpoint_dir: Path,
    device: torch.device,
    L_over_H: float = 20.0,
    x_step: int = 8,
):
    """Run simulation with checkpoint/restart."""
    ny = 2 * m + 1
    H = ny - 1
    L = int(round(L_over_H * H))
    nx = x_step + L
    nu = u_in * H / re
    tau = 0.5 + 3.0 * nu

    # Setup
    solid = build_solid(ny, nx, m, x_step, device)
    u_prof = inlet_profile(ny, m, u_in, device, torch.float32)

    # Sponge
    sponge_peak = 0.3
    sponge_width_H = 3.0
    sponge_w = int(round(sponge_width_H * H))
    sponge_x0 = nx - sponge_w
    sigma = sponge_peak * make_sponge_strength(
        ny, nx, sponge_x0, sponge_w, power=2.0, device=device
    )

    # Check for checkpoint
    checkpoint_file = checkpoint_dir / f"checkpoint_m{m}.pt"
    start_step = 0
    if checkpoint_file.exists():
        print(f"Loading checkpoint from {checkpoint_file}")
        ckpt = torch.load(checkpoint_file, weights_only=True)
        f = ckpt["f"]
        start_step = ckpt["step"]
        print(f"  Resuming from step {start_step}")
    else:
        # Initialize
        rho0 = torch.ones((ny, nx), device=device)
        ux0 = torch.zeros((ny, nx), device=device)
        ux0[:, :x_step] = u_prof.view(ny, 1)
        ux0[:, x_step:] = u_in / 2.0
        ux0[solid] = 0.0
        f = equilibrium(rho0, ux0, torch.zeros_like(ux0))

    # Measurement setup
    K = min(6, m // 2)
    lo_rows, w_lo, up_rows, w_up = _rows_near_walls(ny, m, K)

    opp = OPPOSITE.to(device)
    solid3 = solid.unsqueeze(0)

    # Run in segments
    results = []
    current_step = start_step
    
    while current_step < total_steps:
        seg_end = min(current_step + segment_steps, total_steps)
        seg_start = current_step
        
        print(f"\n=== Segment: step {seg_start} -> {seg_end} ===")
        t0 = time.time()
        
        # Run this segment
        for step in range(seg_start, seg_end):
            # Inlet BC
            zou_he_inlet_velocity(f, u_prof, x=0)
            
            # Pre-stream bounce-back (mode K)
            f[:, solid] = f[:, solid].flip([0])[opp[:, solid]]
            
            # Stream
            stream(f)
            
            # Outlet BC
            zou_he_outlet_pressure(f, rho=1.0, x=nx - 1)
            
            # Sponge
            if sigma is not None:
                feq = equilibrium(*macroscopic(f))
                f -= sigma * (f - feq)
            
            # Collide
            collide_mrt(f, tau, solid3)
            
            # Measure periodically
            if (step + 1) % 1000 == 0 or step == seg_end - 1:
                rho, ux, uy = macroscopic(f)
                mm = measure_profile(ux, solid, x_step, ny, m, K, lo_rows, w_lo, up_rows, w_up)
                mass = float(rho.sum().item())
                
                result = {
                    "step": step + 1,
                    "X1_H": mm["X1_H"],
                    "X1_h": mm["X1_h"],
                    "X2_H": mm["X2_H"],
                    "X3_H": mm["X3_H"],
                    "max_speed": float(torch.sqrt(ux * ux + uy * uy).max().item()),
                    "rho_min": float(rho.min().item()),
                    "rho_max": float(rho.max().item()),
                }
                results.append(result)
                
                print(
                    f"  step={step+1:>7d} X1/H={mm['X1_H']:.4f} X1/h={mm['X1_h']:.4f} "
                    f"X2/H={mm['X2_H']:.3f} X3/H={mm['X3_H']:.3f} "
                    f"max|u|={result['max_speed']:.4f} "
                    f"rho[{result['rho_min']:.4f},{result['rho_max']:.4f}] "
                    f"t={time.time()-t0:.0f}s",
                    flush=True,
                )
        
        current_step = seg_end
        
        # Save checkpoint
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"f": f, "step": current_step}, checkpoint_file)
        print(f"  Checkpoint saved at step {current_step}")
    
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--u", type=float, default=0.06)
    ap.add_argument("--re", type=float, default=800.0)
    ap.add_argument("--total-steps", type=int, default=40000)
    ap.add_argument("--segment-steps", type=int, default=8000)
    ap.add_argument("--checkpoint-dir", type=str, default="/tmp/bfs_checkpoints")
    ap.add_argument("--device", default="sdaa:0")
    ap.add_argument("--out", type=str, default="/tmp/bfs_checkpoint_results.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    checkpoint_dir = Path(args.checkpoint_dir)

    print(f"=== Backward-step Re={args.re} ER=2 with checkpoint/restart ===")
    print(f"m={args.m}, total_steps={args.total_steps}, segment_steps={args.segment_steps}")
    print(f"Reference X1/H = {REF_X1_H}")

    results = run_with_checkpoint(
        m=args.m,
        re=args.re,
        u_in=args.u,
        total_steps=args.total_steps,
        segment_steps=args.segment_steps,
        checkpoint_dir=checkpoint_dir,
        device=device,
    )

    # Save results
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n=== Results saved to {args.out} ===")
    
    # Summary
    if results:
        final = results[-1]
        err = (final["X1_H"] - REF_X1_H) / REF_X1_H * 100.0
        print(f"Final X1/H = {final['X1_H']:.4f} (err = {err:.2f}%)")
        print(f"Reference X1/H = {REF_X1_H}")


if __name__ == "__main__":
    main()
