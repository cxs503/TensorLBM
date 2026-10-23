#!/usr/bin/env python3
"""m186: 3D channel_turb probe run on GPU5.

Runs L1 (64x32x32) first as a fast transition probe.
"""
import sys, os, json, time
sys.path.insert(0, "src")
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
from tensorlbm.wall_resolved_channel3d import (
    WallResolvedChannel3DConfig,
    run_wall_resolved_channel3d,
)

LEVELS = {
    "L1": {"nx": 64,  "ny": 32, "nz": 32,  "steps": 50000, "warmup": 20000},
    "L2": {"nx": 128, "ny": 48, "nz": 64,  "steps": 50000, "warmup": 20000},
    "L3": {"nx": 256, "ny": 64, "nz": 128, "steps": 50000, "warmup": 20000},
}

def run_level(level_name, out_dir):
    p = LEVELS[level_name]
    out_path = f"{out_dir}/{level_name}_result.json"
    ckpt_path = f"{out_dir}/{level_name}_ckpt.pt"
    config = WallResolvedChannel3DConfig(
        nx=p["nx"], ny=p["ny"], nz=p["nz"],
        re_tau=180.0, u_tau=0.003,
        steps=p["steps"], warmup_steps=p["warmup"],
        sample_interval=10, report_interval=500,
        checkpoint_interval=10000,
        collision_model="natural_kbc",
        initialization_mode="spectral_solenoidal",
        perturbation_fraction=1.0,
        random_noise_fraction=0.5,
        spectral_mode_count=32,
        spectral_max_wavenumber=4,
        seed=20260802,
        device="cuda",
        output=out_path,
        checkpoint=ckpt_path,
    )
    print(f"[m186] Starting {level_name}: {p['nx']}x{p['ny']}x{p['nz']}, {p['steps']} steps")
    t0 = time.time()
    result = run_wall_resolved_channel3d(config)
    elapsed = time.time() - t0
    print(f"[m186] {level_name} done in {elapsed:.1f}s")
    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L1", choices=["L1", "L2", "L3"])
    parser.add_argument("--out-dir", default="benchmarks/pending/channel_turb/runs_3d")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    run_level(args.level, args.out_dir)
