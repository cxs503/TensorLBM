#!/usr/bin/env python3
"""m458: 3D channel_turb probe run — domain-length fix + extended statistics.

L1/L2/L3: original levels (replayed for baseline).
L4: DNS-matched domain (Lx/h≈12.6, Lz/h≈4.2) + extended statistics (≥4 eddy turnovers).

Usage:
    python run_probe.py --level L4          # single level
    python run_probe.py --level L4 --resume  # resume from checkpoint
"""

import os
import sys
import time

sys.path.insert(0, "src")

from tensorlbm.wall_resolved_channel3d import (
    WallResolvedChannel3DConfig,
    run_wall_resolved_channel3d,
)

# Grid levels: (nx, ny, nz, steps, warmup)
# L1/L2/L3: original levels from m186
# L4: DNS-matched domain (Lx/h≈12.6, Lz/h≈4.2) + extended statistics
LEVELS = {
    "L1": {"nx": 64, "ny": 32, "nz": 32, "steps": 50000, "warmup": 20000},
    "L2": {"nx": 128, "ny": 48, "nz": 64, "steps": 50000, "warmup": 20000},
    "L3": {"nx": 256, "ny": 64, "nz": 128, "steps": 50000, "warmup": 20000},
    # L4: DNS-matched domain
    #   height = ny - 2 = 62
    #   Lx/h = nx / (0.5*height) = 396 / 31 = 12.77  (DNS: 12.6)
    #   Lz/h = nz / (0.5*height) = 132 / 31 = 4.26   (DNS: 4.2)
    #   steps: 120000 (warmup 40000 + avg 80000) → ~4.0 eddy turnovers
    "L4": {"nx": 396, "ny": 64, "nz": 132, "steps": 125000, "warmup": 40000},
}


def run_level(level_name, out_dir, resume=False):
    p = LEVELS[level_name]
    out_path = f"{out_dir}/{level_name}_result.json"
    ckpt_path = f"{out_dir}/{level_name}_ckpt.pt"
    config = WallResolvedChannel3DConfig(
        nx=p["nx"],
        ny=p["ny"],
        nz=p["nz"],
        re_tau=180.0,
        u_tau=0.003,
        steps=p["steps"],
        warmup_steps=p["warmup"],
        sample_interval=10,
        report_interval=500,
        checkpoint_interval=10000,
        collision_model="natural_kbc",
        initialization_mode="spectral_solenoidal",
        perturbation_fraction=1.0,
        random_noise_fraction=0.5,
        spectral_mode_count=32,
        spectral_max_wavenumber=4,
        minimum_statistics_eddy_turnovers=4.0,
        stationarity_window_eddy_turnovers=2.0,
        seed=20260802,
        device="cuda",
        output=out_path,
        checkpoint=ckpt_path,
        resume=resume,
    )
    height = p["ny"] - 2
    lx_over_h = p["nx"] / (0.5 * height)
    lz_over_h = p["nz"] / (0.5 * height)
    eddy_turnover = height / 0.003
    avg_steps = p["steps"] - p["warmup"]
    eddy_turnovers = avg_steps / eddy_turnover
    print(
        f"[m458] Starting {level_name}: {p['nx']}x{p['ny']}x{p['nz']}, "
        f"Lx/h={lx_over_h:.2f}, Lz/h={lz_over_h:.2f}, "
        f"steps={p['steps']}, warmup={p['warmup']}, "
        f"eddy_turnovers={eddy_turnovers:.2f}"
    )
    t0 = time.time()
    result = run_wall_resolved_channel3d(config)
    elapsed = time.time() - t0
    print(f"[m458] {level_name} done in {elapsed:.1f}s")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="m458: 3D channel_turb with domain-length fix")
    parser.add_argument("--level", default="L4", choices=["L1", "L2", "L3", "L4"])
    parser.add_argument("--out-dir", default="benchmarks/pending/channel_turb_3d/runs_3d")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    run_level(args.level, args.out_dir, resume=args.resume)
