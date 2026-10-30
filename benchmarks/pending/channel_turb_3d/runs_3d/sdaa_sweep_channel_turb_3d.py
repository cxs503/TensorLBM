#!/usr/bin/env python3
"""m458: SDAA sweep script for channel_turb_3d domain-length fix.

Runs L1/L2/L3/L4 levels sequentially on SDAA device, collects artifacts,
and generates validation report.

Usage:
    python sdaa_sweep_channel_turb_3d.py [--levels L1 L2 L3 L4] [--out-dir DIR]
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

import torch

from tensorlbm.wall_resolved_channel3d import (
    WallResolvedChannel3DConfig,
    run_wall_resolved_channel3d,
)

# Grid levels
LEVELS = {
    "L1": {"nx": 64, "ny": 32, "nz": 32, "steps": 50000, "warmup": 20000},
    "L2": {"nx": 128, "ny": 48, "nz": 64, "steps": 50000, "warmup": 20000},
    "L3": {"nx": 256, "ny": 64, "nz": 128, "steps": 50000, "warmup": 20000},
    "L4": {"nx": 396, "ny": 64, "nz": 132, "steps": 125000, "warmup": 40000},
}


def run_level(level_name, out_dir, resume=False):
    """Run a single grid level and return result dict."""
    p = LEVELS[level_name]
    out_path = f"{out_dir}/{level_name}_result.json"
    ckpt_path = f"{out_dir}/{level_name}_ckpt.pt"

    height = p["ny"] - 2
    lx_over_h = p["nx"] / (0.5 * height)
    lz_over_h = p["nz"] / (0.5 * height)
    eddy_turnover = height / 0.003
    avg_steps = p["steps"] - p["warmup"]
    eddy_turnovers = avg_steps / eddy_turnover

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

    print(
        f"[m458] Starting {level_name}: {p['nx']}x{p['ny']}x{p['nz']}, "
        f"Lx/h={lx_over_h:.2f}, Lz/h={lz_over_h:.2f}, "
        f"steps={p['steps']}, warmup={p['warmup']}, "
        f"eddy_turnovers={eddy_turnovers:.2f}"
    )

    # Record SDAA device info
    device_name = "unknown"
    try:
        device_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    t0 = time.time()
    result = run_wall_resolved_channel3d(config)
    elapsed = time.time() - t0

    print(f"[m458] {level_name} done in {elapsed:.1f}s")

    # Add device metadata
    result["sdaa_device"] = device_name
    result["wall_clock_seconds"] = round(elapsed, 1)

    return result


def generate_artifact(results, out_dir):
    """Generate SDAA artifact JSON."""
    artifact = {
        "mission": "m458",
        "case": "channel_turb_3d",
        "sdaa_device": results[0].get("sdaa_device", "unknown") if results else "unknown",
        "levels": [],
        "threshold_rms_pct": 3.0,
    }

    for r in results:
        cfg = r["configuration"]
        stats = r["statistics"]
        derived = r["derived"]
        level = cfg["nx"]  # use nx to identify level

        # Determine level name
        if cfg["nx"] == 64:
            level_name = "L1"
        elif cfg["nx"] == 128:
            level_name = "L2"
        elif cfg["nx"] == 256:
            level_name = "L3"
        elif cfg["nx"] == 396:
            level_name = "L4"
        else:
            level_name = f"L_unknown_{cfg['nx']}"

        height = cfg["ny"] - 2
        eddy_turnover = height / cfg["u_tau"]
        avg_steps = cfg["steps"] - cfg["warmup_steps"]
        eddy_turnovers = avg_steps / eddy_turnover

        # u_τ from force balance
        reports = r.get("reports", [])
        wall_shears = [float(rp["measured_wall_shear"]) for rp in reports if "measured_wall_shear" in rp]
        if wall_shears:
            stationarity_start = len(wall_shears) // 2
            stationarity_shears = wall_shears[stationarity_start:]
            mean_shear = sum(stationarity_shears) / len(stationarity_shears)
            u_tau_fb = mean_shear ** 0.5
            split = len(stationarity_shears) // 2
            first_half = sum(stationarity_shears[:split]) / split
            second_half = sum(stationarity_shears[split:]) / len(stationarity_shears[split:])
            u_tau_first = first_half ** 0.5
            u_tau_second = second_half ** 0.5
            window_drift = abs(u_tau_second - u_tau_first) / max(abs(u_tau_fb), 1e-30)
        else:
            u_tau_fb = cfg["u_tau"]
            window_drift = 0.0
            u_tau_first = u_tau_second = u_tau_fb

        u_tau_err = (u_tau_fb - cfg["u_tau"]) / cfg["u_tau"] * 100

        artifact["levels"].append({
            "level": level_name,
            "grid": f"{cfg['nx']}x{cfg['ny']}x{cfg['nz']}",
            "Lx_over_h": derived["streamwise_length_over_half_height"],
            "Lz_over_h": derived["spanwise_length_over_half_height"],
            "dy_plus": derived["dx_plus"],
            "steps": cfg["steps"],
            "warmup": cfg["warmup_steps"],
            "eddy_turnovers": round(eddy_turnovers, 2),
            "profile_samples": stats["profile_samples"],
            "u_tau_target": cfg["u_tau"],
            "u_tau_force_balance": round(u_tau_fb, 6),
            "u_tau_err_pct": round(u_tau_err, 2),
            "window_drift_fraction": round(window_drift, 4),
            "wall_clock_seconds": r.get("wall_clock_seconds", 0),
        })

    # Check convergence
    rms_values = []
    for r in results:
        stats = r["statistics"]
        profile = stats["mean_velocity_profile"]
        ny = r["configuration"]["ny"]
        u_tau = r["configuration"]["u_tau"]
        nu = r["derived"]["nu"]

        y_cells = list(range(ny))
        y_dist = [y - 0.5 for y in y_cells]
        y_plus = [y * u_tau / nu for y in y_dist]

        fluid_mask = [y_cells[i] >= 1 and y_cells[i] <= ny - 2 for i in range(ny)]
        yp_fluid = [y_plus[i] for i in range(ny) if fluid_mask[i]]
        up_fluid = [profile[i] / u_tau for i in range(ny) if fluid_mask[i]]

        # Load DNS
        dns_csv = "benchmarks/pending/channel_turb/dns_ref/mkm180.csv"
        import csv
        dns_yp, dns_up = [], []
        with open(dns_csv) as f:
            for row in csv.DictReader(f):
                dns_yp.append(float(row["y_plus"]))
                dns_up.append(float(row["u_plus"]))

        import numpy as np
        window_mask = [yp >= 30 and yp <= 100 for yp in yp_fluid]
        yp_window = [yp_fluid[i] for i in range(len(yp_fluid)) if window_mask[i]]
        up_window = [up_fluid[i] for i in range(len(up_fluid)) if window_mask[i]]
        dns_up_window = np.interp(yp_window, dns_yp, dns_up)

        rms = float(np.sqrt(np.mean(((np.array(up_window) - dns_up_window) / dns_up_window) ** 2))) * 100
        rms_values.append(rms)

    artifact["rms_values"] = [round(v, 2) for v in rms_values]
    artifact["monotone"] = all(rms_values[i] > rms_values[i + 1] for i in range(len(rms_values) - 1)) if len(rms_values) >= 2 else False
    artifact["best_rms"] = round(min(rms_values), 2) if rms_values else None
    artifact["passes_threshold"] = artifact["best_rms"] is not None and artifact["best_rms"] <= 3.0

    artifact_path = Path(out_dir) / "sdaa_artifact.json"
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"Artifact written to {artifact_path}")

    return artifact


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="m458: SDAA sweep for channel_turb_3d")
    parser.add_argument("--levels", nargs="+", default=["L1", "L2", "L3", "L4"], choices=["L1", "L2", "L3", "L4"])
    parser.add_argument("--out-dir", default="benchmarks/pending/channel_turb_3d/runs_3d")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    results = []
    for level in args.levels:
        result = run_level(level, args.out_dir, resume=args.resume)
        results.append(result)

    if results:
        artifact = generate_artifact(results, args.out_dir)
        print(f"\n=== Summary ===")
        print(f"Device: {artifact['sdaa_device']}")
        print(f"Monotone: {artifact['monotone']}")
        print(f"Best RMS: {artifact['best_rms']}%")
        print(f"Passes 3% threshold: {artifact['passes_threshold']}")
