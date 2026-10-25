#!/usr/bin/env python3
"""Post-process 3D channel results and generate validation report."""

import json
import sys

sys.path.insert(0, "src")
import csv
from pathlib import Path

import numpy as np


def load_dns(csv_path):
    """Load MKM1999 DNS profile."""
    yp, up = [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            yp.append(float(row["y_plus"]))
            up.append(float(row["u_plus"]))
    return np.array(yp), np.array(up)


def analyze_level(result_path, dns_yp, dns_up, level_name):
    """Analyze one grid level and return metrics."""
    result = json.load(open(result_path))
    cfg = result["configuration"]
    derived = result["derived"]
    stats = result["statistics"]

    ny = cfg["ny"]
    u_tau = cfg["u_tau"]
    nu = derived["nu"]

    # Mean velocity profile
    profile = np.array(stats["mean_velocity_profile"])

    # Wall distance
    y_cells = np.arange(ny)
    y_dist = y_cells - 0.5
    y_plus = y_dist * u_tau / nu

    # Fluid region
    fluid_mask = (y_cells >= 1) & (y_cells <= ny - 2)
    yp_fluid = y_plus[fluid_mask]
    up_fluid = profile[fluid_mask] / u_tau

    # DNS interpolation
    dns_up_interp = np.interp(yp_fluid, dns_yp, dns_up)

    # RMS error in y+ ∈ [30, 100]
    window_mask = (yp_fluid >= 30) & (yp_fluid <= 100)
    if window_mask.sum() == 0:
        return None

    yp_window = yp_fluid[window_mask]
    up_window = up_fluid[window_mask]
    dns_up_window = np.interp(yp_window, dns_yp, dns_up)

    rms_error = np.sqrt(np.mean(((up_window - dns_up_window) / dns_up_window) ** 2)) * 100

    # Friction velocity
    u_tau_meas = stats["recent_friction_velocity_mean"]
    u_tau_err = (u_tau_meas - u_tau) / u_tau * 100

    # Centerline velocity
    center_idx = ny // 2
    u_cl_plus = profile[center_idx] / u_tau

    return {
        "level": level_name,
        "grid": f"{cfg['nx']}x{cfg['ny']}x{cfg['nz']}",
        "Lx_over_h": derived["streamwise_length_over_half_height"],
        "Lz_over_h": derived["spanwise_length_over_half_height"],
        "dy_plus": derived.get("dx_plus", 0),
        "u_tau_target": u_tau,
        "u_tau_measured": u_tau_meas,
        "u_tau_err_pct": u_tau_err,
        "u_cl_plus": u_cl_plus,
        "rms_y30_100_pct": rms_error,
        "eddy_turnovers": stats["statistics_eddy_turnovers"],
        "profile_samples": stats["profile_samples"],
    }


def generate_report(results, output_path):
    """Generate validation report."""
    report = {
        "schema": "channel_turb_3d_validation_v1",
        "case": "channel_turb_3d",
        "reference": {
            "source": "Moser, Kim & Mansour (1999), Phys. Fluids 11, 943",
            "re_tau": 180.0,
            "file": "benchmarks/pending/channel_turb/dns_ref/mkm180.csv",
        },
        "grid_levels": results,
        "convergence": {"monotone": False, "trend": []},
        "verdict": "pending",
        "reason": "",
    }

    # Check monotone convergence
    if len(results) >= 2:
        rms_values = [r["rms_y30_100_pct"] for r in results]
        report["convergence"]["trend"] = rms_values
        report["convergence"]["monotone"] = all(
            rms_values[i] > rms_values[i + 1] for i in range(len(rms_values) - 1)
        )

    # Verdict
    if len(results) >= 2:
        best_rms = min(r["rms_y30_100_pct"] for r in results)
        if report["convergence"]["monotone"] and best_rms <= 3.0:
            report["verdict"] = "promote"
            report["reason"] = (
                f"≥2 grid levels with monotone convergence, best RMS={best_rms:.2f}% ≤ 3%"
            )
        else:
            report["verdict"] = "terminal"
            report["reason"] = f"Best RMS={best_rms:.2f}% > 3% or non-monotone"

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report written to {output_path}")
    print(f"Verdict: {report['verdict']}")
    print(f"Reason: {report['reason']}")

    return report


if __name__ == "__main__":
    dns_csv = "benchmarks/pending/channel_turb/dns_ref/mkm180.csv"
    dns_yp, dns_up = load_dns(dns_csv)

    results = []
    for level in ["L1", "L2", "L3"]:
        result_path = f"benchmarks/pending/channel_turb/runs_3d/{level}_result.json"
        if Path(result_path).exists():
            print(f"\nAnalyzing {level}...")
            r = analyze_level(result_path, dns_yp, dns_up, level)
            if r:
                results.append(r)
                print(f"  Grid: {r['grid']}")
                print(f"  Domain: Lx/h={r['Lx_over_h']:.2f}, Lz/h={r['Lz_over_h']:.2f}")
                print(f"  u_tau error: {r['u_tau_err_pct']:+.2f}%")
                print(f"  RMS error y+∈[30,100]: {r['rms_y30_100_pct']:.2f}%")
                print(f"  Eddy turnovers: {r['eddy_turnovers']:.2f}")

    if results:
        report = generate_report(results, "benchmarks/pending/channel_turb/validation_report.json")
