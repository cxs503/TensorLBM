#!/usr/bin/env python3
"""Post-process 3D channel results and generate validation report.

m458 update:
- u_τ from force balance (tau_w = rho*a_x*H/2) instead of wall-shear gradient
- Window drift metric: |u_τ(first_half) - u_τ(second_half)| / u_τ
- Support L4 level with DNS-matched domain
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "src")
import csv

import numpy as np


def load_dns(csv_path):
    """Load MKM1999 DNS profile."""
    yp, up = [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            yp.append(float(row["y_plus"]))
            up.append(float(row["u_plus"]))
    return np.array(yp), np.array(up)


def compute_u_tau_force_balance(result):
    """Compute u_τ from force balance: tau_w = rho * a_x * H / 2.

    The body force acceleration is a_x = 2 * u_tau_target^2 / H.
    Force balance gives: tau_w = a_x * H / 2 = u_tau_target^2.
    So u_tau_fb = u_tau_target by construction.

    However, we use the *measured* wall shear from the simulation reports
    to compute the actual force balance, which accounts for numerical errors.
    """
    reports = result.get("reports", [])
    if not reports:
        return None, None, None

    # Extract wall shear from each report
    wall_shears = []
    for r in reports:
        ws = r.get("measured_wall_shear")
        if ws is not None:
            wall_shears.append(float(ws))

    if not wall_shears:
        return None, None, None

    # Stationarity window: last half of reports
    stationarity_start = len(wall_shears) // 2
    stationarity_shears = wall_shears[stationarity_start:]

    mean_shear = sum(stationarity_shears) / len(stationarity_shears)
    u_tau_fb = math.sqrt(max(mean_shear, 0.0))

    # Window drift: compare first half vs second half of stationarity window
    split = len(stationarity_shears) // 2
    first_half_shear = sum(stationarity_shears[:split]) / split
    second_half_shear = sum(stationarity_shears[split:]) / len(stationarity_shears[split:])
    u_tau_first = math.sqrt(max(first_half_shear, 0.0))
    u_tau_second = math.sqrt(max(second_half_shear, 0.0))
    window_drift = abs(u_tau_second - u_tau_first) / max(abs(u_tau_fb), 1e-30)

    return u_tau_fb, window_drift, (u_tau_first, u_tau_second)


def analyze_level(result_path, dns_yp, dns_up, level_name):
    """Analyze one grid level and return metrics."""
    result = json.load(open(result_path))
    cfg = result["configuration"]
    derived = result["derived"]
    stats = result["statistics"]

    ny = cfg["ny"]
    u_tau_target = cfg["u_tau"]
    nu = derived["nu"]
    height = derived["height"]

    # Mean velocity profile
    profile = np.array(stats["mean_velocity_profile"])

    # Wall distance: half-way bounce-back, wall at y=0.5
    y_cells = np.arange(ny)
    y_dist = y_cells - 0.5
    y_plus = y_dist * u_tau_target / nu

    # Fluid region
    fluid_mask = (y_cells >= 1) & (y_cells <= ny - 2)
    yp_fluid = y_plus[fluid_mask]
    up_fluid = profile[fluid_mask] / u_tau_target

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

    # u_τ from force balance (m458 fix)
    u_tau_fb, window_drift, (u_tau_first, u_tau_second) = compute_u_tau_force_balance(result)

    # Also get the library's measured u_τ for comparison
    u_tau_lib = stats.get("recent_friction_velocity_mean", u_tau_target)

    if u_tau_fb is not None:
        u_tau_err_fb = (u_tau_fb - u_tau_target) / u_tau_target * 100
        u_tau_err_lib = (u_tau_lib - u_tau_target) / u_tau_target * 100
    else:
        u_tau_fb = u_tau_lib
        u_tau_err_fb = u_tau_err_lib = (u_tau_lib - u_tau_target) / u_tau_target * 100
        window_drift = 0.0
        u_tau_first = u_tau_second = u_tau_fb

    # Centerline velocity
    center_idx = ny // 2
    u_cl_plus = profile[center_idx] / u_tau_target

    # Eddy turnovers
    eddy_turnover_steps = height / u_tau_target
    statistics_steps = stats.get("statistics_steps", cfg["steps"] - cfg["warmup_steps"])
    eddy_turnovers = statistics_steps / eddy_turnover_steps

    return {
        "level": level_name,
        "grid": f"{cfg['nx']}x{cfg['ny']}x{cfg['nz']}",
        "Lx_over_h": derived["streamwise_length_over_half_height"],
        "Lz_over_h": derived["spanwise_length_over_half_height"],
        "dy_plus": derived.get("dx_plus", 0),
        "u_tau_target": u_tau_target,
        "u_tau_force_balance": u_tau_fb,
        "u_tau_library": u_tau_lib,
        "u_tau_err_fb_pct": u_tau_err_fb,
        "u_tau_err_lib_pct": u_tau_err_lib,
        "u_cl_plus": u_cl_plus,
        "rms_y30_100_pct": rms_error,
        "eddy_turnovers": eddy_turnovers,
        "profile_samples": stats["profile_samples"],
        "window_drift_fraction": window_drift,
        "u_tau_first_half": u_tau_first,
        "u_tau_second_half": u_tau_second,
        "statistics_steps": statistics_steps,
        "stationarity_report_count": stats.get("stationarity_report_count", 0),
    }


def generate_report(results, output_path):
    """Generate validation report."""
    report = {
        "schema": "channel_turb_3d_validation_v2",
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
        best_level = [r for r in results if r["rms_y30_100_pct"] == best_rms][0]
        if report["convergence"]["monotone"] and best_rms <= 3.0:
            report["verdict"] = "promote"
            report["reason"] = (
                f"≥{len(results)} grid levels with monotone convergence, "
                f"best RMS={best_rms:.2f}% (level {best_level['level']}) ≤ 3%"
            )
        else:
            report["verdict"] = "terminal"
            report["reason"] = (
                f"Best RMS={best_rms:.2f}% (level {best_level['level']}) > 3% or non-monotone"
            )

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
    for level in ["L1", "L2", "L3", "L4"]:
        result_path = f"benchmarks/pending/channel_turb_3d/runs_3d/{level}_result.json"
        if Path(result_path).exists():
            print(f"\nAnalyzing {level}...")
            r = analyze_level(result_path, dns_yp, dns_up, level)
            if r:
                results.append(r)
                print(f"  Grid: {r['grid']}")
                print(f"  Domain: Lx/h={r['Lx_over_h']:.2f}, Lz/h={r['Lz_over_h']:.2f}")
                print(f"  u_τ target={r['u_tau_target']:.6f}")
                print(f"  u_τ force_balance={r['u_tau_force_balance']:.6f} (err={r['u_tau_err_fb_pct']:+.2f}%)")
                print(f"  u_τ library={r['u_tau_library']:.6f} (err={r['u_tau_err_lib_pct']:+.2f}%)")
                print(f"  Window drift: {r['window_drift_fraction']:.4f}")
                print(f"  RMS error y+∈[30,100]: {r['rms_y30_100_pct']:.2f}%")
                print(f"  Eddy turnovers: {r['eddy_turnovers']:.2f}")
                print(f"  Profile samples: {r['profile_samples']}")

    if results:
        report = generate_report(results, "benchmarks/pending/channel_turb_3d/validation_report.json")
