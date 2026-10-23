#!/usr/bin/env python3
"""Analyze 3D channel results vs DNS reference."""
import sys, json
sys.path.insert(0, "src")
import numpy as np
import csv
from pathlib import Path

def load_dns(csv_path):
    """Load MKM1999 DNS profile."""
    yp, up = [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            yp.append(float(row["y_plus"]))
            up.append(float(row["u_plus"]))
    return np.array(yp), np.array(up)

def analyze_level(result_path, dns_yp, dns_up, level_name):
    """Analyze one grid level."""
    result = json.load(open(result_path))
    cfg = result["configuration"]
    derived = result["derived"]
    stats = result["statistics"]
    
    ny = cfg["ny"]
    u_tau = cfg["u_tau"]
    nu = derived["nu"]
    height = derived["height"]
    
    # Mean velocity profile (ny points, including walls)
    profile = np.array(stats["mean_velocity_profile"])
    
    # Wall distance: y=0 is wall, y=1..ny-2 are fluid, y=ny-1 is wall
    # First fluid cell at y=1, distance = 0.5 (half-way bounce-back)
    y_cells = np.arange(ny)
    y_dist = y_cells - 0.5  # distance from bottom wall
    y_plus = y_dist * u_tau / nu
    
    # Extract fluid region (exclude walls)
    fluid_mask = (y_cells >= 1) & (y_cells <= ny-2)
    yp_fluid = y_plus[fluid_mask]
    up_fluid = profile[fluid_mask] / u_tau
    
    # Interpolate DNS to fluid y+ locations
    dns_up_interp = np.interp(yp_fluid, dns_yp, dns_up)
    
    # Compute RMS error in y+ ∈ [30, 100]
    window_mask = (yp_fluid >= 30) & (yp_fluid <= 100)
    if window_mask.sum() == 0:
        print(f"[{level_name}] WARNING: No points in y+∈[30,100] window!")
        return None
    
    yp_window = yp_fluid[window_mask]
    up_window = up_fluid[window_mask]
    dns_up_window = np.interp(yp_window, dns_yp, dns_up)
    
    rms_error = np.sqrt(np.mean(((up_window - dns_up_window) / dns_up_window)**2)) * 100
    
    # Also compute full profile error
    full_rms = np.sqrt(np.mean(((up_fluid - dns_up_interp) / dns_up_interp)**2)) * 100
    
    # Reynolds stress
    reynolds = np.array(stats["reynolds_stress_profiles_uu_vv_ww_uv"])
    # Shape: (4, ny) -> uu, vv, ww, uv
    uv_profile = reynolds[3, :]  # uv component
    uv_plus = uv_profile[fluid_mask] / (u_tau**2)
    
    # Friction velocity
    u_tau_meas = stats["recent_friction_velocity_mean"]
    u_tau_err = (u_tau_meas - u_tau) / u_tau * 100
    
    # Centerline velocity
    center_idx = ny // 2
    u_cl_plus = profile[center_idx] / u_tau
    
    print(f"\n[{level_name}] Grid: {cfg['nx']}x{cfg['ny']}x{cfg['nz']}")
    print(f"  Domain: Lx/h={derived['streamwise_length_over_half_height']:.2f}, Lz/h={derived['spanwise_length_over_half_height']:.2f}")
    print(f"  dy+={derived.get('dx_plus', 'N/A'):.2f}")
    print(f"  u_tau target={u_tau:.6f}, measured={u_tau_meas:.6f}, error={u_tau_err:+.2f}%")
    print(f"  U_cl+ = {u_cl_plus:.2f} (DNS ref ~17.5)")
    print(f"  RMS error y+∈[30,100]: {rms_error:.2f}%")
    print(f"  RMS error full profile: {full_rms:.2f}%")
    print(f"  Reynolds stress <uv>+ range: [{uv_plus.min():.3f}, {uv_plus.max():.3f}]")
    print(f"  Statistics samples: {stats['profile_samples']}")
    print(f"  Eddy turnovers: {stats['statistics_eddy_turnovers']:.2f}")
    
    return {
        "level": level_name,
        "grid": f"{cfg['nx']}x{cfg['ny']}x{cfg['nz']}",
        "u_tau_err_pct": u_tau_err,
        "rms_y30_100_pct": rms_error,
        "rms_full_pct": full_rms,
        "u_cl_plus": u_cl_plus,
        "yp": yp_fluid,
        "up": up_fluid,
        "uv_plus": uv_plus,
    }

if __name__ == "__main__":
    dns_csv = "benchmarks/pending/channel_turb/dns_ref/mkm180.csv"
    dns_yp, dns_up = load_dns(dns_csv)
    
    results = []
    for level in ["L1", "L2"]:
        result_path = f"benchmarks/pending/channel_turb/runs_3d/{level}_result.json"
        if Path(result_path).exists():
            r = analyze_level(result_path, dns_yp, dns_up, level)
            if r:
                results.append(r)
    
    if len(results) >= 2:
        print("\n=== Convergence check ===")
        print(f"L1 RMS: {results[0]['rms_y30_100_pct']:.2f}%")
        print(f"L2 RMS: {results[1]['rms_y30_100_pct']:.2f}%")
        if results[1]['rms_y30_100_pct'] < results[0]['rms_y30_100_pct']:
            print("✓ Monotone convergence observed")
        else:
            print("✗ Non-monotone or diverging")
