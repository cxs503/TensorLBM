#!/usr/bin/env python3
"""Plot figures for channel_turb 3D terminal verdict."""
import sys, json, csv
sys.path.insert(0, "src")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIG_DIR = Path("benchmarks/pending/channel_turb/figures")
FIG_DIR.mkdir(exist_ok=True)

def load_dns(csv_path):
    yp, up = [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            yp.append(float(row["y_plus"]))
            up.append(float(row["u_plus"]))
    return np.array(yp), np.array(up)

def extract_profile(result_path):
    result = json.load(open(result_path))
    cfg = result["configuration"]
    derived = result["derived"]
    stats = result["statistics"]
    ny = cfg["ny"]
    u_tau = cfg["u_tau"]
    nu = derived["nu"]
    profile = np.array(stats["mean_velocity_profile"])
    y_cells = np.arange(ny)
    y_dist = y_cells - 0.5
    y_plus = y_dist * u_tau / nu
    fluid_mask = (y_cells >= 1) & (y_cells <= ny-2)
    return y_plus[fluid_mask], profile[fluid_mask] / u_tau

def extract_convergence(result_path):
    """Extract u_tau error vs step from log file."""
    log_path = result_path.replace("_result.json", "_probe.log")
    steps, errs = [], []
    with open(log_path) as f:
        for line in f:
            if line.startswith("channel3d "):
                try:
                    d = json.loads(line.split("channel3d ")[1])
                    steps.append(d["step"])
                    errs.append(d["friction_velocity_error_pct"])
                except:
                    pass
    return np.array(steps), np.array(errs)

# Load DNS
dns_yp, dns_up = load_dns("benchmarks/pending/channel_turb/dns_ref/mkm180.csv")

# --- Figure 1: u+ profile comparison (curve) ---
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(dns_up, dns_yp, 'k-', linewidth=2, label='DNS (Moser 1999)')

# Log law
yp_log = np.linspace(30, 100, 100)
up_log = np.log(yp_log) / 0.41 + 5.0
ax.plot(up_log, yp_log, 'k--', linewidth=1, alpha=0.5, label='Log law')

colors = {'L1': 'red', 'L2': 'blue', 'L3': 'green'}
for level in ['L1', 'L2', 'L3']:
    result_path = f"benchmarks/pending/channel_turb/runs_3d/{level}_result.json"
    if Path(result_path).exists():
        yp, up = extract_profile(result_path)
        ax.plot(up, yp, color=colors[level], linewidth=1.5, alpha=0.8, label=f'{level}')

ax.set_xlabel('u⁺', fontsize=14)
ax.set_ylabel('y⁺', fontsize=14)
ax.set_title('3D Channel Turbulence: u⁺ Profile vs DNS (Re_τ=180)', fontsize=14)
ax.set_xlim(0, 25)
ax.set_ylim(0, 65)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_uplus_profile.png", dpi=150)
plt.close()
print(f"Saved {FIG_DIR / 'fig_uplus_profile.png'}")

# --- Figure 2: Convergence history (curve) ---
fig, ax = plt.subplots(figsize=(8, 5))
for level in ['L1', 'L2', 'L3']:
    result_path = f"benchmarks/pending/channel_turb/runs_3d/{level}_result.json"
    if Path(result_path).exists():
        steps, errs = extract_convergence(result_path)
        if len(steps) > 0:
            ax.plot(steps, np.abs(errs), color=colors[level], linewidth=1.5, 
                   alpha=0.8, label=f'{level}')

ax.axhline(y=3.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='3% threshold')
ax.set_xlabel('Step', fontsize=14)
ax.set_ylabel('|u_τ error| [%]', fontsize=14)
ax.set_title('3D Channel: Friction Velocity Convergence History', fontsize=14)
ax.set_yscale('log')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_convergence.png", dpi=150)
plt.close()
print(f"Saved {FIG_DIR / 'fig_convergence.png'}")

# --- Figure 3: RMS error convergence (curve) ---
fig, ax = plt.subplots(figsize=(8, 5))
validation = json.load(open("benchmarks/pending/channel_turb/validation_report.json"))
levels = [g["level"] for g in validation["grid_levels"]]
rms_vals = [g["rms_y30_100_pct"] for g in validation["grid_levels"]]
ax.plot(range(len(levels)), rms_vals, 'bo-', linewidth=2, markersize=10, label='RMS error')
ax.axhline(y=3.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='3% threshold')
ax.set_xlabel('Grid Level', fontsize=14)
ax.set_ylabel('RMS error vs DNS [%]', fontsize=14)
ax.set_title('3D Channel: Grid Convergence (Monotone)', fontsize=14)
ax.set_xticks(range(len(levels)))
ax.set_xticklabels(levels)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_rms_convergence.png", dpi=150)
plt.close()
print(f"Saved {FIG_DIR / 'fig_rms_convergence.png'}")

print("\nAll figures generated successfully.")
