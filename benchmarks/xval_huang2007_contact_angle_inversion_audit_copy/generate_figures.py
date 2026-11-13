#!/usr/bin/env python3
"""Generate figures for XVAL-LBM-HUANG2007-CONTACT-ANGLE-INVERSION report."""

import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Load results
out_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(out_dir, "results.json")) as f:
    results = json.load(f)

fig_dir = os.path.join(out_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Figure 1: Calibration curve (f_w vs θ)
# ---------------------------------------------------------------------------

cal_data = results["analytical_comparison"]
cal_coarse = [
    {"f_w": 0.30, "theta_measured": 161.47, "theta_young": 81.09},
    {"f_w": 0.35, "theta_measured": 143.49, "theta_young": 77.57},
    {"f_w": 0.40, "theta_measured": 127.94, "theta_young": 74.01},
    {"f_w": 0.45, "theta_measured": 112.90, "theta_young": 70.37},
    {"f_w": 0.50, "theta_measured": 98.01, "theta_young": 66.65},
    {"f_w": 0.55, "theta_measured": 81.40, "theta_young": 62.83},
    {"f_w": 0.60, "theta_measured": 62.68, "theta_young": 58.87},
    {"f_w": 0.65, "theta_measured": 36.30, "theta_young": 54.73},
]

fig, ax = plt.subplots(figsize=(10, 6))

f_ws = [d["f_w"] for d in cal_coarse]
theta_meas = [d["theta_measured"] for d in cal_coarse]
theta_young = [d["theta_young"] for d in cal_coarse]

ax.plot(f_ws, theta_meas, "ro-", label="Measured θ (LBM, coarse grid)", linewidth=2, markersize=8)
ax.plot(f_ws, theta_young, "b--", label="Young prediction θ", linewidth=2)

# Neutral wall markers
ax.axhline(y=90, color="gray", linestyle=":", alpha=0.5)
ax.axvline(x=0.1718, color="blue", linestyle=":", alpha=0.5, label="Young neutral wall (f_w=0.172)")
ax.axvline(x=0.5117, color="red", linestyle=":", alpha=0.5, label="Measured neutral wall (f_w=0.512)")

ax.set_xlabel("Wall pseudopotential ratio $f_w = \\psi_w / \\psi_l$", fontsize=12)
ax.set_ylabel("Contact angle $\\theta$ (degrees)", fontsize=12)
ax.set_title("Contact Angle Calibration Curve: Young vs Measured (Huang2007)", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.25, 0.70)
ax.set_ylim(30, 170)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "calibration_curve.png"), dpi=150)
plt.close()
print("Saved calibration_curve.png")

# ---------------------------------------------------------------------------
# Figure 2: Error comparison (Young vs Huang2007)
# ---------------------------------------------------------------------------

target_angles = [30, 45, 60, 75, 90, 105, 120, 135, 150]

# Compute errors using calibration data
def forward_cal(f_w, data):
    valid = sorted(data, key=lambda d: d["f_w"])
    f_ws = [d["f_w"] for d in valid]
    thetas = [d["theta_measured"] for d in valid]
    if f_w <= f_ws[0]:
        return thetas[0]
    if f_w >= f_ws[-1]:
        return thetas[-1]
    for i in range(len(f_ws) - 1):
        if f_ws[i] <= f_w <= f_ws[i + 1]:
            t = (f_w - f_ws[i]) / (f_ws[i + 1] - f_ws[i])
            return thetas[i] + t * (thetas[i + 1] - thetas[i])
    return thetas[-1]

PSI_L = 0.8587
PSI_V = 0.1475

def young_f_w(theta_deg):
    cos_t = math.cos(math.radians(theta_deg))
    f_w = (PSI_V + cos_t * (PSI_L - PSI_V)) / PSI_L
    return max(0.0, min(1.0, f_w))

def build_invert(data):
    valid = sorted([d for d in data], key=lambda d: d["theta_measured"])
    thetas = [d["theta_measured"] for d in valid]
    f_ws = [d["f_w"] for d in valid]
    def invert(theta_target):
        if theta_target <= thetas[0]:
            return f_ws[0]
        if theta_target >= thetas[-1]:
            return f_ws[-1]
        for i in range(len(thetas) - 1):
            if thetas[i] <= theta_target <= thetas[i + 1]:
                t = (theta_target - thetas[i]) / (thetas[i + 1] - thetas[i])
                return f_ws[i] + t * (f_ws[i + 1] - f_ws[i])
        return f_ws[-1]
    return invert

invert = build_invert(cal_coarse)

young_errors = []
huang_errors = []
for theta in target_angles:
    f_w_y = young_f_w(theta)
    theta_meas_y = forward_cal(f_w_y, cal_coarse)
    err_y = abs(theta_meas_y - theta) / theta * 100 if theta > 0 else float("nan")
    young_errors.append(err_y)

    try:
        f_w_h = invert(theta)
        theta_meas_h = forward_cal(f_w_h, cal_coarse)
        err_h = abs(theta_meas_h - theta) / theta * 100 if theta > 0 else float("nan")
    except:
        err_h = float("nan")
    huang_errors.append(err_h)

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(target_angles))
width = 0.35

bars1 = ax.bar(x - width/2, young_errors, width, label="Young relation", color="red", alpha=0.7)
bars2 = ax.bar(x + width/2, huang_errors, width, label="Huang2007 calibration", color="green", alpha=0.7)

# 15% threshold line
ax.axhline(y=15, color="orange", linestyle="--", linewidth=2, label="15% threshold (owner criterion)")

ax.set_xlabel("Target contact angle (degrees)", fontsize=12)
ax.set_ylabel("Relative error (%)", fontsize=12)
ax.set_title("Contact Angle Error: Young vs Huang2007 Calibration", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels([str(t) + "°" for t in target_angles])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "error_comparison.png"), dpi=150)
plt.close()
print("Saved error_comparison.png")

# ---------------------------------------------------------------------------
# Figure 3: cos(θ) vs f_w (linear fit comparison)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

# Young relation line
f_w_range = np.linspace(0, 1, 100)
young_slope = PSI_L / (PSI_L - PSI_V)
young_intercept = -PSI_V / (PSI_L - PSI_V)
cos_young = young_slope * f_w_range + young_intercept
ax.plot(f_w_range, cos_young, "b--", label=f"Young: cos(θ) = {young_slope:.3f}·f_w + ({young_intercept:.3f})", linewidth=2)

# Measured data
cos_meas = [math.cos(math.radians(d["theta_measured"])) for d in cal_coarse]
ax.plot(f_ws, cos_meas, "ro-", label="Measured (LBM)", linewidth=2, markersize=8)

# Linear fit to measured data
z = np.polyfit(f_ws, cos_meas, 1)
p = np.poly1d(z)
ax.plot(f_w_range, p(f_w_range), "g-", label=f"Measured fit: cos(θ) = {z[0]:.3f}·f_w + ({z[1]:.3f})", linewidth=2)

ax.set_xlabel("Wall pseudopotential ratio $f_w$", fontsize=12)
ax.set_ylabel("cos(θ)", fontsize=12)
ax.set_title("cos(θ) vs f_w: Young Theory vs LBM Measurement", fontsize=14)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.25, 0.70)
ax.set_ylim(-1.1, 1.1)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "cos_theta_vs_fw.png"), dpi=150)
plt.close()
print("Saved cos_theta_vs_fw.png")

print("All figures generated.")
