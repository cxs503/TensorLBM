#!/usr/bin/env python3
"""Generate comparison plots for IN99-N03 turbulent channel validation.

Produces:
- U+(y+) comparison vs MKM-1999 DNS
- Reynolds stress profiles (-<u'v'>+, u_rms+, v_rms+, w_rms+)
- Anisotropy signature plot
- Gate summary bar chart
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_PILOT_DIR = _REPO / "runs" / "in99_n03_pilot"
_FIGURES_DIR = _REPO / "figures" / "in99_n03"
_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(_REPO / "src"))
from tensorlbm.validation import mkm1999


def _load_case(name: str) -> dict | None:
    path = _PILOT_DIR / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_gates() -> dict | None:
    path = _PILOT_DIR / "gates.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def plot_mean_velocity(mkm, result, case_name: str, out_path: Path):
    """Plot U+(y+) comparison."""
    y_plus_sim = np.asarray(result["profiles"]["y"], dtype=float)
    # Compute y+ from the simulation
    nu = result["derived"]["nu"]
    u_tau = np.sqrt(result["derived"]["accel"] * result["derived"]["half_height"])
    y_w = 0.5
    y_plus_sim = (y_plus_sim - y_w) * u_tau / nu
    u_plus_sim = np.asarray(result["profiles"]["mean_ux"], dtype=float) / u_tau

    # Only take half channel
    half = y_plus_sim <= mkm.re_tau * 1.02
    y_plus_half = y_plus_sim[half]
    u_plus_half = u_plus_sim[half]

    fig, ax = plt.subplots(figsize=(8, 6))

    # DNS reference
    ax.semilogx(mkm.y_plus[mkm.y_plus > 0], mkm.u_plus[mkm.y_plus > 0],
                'k-', linewidth=2, label=f'MKM-1999 DNS (Re_τ={mkm.re_tau:.1f})')

    # Simulation
    ax.semilogx(y_plus_half[y_plus_half > 0], u_plus_half[y_plus_half > 0],
                'ro-', markersize=4, linewidth=1, label=f'TensorLBM LES ({case_name})')

    # Log law reference
    y_log = np.logspace(0.5, np.log10(mkm.re_tau), 100)
    u_log = (1.0 / 0.41) * np.log(y_log) + 5.2
    ax.semilogx(y_log, u_log, 'b--', linewidth=1, alpha=0.7, label='Log law: U⁺ = (1/κ)ln(y⁺) + B')

    # Viscous sublayer
    y_vis = np.linspace(0.1, 5, 50)
    ax.semilogx(y_vis, y_vis, 'g--', linewidth=1, alpha=0.7, label='Viscous sublayer: U⁺ = y⁺')

    ax.set_xlabel('y⁺', fontsize=12)
    ax.set_ylabel('U⁺', fontsize=12)
    ax.set_title(f'Mean Velocity Profile — Re_τ = {mkm.re_tau:.1f}', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(1, mkm.re_tau * 1.1)
    ax.grid(True, alpha=0.3)

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_reynolds_stress(mkm, result, case_name: str, out_path: Path):
    """Plot Reynolds stress profiles."""
    nu = result["derived"]["nu"]
    u_tau = np.sqrt(result["derived"]["accel"] * result["derived"]["half_height"])
    y_w = 0.5
    y = np.asarray(result["profiles"]["y"], dtype=float)
    y_plus = (y - y_w) * u_tau / nu

    r_uu = np.asarray(result["profiles"]["r_uu"], dtype=float)
    r_vv = np.asarray(result["profiles"]["r_vv"], dtype=float)
    r_ww = np.asarray(result["profiles"]["r_ww"], dtype=float)
    r_uv = np.asarray(result["profiles"]["r_uv"], dtype=float)

    half = y_plus <= mkm.re_tau * 1.02
    y_plus_half = y_plus[half]
    uu_plus = r_uu[half] / u_tau**2
    vv_plus = r_vv[half] / u_tau**2
    ww_plus = r_ww[half] / u_tau**2
    uv_plus = -r_uv[half] / u_tau**2  # negative of <u'v'>

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: -<u'v'>+
    ax = axes[0]
    ax.plot(mkm.y_plus, -mkm.r_uv, 'k-', linewidth=2, label='MKM-1999 DNS')
    ax.plot(y_plus_half, uv_plus, 'ro-', markersize=4, linewidth=1, label=f'TensorLBM ({case_name})')
    ax.set_xlabel('y⁺', fontsize=11)
    ax.set_ylabel('-⟨u\'v\'⟩⁺', fontsize=11)
    ax.set_title('Reynolds Shear Stress', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: RMS profiles
    ax = axes[1]
    ax.plot(mkm.y_plus, mkm.u_rms_plus, 'r-', linewidth=2, label='MKM u_rms⁺')
    ax.plot(mkm.y_plus, mkm.v_rms_plus, 'g-', linewidth=2, label='MKM v_rms⁺')
    ax.plot(mkm.y_plus, mkm.w_rms_plus, 'b-', linewidth=2, label='MKM w_rms⁺')
    ax.plot(y_plus_half, np.sqrt(np.maximum(uu_plus, 0)), 'r^--', markersize=3, linewidth=1, label='LBM u_rms⁺')
    ax.plot(y_plus_half, np.sqrt(np.maximum(vv_plus, 0)), 'g^--', markersize=3, linewidth=1, label='LBM v_rms⁺')
    ax.plot(y_plus_half, np.sqrt(np.maximum(ww_plus, 0)), 'b^--', markersize=3, linewidth=1, label='LBM w_rms⁺')
    ax.set_xlabel('y⁺', fontsize=11)
    ax.set_ylabel('RMS⁺', fontsize=11)
    ax.set_title('Turbulence Intensity Profiles', fontsize=12)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Reynolds Stress & Turbulence Intensity — Re_τ = {mkm.re_tau:.1f}', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_anisotropy(mkm, result, case_name: str, out_path: Path):
    """Plot anisotropy signature: u_rms > w_rms > v_rms."""
    nu = result["derived"]["nu"]
    u_tau = np.sqrt(result["derived"]["accel"] * result["derived"]["half_height"])
    y_w = 0.5
    y = np.asarray(result["profiles"]["y"], dtype=float)
    y_plus = (y - y_w) * u_tau / nu

    r_uu = np.asarray(result["profiles"]["r_uu"], dtype=float)
    r_vv = np.asarray(result["profiles"]["r_vv"], dtype=float)
    r_ww = np.asarray(result["profiles"]["r_ww"], dtype=float)

    half = y_plus <= mkm.re_tau * 1.02
    y_plus_half = y_plus[half]
    u_rms = np.sqrt(np.maximum(r_uu[half], 0)) / u_tau
    v_rms = np.sqrt(np.maximum(r_vv[half], 0)) / u_tau
    w_rms = np.sqrt(np.maximum(r_ww[half], 0)) / u_tau

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(mkm.y_plus, mkm.u_rms_plus, 'r-', linewidth=2, label='MKM u_rms⁺')
    ax.plot(mkm.y_plus, mkm.w_rms_plus, 'b-', linewidth=2, label='MKM w_rms⁺')
    ax.plot(mkm.y_plus, mkm.v_rms_plus, 'g-', linewidth=2, label='MKM v_rms⁺')

    ax.plot(y_plus_half, u_rms, 'r^--', markersize=4, linewidth=1, label='LBM u_rms⁺')
    ax.plot(y_plus_half, w_rms, 'b^--', markersize=4, linewidth=1, label='LBM w_rms⁺')
    ax.plot(y_plus_half, v_rms, 'g^--', markersize=4, linewidth=1, label='LBM v_rms⁺')

    ax.set_xlabel('y⁺', fontsize=12)
    ax.set_ylabel('RMS⁺', fontsize=12)
    ax.set_title(f'Anisotropy Signature: u_rms > w_rms > v_rms — Re_τ = {mkm.re_tau:.1f}', fontsize=13)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_gate_summary(gates, out_path: Path):
    """Plot gate evaluation summary."""
    if gates is None:
        return

    gate_names = []
    statuses = []
    for g in gates["gates"]:
        gate_names.append(g["gate"].replace("C", "C"))
        if g["status"] == "pass":
            statuses.append(2)
        elif g["status"] == "fail":
            statuses.append(1)
        else:
            statuses.append(0)

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['green' if s == 2 else ('red' if s == 1 else 'gray') for s in statuses]
    bars = ax.barh(gate_names, statuses, color=colors, edgecolor='black', height=0.5)

    ax.set_xlim(-0.5, 2.5)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Withheld', 'FAIL', 'PASS'])
    ax.set_xlabel('Gate Status', fontsize=12)
    ax.set_title('IN99-N03 Gate Evaluation Summary (Pilot: R180_h32)', fontsize=13)
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for bar, status in zip(bars, statuses):
        label = ['Withheld', 'FAIL', 'PASS'][status]
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                label, va='center', fontsize=10, fontweight='bold')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    print("Generating IN99-N03 comparison plots...")

    mkm = mkm1999.load_profile(180)
    result = _load_case("R180_h32")
    gates = _load_gates()

    if result is None:
        print("ERROR: R180_h32.json not found")
        return

    print("\n1. Mean velocity profile...")
    plot_mean_velocity(mkm, result, "R180_h32", _FIGURES_DIR / "mean_velocity_retau180.png")

    print("\n2. Reynolds stress profiles...")
    plot_reynolds_stress(mkm, result, "R180_h32", _FIGURES_DIR / "reynolds_stress_retau180.png")

    print("\n3. Anisotropy signature...")
    plot_anisotropy(mkm, result, "R180_h32", _FIGURES_DIR / "anisotropy_retau180.png")

    print("\n4. Gate summary...")
    plot_gate_summary(gates, _FIGURES_DIR / "gate_summary.png")

    print("\nAll plots saved to:", _FIGURES_DIR)


if __name__ == "__main__":
    main()
