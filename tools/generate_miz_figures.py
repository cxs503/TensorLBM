#!/usr/bin/env python3
"""Generate figures for IN101-N04 MIZ coupling report.

Produces 6 figures:
1. Wave attenuation profile (Hs vs x for different C_floe)
2. Floe size distribution P(a_floe) + Squire window
3. Roll RAO comparison (OW / MIZ / continuous ice)
4. Resistance components (bar chart)
5. 6-DOF time history (roll, heave, pitch)
6. Energy budget (bar chart)
"""

from __future__ import annotations

import base64
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tensorlbm.miz_coupling import (
    ShipParams,
    WaveSpectrum,
    characteristic_floe_size,
    floe_size_distribution,
    run_miz_simulation,
    verify_floe_size_scaling,
)
from tensorlbm.miz_6dof import run_6dof_simulation


def save_fig(fig, out_dir: Path, name: str, caption: str) -> dict:
    """Save figure to PNG and return a dict with base64 encoding."""
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('ascii')
    return {'name': name, 'caption': caption, 'png_base64': b64, 'path': str(path)}


def main() -> None:
    out_dir = Path("runs/miz_sweep/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    ship = ShipParams()
    figures = []

    # ------------------------------------------------------------------
    # Fig 1: Wave attenuation profile
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for C, color in zip([0.1, 0.3, 0.5, 0.7, 0.9],
                         ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']):
        res = run_miz_simulation(Hs=1.5, Tp=8.0, C_floe=C, ship=ship)
        ax.plot(res.x_array / 1000.0, res.Hs_profile, label=f'C_floe={C}', color=color, lw=2)
    ax.set_xlabel('Distance into ice field [km]')
    ax.set_ylabel('Significant wave height Hs [m]')
    ax.set_title('Wave attenuation in MIZ (Hs=1.5 m, Tp=8 s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    figures.append(save_fig(fig, out_dir, 'fig1_wave_attenuation',
                            'Wave attenuation in MIZ for different floe concentrations'))
    print("Fig 1: wave attenuation")

    # ------------------------------------------------------------------
    # Fig 2: Floe size distribution + Squire window
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    a_arr, P_arr = floe_size_distribution(a_mean=5.0)
    ax1.plot(a_arr, P_arr, 'b-', lw=2)
    ax1.axvline(5.0, color='r', ls='--', label='a_floe=5 m')
    ax1.set_xlabel('Floe diameter a [m]')
    ax1.set_ylabel('P(a) [1/m]')
    ax1.set_title('Floe size distribution (log-normal)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    Tp_arr, a_arr2, ratios = verify_floe_size_scaling()
    ax2.plot(Tp_arr, ratios, 'b-', lw=2)
    ax2.axhline(0.05, color='g', ls='--')
    ax2.axhline(0.2, color='g', ls='--')
    ax2.fill_between(Tp_arr, 0.05, 0.2, alpha=0.1, color='g', label='Squire window')
    ax2.set_xlabel('Peak period Tp [s]')
    ax2.set_ylabel('a_floe / λ_peak [-]')
    ax2.set_title('Squire scaling window')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    figures.append(save_fig(fig, out_dir, 'fig2_floe_size',
                            'Floe size distribution and Squire scaling window'))
    print("Fig 2: floe size")

    # ------------------------------------------------------------------
    # Fig 3: Roll RAO comparison
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    res = run_miz_simulation(Hs=1.5, Tp=8.0, C_floe=0.8, a_floe_input=15.0, h_ice=1.2, ship=ship)
    ax.plot(res.freq_rao, res.RAO_roll_ow, 'b-', lw=2, label='Open water')
    ax.plot(res.freq_rao, res.RAO_roll_miz, 'r-', lw=2, label='MIZ (C=0.8)')
    ax.plot(res.freq_rao, res.RAO_roll_cont, 'g-', lw=2, label='Continuous ice')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Roll RAO [rad/rad]')
    ax.set_title('Roll RAO: open water vs MIZ vs continuous ice')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    figures.append(save_fig(fig, out_dir, 'fig3_rao_comparison',
                            'Roll RAO: open water vs MIZ vs continuous ice'))
    print("Fig 3: RAO comparison")

    # ------------------------------------------------------------------
    # Fig 4: Resistance components
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ['Open water', 'Wave added', 'Ice', 'Total']
    values = [res.R_open_water, res.R_wave, res.R_ice, res.R_total]
    colors = ['#2196F3', '#4CAF50', '#F44336', '#9C27B0']
    bars = ax.bar(categories, values, color=colors)
    ax.set_ylabel('Resistance [N]')
    ax.set_title('Resistance components (C_floe=0.8, a_floe=15 m, h=1.2 m)')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.0f}', ha='center', va='bottom')
    figures.append(save_fig(fig, out_dir, 'fig4_resistance_components',
                            'Resistance components in MIZ'))
    print("Fig 4: resistance components")

    # ------------------------------------------------------------------
    # Fig 5: 6-DOF time history
    # ------------------------------------------------------------------
    res6 = run_6dof_simulation(Hs=1.5, Tp=8.0, C_floe=0.5, a_floe=5.0,
                               h_ice=0.5, duration=300.0, dt=0.05)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(res6.t, np.degrees(res6.xi[3]), 'r-', lw=1)
    axes[0].set_ylabel('Roll [deg]')
    axes[0].set_title(f'6-DOF time history (C_floe=0.5, {res6.n_impacts} ice impacts)')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(res6.t, res6.xi[2], 'b-', lw=1)
    axes[1].set_ylabel('Heave [m]')
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(res6.t, np.degrees(res6.xi[4]), 'g-', lw=1)
    axes[2].set_ylabel('Pitch [deg]')
    axes[2].set_xlabel('Time [s]')
    axes[2].grid(True, alpha=0.3)
    figures.append(save_fig(fig, out_dir, 'fig5_6dof_time_history',
                            '6-DOF time history with ice impacts'))
    print("Fig 5: 6-DOF time history")

    # ------------------------------------------------------------------
    # Fig 6: Energy budget
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ['Wave input', 'Ice breaking', 'Dissipation', 'Ship motion']
    E_ship_area = res6.E_ship_kinetic / (ship.L * ship.B)
    values = [res6.E_wave_input, res6.E_ice_breaking, res6.E_dissipated, E_ship_area]
    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']
    bars = ax.bar(categories, values, color=colors)
    ax.set_ylabel('Energy [J/m²]')
    ax.set_title('Energy budget (per unit area)')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.1f}', ha='center', va='bottom')
    figures.append(save_fig(fig, out_dir, 'fig6_energy_budget',
                            'Energy budget (per unit area)'))
    print("Fig 6: energy budget")

    # Save figure metadata (without base64 to keep the file small)
    fig_meta = [{'name': f['name'], 'caption': f['caption'], 'path': f['path']}
                for f in figures]
    with open(out_dir / "figures.json", 'w') as f:
        json.dump(fig_meta, f, indent=2)
    print(f"Figures saved to {out_dir}")


if __name__ == "__main__":
    main()
