#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K2-BUDGET-B 图表：能量去向、逐项收支、通道份额、坑形剖面、动量账。

所有数字从 runs/k2budget_verdict/verdict.json 程序化读取（零手抄）。
用法：
    python3 tools/k2_budget_figures.py --tree .
输出：
    docs/audits/k2_budget_20261121/figs/*.png
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "docs", "audits", "k2_budget_20261121", "figs")


def load(tree):
    with open(os.path.join(tree, "runs", "k2budget_verdict", "verdict.json")) as handle:
        return json.load(handle)


def fig_ke(v, out):
    rows = v["rows"]
    t = np.array([r["time_ms"] for r in rows])
    fig, ax = plt.subplots(2, 1, figsize=(8.4, 6.4), sharex=True)
    series = [
        ("ke_liq", "total liquid KE $E_{KE}$", "#1f77b4"),
        ("ke_axial", "axial (column) $E_z$", "#d62728"),
        ("ke_lat", "lateral $E_\\perp$", "#2ca02c"),
        ("ke_drop", "drop-material residual $E_{drop}$", "#9467bd"),
    ]
    for key, label, color in series:
        ax[0].plot(t, [r[key] for r in rows], label=label, color=color, lw=1.6)
    ax[0].set_ylabel("kinetic energy (lattice energy units)")
    ax[0].legend(fontsize=8, ncol=2)
    ax[0].grid(alpha=.3)
    ax[0].set_title("Where the impact energy sits in the liquid (contact -> depth peak)")
    for key, label, color in [
        ("cav_volume", "cavity volume (lattice cells)", "#ff7f0e"),
        ("area_total", "total interface area $\\int|\\nabla\\phi|$", "#8c564b"),
    ]:
        ax[1].plot(t, [r[key] for r in rows], label=label, color=color, lw=1.4)
    ax[1].set_xlabel("time (ms, from t=0)")
    ax[1].set_ylabel("geometry")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3)
    for axis in ax:
        axis.axvline(v["window"]["t_star_time_ms"], color="k", ls="--", lw=1)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_ledger(v, out):
    rows = v["rows"]
    t = np.array([r["time_ms"] for r in rows])
    e0 = rows[0]["ke_liq"]
    sigma = None
    fig, ax = plt.subplots(2, 1, figsize=(8.4, 6.4), sharex=True)
    # cumulative terms
    def cum(key):
        vals = [r[key] for r in rows]
        return np.cumsum(vals) - 0.5 * np.array(vals[0])

    ax[0].plot(t, cum("grav_power"), label="gravity work $W_g$", lw=1.5)
    ax[0].plot(t, cum("diss_rate"), label="viscous dissipation $E_{diss}$", lw=1.5)
    ax[0].plot(t, cum("conv_loss"), label="conversion loss $E_{conv}$", lw=1.5)
    ax[0].plot(t, cum("conv_gain"), label="conversion gain", lw=1.2, ls=":")
    ax[0].plot(t, cum("gas_power_wall"), label="cavity-wall local pressure work (local form)", lw=1.2, ls="--")
    ax[0].set_ylabel("accumulated (lattice energy)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)
    ax[0].set_title("Time-integrated budget terms (contact -> depth peak)")
    # running residual
    ke = np.array([r["ke_liq"] for r in rows])
    res = (ke - ke[0]) - (cum("grav_power") - cum("diss_rate") + cum("conv_gain") - cum("conv_loss"))
    ax[1].plot(t, res, color="crimson", label="residual $R$ (without the atmospheric term)")
    ax[1].axhline(0, color="k", lw=.8)
    ax[1].axhline(0.1 * e0, color="grey", ls="--", lw=1, label="closure gate +-10 % of $E_{scale}$")
    ax[1].axhline(-0.1 * e0, color="grey", ls="--", lw=1)
    ax[1].set_xlabel("time (ms, from t=0)")
    ax[1].set_ylabel("residual (lattice energy)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3)
    for axis in ax:
        axis.axvline(v["window"]["t_star_time_ms"], color="k", ls="--", lw=1)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_shares(v, out):
    ch = v["channels"]
    names = [
        ("2_surface_energy_increment", "2 surface energy"),
        ("3_interface_viscous_dissipation", "3 viscous dissipation"),
        ("4_lateral_jet_kinetic_energy", "4 lateral jet KE"),
        ("5_droplet_residual_kinetic_energy", "5 droplet residual KE"),
        ("6_axial_water_column_kinetic_energy", "6 axial column KE"),
        ("7_mass_exchange_conversion", "7 mass-exchange conversion"),
        ("8_gravity_work", "8 gravity work"),
    ]
    labels, values, colors = [], [], []
    palette = ["#9467bd", "#d62728", "#2ca02c", "#8c564b", "#1f77b4", "#ff7f0e", "#7f7f7f"]
    for (key, label), color in zip(names, palette):
        entry = ch[key]
        if key == "4_lateral_jet_kinetic_energy":
            value = entry["ke_lat_gain"]
        else:
            value = entry.get("share", 0.0)
            if key == "8_gravity_work":
                value = entry["work_lattice"]
            elif key == "2_surface_energy_increment":
                value = entry["cavity_wall_sigma_dA_lattice"]
            elif key == "3_interface_viscous_dissipation":
                value = entry["total_lattice"]
            elif key == "5_droplet_residual_kinetic_energy":
                value = entry["ke_drop_final"]
            elif key == "6_axial_water_column_kinetic_energy":
                value = entry["ke_axial_final"]
            elif key == "7_mass_exchange_conversion":
                value = entry["conv_loss_lattice"]
        labels.append(label)
        values.append(value / v["closure"]["E_scale_lattice"])
        colors.append(color)
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    bars = ax.bar(range(len(values)), values, color=colors)
    ax.axhline(0.5, color="crimson", ls="--", lw=1.2, label="dominance gate 50 %")
    ax.axhline(0.0, color="k", lw=.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("share of the impact energy $E_{scale}$")
    ax.set_title("K2-BUDGET-B: channel shares at the depth peak (E_scale = liquid KE at contact)")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:+.1%}",
                ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_momentum(v, out):
    rows = v["rows"]
    t = np.array([r["time_ms"] for r in rows])
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(t, [r["p_liq_z"] for r in rows], label="total liquid momentum $P_z$", lw=1.6)
    ax.plot(t, [r["p_drop_z"] for r in rows], label="drop-material momentum $P^{drop}_z$", lw=1.5)
    ax.plot(t, [r["gas_force_wall_z"] for r in rows], label="cavity-wall gas force $F_{gas,z}$", lw=1.2)
    ax.plot(t, [r["conv_mom_z"] for r in rows], label="conversion momentum flux", lw=1.0, ls=":")
    ax.axhline(0, color="k", lw=.8)
    ax.set_xlabel("time (ms, from t=0)")
    ax.set_ylabel("momentum / force (lattice units)")
    ax.set_title("Momentum bookkeeping (contact -> depth peak)")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    ax.axvline(v["window"]["t_star_time_ms"], color="k", ls="--", lw=1)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_crater(tree, v, out):
    path = os.path.join(tree, "runs", "k2budget_planes", "planes.npz")
    if not os.path.exists(path):
        return False
    data = np.load(path)
    times = data["time_ms"]
    fill = data["fill"]
    x = data["x_over_D"]
    z = data["z_over_D"]
    exp_peak = v["experiment_comparison"]["experiment_peak_D"]
    lbm_peak = v["experiment_comparison"]["lbm_peak_D"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    for k, t in enumerate(times):
        field = fill[k]
        depth = np.zeros_like(x)
        for j in range(field.shape[1]):
            col = field[:, j]
            idx = np.where(col <= 0.5)[0]
            if len(idx) == 0 or idx[0] == 0:
                continue
            i = int(idx[0])
            lo, hi = float(col[i - 1]), float(col[i])
            frac = 0.0 if lo == hi else (lo - 0.5) / (lo - hi)
            z_if = z[i - 1] + frac * (z[i] - z[i - 1])
            depth[j] = -float(z_if)
        axes[0].plot(x, depth, lw=1.2, label=f"t={t:.2f} ms")
    axes[0].axhline(exp_peak, color="crimson", ls="--", lw=1.2, label=f"experiment peak {exp_peak:.4f} D")
    axes[0].axhline(lbm_peak, color="navy", ls="--", lw=1.2, label=f"LBM peak {lbm_peak:.4f} D")
    axes[0].set_xlabel("x / D")
    axes[0].set_ylabel("crater depth / D")
    axes[0].set_title("mid-plane crater profile")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=.3)
    # energy requirement bar
    comp = v["experiment_comparison"]
    labels = ["impact energy\n$E_{scale}$", "gravitational PE need\n$\\Delta E_{PE}$", "extra axial KE\n(declared assumption)"]
    vals = [v["closure"]["E_scale_SI_J"],
            comp["gravitational_PE_requirement_J"], comp["extra_axial_KE_needed_J"]]
    axes[1].bar(labels, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("energy (J)")
    axes[1].set_title("Energy comparison: experimental depth vs impact energy")
    for i, value in enumerate(vals):
        axes[1].text(i, value, f"{value:.3e}", ha="center", va="bottom", fontsize=8)
    axes[1].grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", default=ROOT)
    args = ap.parse_args()
    tree = os.path.abspath(args.tree)
    os.makedirs(FIGDIR, exist_ok=True)
    v = load(tree)
    fig_ke(v, os.path.join(FIGDIR, "fig1_ke_channels.png"))
    fig_ledger(v, os.path.join(FIGDIR, "fig2_ledger_terms.png"))
    fig_shares(v, os.path.join(FIGDIR, "fig3_channel_shares.png"))
    fig_momentum(v, os.path.join(FIGDIR, "fig4_momentum.png"))
    have_crater = fig_crater(tree, v, os.path.join(FIGDIR, "fig5_crater_and_energy.png"))
    print("figures written to", FIGDIR, "crater_figure:", have_crater)


if __name__ == "__main__":
    main()
