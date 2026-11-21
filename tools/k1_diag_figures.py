#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K1-DIAG-B figures: crater-depth decomposition against the Santini case-3 data.

Inputs : runs/k1diag_verdict/verdict.json, runs/k1diag_ctrl/result.json,
         runs/k1diag_diag/result.json, runs/k1diag_planes/planes.npz
Outputs: docs/audits/k1_diag_cavity_20261121/figs/*.png
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "docs", "audits", "k1_diag_cavity_20261121", "figs")
os.makedirs(FIGDIR, exist_ok=True)

verdict = json.load(open(os.path.join(ROOT, "runs", "k1diag_verdict", "verdict.json")))
ctrl = json.load(open(os.path.join(ROOT, "runs", "k1diag_ctrl", "result.json")))
diag = json.load(open(os.path.join(ROOT, "runs", "k1diag_diag", "result.json")))

exp = np.array([[p["time_ms"], p["depth_over_D"]] for p in verdict["experiment_points"]])
rows = verdict["diag_rows"]
contact = diag["contact_step"]
t0 = next(r["time_ms"] for r in diag["history"] if r["step"] == contact)
hist = [r for r in diag["history"] if r["step"] >= contact]
t_sim = np.array([r["time_ms"] - t0 for r in hist])
depth = np.array([r["depth_over_D"] for r in hist])

t_diag = np.array([r["time_ms"] - t0 for r in rows])
d_open = np.array([np.nan if r["d_cav_open"] is None else r["d_cav_open"] for r in rows])
d_all = np.array([np.nan if r["d_cav_all"] is None else r["d_cav_all"] for r in rows])
d_drop = np.array([np.nan if r["d_drop"] is None else r["d_drop"] for r in rows])
d_drop_live = np.array([np.nan if r["d_drop_max_live"] is None else r["d_drop_max_live"] for r in rows])
p_cav = np.array([np.nan if r["p_cav"] is None else r["p_cav"] for r in rows])
gap = np.array([np.nan if r["d_cav_gap"] is None else r["d_cav_gap"] for r in rows])
u_back = np.array([np.nan if r["u_back"] is None else r["u_back"] for r in rows])

EXP_PEAK = verdict["verdict"]["experiment_peak_D"]
EXP_PEAK_T = verdict["verdict"]["experiment_peak_time_ms_post_impact"]
TH = verdict["verdict"]["thresholds"]
T_STAR = verdict["verdict"]["t_star_time_ms"] - t0

# ---------------------------------------------------------------- figure 1
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.plot(t_sim, depth, lw=1.8, color="#1f4e79", label="LBM cavity depth (ctrl, depth_over_D)")
ax.plot(exp[:, 0], exp[:, 1], "o", ms=5, color="#c0392b", label="Santini case-3 experiment (18 pts)")
i_exp = int(np.argmax(exp[:, 1]))
ax.annotate(f"experiment peak {exp[i_exp,1]:.4f} D @ {exp[i_exp,0]:.3f} ms",
            xy=(exp[i_exp, 0], exp[i_exp, 1]), xytext=(exp[i_exp, 0] - 7.5, exp[i_exp, 1] + 0.42),
            color="#c0392b", arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.1), fontsize=9)
j = int(np.argmax(depth))
ax.plot([t_sim[j]], [depth[j]], "v", ms=8, color="#1f4e79")
ax.annotate(f"LBM peak {depth[j]:.4f} D @ {t_sim[j]:.3f} ms\n(peak error {abs(depth[j]/exp[i_exp,1]-1)*100:.2f} %)",
            xy=(t_sim[j], depth[j]), xytext=(t_sim[j] + 1.6, depth[j] - 0.45),
            color="#1f4e79", arrowprops=dict(arrowstyle="->", color="#1f4e79", lw=1.1), fontsize=9)
ax.set_xlabel("time after contact (ms)")
ax.set_ylabel("crater depth / D")
ax.set_title("K1-DIAG-B | baseline crater depth vs Santini case-3 experiment")
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig1_baseline_vs_experiment.png"), dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- figure 2
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.plot(t_diag, d_open, lw=1.6, color="#1f4e79", label="d_cav_open  (headspace-connected = benchmark metric)")
ax.plot(t_diag, d_all, lw=1.4, ls="--", color="#e67e22", label="d_cav_all   (ignoring connectivity)")
ax.plot(t_diag, d_drop, lw=1.4, color="#27ae60", label="d_drop      (drop material still in liquid)")
ax.plot(t_diag, d_drop_live, lw=1.1, ls=":", color="#145a32", label="d_drop_max_live")
ax.axhline(EXP_PEAK, color="#c0392b", lw=1.2, label=f"experiment peak {EXP_PEAK:.4f} D")
ax.axhline(TH["match_lo_D"], color="#7f8c8d", lw=0.9, ls=":")
ax.axhline(TH["match_hi_D"], color="#7f8c8d", lw=0.9, ls=":")
ax.axhline(TH["shallow20_D"], color="#7f8c8d", lw=0.9, ls=":")
ax.axvline(T_STAR, color="#8e44ad", lw=1.0, ls="-.")
ax.text(T_STAR + 0.25, 0.06, f"t* = {T_STAR:.3f} ms", color="#8e44ad", fontsize=9)
ax.set_xlabel("time after contact (ms)")
ax.set_ylabel("depth / D")
ax.set_title("K1-DIAG-B | closure split: drop penetration vs trapped-cavity depth")
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=8.5)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig2_decomposition.png"), dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- figure 3
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True)
ax1.plot(t_diag, gap, lw=1.6, color="#e67e22", label="d_cav_gap = d_cav_all − d_cav_open")
ax1.plot(t_diag, p_cav, lw=1.4, color="#8e44ad", label="p_cav = d_cav_open / d_cav_all")
ax1.axvline(T_STAR, color="#8e44ad", lw=1.0, ls="-.")
ax1.set_ylabel("D  /  ratio")
ax1.grid(alpha=0.3)
ax1.legend(loc="upper left", fontsize=8.5)
ax1.set_title("K1-DIAG-B · how much depth the headspace-connected measure loses")
ax2.plot(t_diag, u_back, lw=1.4, color="#2874a6", label="u_back (max upward liquid velocity)")
ax2.axvline(T_STAR, color="#8e44ad", lw=1.0, ls="-.")
ax2.set_xlabel("time after contact (ms)")
ax2.set_ylabel("m/s")
ax2.grid(alpha=0.3)
ax2.legend(loc="upper left", fontsize=8.5)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig3_closure_and_rebound.png"), dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- figure 4 (field)
planes_path = os.path.join(ROOT, "runs", "k1diag_planes", "planes.npz")
if os.path.exists(planes_path):
    data = np.load(planes_path)
    fill = data["fill"]
    x_over_D = data["x_over_D"]
    z_over_D = data["z_over_D"]
    times = data["time_ms"]
    fig, axes = plt.subplots(1, len(times), figsize=(3.0 * len(times), 3.6), sharey=True)
    if len(times) == 1:
        axes = [axes]
    for ax, field, t_ms in zip(axes, fill, times):
        ax.imshow(field.T, origin="lower", cmap="coolwarm_r", vmin=0, vmax=1,
                  extent=[x_over_D[0], x_over_D[-1], z_over_D[0], z_over_D[-1]],
                  aspect="equal", interpolation="nearest")
        ax.axhline(0.0, color="k", lw=0.8, ls="--")
        ax.set_title(f"t = {t_ms:.3f} ms", fontsize=9)
        ax.set_xlabel("x / D")
    axes[0].set_ylabel("z / D  (0 = initial surface)")
    fig.suptitle("K1-DIAG-B | mid-plane liquid fraction (blue = liquid, red = gas)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig4_midplane_fields.png"), dpi=160)
    plt.close(fig)

print("figures written to", FIGDIR)
for name in sorted(os.listdir(FIGDIR)):
    print(" ", name, os.path.getsize(os.path.join(FIGDIR, name)))
