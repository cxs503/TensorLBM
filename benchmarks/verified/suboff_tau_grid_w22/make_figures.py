#!/usr/bin/env python3
"""W22 /**/ make_figures.py — figures for the tau x grid matrix (zero hand-typed numbers).

Everything is read from the machine-written artifacts:
  tau_grid_matrix.json, negative_drag_surface.json,
  raw/case_*.npz (force time series), raw_calib/*.npz (transient calibration)

Figures (benchmarks/verified/suboff_tau_grid_w22/figures/):
  fig1_ct_vs_re.png            C_T vs simulated Re for all 18 cases, anchor line
  fig2_negative_surface.png    (tau,N) sign map + C_T heat map
  fig3_ct_vs_tau_per_grid.png  C_T(tau) at each N (C3b monotonicity evidence)
  fig4_force_series.png        F_x / C_T time series of representative cases
  fig5_iso_re_collapse.png     iso-Re groups: same Re, different (tau,N)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
RAW = HERE / "raw"
CALIB = HERE / "raw_calib"
TAU_VALUES = [0.5005, 0.505, 0.510, 0.515, 0.520, 0.530]
GRIDS = [96, 144, 192]
U_LB = 0.05


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    m = json.loads((HERE / "tau_grid_matrix.json").read_text())
    nds = json.loads((HERE / "negative_drag_surface.json").read_text())
    cases = [c for c in m["cases"] if c.get("status") == "OK"]
    if not cases:
        print("[fig] no completed cases, nothing to plot")
        return
    anchor = m["anchor"]

    # ---- fig1: C_T vs Re_sim ---------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for N in GRIDS:
        sel = sorted([c for c in cases if c["N"] == N], key=lambda c: c["Re_sim"])
        if not sel:
            continue
        ax.plot(
            [c["Re_sim"] for c in sel],
            [c["C_T"] for c in sel],
            "o-",
            ms=5,
            label=f"N={N}",
        )
    ax.axhline(anchor["CT_anchor"], color="k", ls="--", lw=1,
               label=f"AN-SUBOFF {anchor['anchor_used']} (Re={anchor['Re_anchor']:.3g})")
    ax.set_xscale("log")
    ax.set_xlabel("Re simulated  (= 3 u_lb N /(tau-0.5))")
    ax.set_ylabel("C_T = F_x /(0.5 rho u^2 pi D L)")
    ax.set_title("SUBOFF bare hull: C_T over the (tau,N) matrix\n"
                 "each case = one point; colour/series = grid level", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_ct_vs_re.png", dpi=150)
    plt.close(fig)

    # ---- fig2: negative-drag surface -------------------------------------
    sign = np.array(
        [[next((s["negative"] for s in nds["cells"] if s["tau"] == tau and s["N"] == N), False)
          for tau in TAU_VALUES] for N in GRIDS],
        dtype=float,
    )
    ct = np.array(
        [[next((s["C_T"] for s in nds["cells"] if s["tau"] == tau and s["N"] == N), np.nan)
          for tau in TAU_VALUES] for N in GRIDS],
        dtype=float,
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    im0 = axes[0].imshow(sign, cmap="RdYlGn_r", vmin=-1, vmax=1, aspect="auto")
    axes[0].set_xticks(range(len(TAU_VALUES)))
    axes[0].set_xticklabels([f"{t:.4f}" for t in TAU_VALUES], rotation=45, fontsize=8)
    axes[0].set_yticks(range(len(GRIDS)))
    axes[0].set_yticklabels([f"N={n}" for n in GRIDS])
    axes[0].set_title("negative-drag surface (red = C_T<0)", fontsize=10)
    for i in range(len(GRIDS)):
        for j in range(len(TAU_VALUES)):
            axes[0].text(j, i, "NEG" if sign[i, j] else "POS", ha="center", va="center", fontsize=8)
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(ct, cmap="viridis", aspect="auto")
    axes[1].set_xticks(range(len(TAU_VALUES)))
    axes[1].set_xticklabels([f"{t:.4f}" for t in TAU_VALUES], rotation=45, fontsize=8)
    axes[1].set_yticks(range(len(GRIDS)))
    axes[1].set_yticklabels([f"N={n}" for n in GRIDS])
    axes[1].set_title("C_T field on the same (tau,N) grid", fontsize=10)
    for i in range(len(GRIDS)):
        for j in range(len(TAU_VALUES)):
            if math.isfinite(ct[i, j]):
                axes[1].text(j, i, f"{ct[i, j]:.4f}", ha="center", va="center", fontsize=7,
                             color="w")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_negative_surface.png", dpi=150)
    plt.close(fig)

    # ---- fig3: C_T(tau) per grid -----------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for N in GRIDS:
        sel = sorted([c for c in cases if c["N"] == N], key=lambda c: c["tau"])
        if not sel:
            continue
        ax.plot([c["tau"] for c in sel], [c["C_T"] for c in sel], "o-", ms=5, label=f"N={N}")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("tau (relaxation time; nu_lb=(tau-0.5)/3)")
    ax.set_ylabel("C_T")
    ax.set_title("C_T vs tau at fixed grid (C3b monotonicity evidence)", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_ct_vs_tau_per_grid.png", dpi=150)
    plt.close(fig)

    # ---- fig4: force time series -----------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    pick = [(0.530, 96), (0.510, 96), (0.5005, 96)]
    for tau, N in pick:
        npz = RAW / f"case_tau{tau:.4f}_N{N}.npz"
        if not npz.exists():
            continue
        d = np.load(npz)
        rescale = float(d["dpS_front"]) / float(d["dpS_wet"])
        axes[0].plot(d["step"], d["cd_total_frontal"] * rescale, lw=1.1,
                     label=f"tau={tau:.4f}, N={N}")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("C_T (instantaneous)")
    axes[0].set_title("force series, N=96", fontsize=10)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    calibs = sorted(CALIB.glob("case_calibN*.npz")) if CALIB.exists() else []
    for npz in calibs:
        d = np.load(npz)
        rescale = float(d["dpS_front"]) / float(d["dpS_wet"])
        axes[1].plot(d["step"], d["cd_total_frontal"] * rescale, lw=1.2,
                     label=f"{npz.stem} (transient calibration)")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("C_T")
    axes[1].set_title("transient calibration (sub-matrix grid, not a matrix case)", fontsize=10)
    axes[1].grid(alpha=0.3)
    if calibs:
        axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_force_series.png", dpi=150)
    plt.close(fig)

    # ---- fig5: iso-Re collapse -------------------------------------------
    iso = [g for g in m["decoupling"]["iso_Re_groups"] if g["n_members"] >= 2]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for c in cases:
        ax.semilogx(c["Re_sim"], c["C_T"], "o", ms=7, mfc="none", mec="0.4")
    for g in iso:
        ax.semilogx([g["Re_sim"]] * g["n_members"],
                    [mm["C_T"] for mm in g["members"]], "s-", ms=6, lw=0.8,
                    label=f"Re={g['Re_sim']:.0f} spread {g['spread_pct']:.1f}%")
    ax.set_xlabel("Re simulated")
    ax.set_ylabel("C_T")
    ax.set_title("iso-Re groups: same Re reached by different (tau,N) pairs", fontsize=10)
    ax.grid(alpha=0.3)
    if iso:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_iso_re_collapse.png", dpi=150)
    plt.close(fig)

    # ---- fig6: window/transient diagram ----------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for c in cases:
        w = c.get("C_T_windows") or {}
        xs = [0, 1, 2]
        ys = [w.get("first_third"), w.get("mid_third"), w.get("last_window")]
        ax.plot(xs, ys, "o-", lw=0.9, ms=4, label=f"tau={c['tau']:.4f},N={c['N']}" if c["N"] == 96 else None)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["first third", "mid third", "last window (reported)"])
    ax.set_ylabel("C_T window mean")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_title("in-run transient indicator (per case, 400-step protocol)", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_transient_windows.png", dpi=150)
    plt.close(fig)

    print(f"[fig] wrote {len(list(FIG.glob('*.png')))} figures to {FIG}")


if __name__ == "__main__":
    main()