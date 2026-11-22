#!/usr/bin/env python3
"""K1-CAVCLOS-R2 report figures (matplotlib, programmatic).

Usage: python3 tools/k1r2_figures.py <result.json> [--alpha 0.5] [--out-dir runs/K1-CAVCLOS-R2]
Produces:
  fig_depth_curve.png  - depth-over-time (pc) model vs experiment + pre-reg band
  fig_field_planes.png - mid-plane fill/uz at 4 registered post-contact times
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP_PEAK_D = 1.857434
EXP_PEAK_MS_PC = 10.394
CTRL_PEAK_D = 1.435220
CTRL_PEAK_MS_PC = 10.543339  # CMB3 card ctrl (same lane, bit-parity verified)
PREREG_LO, PREREG_HI = 1.50, 1.70


def main():
    argv = sys.argv[1:]
    path = argv[0]
    alpha = 0.5
    if "--alpha" in argv:
        alpha = float(argv[argv.index("--alpha") + 1])
    out_dir = "runs/K1-CAVCLOS-R2"
    if "--out-dir" in argv:
        out_dir = argv[argv.index("--out-dir") + 1]

    res = json.load(open(path))
    h = res["history"]
    sc = res["scales"]
    dt_ms = 1000.0 * sc["dt_s"]
    contact = res.get("contact_step") or 0
    t_pc = np.array([r["time_ms"] - contact * dt_ms for r in h])
    depth = np.array([r.get("depth_over_D", np.nan) for r in h])

    # ---- fig 1: depth curve ----
    exp = np.genfromtxt(
        "benchmarks/water_entry/single_drop/fig5_experiment.csv",
        delimiter=",", names=True,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_pc, depth, lw=1.6, color="tab:blue",
            label=f"model K1-CAVCLOS-R2 (alpha={alpha})")
    ax.plot(exp["time_ms"], exp["depth_over_D"], "o", ms=5, color="tab:red",
            label="experiment (fig5, 18 pts, pc)")
    ax.axhline(PREREG_LO, ls=":", lw=1, color="gray")
    ax.axhline(PREREG_HI, ls=":", lw=1, color="gray")
    ax.axhspan(PREREG_LO, PREREG_HI, color="tab:green", alpha=0.12,
              label="pre-reg band [1.50, 1.70] D")
    ax.axvline(EXP_PEAK_MS_PC, ls="--", lw=1, color="tab:red", alpha=0.6,
              label="exp peak time 10.394 ms pc")
    ax.plot([CTRL_PEAK_MS_PC], [CTRL_PEAK_D], "s", ms=8, mfc="none",
            color="tab:orange", label="ctrl anchor 1.435220 D (CMB3 lane)")
    pk_i = int(np.nanargmax(depth))
    ax.plot([t_pc[pk_i]], [depth[pk_i]], "*", ms=14, color="tab:blue",
            label=f"model peak {depth[pk_i]:.4f} D @ {t_pc[pk_i]:.3f} ms pc")
    ax.set_xlabel("time (ms, post-contact)")
    ax.set_ylabel("crater depth (D)")
    ax.set_title(f"K1-CAVCLOS-R2 alpha={alpha}: depth vs time (cells=8, full window)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig_depth_curve.png", dpi=140)
    plt.close(fig)

    # ---- fig 2: field planes ----
    try:
        import os
        npz = np.load(f"{out_dir}/planes.npz")
        times = npz["time_ms"]
        fill = npz["fill"]
        uz = npz["uz_m_s"]
        x = npz["x_over_D"]
        z = npz["z_over_D"]
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for i, axi in enumerate(axes.ravel()):
            if i < len(times):
                im = axi.contourf(x, z, fill[i], levels=12, cmap="RdYlBu")
                fig.colorbar(im, ax=axi, shrink=0.8)
                axi.set_title(f"fill @ {times[i]:.3f} ms pc")
                axi.set_xlabel("x/D"); axi.set_ylabel("z/D")
        fig.suptitle(f"K1-CAVCLOS-R2 alpha={alpha}: mid-plane fill (post-contact times)")
        fig.tight_layout()
        fig.savefig(f"{out_dir}/fig_field_planes.png", dpi=140)
        plt.close(fig)
    except Exception as e:  # planes optional
        print("planes figure skipped:", e)
    print("figures written to", out_dir)


if __name__ == "__main__":
    main()
