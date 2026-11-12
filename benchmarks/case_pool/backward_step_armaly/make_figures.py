#!/usr/bin/env python3
"""Figures for CASE-BACKWARD-STEP-LBM (matplotlib only, no hand-entered numbers).

    PYTHONPATH=src python3 benchmarks/case_pool/backward_step_armaly/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
CASE = REPO / "benchmarks/case_pool/backward_step_armaly"
ART = REPO / "artifacts" / "case_backward_step_armaly"
FIG = ART / "figures"


def load(tag):
    p = ART / tag / "result.json"
    return json.loads(p.read_text()) if p.is_file() else None


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    refs = json.loads((CASE / "refs.json").read_text())["adopted_numeric_reference"]["points"]
    tags = ["g1_re100", "g1_re200", "g1_re400", "g2_re100", "g2_re200", "g2_re400"]

    # ---- fig1: time-averaged ux field + streamlines (finest grid, Re=200) ----
    for tag in ("g2_re200", "g2_re100", "g2_re400"):
        f = ART / tag / "field.npz"
        if not f.is_file():
            continue
        d = np.load(f)
        ux, solid = d["ux_mean"], d["solid"].astype(bool)
        ny, nx = ux.shape
        ys = np.arange(1, ny - 1)
        fig, ax = plt.subplots(figsize=(13, 3.2), constrained_layout=True)
        X, Y = np.meshgrid(np.arange(nx), np.arange(1, ny - 1))
        umax = float(np.abs(ux[1:-1]).max())
        im = ax.pcolormesh(X, Y, ux[1:-1], cmap="RdBu_r", vmin=-0.25 * umax, vmax=0.25 * umax, shading="auto")
        ax.contour(X, Y, ux[1:-1], levels=[0.0], colors="k", linewidths=0.8)
        ax.set_title(f"{tag}: time-averaged ux (lattice) with ux=0 contour")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, fraction=0.02)
        fig.savefig(FIG / f"fig1_field_{tag}.png", dpi=130)
        plt.close(fig)

    # ---- fig2: time-averaged wall shear along the lower wall ----
    have = [t for t in tags if (ART / t / "wall_shear.csv").is_file()]
    if have:
        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        for tag in have:
            d = np.loadtxt(ART / tag / "wall_shear.csv", delimiter=",", skiprows=1)
            res = load(tag)
            x0 = res["config"]["x_step"]
            s = res["config"]["s"]
            xs = (d[:, 0] - (x0 - 0.5)) / s
            ax.plot(xs, d[:, 1], lw=1.2, label=f"{tag} (Re={res['config']['re']:g})")
            x1 = res["result"]["x1_over_s"]
            if np.isfinite(x1):
                ax.plot([x1], [0.0], "ko", ms=4)
        ax.axhline(0.0, color="grey", lw=0.8, ls="--")
        ax.set_xlabel("x / s  (s = step height, measured from the step face)")
        ax.set_ylabel("wall shear  du_x/dy|wall  (-2u1+3u2-u3)")
        ax.set_xlim(-0.5, 16)
        ax.set_title("Lower-wall shear: reattachment = zero crossing (time averaged)")
        ax.legend(fontsize=8)
        fig.savefig(FIG / "fig2_wall_shear.png", dpi=130)
        plt.close(fig)

    # ---- fig3: reference vs obtained ----
    rows = []
    for tag in tags:
        r = load(tag)
        if not r:
            continue
        re_ = r["config"]["re"]
        ref = refs.get(str(int(re_)))
        if not ref:
            continue
        rows.append((tag, re_, ref["x1_over_s"], r["result"]["x1_over_s"]))
    if rows:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        w = 0.35
        idx = np.arange(len(rows))
        ax1.bar(idx - w / 2, [r[2] for r in rows], w, label="ref (Armaly 1983, adopted)")
        ax1.bar(idx + w / 2, [r[3] for r in rows], w, label="got (LBM)")
        ax1.set_xticks(idx, [r[0] for r in rows], rotation=30, fontsize=8)
        ax1.set_ylabel("X1 / s"); ax1.legend(fontsize=8)
        ax1.set_title("re- attachment length: reference vs obtained")
        err = [100 * (r[3] - r[2]) / r[2] for r in rows]
        colors = ["tab:green" if abs(e) <= 15 else "tab:red" for e in err]
        ax2.bar(idx, err, color=colors)
        ax2.axhline(15, ls="--", c="grey"); ax2.axhline(-15, ls="--", c="grey")
        ax2.set_xticks(idx, [r[0] for r in rows], rotation=30, fontsize=8)
        ax2.set_ylabel("error vs ref [%]"); ax2.set_title("error, 15% tolerance band")
        fig.savefig(FIG / "fig3_ref_vs_got.png", dpi=130)
        plt.close(fig)

    # ---- fig4: grid sensitivity ----
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for re_ in (100.0, 200.0, 400.0):
        vals = [load(f"g{g}_re{int(re_)}") for g in (1, 2)]
        if not all(vals):
            continue
        pts = [(v["derived"]["n_cells"], v["result"]["x1_over_s"]) for v in vals]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=f"Re={re_:g}")
        ref = refs.get(str(int(re_)))
        if ref:
            ax.axhline(ref["x1_over_s"], ls=":", lw=1)
    ax.set_xlabel("cells in domain"); ax.set_ylabel("X1 / s")
    ax.set_title("grid sensitivity (dotted = reference)"); ax.legend(fontsize=8)
    fig.savefig(FIG / "fig4_grid_sensitivity.png", dpi=130)
    plt.close(fig)

    print("figures ->", FIG, sorted(p.name for p in FIG.glob("*.png")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
