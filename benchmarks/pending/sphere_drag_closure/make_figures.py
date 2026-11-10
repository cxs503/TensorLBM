#!/usr/bin/env python3
"""Report figures for node P54-W3-N11.  Every plotted number comes from
``results.json`` / ``dem_verification.json`` / ``field_snapshot.json`` -- no
value is typed in by hand."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_HERE = Path(__file__).resolve().parent
FIG = _HERE / "figures"
FIG.mkdir(exist_ok=True)

RED, BLUE, GREEN, ORANGE, PURPLE, GREY = (
    "#c0392b", "#2c6fb5", "#2e8b57", "#e08a1e", "#7d3c98", "#666666")


def load(name, default=None):
    p = _HERE / name
    return json.loads(p.read_text()) if p.exists() else default


def fig_cd_vs_re(res):
    rows = res["cd_table"]
    if not rows:
        return None
    Re = [r["Re"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    grid = np.logspace(-0.1, 1.9, 200)
    from tensorlbm.drag_closure import (
        cd_stokes, cd_oseen, cd_schiller_naumann, cd_clift_gauvin, cd_oseen_bridge,
    )
    for f, lab, col, ls in (
        (cd_stokes, "Stokes $24/Re$", GREY, ":"),
        (cd_oseen, "Oseen $(24/Re)(1+3Re/16)$", BLUE, "--"),
        (cd_schiller_naumann, "Schiller-Naumann", GREEN, "-."),
        (cd_clift_gauvin, "Clift-Gauvin (standard curve)", ORANGE, "-"),
        (cd_oseen_bridge, "bridged closure (this work)", RED, "-"),
    ):
        ax.plot(grid, [f(x) for x in grid], ls, color=col, label=lab, lw=1.6)
    ax.plot(Re, [r["cd_lbm_lambda_inf"] for r in rows], "o", ms=7, mfc="white",
            mec="k", mew=1.4, label="LBM, $\\lambda=R/L\\to0$ (this work)")
    ax.plot(Re, [r["cd_lbm_lam_1_8"] for r in rows], "s", ms=4, color="k",
            alpha=0.5, label="LBM, raw $\\lambda=1/8$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Re")
    ax.set_ylabel("$C_D$")
    ax.set_title("Sphere drag: LBM vs analytic hierarchy (Re = 1 … 50)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    p = FIG / "fig1_cd_vs_re.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def fig_confinement(res):
    conf = res["confinement"]
    keys = [k for k in conf if conf[k]["points"]]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(keys)))
    for col, k in zip(cols, keys):
        pts = conf[k]["points"]
        lam = [p["lambda"] for p in pts]
        cd = [p["cd"] for p in pts]
        ax.plot(lam, cd, "o", color=col, ms=6)
        fit = conf[k]["fit_lambda"]
        if fit:
            xs = np.linspace(0.0, max(lam) * 1.08, 20)
            ax.plot(xs, [fit["intercept"] + fit["slope"] * x for x in xs], "-",
                    color=col, lw=1.1, alpha=0.9,
                    label=f"Re={k}: $C_D(\\lambda\\to0)$={fit['intercept']:.3f} "
                          f"$R^2$={fit['r2']:.3f}")
            ax.plot([0.0], [fit["intercept"]], "*", color=col, ms=11)
    ax.set_xlabel("$\\lambda = R/L$")
    ax.set_ylabel("$C_D$ (momentum balance)")
    ax.set_title("Confinement study at D = 12 lattices")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    p = FIG / "fig2_confinement.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def fig_resolution(res):
    rsl = res["resolution"]
    keys = [k for k in rsl if rsl[k].get("fit_inv_D")]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    cols = plt.cm.plasma(np.linspace(0.05, 0.75, max(1, len(keys))))
    for col, k in zip(cols, keys):
        pts = rsl[k]["points"]
        x = [p["one_over_D"] for p in pts]
        cd = [p["cd"] for p in pts]
        ax.plot(x, cd, "o", color=col, ms=6)
        fit = rsl[k]["fit_inv_D"]
        xs = np.linspace(0.0, max(x) * 1.1, 20)
        ax.plot(xs, [fit["intercept"] + fit["slope"] * v for v in xs], "-",
                color=col, lw=1.2,
                label=(f"Re={k}: $C_D(1/D\\to0)$={fit['intercept']:.3f}, "
                       + (f"$R^2$={fit['r2']:.4f}" if fit.get('r2') is not None
                          else f"{fit['n']} points (no $R^2$)")))
        for p, xi, yi in zip(pts, x, cd):
            ax.annotate(f"D={p['D']:g}", (xi, yi), textcoords="offset points",
                        xytext=(4, 6), fontsize=7, color=col)
    ax.set_xlabel("$1/D$  (lattice cells)")
    ax.set_ylabel("$C_D$ at $\\lambda = 1/6$")
    ax.set_title("Resolution convergence of the voxel sphere")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIG / "fig3_resolution.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def fig_errors(res):
    rows = res["cd_table"]
    if not rows:
        return None
    Re = [r["Re"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for key, lab, col, mk in (
        ("stokes", "Stokes", GREY, "o"),
        ("oseen", "Oseen", BLUE, "s"),
        ("schiller_naumann", "Schiller-Naumann", GREEN, "^"),
        ("bridge", "bridged closure", RED, "D"),
    ):
        ax.plot(Re, [r[f"err_pct_{key}"] for r in rows], mk + "-", color=col,
                ms=5, lw=1.3, label=lab)
    ax.axhspan(-5, 5, color="#cccccc", alpha=0.35, label="±5 % band")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Re")
    ax.set_ylabel("error vs LBM, %")
    ax.set_title("Closure accuracy against the LBM drag (Re = 1 … 50)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIG / "fig4_closure_error.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def fig_field(snap):
    if not snap:
        return None
    ux = np.array(snap["ux"])
    solid = np.array(snap["solid"]).astype(bool)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    umax = np.abs(ux).max()
    im = ax.imshow(ux, origin="lower", cmap="RdBu_r", vmin=-umax, vmax=umax)
    ax.contour(solid, levels=[0.5], colors="k", linewidths=1.2)
    ax.set_title(f"$u_x$ mid-plane, D={snap['D']:g}, Re={snap['Re_target']:g}, "
                 f"L={snap['L']}, step={snap['steps']}")
    ax.set_xlabel("x [lattice]")
    ax.set_ylabel("y [lattice]")
    fig.colorbar(im, ax=ax, label="$u_x$")
    fig.tight_layout()
    p = FIG / "fig5_field_snapshot.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def fig_dem(dem):
    rows = dem["rows"]
    d = np.array([r["d_m"] for r in rows]) * 1e3
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(d, [r["u_reference_clift_gauvin"] for r in rows], "-", color=ORANGE,
            lw=1.8, label="standard curve (Clift-Gauvin)")
    ax.plot(d, [r["u_reference_bridge"] for r in rows], "-", color=RED, lw=1.4,
            label="bridged closure")
    ax.plot(d, [r["u_dem_bridge"] for r in rows], "o", mfc="white", mec="k",
            mew=1.4, ms=7, label="DEM solver, drag_model=oseen_bridge")
    ax.plot(d, [r["u_dem_stokes"] for r in rows], "s", color=GREY, ms=5,
            label="DEM solver, drag_model=stokes (the m7441 model)")
    ax.set_xlabel("particle diameter [mm]")
    ax.set_ylabel("terminal settling velocity [m/s]")
    ax.set_title("DEM coupling verification (quartz in water)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax2 = ax.twinx()
    ax2.set_ylabel("Re at terminal velocity")
    ax2.plot(d, [r["Re_terminal"] for r in rows], ":", color=PURPLE, lw=1.0)
    ax2.set_yscale("log")
    fig.tight_layout()
    p = FIG / "fig6_dem_settling.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def main() -> int:
    res = load("results.json")
    dem = load("dem_verification.json")
    snap = load("field_snapshot.json")
    made = []
    if res:
        made += [p for p in (fig_cd_vs_re(res), fig_confinement(res),
                             fig_resolution(res), fig_errors(res)) if p]
    if dem:
        made.append(fig_dem(dem))
    f = fig_field(snap)
    if f:
        made.append(f)
    for p in made:
        print("wrote", p)
    return 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(_HERE.parents[2] / "src"))
    raise SystemExit(main())