#!/usr/bin/env python3
"""Render the IN70-N01 figures from the JSON bundle + stashed numpy fields.

    PYTHONPATH=src python scripts/plot_dg_verification.py

Every figure is produced from ``artifacts/IN70_N01/*.json`` and
``artifacts/IN70_N01/fields.npz`` (written by ``run_dg_verification.py``) -- no
number is typed by hand.
"""

from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(HERE, "artifacts", "IN70_N01")
FIG = os.path.join(ART, "figures")


def _load(name: str):
    path = os.path.join(ART, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _fields():
    """Merge every ``fields_<bench>.npz`` written by the runner."""
    merged: dict[str, np.ndarray] = {}
    paths = sorted(glob.glob(os.path.join(ART, "fields_*.npz"))) + [
        p for p in [os.path.join(ART, "fields.npz")] if os.path.exists(p)
    ]
    for path in paths:
        with np.load(path) as data:
            for k in data.files:
                merged[k] = data[k]
    return merged or None


def fig_convergence(cone, mms, gauss=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, data, title in (
        (axes[0], mms, "MMS  u=sin(πx)sin(πy)e^{-t}, ν=0.05"),
        (axes[1], cone, "rotating cone, t=2π (LeVeque)"),
    ):
        if not data:
            continue
        for key, style in (("P1", "o-"), ("P2", "s-"), ("P3", "^-")):
            rows = [r for r in data["rows"] if f"P{ r['degree']}" == key]
            if not rows:
                continue
            h = [r["h"] for r in rows]
            e = [r["L2"] for r in rows]
            ax.loglog(h, e, style, label=f"{key} measured (order "
                                         f"{data['orders'][key]['L2']['order_last_pair']:.2f})")
            if "theory" in data["orders"][key] and rows:
                th = data["orders"][key]["theory"]
                ref = e[0] * (np.array(h) / h[0]) ** th
                ax.loglog(h, ref, "k--", lw=0.8, label=f"{key} O(h^{th})")
        if gauss:
            g = gauss
            for key, style in (("P1", "o:"), ("P2", "s:")):
                rows = [r for r in g["rows"] if f"P{ r['degree']}" == key]
                if not rows:
                    continue
                ax.loglog([r["h"] for r in rows], [r["L2"] for r in rows], style,
                          label=f"{key} smooth Gaussian")
        ax.set_xlabel("h = 1/N")
        ax.set_ylabel("L2 error")
        ax.set_title(title, fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    out = os.path.join(FIG, "fig1_convergence.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_cone_fields(fl):
    if fl is None or "cone_cone_P2_n64_initial" not in fl:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, key, title in (
        (axes[0], "cone_cone_P2_n64_initial", "P2, N=64  t=0"),
        (axes[1], "cone_cone_P2_n64_final", "P2, N=64  t=2π (numerical)"),
        (axes[2], "cone_cone_P1_n64_final", "P1, N=64  t=2π (numerical)"),
    ):
        if key not in fl:
            continue
        a = fl[key]
        im = ax.imshow(a.mean(axis=(-1, -2)), origin="lower", extent=(0, 1, 0, 1),
                       vmin=0, vmax=1, cmap="viridis")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("LeVeque rotating cone: one full revolution, DG upwind", fontsize=10)
    fig.tight_layout()
    out = os.path.join(FIG, "fig2_cone_fields.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_sod(sod, iso, fl):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    if fl is not None and "sod_exact_x" in fl:
        x = fl["sod_exact_x"]
        axes[0].plot(x, fl["sod_exact_rho"], "k-", lw=1.0, label="exact Riemann")
        for key, style, lab in (("sod_P1_n200_rho", "b--", "DG P1 N=200"),
                                ("sod_P2_n200_rho", "r-.", "DG P2 N=200")):
            if key in fl:
                xc = fl[key.replace("_rho", "_xc")]
                axes[0].plot(xc, fl[key], style, lw=1.0, label=lab)
        axes[0].set_title("Sod, γ=1.4, t=0.2 (LF flux: honest FAIL)", fontsize=9)
        axes[0].legend(fontsize=7)
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("density")
    if fl is not None and "iso_P1_t100_xc" in fl:
        xc = fl["iso_P1_t100_xc"]
        axes[1].plot(xc, fl["iso_P1_t100_rho_exact"], "k-", lw=1.0, label="isothermal exact")
        for key, style, lab in (("iso_P1_t100_rho", "b--", "DG P1"),
                                ("iso_P2_t100_rho", "r-.", "DG P2")):
            if key in fl:
                axes[1].plot(fl[key.replace("_rho", "_xc")], fl[key], style, lw=1.0, label=lab)
        axes[1].set_xlim(900, 1200)
        axes[1].set_title("isothermal Sod (p=ρ/3), t=100", fontsize=9)
        axes[1].set_xlabel("x (lattice units)")
        axes[1].legend(fontsize=7)
    if iso:
        rows = iso["rows"]
        for p, style in ((1, "o-"), (2, "s-")):
            r = [x for x in rows if x["degree"] == p]
            if not r:
                continue
            axes[2].plot([x["t"] for x in r], [100 * x["shock_pos_err_rel"] for x in r], style,
                         label=f"DG P{p} vs exact")
        lbm_err = []
        lbm_t = []
        for x in rows:
            if x.get("lbm") and x["lbm"].get("shock_x"):
                lbm_t.append(x["t"])
                lbm_err.append(
                    100 * abs(x["lbm"]["shock_x"] - x["shock_x_exact"]) / x["shock_x_exact"]
                )
        if lbm_t:
            axes[2].plot(lbm_t, lbm_err, "k^--", label="verified LBM D2Q9")
        axes[2].axhline(2.0, color="r", ls=":", lw=1.0, label="2% gate")
        axes[2].set_xlabel("t (lattice units)")
        axes[2].set_ylabel("shock position error [%]")
        axes[2].set_title("DG ⇄ LBM ⇄ exact (isothermal)", fontsize=9)
        axes[2].legend(fontsize=7)
    fig.tight_layout()
    out = os.path.join(FIG, "fig3_sod.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_square(square, fl):
    if not square or fl is None:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for n, style in ((100, "-"), (200, "--"), (400, ":")):
        if f"square_lim_n{n}_x" in fl:
            axes[0].step(fl[f"square_lim_n{n}_x"], fl[f"square_lim_n{n}_mean"], style,
                         where="mid", lw=1.2, label=f"TVB minmod N={n}")
            axes[1].step(fl[f"square_unlim_n{n}_x"], fl[f"square_unlim_n{n}_mean"], style,
                         where="mid", lw=1.2, label=f"unlimited N={n}")
    for ax, title in ((axes[0], "limiter ON: no new extrema (TVD)"),
                      (axes[1], "limiter OFF: Gibbs oscillations")):
        ax.axhline(1.0, color="k", lw=0.6)
        ax.axhline(0.0, color="k", lw=0.6)
        ax.set_ylim(-0.15, 1.15)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x")
        ax.legend(fontsize=7)
    fig.tight_layout()
    out = os.path.join(FIG, "fig4_square_wave.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_mass(mass):
    if not mass:
        return None
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for run in mass["runs"]:
        if not run.get("history"):
            continue
        h = np.array(run["history"])
        ax.semilogy(h[:, 0], np.maximum(h[:, 1], 1e-18), "o-", ms=3, label=run["case"])
    ax.axhline(mass["criterion"], color="r", ls=":", label="1e-10 gate")
    ax.set_xlabel("t")
    ax.set_ylabel("relative mass drift")
    ax.set_title("periodic mass ledger (float64)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG, "fig5_mass_ledger.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    os.makedirs(FIG, exist_ok=True)
    cone = _load("rotating_cone")
    gauss = _load("rotating_gaussian")
    mms = _load("mms")
    sod = _load("sod")
    iso = _load("sod_isothermal")
    square = _load("square_wave")
    mass = _load("mass_ledger")
    fl = _fields()
    made = [
        fig_convergence(cone, mms, gauss),
        fig_cone_fields(fl),
        fig_sod(sod, iso, fl),
        fig_square(square, fl),
        fig_mass(mass),
    ]
    for m in made:
        if m:
            print("wrote", m)
    print("figures:", sorted(os.path.basename(p) for p in glob.glob(os.path.join(FIG, "*.png"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())