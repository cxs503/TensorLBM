#!/usr/bin/env python3
"""Figures for the spurious-current suppression campaign (node p362w2-n06).

Every figure is generated from ``results.json`` written by ``run.py``; no number
in any figure is typed by hand.

Usage::

    python benchmarks/spurious_current_suppression/plot.py --results <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {
    "baseline_bgk_lattice": "#1f4e79",
    "mrt_lattice": "#c00000",
    "bgk_geometric": "#2e7d32",
    "mrt_geometric": "#8e44ad",
}
MARKERS = {
    "baseline_bgk_lattice": "o",
    "mrt_lattice": "s",
    "bgk_geometric": "^",
    "mrt_geometric": "D",
}


def _short(method: str) -> str:
    return {
        "baseline_bgk_lattice": "baseline (BGK+lattice)",
        "mrt_lattice": "MRT+lattice",
        "bgk_geometric": "BGK+geometric",
        "mrt_geometric": "MRT+geometric",
    }.get(method, method)


def fig_w_scan(payload: dict, out: Path) -> Path:
    scan = payload["summary"]["scan"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for m, rows in scan.items():
        rows = sorted(rows, key=lambda r: r["width_10_90"])
        w = [r["width_10_90"] for r in rows]
        u = [r["u_spurious"] for r in rows]
        ax.plot(w, u, marker=MARKERS.get(m, "o"), color=COLORS.get(m, "k"),
                label=_short(m), lw=1.6, ms=6)
    ax.set_yscale("log")
    ax.set_xlabel(r"measured interface thickness $W_{10\text{-}90}$ [lattice units]")
    ax.set_ylabel(r"$U_{\rm spurious}=\max|u|$ [lattice units]")
    ax.set_title("Spurious current vs interface thickness (R=50, G12=2.5)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    p = out / "fig1_spurious_vs_W.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_pareto(payload: dict, out: Path) -> Path:
    scan = payload["summary"]["scan"]
    lap = payload["summary"]["laplace"]
    front = {(f["method"], round(f["W_nominal"], 3)) for f in payload["summary"]["pareto"]["front"]}
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for m, rows in scan.items():
        e = lap.get(m, {}).get("max_rel_error", float("nan"))
        for r in rows:
            u = r["u_spurious"]
            on = (m, round(r["W_nominal"], 3)) in front
            ax.scatter(u, e, s=110 if on else 55,
                       facecolors="none" if on else COLORS.get(m, "k"),
                       edgecolors=COLORS.get(m, "k"), marker=MARKERS.get(m, "o"),
                       linewidths=1.8 if on else 1.0, alpha=0.95)
            ax.annotate(f"W={r['W_nominal']:.0f}", (u, e), fontsize=6,
                        xytext=(4, 3), textcoords="offset points", alpha=0.7)
    for m in scan:
        ax.scatter([], [], marker=MARKERS.get(m, "o"), color=COLORS.get(m, "k"),
                   label=_short(m))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(0.03, color="grey", ls="--", lw=1)
    ax.text(ax.get_xlim()[0] * 1.05, 0.031, "3% acceptance line", fontsize=7, color="grey")
    ax.set_xlabel(r"$U_{\rm spurious}$ [lattice units]  (minimise)")
    ax.set_ylabel("Laplace-law max relative error  (minimise)")
    ax.set_title("Pareto plane (hollow markers = non-dominated)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    p = out / "fig2_pareto.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_laplace(payload: dict, out: Path) -> Path:
    lap = payload["summary"]["laplace"]
    fig, (ax, axr) = plt.subplots(1, 2, figsize=(9.6, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    for m, d in lap.items():
        inv = [1.0 / r for r in d["radii"]]
        ax.plot(inv, d["dp"], marker=MARKERS.get(m, "o"), ls="none",
                color=COLORS.get(m, "k"), label=_short(m), ms=7)
        rr = np.linspace(min(inv) * 0.95, max(inv) * 1.05, 50)
        ax.plot(rr, d["sigma"] * rr + d["intercept"], color=COLORS.get(m, "k"),
                lw=1.2, alpha=0.8)
        axr.semilogy(d["radii"], np.maximum(d["rel_error"], 1e-16),
                     marker=MARKERS.get(m, "o"), color=COLORS.get(m, "k"), ls="-",
                     ms=6, label=_short(m))
    axr.axhline(0.03, color="grey", ls="--", lw=1)
    ax.set_xlabel(r"$1/R$ [1/lattice units]")
    ax.set_ylabel(r"$\Delta p = p_{\rm in}-p_{\rm out}$ [lattice units]")
    ax.set_title("Young-Laplace: $\\Delta p=\\sigma/R+c$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, frameon=False)
    axr.set_xlabel("droplet radius R [lattice units]")
    axr.set_ylabel("relative residual vs fit")
    axr.set_title("residuals (3% line)")
    axr.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    p = out / "fig3_laplace.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_convergence(payload: dict, out: Path) -> Path:
    conv = payload["summary"]["grid_convergence"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for m, d in conv.items():
        h = np.asarray(d["spacing"])
        u = np.asarray(d["u_spurious"])
        ax.loglog(h, u, marker=MARKERS.get(m, "o"), color=COLORS.get(m, "k"),
                  label=f"{_short(m)} (order {d.get('order', float('nan')):.2f})")
        # slope-1 guide through the finest point
        guide = u[-1] * (h / h[-1]) ** 1.0
        ax.loglog(h, guide, ls=":", color=COLORS.get(m, "k"), lw=1)
    ax.set_xlabel("grid spacing h = 1/N")
    ax.set_ylabel(r"$U_{\rm spurious}$ [lattice units]")
    ax.set_title("Grid convergence (R=0.25N, dotted = slope 1)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    p = out / "fig4_grid_convergence.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_fields(payload: dict, out: Path) -> Path:
    arts = payload.get("field_artifacts", {})
    methods = [m for m in ("baseline_bgk_lattice", "bgk_geometric") if m in arts]
    if not methods:
        return out / "fig5_fields_missing.txt"
    fig, axes = plt.subplots(1, len(methods), figsize=(4.6 * len(methods), 4.0))
    axes = np.atleast_1d(axes)
    vmax = 0.0
    data = {}
    for m in methods:
        with np.load(Path(arts[m])) as z:
            data[m] = (z["umag"], float(z["u_max"]))
            vmax = max(vmax, float(z["umag"].max()))
    for ax, m in zip(axes, methods):
        umag, umax = data[m]
        im = ax.imshow(umag, origin="lower", cmap="inferno", vmin=0, vmax=vmax)
        ax.set_title(f"{_short(m)}\nmax|u| = {umax:.2e}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.85, label=r"$|u|$ [lattice units]")
    p = out / "fig5_velocity_fields.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="artifacts/spurious_current_p362w2_n06")
    args = ap.parse_args()
    out = Path(__file__).resolve().parents[2] / args.results
    payload = json.loads((out / "results.json").read_text())
    made = [
        fig_w_scan(payload, out),
        fig_pareto(payload, out),
        fig_laplace(payload, out),
        fig_convergence(payload, out),
        fig_fields(payload, out),
    ]
    for p in made:
        print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
