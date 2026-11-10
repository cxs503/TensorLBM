#!/usr/bin/env python3
"""Figures for node P151-N01 (D3Q27 activation) — reads only benchmark JSON/NPZ.

Outputs (``artifacts/m6648_d3q27/``):
  fig_tgv_order.png      L2 error vs grid spacing + observed orders (gate f)
  fig_cd_history.png     Cd(t) for D3Q19 vs D3Q27 at Re=100 / Re=200
  fig_cd_error.png       Cd error bars vs the 10% / 15% gate and the G12
                         pressure-extrapolation diagnostic
  fig_fields.png         mid-plane |u| and pressure, D3Q19 vs D3Q27 (Re=100)

Every number comes from ``benchmarks/d3q27_activation/runs/*.json`` / ``*.npz``
written by the run scripts — no hand transcription.
"""

from __future__ import annotations

import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "artifacts", "m6648_d3q27"))
os.makedirs(OUT, exist_ok=True)


def load(name):
    p = os.path.join(RUNS, name)
    return json.load(open(p)) if os.path.exists(p) else None


def fig_tgv():
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    colors = {"d3q27": "#1f77b4", "d3q19": "#d62728"}
    for lat in ("d3q27", "d3q19"):
        d = load(f"tgv_convergence_{lat}.json")
        if not d:
            continue
        ns = [int(k) for k in d["l2_errors"]]
        h = [2 * math.pi / n for n in ns]
        e = [d["l2_errors"][str(n)] for n in ns]
        ax.loglog(h, e, "o-", color=colors[lat], label=f"{lat.upper()}  order={d['observed_order_primary']:.2f}")
    hh = np.array([0.06, 0.30])
    ax.loglog(hh, 1.0e-2 * (hh / 0.26) ** 2, "k--", lw=1, label="slope 2 (2nd order)")
    ax.set_xlabel("grid spacing h = 2π/N  [lattice units]")
    ax.set_ylabel(r"$L_2$ velocity error / $\|u_{\rm exact}\|$")
    ax.set_title("Taylor-Green (exact reference): D3Q27 vs D3Q19")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_tgv_order.png"), dpi=150)
    plt.close(fig)


def _cd_curves():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, re in zip(axes, (100, 200)):
        for lat, c in (("d3q19", "#d62728"), ("d3q27", "#1f77b4")):
            d = load(f"sphere_{lat}_re{re}_quadratic.json")
            if not d:
                continue
            steps = [e["step"] for e in d["force_history"]]
            cd = [e["cd_total"] for e in d["force_history"]]
            ax.plot(steps, cd, color=c, lw=1.2, label=f"{lat.upper()}  Cd={d['cd_tail_mean']:.4f}")
            ax.axhline(d["cd_ref_schiller_naumann"], color=c, ls=":", lw=1)
        ax.set_title(f"Sphere drag, Re={re}, D=20, MRT")
        ax.set_xlabel("step")
        ax.set_ylabel(r"$C_d$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.text(
            0.02,
            0.96,
            "dotted: Schiller-Naumann",
            transform=ax.transAxes,
            fontsize=7,
            va="top",
        )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_cd_history.png"), dpi=150)
    plt.close(fig)


def fig_cd_error():
    labels, errs, cols = [], [], []
    for re in (100, 200):
        for lat, c in (("d3q19", "#d62728"), ("d3q27", "#1f77b4")):
            d = load(f"sphere_{lat}_re{re}_quadratic.json")
            if not d:
                continue
            labels.append(f"{lat.upper()}\nRe={re}")
            errs.append(d["cd_error_pct"])
            cols.append(c)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    x = np.arange(len(labels))
    ax.bar(x, errs, color=cols, width=0.6)
    ax.axhline(10.0, color="k", ls="--", lw=1, label="Re=100 gate 10%")
    ax.axhline(15.0, color="0.5", ls=":", lw=1.2, label="Re=200 gate 15%")
    for xi, e in zip(x, errs):
        ax.text(xi, e + 0.2, f"{e:.2f}%", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("|Cd - Cd_SN| / Cd_SN  [%]")
    ax.set_title("Sphere drag error at identical grid (D=20, quadratic wall-pressure extrap.)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_cd_error.png"), dpi=150)
    plt.close(fig)


def fig_fields():
    panels = []
    for lat in ("d3q19", "d3q27"):
        p = os.path.join(RUNS, f"sphere_{lat}_re100_quadratic_fields.npz")
        if os.path.exists(p):
            panels.append((lat, np.load(p)))
    if not panels:
        return
    fig, axes = plt.subplots(2, len(panels), figsize=(5.0 * len(panels), 7.2))
    if len(panels) == 1:
        axes = axes.reshape(2, 1)
    for j, (lat, d) in enumerate(panels):
        ux, uy = d["ux"], d["uy"]
        rho = d["rho"]
        umag = np.sqrt(ux**2 + uy**2)
        solid = d["solid_mid"]
        umag = np.where(solid, np.nan, umag)
        press = np.where(solid, np.nan, (rho - 1.0) / 3.0)
        im0 = axes[0, j].imshow(umag, origin="lower", cmap="viridis")
        axes[0, j].set_title(f"{lat.upper()} |u| (mid plane)")
        fig.colorbar(im0, ax=axes[0, j], fraction=0.046)
        im1 = axes[1, j].imshow(press, origin="lower", cmap="coolwarm")
        axes[1, j].set_title(f"{lat.upper()} pressure p=(ρ-1)/3")
        fig.colorbar(im1, ax=axes[1, j], fraction=0.046)
        for ax in axes[:, j]:
            ax.set_xlabel("x [lu]")
            ax.set_ylabel("z [lu]")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_fields.png"), dpi=150)
    plt.close(fig)


def main() -> int:
    fig_tgv()
    _cd_curves()
    fig_cd_error()
    fig_fields()
    print("figures written to", OUT)
    for f in sorted(os.listdir(OUT)):
        if f.endswith(".png"):
            print(" ", f, os.path.getsize(os.path.join(OUT, f)) // 1024, "kB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())