#!/usr/bin/env python
"""Figures for the P162W1-N07 flapping-foil propulsion verification face.

Figures are generated purely from the artefacts in ``results/`` (results.json,
per-case JSON, the field snapshot NPZ) so every number in the report is
traceable to a program-written file.

    python make_figures.py --outdir artifacts_p162n07 --figdir figures
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from tensorlbm.flapping_propulsion import (  # noqa: E402
    FlappingKinematics,
    garrick_coefficients,
    theodorsen_table,
)


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def fig_theodorsen(figdir):
    rows = theodorsen_table(np.linspace(0.02, 5.0, 200))
    k = [r["k"] for r in rows]
    F = [r["F"] for r in rows]
    G = [r["G"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax[0].plot(k, F, label="F(k) real part")
    ax[0].plot(k, G, label="G(k) imaginary part")
    ax[0].axhline(0.5, ls=":", c="k", lw=0.8, label=r"$C(\infty)=1/2$")
    ax[0].set_xlabel("reduced frequency k = $\\omega c/2U$")
    ax[0].set_ylabel("Theodorsen C(k)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[0].set_title("Theodorsen function (Hankel closed form)")
    ax[1].plot(F, G, lw=1.0)
    ax[1].plot([1.0], [0.0], "o", ms=4, label="k$\\to$0: C=1")
    ax[1].plot([0.5], [0.0], "s", ms=4, label="k$\\to\\infty$: C=1/2")
    ax[1].set_xlabel("F")
    ax[1].set_ylabel("G")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    ax[1].set_title("Argand diagram")
    fig.tight_layout()
    out = os.path.join(figdir, "fig1_theodorsen.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_eta_and_thrust(figdir):
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ks = np.linspace(0.1, 4.0, 120)
    for h0c, a0, phi, lab in ((0.02, 0.0, 0.0, "heave only, h0/c=0.02"),
                              (0.0, 2.0, 0.0, "pitch only, $\\alpha_0$=2$^\\circ$"),
                              (0.02, 2.0, 90.0, "feathering, h0/c=0.02, 2$^\\circ$"),
                              (0.15, 0.0, 0.0, "heave only, h0/c=0.15")):
        etas, cts = [], []
        for k in ks:
            kin = FlappingKinematics(U=1.0, c=1.0, h0=h0c,
                                     alpha0=math.radians(a0),
                                     phi=math.radians(phi), a=-0.5)
            g = garrick_coefficients(kin, float(k))
            etas.append(g.eta)
            cts.append(g.C_T)
        ax[0].plot(ks, etas, label=lab)
        ax[1].plot(ks, cts, label=lab)
    ax[0].set_xlabel("k")
    ax[0].set_ylabel("Garrick efficiency $\\eta=C_T/C_P$")
    ax[0].set_ylim(0, 1.05)
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)
    ax[0].set_title("Closed-form efficiency")
    ax[1].set_xlabel("k")
    ax[1].set_ylabel("$C_T$ (inviscid)")
    ax[1].axhline(0.0, c="k", lw=0.8)
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)
    ax[1].set_title("Closed-form thrust")
    fig.tight_layout()
    out = os.path.join(figdir, "fig2_garrick.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_lbm_vs_theory(figdir, results):
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    rows = results["criteria"].get("C1_rows", [])
    kinds = {"heave": "o", "pitch": "s", "comb": "^"}
    for kind, mk in kinds.items():
        sel = [r for r in rows if r["kind"] == kind and r["C_T_theory"] == r["C_T_theory"]]
        if not sel:
            continue
        ax[0].plot([r["k"] for r in sel], [r["C_T_theory"] for r in sel],
                   mk + "-", label=f"{kind} theory", ms=4)
        ax[0].plot([r["k"] for r in sel], [r["dC_T_increment"] for r in sel],
                   mk + "--", label=f"{kind} LBM $\\Delta C_T$", ms=4, alpha=0.8)
    ax[0].axhline(0, c="k", lw=0.7)
    ax[0].set_xlabel("k")
    ax[0].set_ylabel("$C_T$")
    ax[0].set_yscale("symlog", linthresh=1e-3)
    ax[0].legend(fontsize=6)
    ax[0].grid(alpha=0.3)
    ax[0].set_title("small amplitude h0/c=0.02, $\\alpha_0$=2$^\\circ$")

    e = results["criteria"].get("C2", {}).get("rows", [])
    if e:
        kk = [r["k"] for r in e]
        ax[1].plot(kk, [r["eta_theory"] for r in e], "o-", label="Garrick $\\eta$")
        ax[1].plot(kk, [r["eta_lbm"] for r in e], "s--", label="LBM $\\eta$")
        ax[1].set_xlabel("k")
        ax[1].set_ylabel("propulsive efficiency $\\eta$")
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3)
        ax[1].set_title("thrust-dominated case h0/c=0.15")
    fig.tight_layout()
    out = os.path.join(figdir, "fig3_lbm_vs_theory.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_fields(figdir, outdir):
    npz = os.path.join(outdir, "big_heave_k1_fields.npz")
    if not os.path.exists(npz):
        return None
    d = np.load(npz)
    ux, uy = d["ux"], d["uy"]
    t = d["t_over_T"]
    ny, nx = ux.shape[1], ux.shape[2]
    fig, axes = plt.subplots(len(ux), 1, figsize=(9.0, 1.9 * len(ux)))
    if len(ux) == 1:
        axes = [axes]
    for ax, i in zip(axes, range(len(ux))):
        vort = np.zeros_like(ux[i])
        vort[1:-1, 1:-1] = (
            (uy[i][2:, 1:-1] - uy[i][:-2, 1:-1]) - (ux[i][1:-1, 2:] - ux[i][1:-1, :-2])
        )
        m = np.abs(vort).max() or 1.0
        im = ax.imshow(vort, cmap="RdBu_r", vmin=-m, vmax=m, origin="lower")
        ax.set_title(f"vorticity, t/T = {t[i]:.2f}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    fig.suptitle("heaving foil h0/c=0.15, k=1 (LBM, chord=32 cells)", fontsize=9)
    fig.tight_layout()
    out = os.path.join(figdir, "fig4_vorticity.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_energy_and_gci(figdir, results):
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    c4 = results["criteria"].get("C4", {})
    rows = c4.get("rows", [])
    labels = [r["case"] for r in rows][:14]
    vals = [abs(r["relative_residual"]) if r["relative_residual"] == r["relative_residual"]
            else np.nan for r in rows][:14]
    ax[0].barh(range(len(vals)), vals, color="tab:blue")
    ax[0].set_yticks(range(len(labels)))
    ax[0].set_yticklabels(labels, fontsize=6)
    ax[0].set_xscale("log")
    ax[0].axvline(1e-3, c="r", ls="--", lw=1, label="target 1e-3")
    ax[0].set_xlabel("relative energy-budget residual")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)
    ax[0].set_title("C4 energy budget")

    c6 = results["criteria"].get("C6", {})
    if c6:
        ch = [16, 32, 64]
        vals = [c6.get("err_c16"), c6.get("err_c32"), c6.get("err_c64")]
        ax[1].loglog(ch, vals, "o-")
        for x, y, lab in zip(ch, vals, ["c16", "c32", "c64"]):
            ax[1].annotate(lab, (x, y), fontsize=7)
        ax[1].set_xlabel("chord cells")
        ax[1].set_ylabel("|C_T - Garrick|")
        ax[1].grid(alpha=0.3, which="both")
        ax[1].set_title(f"C6 grid convergence (ratio e32/e64 = "
                        f"{c6.get('error_ratio_medium_over_fine')})")
    fig.tight_layout()
    out = os.path.join(figdir, "fig5_energy_gci.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(HERE, "artifacts_p162n07"))
    ap.add_argument("--figdir", default=os.path.join(HERE, "figures"))
    args = ap.parse_args(argv)
    os.makedirs(args.figdir, exist_ok=True)
    results = _load(os.path.join(args.outdir, "results.json"))
    made = [fig_theodorsen(args.figdir), fig_eta_and_thrust(args.figdir),
            fig_lbm_vs_theory(args.figdir, results),
            fig_fields(args.figdir, args.outdir),
            fig_energy_and_gci(args.figdir, results)]
    made = [m for m in made if m]
    print(json.dumps({"figures": made}, indent=1))


if __name__ == "__main__":
    raise SystemExit(main())