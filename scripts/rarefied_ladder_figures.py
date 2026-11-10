"""Figures for the P218-N06 rarefied / slip-flow ladder.

Reads ``<outdir>/ladder.json`` produced by ``python -m tensorlbm.rarefied.ladder``
and writes publication-quality PNGs into ``<outdir>/figures``.  All plotted
numbers come from the JSON (or from cheap recomputation of the same solvers), so
no value is hand-copied.

    python scripts/rarefied_ladder_figures.py --outdir runs/P218-N06_ladder
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from tensorlbm.rarefied import knudsen as K  # noqa: E402
from tensorlbm.rarefied.bgk_slab import LinearizedBGKSlab  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 160, "axes.titlesize": 10})


def _get(payload, key):
    for crit in payload.get("ladder", []):
        if crit["key"] == key:
            return crit
    return None


def fig_couette_profiles(outdir):
    kns = [0.01, 0.1, 1.0, 10.0]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for kn in kns:
        sol = LinearizedBGKSlab(kn, sigma=1.0, ny=96 if kn >= 1 else 128, nv=32).solve_couette(1.0)
        ax.plot(sol.u, sol.y, label=f"Kn = {kn:g}")
    ax.plot([1, 1], [0, 1], "k--", lw=0.8, label="wall U = 1")
    ax.plot([0, 0], [0, 1], "k--", lw=0.8)
    ax.plot([0.5, 0.5], [0, 1], ":", color="grey", lw=0.8, label="FM plug U/2")
    ax.set_xlabel(r"$u_x$")
    ax.set_ylabel(r"$y/H$")
    ax.set_title("Plane Couette flow, diffuse walls (TMAC = 1)\nlinearised BGK DVM")
    ax.legend(fontsize=7, loc="lower right")
    p = os.path.join(outdir, "figures", "fig1_couette_profiles.png")
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_slip_length(outdir, payload):
    crit = _get(payload, "C1")
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    sig = np.linspace(0.12, 1.0, 200)
    ax.plot(sig, (2 - sig) / sig, "k-", lw=1.2, label=r"Maxwell $(2-\sigma)/\sigma$")
    if crit:
        for kn, mk, col in ((0.01, "o", "tab:blue"), (0.1, "s", "tab:red")):
            cs = [c for c in crit["detail"]["cases"] if c["kn"] == kn]
            if not cs:
                continue
            s = [c["sigma"] for c in cs]
            v = [c["l_s_over_lambda"] for c in cs]
            ax.plot(s, v, mk, color=col, ms=6,
                    label=f"kinetic DVM, Kn = {kn:g}")
    ax.set_xlabel(r"TMAC $\sigma$")
    ax.set_ylabel(r"$l_s/\lambda$")
    ax.set_yscale("log")
    ax.set_title("Slip length vs tangential momentum accommodation")
    ax.legend(fontsize=7)
    p = os.path.join(outdir, "figures", "fig2_slip_length.png")
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_flow_rate(outdir, payload):
    crit = _get(payload, "C3")
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    if crit:
        rows = crit["detail"]["rows"]
        dvm = [(r["kn"], r["G_dvm"]) for r in rows if r.get("G_dvm")]
        dsmc = [(r["kn"], r["G_dsmc"]) for r in rows if r.get("G_dsmc")]
        ns = [(r["kn"], r["G_ns_slip"]) for r in rows if r.get("G_ns_slip")]
        if dvm:
            x, y = zip(*dvm)
            ax.loglog(x, y, "o-", color="tab:blue", ms=5, lw=1.2,
                      label="linearised BGK DVM (Richardson)")
        if dsmc:
            x, y = zip(*dsmc)
            ax.loglog(x, y, "s", color="tab:red", ms=7, mfc="none", mew=1.4,
                      label="DSMC (independent)")
        if ns:
            x, y = zip(*ns)
            ax.loglog(x, y, "k--", lw=1.0, label="NS + Maxwell slip")
        ax.axhline(1.0, color="grey", ls=":", lw=1.0, label="free-molecular level")
        km = crit["detail"]["knudsen_minimum"]
        ax.axvline(km["kn_min"], color="tab:green", ls="-.", lw=1.0,
                   label=f"Kn_min = {km['kn_min']:.2f}")
        ax.plot([km["kn_min"]], [km["q_min"] / (crit["detail"]["rows"][0]["q_fm"])], "*",
                color="tab:green", ms=12)
    ax.set_xlabel(r"$Kn=\lambda/H$")
    ax.set_ylabel(r"$G=Q/Q_{FM}$")
    ax.set_title("Normalised mass flow rate of plane Poiseuille flow\n(force driven, diffuse walls)")
    ax.legend(fontsize=7)
    p = os.path.join(outdir, "figures", "fig3_flow_rate.png")
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_knudsen_layer(outdir, payload):
    crit = _get(payload, "C2")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.2))
    if crit:
        y = np.asarray(crit["detail"]["y"])
        u = np.asarray(crit["detail"]["u_kinetic"])
        uns = np.asarray(crit["detail"]["u_ns_slip"])
        dev = np.asarray(crit["detail"]["deviation"])
        u = u / u.max()
        uns = uns / u.max()
        ax.plot(u, y, "o-", ms=2.5, color="tab:blue", label="kinetic (BGK DVM)")
        ax.plot(uns, y, "k--", lw=1.0, label="NS + Maxwell slip")
        ax.set_xlabel(r"$u_x/u_{x,max}$")
        ax.set_ylabel(r"$y/H$")
        ax.set_title(f"Poiseuille, Kn = {crit['detail']['kn']:g}")
        ax.legend(fontsize=7)
        ax2.semilogy(y, np.abs(dev) / np.max(np.abs(u)), "o-", ms=2.5, color="tab:purple")
        ax2.axvline(crit["detail"]["delta_fit_over_lambda_richardson"] * crit["detail"]["kn"],
                    color="tab:green", ls="--", lw=1.0, label=r"$\delta_{Kn}$ fit")
        ax2.set_xlabel(r"$y/H$")
        ax2.set_ylabel(r"$|u_{kin}-u_{NS+slip}|/\max u$")
        ax2.set_title("Knudsen-layer deviation")
        ax2.legend(fontsize=7)
    p = os.path.join(outdir, "figures", "fig4_knudsen_layer.png")
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_fm_degeneration(outdir, payload):
    crit = _get(payload, "C5")
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    if crit:
        for tag, mk, col in (("dvm", "o", "tab:blue"), ("dsmc", "s", "tab:red")):
            pts = [(r["kn"], r["dev_max_pct"]) for r in crit["detail"][tag]]
            if not pts:
                continue
            # keep the finest grid per Kn
            best = {}
            for kn, d in pts:
                best[kn] = d
            x = sorted(best)
            ax.loglog(x, [best[k] for k in x], mk + "-" if tag == "dvm" else mk,
                      color=col, ms=5, mfc="none" if tag == "dsmc" else col,
                      label="DVM" if tag == "dvm" else "DSMC")
        ax.axhline(15.0, color="tab:green", ls="--", lw=1.0, label="15% criterion")
        ax.axvline(1.0, color="grey", ls=":", lw=1.0, label="Kn = 1 (task target)")
    ax.set_xlabel(r"$Kn=\lambda/H$")
    ax.set_ylabel(r"$\max_y |u - U/2| \,/\, (U/2)$  [%]")
    ax.set_title("Couette approach to the free-molecular plug")
    ax.legend(fontsize=7)
    p = os.path.join(outdir, "figures", "fig5_fm_degeneration.png")
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="runs/P218-N06_ladder")
    args = ap.parse_args(argv)
    os.makedirs(os.path.join(args.outdir, "figures"), exist_ok=True)
    with open(os.path.join(args.outdir, "ladder.json"), encoding="utf-8") as fh:
        payload = json.load(fh)
    made = [fig_couette_profiles(args.outdir), fig_slip_length(args.outdir, payload),
            fig_flow_rate(args.outdir, payload), fig_knudsen_layer(args.outdir, payload),
            fig_fm_degeneration(args.outdir, payload)]
    for p in made:
        print("wrote", p)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())