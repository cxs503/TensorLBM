#!/usr/bin/env python
"""Figures for node ``p161w2-arb-n05`` (oscillating-foil propulsion).

Reads ``artifacts/results.json`` (+ per-case payloads, analytic anchor and field
snapshots) and writes PNGs to ``figures/``.  Every number in the figures comes
from those JSON files — nothing is typed by hand.

Usage::

    python make_figures.py --artifacts artifacts --figures figures
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "src")
sys.path.insert(0, SRC)

from tensorlbm.flapping_propulsion import (  # noqa: E402
    FlappingKinematics,
    garrick_coefficients,
    theodorsen_C,
)

COL = {"lbm": "#c0392b", "theory": "#1f4e79", "aux": "#7f8c8d", "ok": "#27ae60"}


def load(artdir):
    with open(os.path.join(artdir, "results.json")) as fh:
        results = json.load(fh)
    with open(os.path.join(artdir, "analytic.json")) as fh:
        analytic = json.load(fh)
    payloads = {}
    for path in glob.glob(os.path.join(artdir, "*.json")):
        name = os.path.basename(path)[:-5]
        if name in {"results", "analytic"}:
            continue
        with open(path) as fh:
            d = json.load(fh)
        if "label" in d:
            payloads[d["label"]] = d
    return results, analytic, payloads


def fig_theodorsen(analytic, out):
    rows = analytic["theodorsen"]
    ks = [r["k"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].semilogx(ks, [r["F"] for r in rows], "o-", color=COL["theory"], label=r"$F(k)$")
    ax[0].semilogx(ks, [r["G"] for r in rows], "s-", color=COL["lbm"], label=r"$G(k)$")
    ax[0].axhline(1.0, ls=":", color=COL["aux"])
    ax[0].axhline(0.5, ls=":", color=COL["aux"])
    ax[0].set_xlabel("reduced frequency $k$")
    ax[0].set_ylabel("Theodorsen function")
    ax[0].legend()
    ax[0].set_title("Theodorsen $C(k)=F+iG$: $C(0)=1$, $C(\\infty)=1/2$")
    ax[1].semilogx(ks, [r["absC"] for r in rows], "o-", color=COL["theory"])
    ax[1].set_xlabel("reduced frequency $k$")
    ax[1].set_ylabel(r"$|C(k)|$")
    ax[1].set_title("circulatory efficiency factor")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_coefficients(results, payloads, out):
    """Measured C_T and eta vs k against the Garrick closed form (pure heave)."""
    labels = [
        "heave_h015_k0.75",
        "heave_h015_k1",
        "heave_h015_k1.5",
        "heave_h015_k2",
    ]
    ks, ct, th, eta, eta_th, cp = [], [], [], [], [], []
    for lbl in labels:
        if lbl not in payloads:
            continue
        p = payloads[lbl]
        ks.append(p["config"]["k"])
        ct.append(p["coeffs"]["C_T"])
        th.append(p["theory"]["C_T"])
        eta.append(p["coeffs"]["eta"])
        eta_th.append(p["theory"]["eta"])
        cp.append(p["coeffs"]["C_P"])
    cd0 = -payloads["lock_ref"]["coeffs"]["C_T"]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    ax[0].plot(ks, th, "o-", color=COL["theory"], label="Garrick (inviscid, linear)")
    ax[0].plot(ks, ct, "s-", color=COL["lbm"], label="LBM (net, c=32)")
    ax[0].axhline(-cd0, ls="--", color=COL["aux"], label=r"$-C_{D0}$ locked body")
    ax[0].plot(ks, np.array(ct) + cd0, "^:", color=COL["ok"], label=r"$\Delta C_T$ increment")
    ax[0].set_xlabel("reduced frequency $k$")
    ax[0].set_ylabel("$C_T$")
    ax[0].set_title("thrust coefficient, heave $h_0/c=0.15$")
    ax[0].legend(fontsize=7)
    ax[1].plot(ks, cp, "s-", color=COL["lbm"], label="LBM")
    ax[1].set_xlabel("reduced frequency $k$")
    ax[1].set_ylabel("$C_P$")
    ax[1].set_title("input power coefficient")
    ax[1].legend(fontsize=7)
    ax[2].plot(ks, eta_th, "o-", color=COL["theory"], label="Garrick")
    ax[2].plot(ks, eta, "s-", color=COL["lbm"], label="LBM")
    ax[2].axhline(0.0, ls=":", color=COL["aux"])
    ax[2].axhline(1.0, ls=":", color=COL["aux"])
    ax[2].set_xlabel("reduced frequency $k$")
    ax[2].set_ylabel(r"$\eta = C_T/C_P$")
    ax[2].set_title("propulsive efficiency")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_ct_vs_st(results, payloads, out):
    labels = ["heave_h005_k1", "heave_h010_k1", "heave_h015_k1", "heave_h020_k1",
              "heave_h025_k1"]
    st, ct, th, dct = [], [], [], []
    cd0 = -payloads["lock_ref"]["coeffs"]["C_T"]
    for lbl in labels:
        if lbl not in payloads:
            continue
        p = payloads[lbl]
        st.append(p["coeffs"]["strouhal"])
        ct.append(p["coeffs"]["C_T"])
        th.append(p["theory"]["C_T"])
        dct.append(p["coeffs"]["C_T"] + cd0)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    ax[0].plot(st, th, "o-", color=COL["theory"], label="Garrick")
    ax[0].plot(st, ct, "s-", color=COL["lbm"], label="LBM net")
    ax[0].plot(st, dct, "^:", color=COL["ok"], label=r"LBM $\Delta C_T$")
    ax[0].axhline(0.0, ls="--", color=COL["aux"])
    ax[0].set_xlabel("Strouhal number $St=fA/U$")
    ax[0].set_ylabel("$C_T$")
    ax[0].set_title("thrust vs Strouhal number ($k=1$, pure heave)")
    ax[0].legend(fontsize=7)

    # pitch reversal
    ks, cp, pt = [], [], []
    for lbl in ("pitch_a10_k0.75", "pitch_a10_k1", "pitch_a10_k2"):
        if lbl in payloads:
            p = payloads[lbl]
            ks.append(p["config"]["k"])
            cp.append(p["coeffs"]["C_T"])
            pt.append(p["theory"]["C_T"])
    ax[1].plot(ks, pt, "o-", color=COL["theory"], label="Garrick")
    ax[1].plot(ks, cp, "s-", color=COL["lbm"], label="LBM")
    ax[1].axhline(0.0, ls="--", color=COL["aux"])
    ax[1].set_xlabel("reduced frequency $k$")
    ax[1].set_ylabel("$C_T$")
    ax[1].set_title("thrust reversal of a pitching section ($\\alpha_0=10^\\circ$)")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_timeseries(payloads, out):
    lbl = "heave_h015_k1"
    if lbl not in payloads:
        return
    p = payloads[lbl]
    per = p["config"]["steps_per_cycle"]
    q = 0.5 * p["coeffs"]["strouhal"] ** 0 * 1.0  # placeholder, coeffs already normalised
    s = p["series"]
    steps = np.array([r["s"] for r in s])
    fx = np.array([r["fx"] for r in s])
    fy = np.array([r["fy"] for r in s])
    w = np.array([r["w"] for r in s])
    cfg = p["config"]
    qn = 0.5 * cfg["U"] ** 2 * cfg["chord"]
    qp = 0.5 * cfg["U"] ** 3 * cfg["chord"]
    keep = steps > cfg["warmup_cycles"] * per
    steps, fx, fy, w = steps[keep], fx[keep], fy[keep], w[keep]
    t = (steps % per) / per
    order = np.argsort(t)
    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax[0].plot(t[order], fx[order] / qn, ".", ms=2, color=COL["lbm"])
    ax[0].set_ylabel("$C_T(t)$")
    ax[0].set_title(f"phase-averaged force and power, {lbl} (all averaged cycles folded)")
    ax[1].plot(t[order], w[order] / qp, ".", ms=2, color=COL["theory"])
    ax[1].set_ylabel("$C_P(t)$")
    ax[1].set_xlabel("t / T")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_grid(results, payloads, out):
    rows = results["criteria"]
    gci = next(c for c in rows if c["id"] == "C6")
    chords = gci["value"]["chords"]
    vals = gci["value"]["values"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].plot(chords, vals, "o-", color=COL["lbm"])
    if np.isfinite(gci["value"].get("f_extrapolated", float("nan"))):
        ax[0].axhline(gci["value"]["f_extrapolated"], ls="--", color=COL["ok"],
                      label=f"Richardson $\\Delta C_T$ = {gci['value']['f_extrapolated']:.4f}")
        ax[0].legend(fontsize=7)
    ax[0].set_xlabel("chord (lattice cells)")
    ax[0].set_ylabel(r"$\Delta C_T$")
    ax[0].set_title(
        f"grid convergence: $p$ = {gci['value']['p_observed']:.2f}, "
        f"GCI = {gci['value']['gci_fine']:.3f}"
    )
    c4 = next(c for c in rows if c["id"] == "C4")
    res = c4["value"].get("resolutions", [])
    if res:
        xs = [float(r["label"].split("_c")[-1]) for r in res]
        ys = [r["rel_residual"] for r in res]
        ax[1].plot(xs, ys, "s-", color=COL["theory"])
        ax[1].axhline(0.05, ls="--", color=COL["aux"], label="5% tolerance")
        ax[1].set_xlabel("chord (lattice cells)")
        ax[1].set_ylabel("energy-ledger residual / $|W|$")
        ax[1].set_title("energy-ledger closure vs resolution")
        ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_fields(artdir, out):
    paths = sorted(glob.glob(os.path.join(artdir, "fields_*.npz")))
    if not paths:
        return
    data = np.load(paths[0])
    keys = [k for k in data.files if k.startswith("vort_")]
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 2.9))
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        w = data[f"vort_{i}"]
        mx, my = data[f"mx_{i}"], data[f"my_{i}"]
        ny, nx = w.shape
        x0, x1 = int(mx.min()) - 3 * 32, int(mx.max()) + 6 * 32
        y0, y1 = int(my.mean()) - 2 * 32, int(my.mean()) + 2 * 32
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(nx, x1), min(ny, y1)
        lim = float(np.percentile(np.abs(w[y0:y1, x0:x1]), 99.5)) or 1.0
        im = ax.imshow(
            w[y0:y1, x0:x1], origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim,
            extent=[x0, x1, y0, y1],
        )
        ax.plot(mx, my, "k-", lw=1.2)
        ax.set_title(f"$t/T$ = {data['phases'][i]:.2f}", fontsize=9)
        ax.set_xlabel("x")
        if i == 0:
            ax.set_ylabel("y")
    fig.colorbar(im, ax=axes, shrink=0.8, label=r"$\omega_z$")
    fig.suptitle("vorticity field around the oscillating NACA 0012 (phase-locked)", fontsize=10)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default=os.path.join(HERE, "artifacts"))
    ap.add_argument("--figures", default=os.path.join(HERE, "figures"))
    args = ap.parse_args(argv)
    os.makedirs(args.figures, exist_ok=True)
    results, analytic, payloads = load(args.artifacts)
    fig_theodorsen(analytic, os.path.join(args.figures, "fig1_theodorsen.png"))
    fig_coefficients(results, payloads, os.path.join(args.figures, "fig2_coefficients.png"))
    fig_ct_vs_st(results, payloads, os.path.join(args.figures, "fig3_ct_vs_st.png"))
    fig_timeseries(payloads, os.path.join(args.figures, "fig4_timeseries.png"))
    fig_grid(results, payloads, os.path.join(args.figures, "fig5_grid_energy.png"))
    fig_fields(args.artifacts, os.path.join(args.figures, "fig6_fields.png"))
    print("figures written to", args.figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())