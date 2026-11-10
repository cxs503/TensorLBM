"""Figures for the P82W5-N04 report (D3Q19 yield-stress LBM with moving walls).

Reads ``artifacts/P82W5-N04/*.json`` + the ``.npy`` fields written by
``scripts/bench_yield_stress_moving_wall_3d.py`` and writes PNGs into
``figures/p82w5_n04/``.

    python scripts/gen_yield_stress_moving_wall_figures.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts", "P82W5-N04")
FIG = os.path.join(ROOT, "figures", "p82w5_n04")


def _load(name):
    path = os.path.join(ART, name)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _npy(name):
    path = os.path.join(ART, name)
    return np.load(path) if os.path.exists(path) else None


def fig_duct():
    """Square-duct Bingham field: contour of the sim and reference, plus a cut."""
    sim = _npy("duct_N32_sim.npy")
    ref = _npy("duct_N32_ref.npy")
    if sim is None or ref is None:
        return None
    N = sim.shape[0]
    y = np.arange(1, N + 1) - 0.5
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    vmax = max(sim.max(), ref.max())
    for ax, field, title in ((axes[0], sim, "LBM (D3Q19, m=1e5)"),
                             (axes[1], ref, "AL reference (regularisation-free)")):
        im = ax.contourf(y, y, field, levels=24, vmin=0, vmax=vmax)
        ax.set_aspect("equal")
        ax.set_xlabel("y"); ax.set_ylabel("z")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[2]
    ax.plot(y, sim[:, N // 2], "o-", ms=3, label="LBM")
    ax.plot(y, ref[:, N // 2], "k--", label="AL reference")
    ax.plot(y, sim[N // 2, :], "s-", ms=3, alpha=0.6, label="LBM (z-cut)")
    ax.set_xlabel("y (lattice)"); ax.set_ylabel("$u_x$")
    ax.set_title("cut through the duct centre")
    ax.legend()
    err = np.abs(sim - ref).max() / ref.max()
    ax.text(0.03, 0.92, "Linf err = %.2f%%" % (100 * err), transform=ax.transAxes)
    fig.suptitle("P82W5-N04 (a): Bingham square duct, N=%d, G=1e-5, $\\tau_y$=2e-5" % N)
    fig.tight_layout()
    out = os.path.join(FIG, "a_duct_field.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_couette():
    """Couette--Poiseuille profile with the plug region and its exact bounds."""
    data = _load("case_b.json")
    if data is None:
        return None
    runs = data["cases"]["b"]["runs"]
    keys = [k for k in ("H32", "H16", "H8") if k in runs]
    if not keys:
        return None
    fig, axes = plt.subplots(1, len(keys), figsize=(4.6 * len(keys), 4.0),
                             squeeze=False)
    for ax, k in zip(axes[0], keys):
        H = int(k[1:])
        sim = _npy("couette_poiseuille_H%d_sim.npy" % H)
        exact = _npy("couette_poiseuille_H%d_exact.npy" % H)
        if sim is None:
            continue
        y = np.arange(1, H + 1) - 0.5
        y1, y2 = runs[k]["yield_surfaces_exact"]
        ax.axvspan(y1, y2, color="0.85", label="exact plug $\\delta_p$=%.1f" % (y2 - y1))
        ax.plot(y, exact, "k--", label="exact Bingham")
        ax.plot(y, sim, "o-", ms=3, label="LBM (D3Q19+mov.Ladd)")
        ax.axhline(runs[k]["u_wall"], color="C3", lw=0.8, ls=":",
                   label="$U_w$=%.4f" % runs[k]["u_wall"])
        ax.set_xlabel("y"); ax.set_ylabel("$u_x$")
        ysim = runs[k].get("yield_surfaces_sim") or [None, None]
        if ysim[0] is not None:
            ax.axvline(ysim[0], color="C0", lw=0.8, ls="--")
            ax.axvline(ysim[1], color="C0", lw=0.8, ls="--")
        ax.set_title("H=%d  $\\delta_p$ err=%.1f%%, CV=%.1e" % (
            H, 100 * (runs[k].get("delta_p_err_stress") or float("nan")),
            runs[k]["plug_cv"] or float("nan")))
        ax.legend(fontsize=7)
    fig.suptitle("P82W5-N04 (b): Bingham Couette--Poiseuille, one wall moving")
    fig.tight_layout()
    out = os.path.join(FIG, "b_couette_poiseuille.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_ladder():
    """Yield-surface / profile error vs resolution with the fitted slopes."""
    data = _load("case_c.json")
    if data is None:
        return None
    runs = data["cases"]["c"]["runs"]
    Ns = sorted(int(k[1:]) for k in runs)
    e1 = [runs["N%d" % n]["err_umax"] for n in Ns]
    e2 = [runs["N%d" % n]["err_gamma"] for n in Ns]
    e3 = [runs["N%d" % n]["profile_linf_rel"] for n in Ns]
    sl = data["cases"]["c"]["slopes"]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.loglog(Ns, np.maximum(e1, 1e-16), "o-", label="plug half-width (u_max inversion), slope=%.2f"
              % (sl["umax_inversion"] if sl["umax_inversion"] else float("nan")))
    ax.loglog(Ns, np.maximum(e2, 1e-16), "s-", label="$\\dot\\gamma$ crossing, slope=%.2f"
              % (sl["gamma_crossing"] if sl["gamma_crossing"] else float("nan")))
    ax.loglog(Ns, np.maximum(e3, 1e-16), "^-", label="profile $L_\\infty$ rel, slope=%.2f"
              % (sl["profile_linf"] if sl["profile_linf"] else float("nan")))
    ref = [e3[0] * (Ns[0] / n) ** 2 for n in Ns]
    ax.loglog(Ns, ref, "k:", lw=1, label="2nd order reference")
    ax.set_xlabel("N (cells across the gap)")
    ax.set_ylabel("error (lattice units / relative)")
    ax.set_title("P82W5-N04 (c): yield-surface convergence, $y_p$=%.1f" % runs["N%d" % Ns[0]]["yp_exact"])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = os.path.join(FIG, "c_resolution_ladder.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_ledger_ratio():
    """(d) ledger closures and (e) 2D/3D shape factor, (f) Newtonian limit."""
    d = _load("case_d.json")
    e = _load("case_e.json")
    f = _load("case_f.json")
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0))
    # (d)
    ax = axes[0]
    labels, vals = [], []
    if d:
        m = d["cases"]["d"]["metric"]
        for k, v in m.items():
            if v is not None:
                labels.append(k.replace("_", "\n"))
                vals.append(v)
    if vals:
        ax.bar(range(len(vals)), vals, color="C0")
        ax.axhspan(0.97, 1.03, color="C2", alpha=0.2, label="acceptance [0.97,1.03]")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=6)
        ax.set_ylim(0.8, 1.2)
        ax.set_ylabel("closure rate")
        ax.legend(fontsize=7)
    ax.set_title("(d) momentum ledger closures")
    # (e)
    ax = axes[1]
    if e:
        runs = e["cases"]["e"]["runs"]
        for tag, marker in (("newtonian_limit", "o"), ("bingham", "s")):
            ks = [k for k in runs if k.startswith(tag)]
            if not ks:
                continue
            Ns = [int(k.split("_N")[1]) for k in ks]
            fs = [runs[k]["Q_3D_sim"] / (runs[k]["Q_2D_sim"] * int(k.split("_N")[1]))
                  for k in ks]
            fr = [runs[k]["Q_3D_ref"] / (runs[k]["Q_2D_exact"] * int(k.split("_N")[1]))
                  for k in ks]
            ax.plot(Ns, fs, marker + "-", label="LBM %s" % tag)
            ax.plot(Ns, fr, marker + "--", alpha=0.6, label="analytic %s" % tag)
        ax.axhline(1.0 / 2.371179480124106, color="k", ls=":",
                   label="Newtonian f=0.4217")
        ax.set_xlabel("N"); ax.set_ylabel("$Q_{3D}/(Q_{2D}N)$ = shape factor")
        ax.set_title("(e) 2D/3D flow ratio / duct shape factor")
        ax.legend(fontsize=6)
    # (f)
    ax = axes[2]
    if f:
        runs = f["cases"]["f"]["runs"]
        ks = sorted(runs)
        ax.bar(range(len(ks)), [100 * runs[k]["linf_rel"] for k in ks], color="C3")
        ax.axhline(1.0, color="k", ls="--", label="1% acceptance")
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels(ks, fontsize=7)
        ax.set_ylabel("$L_\\infty$ rel error [%]")
        ax.set_title("(f) Bn=0 Newtonian limit")
        ax.legend(fontsize=7)
    fig.tight_layout()
    out = os.path.join(FIG, "def_ledger_ratio_newtonian.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main():
    os.makedirs(FIG, exist_ok=True)
    made = []
    for fn in (fig_duct, fig_couette, fig_ladder, fig_ledger_ratio):
        try:
            out = fn()
            if out:
                made.append(out)
                print("wrote", out)
            else:
                print("skipped", fn.__name__, "(missing inputs)")
        except Exception as exc:                     # noqa: BLE001
            print("failed", fn.__name__, exc)
    with open(os.path.join(FIG, "figures.json"), "w") as fh:
        json.dump(made, fh, indent=2)
    return made


if __name__ == "__main__":
    main()