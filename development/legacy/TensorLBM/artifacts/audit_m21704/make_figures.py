"""Figures for the m21704 board audit of XVAL-LBM-TWO-COMPONENT-SC-LAYERED-POISEUILLE-EDM.

fig1: velocity profile of the driven layered case vs the exact piecewise-quadratic
      reference (nominal H/2 interface and the mass-controlled measured interface),
      plus the demixing profile phi_A(y).
fig2: KPI error bars for every case of the pre-registered matrix (m21640 artifact +
      the two cases this audit ran), against the 15 % owner gate and 3 % self target.

Run:  PYTHONPATH=src python artifacts/audit_m21704/make_figures.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

from tensorlbm.verification.layered_poiseuille.analytic import solve_layered  # noqa: E402

OWNER, TARGET = 0.15, 0.03


def load():
    art = json.load(open(os.path.join(HERE, "m21640_results.json")))
    rer = json.load(open(os.path.join(HERE, "results_rerun_all6.json")))
    recs = {r["case"]: r for r in art["records"]}
    for r in rer["records"]:
        recs.setdefault(r["case"], r)
    return recs


def fig1(recs):
    r = recs["layered_r167"]
    y = np.asarray(r["profiles"]["y"])
    u = np.asarray(r["profiles"]["ux_total"])
    H = r["height"]
    nom = solve_layered(H, [0.5 * H], [r["nu_a"], r["nu_b"]], r["config"]["g_x"])
    match = solve_layered(H, [r["metrics"]["y_iface"]], [r["nu_a"], r["nu_b"]], r["config"]["g_x"])
    phi = np.asarray(r["profiles"]["rho_a"]) / (
        np.asarray(r["profiles"]["rho_a"]) + np.asarray(r["profiles"]["rho_b"]) + 1e-12)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6))
    ax1.plot(u, y, "o-", ms=3.5, lw=1.4, label="MCMP LBM (m21640 artifact)")
    ax1.plot(nom.u(y), y, "k--", lw=1.1, label="exact, interface at H/2")
    ax1.plot(match.u(y), y, "r:", lw=1.4, label=f"exact, measured iface y={r['metrics']['y_iface']:.2f}")
    ax1.axhline(0.5 * H, color="0.6", lw=0.7)
    ax1.axhline(r["metrics"]["y_iface"], color="r", lw=0.7, ls=":")
    ax1.set_xlabel(r"$u_x$ [lattice units]")
    ax1.set_ylabel("y [lattice units]")
    ax1.set_title(f"layered Poiseuille, nu=({r['nu_a']:.3f},{r['nu_b']:.3f}), G={r['config']['G']}\n"
                  f"eps_inf={100*r['metrics']['eps_inf']:.2f}%  u_peak_err={100*r['metrics']['u_peak_err']:.2f}%")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(alpha=0.3)

    ax2.plot(phi, y, "s-", ms=3.5, lw=1.3, color="tab:blue")
    ax2.axvline(0.5, color="k", ls=":", lw=0.8)
    ax2.axhline(r["metrics"]["y_iface"], color="r", lw=0.8, ls="--", label="measured interface")
    ax2.axhline(0.5 * H, color="0.5", lw=0.8, label="nominal H/2")
    ax2.set_xlabel(r"$\phi_A=\rho_A/(\rho_A+\rho_B)$")
    ax2.set_ylabel("y [lattice units]")
    ax2.set_title("demixing: A below / B above (interface drifts 3.2 cells)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    p1 = os.path.join(HERE, "fig1_layered_profile_vs_exact.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    return p1


def fig2(recs):
    order = [c for c in ("control_single", "control_mixture", "layered_r167",
                         "layered_r233", "layered_r167_h48") if c in recs]
    kpis = ["eps_inf", "eps_l2_rel", "Q_err", "u_peak_err"]
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    w = 0.2
    x = np.arange(len(order))
    for k, kpi in enumerate(kpis):
        vals = [100 * recs[c]["metrics"].get(kpi, np.nan) for c in order]
        ax.bar(x + (k - 1.5) * w, vals, w, label=kpi)
    ax.axhline(100 * OWNER, color="r", ls="--", lw=1.0, label="owner gate 15%")
    ax.axhline(100 * TARGET, color="g", ls=":", lw=1.0, label="self target 3%")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n", 1) for c in order], fontsize=8)
    ax.set_ylabel("error [%]")
    ax.set_title("XVAL layered-Poiseuille KPIs (first three cases: m21640; last two: run by audit m21704)")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p2 = os.path.join(HERE, "fig2_kpi_errors.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    return p2


if __name__ == "__main__":
    recs = load()
    print(fig1(recs))
    print(fig2(recs))