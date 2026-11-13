#!/usr/bin/env python3
"""Figures for XVAL-LBM-NO-TESTS-NO-CI-NO-ASSERTS (all data measured, none synthetic)."""
from __future__ import annotations
import os, sys, json
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, SRC)
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
torch.set_num_threads(1)


def fig_poiseuille():
    from tensorlbm.verification.xval.registry import XValRegistry
    from tensorlbm.verification.xval.cases import register_default_cases
    reg = XValRegistry(); register_default_cases(reg)
    case = reg.get("poiseuille_2d_H16")
    rec = case.run_fn()
    u, ua = np.asarray(rec["arrays"]["u_profile"]), np.asarray(rec["arrays"]["u_analytical"])
    res = reg.run_case("poiseuille_2d_H16", compute_hash=False)
    y = np.arange(u.size) + 0.5
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4))
    a1.plot(u / u.max(), y / u.size, "o", ms=4, label="LBM (H=16, tau=0.8)")
    a1.plot(ua / ua.max(), y / ua.size, "-", lw=2, label="analytic y(H-y)")
    a1.set_xlabel(r"$u/u_{max}$"); a1.set_ylabel("y/H"); a1.legend(); a1.set_title("plane Poiseuille")
    a2.plot(u / u.max(), (u - ua) / ua.max() * 100, "s-", ms=4)
    a2.set_xlabel(r"$u/u_{max}$"); a2.set_ylabel("(u_LBM - u_exact)/u_max [%]")
    a2.set_title(f"eps_inf={res.metrics['eps_inf']:.2%}, eps_L2={res.metrics['eps_L2']:.2%}")
    fig.tight_layout(); p = os.path.join(FIG, "fig1_poiseuille_vs_analytic.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    return p, res.metrics


def fig_cavity_divergence():
    from tensorlbm.verification.lid_driven_cavity.solver import CavityConfig, CavitySolver
    series = {}
    onsets = {}
    for n in (33, 65):
        s = CavitySolver(CavityConfig(n=n, re=100, max_steps=200, u_lid=0.1))
        mx = []
        for step in range(1, 201):
            s.step()
            m = float(torch.abs(s.f).max())
            if not np.isfinite(m):
                onsets[n] = step
                break
            mx.append(m)
        series[n] = (np.arange(1, len(mx) + 1), np.array(mx))
        onsets.setdefault(n, None)
    fig, ax = plt.subplots(figsize=(6, 4))
    for n, (st, v) in series.items():
        ax.semilogy(st, v, lw=1.5, label=f"n={n} (NaN at step {onsets[n]})")
    ax.set_xlabel("LBM step"); ax.set_ylabel(r"$\max|f_i|$")
    ax.set_title("lid-driven cavity Re=100: state blow-up (no assertion caught it)")
    ax.legend(); ax.grid(True, which="both", alpha=.3)
    fig.tight_layout(); p = os.path.join(FIG, "fig2_cavity_divergence.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    return p, onsets


if __name__ == "__main__":
    p1, m = fig_poiseuille()
    p2, onsets = fig_cavity_divergence()
    print(json.dumps({"poiseuille_metrics": m, "cavity_nan_onset_step": onsets,
                      "figures": [p1, p2]}, indent=2))
