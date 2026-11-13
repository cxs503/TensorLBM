#!/usr/bin/env python3
"""Figures for board audit m17878 (all numbers read from artifacts/verify_m17757.json)."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ART = json.loads((HERE / "artifacts/verify_m17757.json").read_text())
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)


def fig1():
    shipped = ART["C3_degeneracy"]["shipped_kpis"]
    labels = [s["case"] for s in shipped]
    informative = [s["A1_vs_analytic"] for s in shipped]
    cross = [np.nan if s["A1_vs_A2"] is None else s["A1_vs_A2"] for s in shipped]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.bar(x - 0.2, [v * 100 for v in informative], width=0.4,
           color="#2b6cb0", label="A1 vs analytic normal (informative)")
    ax.bar(x + 0.2, [v * 100 for v in cross], width=0.4,
           color="#cbd5e0", edgecolor="#718096",
           label="A1 vs A2 (declared headline: degenerate, 0 by construction)")
    ax.axhline(15.0, color="#c53030", ls="--", lw=1.5, label="owner bar 15%")
    for xi, v in zip(x - 0.2, informative):
        ax.text(xi, v * 100 + 0.25, f"{v * 100:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("direction error  1 - mean|n·n_ref|  [%]")
    ax.set_title("m17757 XVAL-LBM-FILL-GRADIENT-NORMAL: what the KPIs actually measure\n"
                 "analytic anchor passes 15% (max 10.24%); cross-arm KPI is an identity",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = FIG / "fig1_kpi_informative_vs_degenerate.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig2():
    import sys
    sys.path.insert(0, str(HERE.parents[2] / "src"))
    from tensorlbm import free_surface_lbm as fsl
    from tensorlbm import free_surface_common as fsc
    phi = torch.zeros(16, 16, 16, dtype=torch.float64)
    phi[:, :, :8] = 1.0
    ax_, _, _ = fsl._compute_interface_normal(torch.zeros_like(phi, dtype=torch.int32),
                                              phi, torch.ones_like(phi))
    lx, _, _, _ = fsc.interface_normal_3d(phi)
    xs = np.arange(16)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    a.plot(xs, ax_[8, 8, :].numpy(), "o-", color="#2b6cb0", label="audited: _compute_interface_normal (dead)")
    a.plot(xs, lx[8, 8, :].numpy(), "s--", color="#dd6b20", label="live: free_surface_common.interface_normal_3d")
    a.axvline(7.5, color="k", lw=0.8, alpha=0.5)
    a.set_xlabel("x index (liquid $\\phi$=1 for x<8, gas for x>=8)")
    a.set_ylabel("$n_x$")
    a.set_title("Sign conventions are opposite\n(and the board KPI uses |n·n|, so it cannot see this)",
                fontsize=9)
    a.legend(fontsize=7.5)
    a.grid(alpha=0.3)

    rows = ART["C3_degeneracy"]["random_field_rows"]
    names = [r["field"] for r in rows]
    vals = [abs(r["dir_err_A1_vs_A2"]) for r in rows]
    b.bar(names, vals, color="#cbd5e0", edgecolor="#718096")
    for i, v in enumerate(vals):
        b.text(i, v * 1.5, f"{v:.1e}", ha="center", fontsize=8)
    b.set_yscale("log")
    b.set_ylim(1e-20, 1.0)
    b.axhline(0.15, color="#c53030", ls="--", lw=1.2, label="owner bar 15%")
    b.set_ylabel("A1 vs A2 direction error (|·|, log)")
    b.set_title("Degeneracy proof: the cross-arm error is ~1e-17\non arbitrary fields — it must be 0 for ANY input",
                fontsize=9)
    b.tick_params(axis="x", labelsize=8)
    b.legend(fontsize=8)
    b.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = FIG / "fig2_sign_flip_and_degeneracy.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


if __name__ == "__main__":
    ps = [fig1(), fig2()]
    b64 = {p.name: base64.b64encode(p.read_bytes()).decode() for p in ps}
    (HERE / "figures_b64.json").write_text(json.dumps(b64))
    print(json.dumps({p.name: p.stat().st_size for p in ps}, indent=2))