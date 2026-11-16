"""Figures for the R2 gated added-mass ladder (reads ``runs/*/result.json``).

fig1  in-phase coefficient vs beta against the three references
      (inviscid potential same-mask, exact unsteady-Stokes Gamma(beta), analytic 1)
fig2  error against the viscous reference vs. Stokes-layer resolution in cells
fig3  produced by ``make_field_figure.py`` (interior-zeroing defect)
fig4  quadrature (viscous) channel: measured vs -Im Gamma(beta)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorlbm.oscillatory_cylinder_inertia import stokes_cylinder_coefficients

from analyze_r2 import load

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"


def main() -> int:
    FIG.mkdir(exist_ok=True)
    arms = [a for a in load() if a.get("status") == "ok"]
    beta_arms = sorted([a for a in arms if a["_name"] in
                        ("r2_b25_d32", "r2_b50_d32", "r2_b100_d32")],
                       key=lambda a: a["case"]["beta"])
    betas = [a["case"]["beta"] for a in beta_arms]

    # ---- fig1 ------------------------------------------------------------- #
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.9))
    b_fine = np.logspace(math.log10(12), math.log10(400), 200)
    g = [stokes_cylinder_coefficients(float(b)) for b in b_fine]
    ax[0].plot(b_fine, [x["Gamma_re"] for x in g], "k-", lw=2,
               label=r"Stokes $\Gamma_{re}(\beta)$ (exact unsteady)")
    ax[0].plot(b_fine, [x["C_Q_viscous"] for x in g], "k--", lw=1.4,
               label=r"Stokes $-\Gamma_{im}(\beta)$")
    ax[0].axhline(1.0, color="0.5", ls=":", lw=1.2, label=r"potential flow, $C_a=1$")
    if beta_arms:
        c_pot = beta_arms[0]["reference"]["c_a_level1"]
        ax[0].axhline(c_pot, color="#8e44ad", ls="-.", lw=1.2,
                      label=fr"same-mask potential $C_a$={c_pot:.3f}")
    for key, col, mk in (("E1_mem_area", "#1f77b4", "o"), ("E2_trac_area", "#d35400", "s"),
                         ("E3_global_area", "#2ca02c", "^")):
        ax[0].plot(betas, [a["kpi"][key]["C_I"] for a in beta_arms], mk + "-", color=col,
                   label=f"{key.split('_')[0]} in-phase ($C_I$)")
        ax[0].plot(betas, [a["kpi"][key]["C_Q"] for a in beta_arms], mk + ":", color=col,
                   alpha=.65, lw=1.0)
    ax[0].set_xscale("log")
    ax[0].set_xlabel(r"$\beta = D^2/(\nu T)$")
    ax[0].set_ylabel("coefficient")
    ax[0].set_title("R2 fig1 — measured in-phase/quadrature vs the analytic unsteady "
                    "Stokes reference", fontsize=10)
    ax[0].legend(fontsize=7.5, ncol=1, loc="upper right")
    ax[0].grid(alpha=.3, which="both")

    for a in beta_arms:
        e1 = a["err_inphase_pct"]["E1_vs_stokes"]
        ep = a["err_inphase_pct"]["E1_vs_potential"]
        ax[1].plot(a["case"]["beta"], e1, "o", color="#1f77b4", ms=8)
        ax[1].plot(a["case"]["beta"], ep, "x", color="#c0392b", ms=8)
    ax[1].axhspan(-15, 15, color="#2ca02c", alpha=.13, label="owner gate ±15 %")
    ax[1].axhspan(-3, 3, color="#2ca02c", alpha=.25, label="platform caliber ±3 %")
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"$\beta$")
    ax[1].set_ylabel("in-phase error [%]")
    ax[1].set_title("R2 fig1b — same measurement, two references:  o = Stokes viscous,  "
                    "x = inviscid potential", fontsize=10)
    ax[1].legend(fontsize=7.5)
    ax[1].grid(alpha=.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_beta_ladder.png", dpi=130)

    # ---- fig2: error vs Stokes-layer resolution ---------------------------- #
    fig2, ax2 = plt.subplots(figsize=(7.8, 5.0))
    pts = [(a["stokes_layer_cells"], a["err_inphase_pct"]["E1_vs_stokes"], a["_name"])
           for a in arms if a["_name"] != "r2_b50_d32_int4"]
    pts.sort()
    ax2.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color="#1f77b4")
    for x, y, n in pts:
        ax2.annotate(n.replace("r2_", ""), (x, y), textcoords="offset points",
                     xytext=(6, 5), fontsize=7)
    ax2.axhspan(-15, 15, color="#2ca02c", alpha=.13, label="owner gate ±15 %")
    ax2.axhspan(-3, 3, color="#2ca02c", alpha=.25, label="platform caliber ±3 %")
    ax2.axhline(0, color="k", lw=.8)
    ax2.set_xlabel(r"Stokes layer thickness $\delta = \sqrt{2\nu/\omega}$  [lattice cells]")
    ax2.set_ylabel("in-phase error vs $C_a\\cdot\\Gamma_{re}$ [%]")
    ax2.set_title("R2 fig2 — the error is set by Stokes-layer resolution", fontsize=10)
    ax2.grid(alpha=.3)
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(FIG / "fig2_resolution.png", dpi=130)

    # ---- fig4: quadrature channel ------------------------------------------ #
    fig4, ax4 = plt.subplots(figsize=(7.8, 5.0))
    ax4.plot(b_fine, [x["C_Q_viscous"] for x in g], "k-", lw=2, label=r"$-\Gamma_{im}$")
    for key, col, mk in (("E1_mem_area", "#1f77b4", "o"), ("E2_trac_area", "#d35400", "s"),
                         ("E3_global_area", "#2ca02c", "^")):
        ax4.plot(betas, [a["kpi"][key]["C_Q"] for a in beta_arms], mk, color=col,
                 label=f"{key.split('_')[0]} $C_Q$")
    ax4.set_xscale("log")
    ax4.set_xlabel(r"$\beta$")
    ax4.set_ylabel(r"$C_Q$")
    ax4.set_title("R2 fig4 — quadrature (viscous) channel: estimators vs Stokes", fontsize=10)
    ax4.grid(alpha=.3, which="both")
    ax4.legend(fontsize=8)
    fig4.tight_layout()
    fig4.savefig(FIG / "fig4_quadrature.png", dpi=130)

    print("figures written:", sorted(p.name for p in FIG.glob("*.png")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())