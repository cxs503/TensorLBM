"""Figures for the displacement-Aitken XVAL (reads out/results.json)."""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

ARM_LABEL = {
    "ours_old_default": "TensorLBM pre-fix default (clip floor 0.05)",
    "ours_new_default": "TensorLBM fixed default (clip floor 1e-4)",
    "ours_noclip": "TensorLBM, clip=False (raw Aitken)",
    "ours_life_conv": "TensorLBM + LIFE conventions (predictor)",
    "none_control": "plain fixed point (omega = 1), control",
    "oracle_constant": "constant omega = 1/(1+beta) (oracle)",
    "life_ref": "joconnor22/LIFE transcription",
}
STYLE = {
    "ours_old_default": dict(color="crimson", marker="x", ls="--"),
    "ours_new_default": dict(color="tab:blue", marker="o"),
    "ours_noclip": dict(color="tab:cyan", marker="s", ls=":"),
    "ours_life_conv": dict(color="tab:green", marker="^"),
    "none_control": dict(color="grey", marker=".", ls="-."),
    "oracle_constant": dict(color="k", marker="v", ls=":"),
    "life_ref": dict(color="tab:purple", marker="D", ls="--"),
}


def main() -> int:
    d = json.load(open(os.path.join(OUT, "results.json")))
    rows = d["rows"]
    betas = sorted({r["beta_eff"] for r in rows})
    arms = [a for a in ARM_LABEL if any(r["arm"] == a for r in rows)]

    # ---- fig 1: sub-iterations per time step vs beta ---------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for a in arms:
        xs = [r["beta_eff"] for r in rows if r["arm"] == a]
        ys = [r["iters_mean"] for r in rows if r["arm"] == a]
        conv = [r["converged"] for r in rows if r["arm"] == a]
        ax.plot(xs, ys, label=ARM_LABEL[a], ms=5, lw=1.4, **STYLE[a])
        for x, y, c in zip(xs, ys, conv):
            if not c:
                ax.annotate("FAIL", (x, y), textcoords="offset points", xytext=(4, 6),
                            color="crimson", fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"added-mass ratio $\beta_{eff}=m_a/m_{eff}$")
    ax.set_ylabel("sub-iterations per time step (mean)")
    ax.set_title("Interface fixed-point cost, $\\Delta t/T=1/200$, 3 periods")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_iterations_vs_beta.png"), dpi=140)
    plt.close(fig)

    # ---- fig 2: recovered Aitken factor vs analytic optimum --------------
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bb = np.geomspace(min(betas) * 0.5, max(betas) * 1.5, 100)
    ax.plot(bb, 1.0 / (1.0 + bb), "k--", lw=1.2, label=r"analytic $\omega^*=1/(1+\beta)$")
    for a in ("ours_old_default", "ours_new_default", "ours_life_conv", "life_ref"):
        xs = [r["beta_eff"] for r in rows if r["arm"] == a]
        ys = [r["omega_final"] for r in rows if r["arm"] == a]
        ax.plot(xs, ys, label=ARM_LABEL[a], ms=5, lw=1.2, **STYLE[a])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\beta_{eff}$")
    ax.set_ylabel(r"final Aitken factor $\omega$")
    ax.set_title("Recovered dynamic relaxation vs analytic optimum")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_omega_vs_beta.png"), dpi=140)
    plt.close(fig)

    # ---- fig 3: interface displacement history, beta = 25 ---------------
    beta_pick = 25.0
    h = d["histories"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0))
    ref = h[f"ref@{beta_pick}"]
    ax1.plot(ref["t"], ref["d_mono"], "k-", lw=2.2, label="monolithic Newmark (REF-D)")
    ax1.plot(ref["t"], ref["d_ana"], "k:", lw=1.0, label="closed form (REF-C)")
    for a in ("ours_new_default", "life_ref"):
        hh = h[f"{a}@{beta_pick}"]
        ax1.plot(hh["t"], hh["d"], lw=1.0, alpha=0.9, label=ARM_LABEL[a], **STYLE[a])
    ax1.set_xlabel("t [s]")
    ax1.set_ylabel("interface displacement d [m]")
    ax1.set_title(rf"$\beta_{{eff}}={beta_pick:g}$: all converged arms coincide")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=7)

    for a in ("ours_old_default", "life_ref"):
        hh = h[f"{a}@{beta_pick}"]
        res = np.asarray(hh["residuals_first_step"], dtype=float)
        ax2.semilogy(np.arange(1, len(res) + 1), res, label=ARM_LABEL[a], **STYLE[a])
    ax2.axhline(1e-8, color="k", ls=":", label=r"target $10^{-8}$")
    ax2.set_xlabel("sub-iteration k (first time step)")
    ax2.set_ylabel("normalised interface residual")
    ax2.set_title("Residual decay at the first step")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_history_and_residual_beta25.png"), dpi=140)
    plt.close(fig)

    # ---- fig 4: discretisation sensitivity -------------------------------
    sd = d["sensitivity_dt"]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for beta in sorted({r["beta_eff"] for r in sd}):
        sel = [r for r in sd if r["beta_eff"] == beta]
        xs = [r["steps_per_period"] for r in sel]
        ax.loglog(xs, [r["err_vs_analytic_pct_ours"] for r in sel], "o-", label=rf"ours, $\beta$={beta:g}")
        ax.loglog(xs, [r["err_vs_analytic_pct_life"] for r in sel], "s--", label=rf"LIFE, $\beta$={beta:g}")
    ax.set_xlabel("time steps per coupled period")
    ax.set_ylabel("max error vs closed form [%]")
    ax.set_title("Time-discretisation sensitivity (2nd order)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_dt_sensitivity.png"), dpi=140)
    plt.close(fig)
    print("figures written to", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())