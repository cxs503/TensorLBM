"""Figures for round 3, generated only from summary.json (zero hand-copied numbers)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "summary.json").read_text())
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
C1 = S["predeclared"] and json.loads((HERE / "predeclared.json").read_text())["acceptance_criteria"]["C1_code_equivalence"]["threshold"]

# fig1: pointwise error vs step (dense config)
fig, ax = plt.subplots(figsize=(7.2, 4.4))
for curve in S["per_step_curves"]:
    ax.semilogy(range(1, len(curve["eps_inf"]) + 1), curve["eps_inf"], marker=".",
                label=curve["arm"])
ax.axhline(C1, color="k", ls="--", lw=1, label=f"gate {C1:.0e}")
ax.set_xlabel("step n"); ax.set_ylabel(r"$\varepsilon_\infty$ vs lbmpy reference")
ax.set_title("D3Q19 whole step, 8x8x8, tau=0.7, seed 0")
ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(FIG / "fig1_eps_vs_step.png", dpi=140); plt.close(fig)

# fig2: eps at t=30 vs tau, all arms/grids/seeds
fig, ax = plt.subplots(figsize=(7.2, 4.4))
equiv = S["code_equivalence"]
arms = sorted({r["arm"] for r in equiv})
for arm in arms:
    for grid in sorted({tuple(r["grid"]) for r in equiv}):
        sel = [r for r in equiv if r["arm"] == arm and tuple(r["grid"]) == grid]
        taus = [r["tau"] for r in sel]
        vals = [r["eps_inf_t30"] for r in sel]
        ax.semilogy(taus, vals, "o-", ms=4, lw=1,
                    label=f"{arm} {'x'.join(map(str, grid))}")
ax.axhline(C1, color="k", ls="--", lw=1, label=f"gate {C1:.0e}")
ax.set_xlabel("tau"); ax.set_ylabel(r"$\varepsilon_\infty$ at t=30 (worst seed)")
ax.set_title("D3Q19 whole-step pointwise error across relaxation times")
ax.legend(fontsize=5.5, ncol=2); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(FIG / "fig2_eps_vs_tau.png", dpi=140); plt.close(fig)

# fig3: physics arms vs 3% / 15% criteria
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
sh = S["physics_shear_wave"]
axes[0].bar([f"n={r['n']}\ntau={r['tau']}" for r in sh],
            100 * np.array([r["err_vs_continuum"] for r in sh]), color="tab:blue", alpha=.8,
            label=r"vs $\nu k^2$ (continuum)")
axes[0].plot(range(len(sh)), 100 * np.array([r["err_vs_lattice_dispersion"] for r in sh]),
             "ro", label=r"vs $\nu k_{eff}^2$ (lattice)")
axes[0].axhline(3, color="k", ls="--", lw=1, label="3% criterion")
axes[0].axhline(15, color="r", ls=":", lw=1, label="15% owner limit")
axes[0].set_ylabel("shear-wave decay-rate error [%]"); axes[0].set_ylim(0, 16)
axes[0].set_title("3-D shear wave: measured vs analytic decay")
axes[0].legend(fontsize=7); axes[0].grid(alpha=.3, axis="y")

ab = S["physics_abc"]
axes[1].bar([f"n={r['n']}\ntau={r['tau']}" for r in ab],
            100 * np.array([r["err_vs_analytic"] for r in ab]), color="tab:green", alpha=.8)
axes[1].axhline(3, color="k", ls="--", lw=1, label="3% criterion")
axes[1].axhline(15, color="r", ls=":", lw=1, label="15% owner limit")
axes[1].set_ylabel("ABC decay-rate error [%]"); axes[1].set_title(r"ABC flow: measured vs $e^{-\nu t}$")
axes[1].legend(fontsize=7); axes[1].grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig(FIG / "fig3_physics_arms.png", dpi=140); plt.close(fig)

# fig4: equilibrium-convention arm
conv = S["convention"]
fig, ax = plt.subplots(figsize=(6.4, 4.0))
labels = ["discrete eq.\n(lbmpy, moment-conserving)", "continuous eq.\n(lbmpy default)"]
vals = [conv["discrete"]["eps_inf_t30"], conv["continuous"]["eps_inf_t30"]]
ax.bar(labels, vals, color=["tab:blue", "tab:orange"])
for i, v in enumerate(vals):
    ax.text(i, v * 1.15, f"{v:.2e}", ha="center", fontsize=9)
ax.set_yscale("log"); ax.set_ylabel(r"$\varepsilon_\infty$ at t=30")
ax.set_title("Repo vs lbmpy reference under the two equilibrium conventions\n"
             f"(continuous rest mass ratio sum f/rho = {conv['continuous_eq_rest_mass_ratio']:.6f})")
ax.grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig(FIG / "fig4_equilibrium_convention.png", dpi=140); plt.close(fig)
print("figures:", sorted(p.name for p in FIG.glob("*.png")))
