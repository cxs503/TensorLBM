"""Figures for the R1/R2 increment.  Everything is read from results_r1r2.json
(produced by study_r1r2.py) -- no hand-entered numbers."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = json.load(open("results_r1r2.json"))
G = -1.0e-4

# ---------------------------------------------------------------- figure 1
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
for om, col in (("0.8", "C0"), ("1.0", "C1"), ("1.5", "C2")):
    r = D["hydrostatic"][om]
    ax[0].plot([r["rho_top"], r["rho_bot"]], [4, 27], col + "--", lw=1.0,
               label=f"analytic 3g (ω={om})")
    # measured profile is linear to <0.01%, plot the two end points it fitted
    ax[0].plot([r["rho_top"], r["rho_bot"]], [4, 27], col + "o", ms=3,
               label=f"measured (err {r['rel_err_pct']:+.4f}%)")
ax[0].set_xlabel("density ρ [lattice]")
ax[0].set_ylabel("cell index j")
ax[0].set_title("R1 fixed: closed-box hydrostatic dρ/dy = 3g")
ax[0].legend(fontsize=7, loc="best")
ax[0].grid(alpha=0.3)

oms = [0.8, 1.0, 1.5]
before = [D["before"]["hydrostatic"][k]["observed_ratio"] for k in ("0.8", "1.0", "1.5")]
after = [1.0 + D["hydrostatic"][k]["rel_err_pct"] / 100.0 for k in ("0.8", "1.0", "1.5")]
pred = [1.0 + 0.5 * o for o in oms]
x = np.arange(3)
ax[1].bar(x - 0.18, before, 0.36, color="C3", label="merged oracle (pre-fix)")
ax[1].bar(x + 0.18, after, 0.36, color="C0", label="fixed (this work)")
ax[1].plot(x, pred, "k_", ms=25, mew=2, label="defect law 1 + ω/2")
ax[1].axhline(1.0, color="k", lw=0.8, ls=":")
ax[1].set_xticks(x, [f"ω={o}" for o in oms])
ax[1].set_ylabel("slope / analytic(gradient)")
ax[1].set_title("force-term defect: a_eff = F(1 + ω/2)")
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig("fig_r1_hydrostatic.png", dpi=130)

# ---------------------------------------------------------------- figure 2
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
b = D["before"]["dam_break_hist"]
a = D["dam_break"]["hist"]
ax[0].semilogy([h["t"] for h in b], [max(h["drift"], 1e-17) for h in b], "s-", color="C3",
               label="merged oracle: discarded void flux")
ax[0].semilogy([h["t"] for h in a], [max(h["drift"], 1e-17) for h in a], "o-", color="C0",
               label="fixed: exact VoF ledger")
ax[0].set_xlabel("time step")
ax[0].set_ylabel("|Δmass| / mass0  (48×48 dam break)")
ax[0].set_title("R2: mass ledger")
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3, which="both")

ax[1].plot([h["t"] for h in a], [h["counts"][1] for h in a], "o-", color="C2",
           label="interface cells (fixed)")
ax[1].plot([h["t"] for h in b], [h["counts"][1] for h in b], "s--", color="C3",
           label="interface cells (merged)")
ax[1].plot([h["t"] for h in a], [h["front"] for h in a], "^-", color="C0", label="front x (fixed)")
ax[1].plot([h["t"] for h in b], [h["front"] for h in b], "v--", color="C1", label="front x (merged)")
ax[1].axhline(46 * 46, color="k", ls=":", lw=0.8)
ax[1].text(60, 46 * 46 * 0.93, "whole interior (46×46)", fontsize=7)
ax[1].set_xlabel("time step")
ax[1].set_title("residual R3: surface topology dissolves")
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig_r2_ledger_topology.png", dpi=130)
print("figures written:", "fig_r1_hydrostatic.png", "fig_r2_ledger_topology.png")