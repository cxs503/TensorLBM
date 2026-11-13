"""Figures + machine-generated tables for the Porter-forcing cross-benchmark."""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

d = json.load(open(os.path.join(HERE, "artifacts", "result.json")))
runs = d["runs"]


def find(bench, arm, M, ny, mode="fixed_point"):
    for r in runs:
        if r["bench"] == bench and r["arm"] == arm and r["M"] == M and r["ny"] == ny and r.get("mode", "fixed_point") == mode:
            return r
    return None


# ---------------------------------------------------------------- figure 1
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=False)
for ax, M in zip(axes, (1.0, 10.0)):
    r = find("B2_layered_channel", "doc_literal", M, 66)
    rl = find("B2_layered_channel", "legacy", M, 66)
    y = np.array(r["y"])
    ax.plot(y, np.array(r["u_exact"]), "k-", lw=2, label="analytic two-layer Stokes")
    ax.plot(y, np.array(r["ux"]), "o", ms=3, color="tab:blue", label=f"Porter explicit forcing (err {r['err']*100:.1f}%)")
    ax.plot(y, np.array(rl["ux"]), "s", ms=3, color="tab:red", mfc="none",
            label=f"legacy Shan-Chen forcing (err {rl['err']*100:.1f}%)")
    ax.axvline(32.0, color="0.6", ls=":", lw=1)
    ax.set_xlabel("wall distance y [lattice units]")
    ax.set_ylabel(r"$u_x$ [lattice units]")
    ax.set_title(f"layered channel, viscosity ratio M={M:g}, NY=66")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_layered_profiles.png"), dpi=150)

# ---------------------------------------------------------------- figure 2
fig, ax = plt.subplots(figsize=(6.2, 4.2))
for M, col in ((1.0, "tab:blue"), (10.0, "tab:orange")):
    for arm, mk in (("doc_literal", "o"), ("legacy", "s")):
        ns, es = [], []
        for ny in (18, 34, 66):
            r = find("B2_layered_channel", arm, M, ny)
            ns.append(ny - 2)
            es.append(r["err"])
        ax.loglog(ns, es, mk + "-", color=col, label=f"M={M:g}, {arm}")
ax.axhline(0.15, color="k", ls="--", lw=1, label="owner criterion 15%")
ax.axhline(0.03, color="0.5", ls=":", lw=1, label="stretch 3%")
ax.set_xlabel("channel height H = NY-2 [lattice units]")
ax.set_ylabel(r"max profile error / max $u_{exact}$")
ax.set_title("grid convergence of the layered benchmark")
ax.legend(fontsize=7)
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_convergence.png"), dpi=150)

# ---------------------------------------------------------------- figure 3
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
orders = sorted(d["b3"], key=int)
names = [f"order {o}" for o in orders]
vals_t = [d["b3"][o]["flat_max_tangential"] for o in orders]
vals_r = [d["b3"][o]["drop_max_tangential"] for o in orders]
axes[0].bar(names, vals_t, color="tab:blue")
axes[0].set_title("flat interface: max tangential (spurious) force")
axes[0].set_ylabel(r"$\max|\partial_x \psi|$")
axes[1].bar(names, vals_r, color="tab:green")
axes[1].set_title("droplet: max tangential force")
axes[1].set_ylabel(r"$\max|F_t|$")
for ax in axes:
    ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3_stencils.png"), dpi=150)

# ---------------------------------------------------------------- table
lines = ["| bench | arm | NY | M | mode | err | err excl. interface | mass drift | unstable |",
         "|---|---|---|---|---|---|---|---|---|"]
for r in runs:
    lines.append(
        "| {bench} | {arm} | {ny} | {M:g} | {mode} | {err:.3e} | {err_excl:.3e} | {md} | {un} |".format(
            bench=r["bench"], arm=r["arm"], ny=r["ny"], M=r["M"], mode=r.get("mode", "fixed_point"),
            err=r["err"], err_excl=r["err_excl"],
            md=("nan" if r.get("mass_drift") is None or not np.isfinite(r.get("mass_drift", float("nan"))) else f"{r['mass_drift']:.1e}"),
            un=r["unstable"],
        )
    )
lines.append("")
lines.append("| stencil | sum w | sum w cx^2 | A4 | A6 | A8 | flat max tangential | drop tang/radial (shell) |")
lines.append("|---|---|---|---|---|---|---|---|")
for o in orders:
    b = d["b3"][o]
    m = b["isotropy"]
    lines.append(
        f"| {o} | {m['sum_w']:.6f} | {m['sum_w_cx2']:.6f} | {m['A4']:.6f} | {m['A6']:.6f} | {m['A8']:.6f} | "
        f"{b['flat_max_tangential']:.6f} | {b['drop_tangential_over_radial_shell']:.6f} |"
    )
open(os.path.join(HERE, "artifacts", "table.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("figures:", os.listdir(FIG))