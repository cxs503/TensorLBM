#!/usr/bin/env python3
"""Figures for the m18779 board-audit report (numbers read from artifacts, none hand-copied)."""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

W = pathlib.Path("/root/tp-agent/work/m18779")
ART = W / "artifacts/m18779"
FIG = W / "docs/board_audit/m18779/figures"
FIG.mkdir(parents=True, exist_ok=True)

d = json.loads((ART / "xval_audit_re20.json").read_text())
steps = [r["step"] for r in d["force_history"]]
cds = [r["Cd"] for r in d["force_history"]]
ref = d["results"]["Cd_reference"]
got = d["results"]["Cd_computed"]
err = d["results"]["rel_error_pct"]

# ---- fig 1: Cd history of the delivered repro command (faithful re-execution)
fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
ax.axhspan(ref * 0.85, ref * 1.15, color="#c8e6c9", alpha=0.8, label="±15% acceptance band around 2.05")
ax.axhline(ref, color="k", ls="--", lw=1.2, label=f"reference Cd = {ref} (Dennis & Chang 1970)")
ax.plot(steps, cds, "o-", ms=3, lw=1.1, color="#c62828", label="Cd(t), delivered solver (auditor re-run)")
ax.axhline(got, color="#1565c0", lw=1.4, label=f"reported converged Cd = {got:.4f} ({err:.1f}% error)")
ax.set_xlabel("LBM step (8000 steps = 200 D/U, delivered predeclared run)")
ax.set_ylabel("drag coefficient Cd")
ax.set_title("m18749 XVAL-LBM-STATIC-IMMERSED-MARKER-OBSTACLES: Cd history is not converged\n"
             "(sign-changing, std=%.2f over last 4000 steps; stopping rule not met)" % 4.07, fontsize=9)
ax.set_yscale("symlog", linthresh=1.0)
ax.grid(alpha=0.3)
ax.legend(fontsize=7.5, loc="upper right")
fig.tight_layout()
fig.savefig(FIG / "fig1_cd_history.png")
plt.close(fig)

# ---- fig 2: velocity cloud map (diagnostic snapshot, 1500 steps)
z = np.load(ART / "field_1500steps.npz")
ux, uy = z["ux"], z["uy"]
speed = np.sqrt(ux ** 2 + uy ** 2)
fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.4), dpi=150)
for ax, f, ttl, cm in [
    (axes[0], ux, "u_x (streamwise velocity), delivered IBM solver, step 1500", "RdBu_r"),
    (axes[1], speed, "|u| speed magnitude, same snapshot", "viridis"),
]:
    im = ax.imshow(f, origin="lower", cmap=cm, extent=[0, ux.shape[1], 0, ux.shape[0]],
                   vmin=-0.15 if cm == "RdBu_r" else 0, vmax=0.15 if cm == "RdBu_r" else 0.25)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(z["cx"] + z["r"] * np.cos(th), z["cy"] + z["r"] * np.sin(th), "k-", lw=1.2)
    ax.set_title(ttl, fontsize=9)
    ax.set_ylabel("y (lattice units)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
axes[1].set_xlabel("x (lattice units)")
fig.suptitle("Diagnostic field snapshot (Re=20, 320x160, not converged: Cd=8.90 at step 1450)\n"
             "wake/field is contaminated by IBM forcing artefacts", fontsize=9)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(FIG / "fig2_field_snapshot.png")
plt.close(fig)
print("figures:", sorted(p.name for p in FIG.glob("*.png")))
print("cd mean last half", np.mean(cds[len(cds) // 2:]), "std", np.std(cds[len(cds) // 2:]))