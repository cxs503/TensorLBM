import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

A = "audit"
pb = json.load(open(f"{A}/probe_baseline.json")); pg = json.load(open(f"{A}/probe_geometric.json"))
p3 = json.load(open(f"{A}/probe_3d.json"))
m154 = json.load(open("/root/tp-agent/work/m15468/artifacts/mcmp_spurious_2d_quick.json"))
m154_3d = json.load(open("/root/tp-agent/work/m15468/artifacts/mcmp_spurious_single.json"))
rerun = json.load(open("artifacts/mcmp_spurious_2d_quick.json"))

# Fig 1: u_max(t) trajectories
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for d, lab, c in ((pb, "baseline_bgk_lattice (audit, stop_tol=1e-4)", "C0"),
                  (pg, "bgk_geometric (audit, stop_tol=1e-4)", "C3")):
    h = np.asarray(d["history"], float)
    ax.semilogy(h[:, 0], h[:, 1], c, label=f"{lab}; plateau={d['u_max_final']:.2e}", lw=1.6)
ax.axhline(1e-3, color="k", ls="--", lw=1.2, label="tolerance 1e-3 (case criterion)")
ax.axvline(1500, color="gray", ls=":", lw=1.2, label="m15468 stop (1500 steps, converged=false)")
ax.scatter([1500, 1500], [m154["baseline_bgk_lattice"]["u_spurious"], m154["bgk_geometric"]["u_spurious"]],
           marker="x", s=70, c=["C0", "C3"], zorder=5, label="m15468 reported values @1500")
ax.set_xlabel("lattice step"); ax.set_ylabel(r"$u_{spurious}=\max|\mathbf{u}|$ (lattice units)")
ax.set_title("CASE-MCMP-SPURIOUS 2D: transient vs converged plateau (audit re-run)")
ax.grid(alpha=.3, which="both"); ax.legend(fontsize=7.5, loc="upper right")
fig.tight_layout(); fig.savefig(f"{A}/figs/fig1_uspurious_transient_vs_plateau.png", dpi=160); plt.close(fig)

# Fig 2: ref vs got bars (2D + 3D)
fig, ax = plt.subplots(figsize=(7.6, 4.4))
labels = ["2D baseline\n(m15468 @1500)", "2D baseline\n(audit plateau)", "2D geometric\n(m15468 @1500)",
          "2D geometric\n(audit plateau)", "3D EDM n32\n(m15468 artifact)", "3D EDM n32\n(audit re-run)"]
vals = [m154["baseline_bgk_lattice"]["u_spurious"], pb["u_max_final"],
        m154["bgk_geometric"]["u_spurious"], pg["u_max_final"],
        m154_3d["u_max"], p3["u_max"]]
cols = ["C0", "C0", "C3", "C3", "C1", "C1"]
ax.bar(range(6), vals, color=cols, alpha=.75)
ax.axhspan(1e-3, 1e-2, color="green", alpha=.12, label="literature band 1e-3..1e-2 (Kruger ch.9)")
ax.axhline(1e-3, color="k", ls="--", lw=1.2, label="tolerance 1e-3")
ax.set_yscale("log"); ax.set_xticks(range(6)); ax.set_xticklabels(labels, fontsize=7.5)
for i, v in enumerate(vals):
    ax.text(i, v * 1.25, f"{v:.2e}", ha="center", fontsize=7.5)
ax.set_ylabel(r"$u_{spurious}$ (lattice units)")
ax.set_title("CASE-MCMP-SPURIOUS ref vs got: reported vs audit-recomputed")
ax.grid(alpha=.3, axis="y", which="both"); ax.legend(fontsize=7.5)
fig.tight_layout(); fig.savefig(f"{A}/figs/fig2_ref_vs_got.png", dpi=160); plt.close(fig)
print("figures ok", pb["u_max_final"], pg["u_max_final"], p3["u_max"], m154_3d["u_max"])
