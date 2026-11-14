#!/usr/bin/env python3
"""m20151 audit figures — every number comes from the delivered artifact JSON."""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

A = pathlib.Path(__file__).resolve().parent
ART = json.loads((A / "delivered_snapshot/artifacts/diff_boundary_grad_full.json").read_text())
FIG = A / "figures"
FIG.mkdir(exist_ok=True)

GROUPS = [("gradient_matrix", "main 30x96 D=8 200 steps"),
          ("resolution_scan", "second lattice 22x73 D=5.25 200 steps"),
          ("horizon_scan", "second horizon 30x96 D=8 100 steps")]

# ---------------------------------------------------------------- fig 1
rows = []
for grp, label in GROUPS:
    for r in ART[grp]:
        rows.append((label, r["closure"], r["parameter"], r["fd"]["h1"]["rel_err_drag"],
                     r["fd"]["h2"]["rel_err_drag"], r["fd"]["h1"]["rel_err_lift"],
                     r["fd"]["h2"]["rel_err_lift"]))
labels = [f"{c}\n{p.split()[0]}" for _, c, p, *_ in rows]
x = np.arange(len(rows))
w = 0.2
fig, ax = plt.subplots(figsize=(13, 5.2))
ax.bar(x - 1.5 * w, [r[3] for r in rows], w, label="drag, FD h=coarse")
ax.bar(x - 0.5 * w, [r[4] for r in rows], w, label="drag, FD h=fine")
ax.bar(x + 0.5 * w, [r[5] for r in rows], w, label="lift, FD h=coarse")
ax.bar(x + 1.5 * w, [r[6] for r in rows], w, label="lift, FD h=fine")
ax.axhline(1e-2, color="crimson", ls="--", lw=1.4, label="predeclared gate 1e-2")
worst = max(max(r[3:]) for r in rows)
ax.axhline(worst, color="k", ls=":", lw=1.0, label=f"audit worst = {worst:.2e}")
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("|AD - central FD| / |FD|")
ax.set_title("m20151 audit: differentiable boundary-gradient matrix, 18 (closure, closure-parameter) rows\n"
             "three groups (main / second lattice / second horizon); all rows pass the 1e-2 gate")
for i, (grp, label) in enumerate(GROUPS):
    ax.annotate(label, xy=((6 * i + 2.5), 3e-3), ha="center", fontsize=7,
                color="dimgray")
for b in (5.5, 11.5):
    ax.axvline(b, color="0.7", lw=0.8)
ax.legend(fontsize=7, ncol=2, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "fig1_ad_vs_fd_matrix.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- fig 2
cd = ART["anchors"]["dfg"]["samples"]["C_D"]
fig, ax = plt.subplots(figsize=(8.5, 4.6))
ax.plot(np.arange(len(cd)) * 50 + 7000, cd, "-o", ms=2.5, lw=1.2,
        label="delivered run (Re=20, D=8, 35x176, 24% blockage)")
ax.axhspan(5.57, 5.59, color="green", alpha=0.35, label="Schafer-Turek 2-D-1 range 5.57-5.59 (Re=20)")
ax.axhline(3.23, color="tab:orange", ls="--", lw=1.4, label="2-D-2 midpoint 3.23 (Re=100, wrongly quoted in report s4)")
ax.set_xlabel("lattice step")
ax.set_ylabel("C_D (instantaneous)")
ax.set_ylim(0, max(cd) * 1.08)
ax.set_title("m20151 audit: DFG cylinder anchor does NOT converge - honest FAIL\n"
             "C_D drifts monotonically upward (51.2 -> 65+, density diverges): absolute force on a body is unvalidated")
ax.annotate(f"first {cd[0]:.2f} / last {cd[-1]:.2f}", xy=(7000, cd[0]), xytext=(9000, 20),
            arrowprops=dict(arrowstyle="->", lw=0.9), fontsize=8)
ax.legend(fontsize=7.5)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "fig2_dfg_cd_fail.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- fig 3
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
c = ART["anchors"]["couette"]
y = np.arange(len(c["profile_got"]))
axes[0].plot(c["profile_exact"], y, "k-", lw=2, label="exact u = U(y-y_w)/H")
axes[0].plot(c["profile_got"], y, "o", ms=4, mfc="none", label="LBM (momentum-exchange wall)")
axes[0].set_title(f"plane Couette: max rel err {c['profile_max_rel_err']:.2e}\n"
                  f"wall force {c['force_on_moving_wall']:.6f} vs {c['force_analytic_magnitude']:.6f} "
                  f"(rel {c['force_rel_err']:.2e})")
axes[0].set_xlabel("u_x"); axes[0].set_ylabel("wall-normal cell")
p = ART["anchors"]["poiseuille"]
y = np.arange(len(p["profile_got"]))
axes[1].plot(p["profile_exact"], y, "k-", lw=2, label="exact parabola")
axes[1].plot(p["profile_got"], y, "o", ms=4, mfc="none", label="LBM (half-way bounce-back)")
axes[1].set_title(f"plane Poiseuille: max rel err {p['profile_max_rel_err']:.2e}\n"
                  f"per-wall force {p['force_on_lower_wall']:.6f} vs {p['force_analytic_per_wall']:.6f} "
                  f"(rel {p['force_rel_err_recomputed'] if 'force_rel_err_recomputed' in p else p['force_rel_err']:.2e})")
axes[1].set_xlabel("u_x")
for ax in axes:
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
fig.suptitle("m20151 audit: exact-solution anchors of the force probes (these are the anchors that DO hold)", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "fig3_exact_anchors.png", dpi=160)
plt.close(fig)
print("figures:", sorted(p.name for p in FIG.glob("*.png")))