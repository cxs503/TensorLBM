"""Figures for REPORT_V3 (moving-wall Dirichlet term of the alpha family).

Reads ``artifacts_v3/results.json`` (produced by ``verify_v3_moving_wall.py``) and
writes PNGs next to it.  No number is typed by hand: every plotted value comes
from the measurement JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts_v3"
DATA = json.loads((ART / "results.json").read_text())["sweeps"]


def rows(part, **kw):
    out = []
    for r in DATA[part]:
        if all(r[k] == v for k, v in kw.items()):
            out.append(r)
    return out


def fig_profiles():
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for j, arm in enumerate(("term", "none")):
        r = rows("couette", closure="bfl", carrier="bgk", delta_b=0.25, delta_t=0.25, arm=arm)[0]
        ax[j].plot([u * 1e3 for u in r["u_ref"]], r["y"], "k-", lw=1.5, label="analytic $u_w(1-y/H)$")
        ax[j].plot([u * 1e3 for u in r["u"]], r["y"], "o", ms=5, label=f"LBM (arm={arm})")
        ax[j].set_xlabel("$u_x$ [1e-3]")
        ax[j].set_ylabel("y [cell]")
        ax[j].set_title(f"arm={arm}:  rel. L2 = {r['l2_rel']:.2e}", fontsize=10)
        ax[j].grid(alpha=0.3)
        ax[j].legend(fontsize=8)
    fig.suptitle("Plane Couette, walls at $\\delta=0.25$ cells, $U_w=0.02$ (bgk, BFL)", fontsize=11)
    fig.tight_layout()
    fig.savefig(ART / "fig_couette_profiles.png", dpi=140)
    plt.close(fig)


def fig_l2_vs_delta():
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for carrier in ("bgk", "trt_c"):
        for closure in ("bfl", "cli", "halfway"):
            for arm, mk in (("term", "o-"), ("none", "s--")):
                rs = rows("couette", carrier=carrier, closure=closure, arm=arm)
                rs = [r for r in rs if r["delta_b"] == r["delta_t"]]
                if not rs:
                    continue
                xs = [r["delta_b"] for r in rs]
                ys = [max(r["l2_rel"], 1e-16) for r in rs]
                ax.plot(xs, ys, mk, ms=4, lw=1,
                        label=f"{carrier}/{closure}/{arm}")
    ax.set_yscale("log")
    ax.set_xlabel("wall distance $\\delta$ [cell]")
    ax.set_ylabel("rel. $L_2$ error vs analytic Couette")
    ax.set_title("Dirichlet term is what makes a moving wall move", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(ART / "fig_l2_vs_delta.png", dpi=140)
    plt.close(fig)


def fig_forced():
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    combos = [("bgk", "bfl"), ("bgk", "cli"), ("trt_c", "bfl")]
    arms = ["none", "half", "lam", "zero"]
    deltas = [0.1, 0.25]
    width = 0.2
    xi = 0
    ticks, labels = [], []
    for d in deltas:
        for carrier, closure in combos:
            for k, arm in enumerate(arms):
                r = rows("forced", closure=closure, carrier=carrier, delta_b=d, arm=arm)
                if not r:
                    continue
                r = r[0]
                ax.bar(xi + (k - 1.5) * width, r["l2_rel"] * 1e2, width,
                       label=arm if xi == 0 else None)
            ticks.append(xi)
            labels.append(f"{carrier}/{closure}\n$\\delta$={d}")
            xi += 1.0
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("rel. $L_2$ error vs analytic Poiseuille [%]")
    ax.set_title("Force-driven channel, walls at rest: the Dirichlet term is NOT the cure",
                 fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, title="wall value convention")
    fig.tight_layout()
    fig.savefig(ART / "fig_forced_conventions.png", dpi=140)
    plt.close(fig)


def fig_mass_drift():
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for part, marker in (("couette", "o"), ("forced", "s")):
        rs = [r for r in DATA[part] if r["delta_b"] == r["delta_t"]]
        xs = [r["delta_b"] for r in rs]
        ys = [abs(r["dM_per_step"]) for r in rs]
        ax.semilogy(xs, ys, marker, ms=4, label=f"{part} ({len(rs)} arms)")
    cp = [r for r in DATA["couette_poiseuille"]]
    ax.semilogy([r["delta_b"] for r in cp], [abs(r["dM_per_step"]) for r in cp], "^", ms=5,
                label="couette_poiseuille")
    ax.set_xlabel("wall distance $\\delta$ [cell]")
    ax.set_ylabel("relative total-mass leak per step")
    ax.set_title("Residual bias of every closure = mass leak of the solid-side write", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ART / "fig_mass_drift.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    fig_profiles()
    fig_l2_vs_delta()
    fig_forced()
    fig_mass_drift()
    print("figures ->", sorted(p.name for p in ART.glob("*.png")))