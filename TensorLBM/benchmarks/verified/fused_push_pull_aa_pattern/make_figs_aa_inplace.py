#!/usr/bin/env python3
"""Figures for the strict A-A in-place verification (anchor XVAL-LBM-FUSED-PUSH-PULL-AA-PATTERN).

All numbers are read from ``result_aa_inplace.json`` / ``storage_aa.json``
(produced by ``run_aa_inplace.py`` / ``measure_storage.py``); the short series
plotted in fig. 4 are recomputed here because they are cheap and must not be
transcribed by hand.

    python make_figs_aa_inplace.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
FIG = HERE / "figures"
sys.path.insert(0, str(HERE.parents[2] / "src"))

import tensorlbm.aa_fused_push_pull as AB          # noqa: E402
import tensorlbm.aa_fused_push_pull_inplace as AA  # noqa: E402

NU, G = 0.05, 1e-6
TAU = 3.0 * NU + 0.5


def fig_poiseuille(res):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    ax = axes[0]
    for r in res["poiseuille"]:
        n = r["n_fluid"]
        y = np.arange(n) + 0.5
        ax.plot(np.array(r["profile"]) / r["u_max_analytic"], y / n, "o-",
                ms=3, lw=1, label=f"A-A in-place, n={n}")
    r0 = res["poiseuille"][-1]
    n = r0["n_fluid"]
    y = np.arange(n) + 0.5
    ax.plot(np.array(r0["analytic"]) / r0["u_max_analytic"], y / n, "k--", lw=1.5,
            label="analytic parabola")
    ax.set_xlabel("$u_x / u_{max}$")
    ax.set_ylabel("$y / H$")
    ax.set_title("Plane Poiseuille, Guo forcing, halfway bounce-back")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for r in res["poiseuille"]:
        n = r["n_fluid"]
        y = (np.arange(n) + 0.5) / n
        err = np.array(r["profile"]) - np.array(r["analytic"])
        ax.plot(y, err / r["u_max_analytic"] * 100.0, "o-", ms=3, lw=1,
                label=f"n={n}: $u_{{max}}$ err {100*r['u_max_rel_err']:.2f}%")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhline(3.0, color="r", ls=":", lw=0.8)
    ax.axhline(-3.0, color="r", ls=":", lw=0.8, label="±3 % owner gate")
    ax.set_xlabel("$y / H$")
    ax.set_ylabel("$(u_x^{num} - u_x^{ana}) / u_{max}$  [%]")
    ax.set_title("Pointwise profile error")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_aa_inplace_poiseuille.png", dpi=150)
    plt.close(fig)


def fig_ladder(res):
    ns = [r["n_fluid"] for r in res["poiseuille"]]
    umax = [100 * r["u_max_rel_err"] for r in res["poiseuille"]]
    l2 = [100 * r["profile_rel_l2"] for r in res["poiseuille"]]
    fig, ax = plt.subplots(figsize=(5.2, 4))
    ax.loglog(ns, umax, "o-", label="$u_{max}$ relative error")
    ax.loglog(ns, l2, "s-", label="profile $L_2$ relative error")
    ref = l2[0] * (np.array(ns) / ns[0]) ** -2.0
    ax.loglog(ns, ref, "k--", lw=1, label="slope $-2$ (2nd order)")
    ax.axhline(3.0, color="r", ls=":", lw=0.9, label="3 % gate")
    ax.set_xlabel("channel height $H$ [lattice units]")
    ax.set_ylabel("relative error [%]")
    ax.set_title("Grid ladder, A-A in-place kernel")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_aa_inplace_ladder.png", dpi=150)
    plt.close(fig)


def fig_storage(storage):
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    labels = ["fused A-B\n(shipped kernel)", "fused A-A\n(in place)"]
    vals = [storage[0]["extra_fields"], storage[1]["extra_fields"]]
    bars = ax.bar(labels, vals, color=["#c0504d", "#4f81bd"], width=0.55)
    ax.axhline(1.0, color="k", ls="--", lw=1, label="one full $Q\\times N$ field")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.06, f"{v:.2f} fields",
                ha="center", fontsize=9)
    ax.set_ylabel("peak memory above the carried field [fields]")
    ax.set_title("Measured storage, 900×900 D2Q9 float64, VmHWM")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_aa_inplace_storage.png", dpi=150)
    plt.close(fig)


def fig_series():
    """Equivalence and mass-conservation histories (short, recomputed here)."""
    steps = 300
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    solid = AA.channel_masks(4, 10, 1)
    wall = torch.zeros(2, 10, 4, dtype=torch.float64)
    wall[0, -1, :] = 0.02
    force = torch.zeros(2, 10, 4, dtype=torch.float64)
    force[0] = G
    cases = {
        "periodic": dict(nx=8, ny=8, tau=0.9, perturb=True),
        "walled": dict(nx=4, ny=10, tau=0.9, solid=solid),
        "moving wall": dict(nx=4, ny=10, tau=0.9, solid=solid, wall_u=wall),
        "Guo forced": dict(nx=4, ny=10, tau=TAU, solid=solid, force=force),
    }
    ax = axes[0]
    for name, kw in cases.items():
        kw = dict(kw)
        nx, ny = kw.pop("nx"), kw.pop("ny")
        f_ab = AA.rest_equilibrium(ny, nx, perturb=kw.pop("perturb", False))
        f_aa = f_ab.clone()
        hist = []
        for _ in range(steps):
            f_ab = AB.fused_push_pull_step(f_ab, kw["tau"], solid=kw.get("solid"),
                                           wall_u=kw.get("wall_u"), force=kw.get("force"))
            AA.aa_fused_push_pull_step(f_aa, kw["tau"], solid=kw.get("solid"),
                                       wall_u=kw.get("wall_u"), force=kw.get("force"))
            hist.append(float((f_ab - f_aa).abs().max()))
        ax.semilogy(np.arange(1, steps + 1), np.maximum(hist, 1e-18), lw=1, label=name)
    ax.axhline(1e-13, color="r", ls=":", lw=0.9, label="1e-13 gate")
    ax.set_xlabel("step")
    ax.set_ylabel("max $|f^{A-A}_{in\\,place} - f^{A-B}_{fused}|$")
    ax.set_title("Same scheme, different storage discipline")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    n = 16
    solid_box = torch.zeros(n, n, dtype=torch.bool)
    solid_box[0, :] = solid_box[-1, :] = True
    solid_box[:, 0] = solid_box[:, -1] = True
    fluid = ~solid_box
    f = AA.rest_equilibrium(n, n, perturb=True)
    m0 = float(f[:, fluid].sum())
    drift = []
    for _ in range(500):
        AA.aa_fused_push_pull_step(f, 0.9, solid=solid_box)
        drift.append(abs(float(f[:, fluid].sum()) - m0) / m0)
    ax.semilogy(np.arange(1, len(drift) + 1), np.maximum(drift, 1e-18), lw=1)
    ax.axhline(1e-12, color="r", ls=":", lw=0.9, label="1e-12 gate")
    ax.set_xlabel("step")
    ax.set_ylabel("relative mass drift")
    ax.set_title(f"Closed 16×16 box, all-wall bounce-back")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_aa_inplace_equivalence_mass.png", dpi=150)
    plt.close(fig)


def main() -> int:
    FIG.mkdir(exist_ok=True)
    res = json.loads((HERE / "result_aa_inplace.json").read_text())
    storage = json.loads((HERE / "storage_aa.json").read_text())
    fig_poiseuille(res)
    fig_ladder(res)
    fig_storage(storage)
    fig_series()
    print("figures:", sorted(p.name for p in FIG.glob("fig*aa_inplace*.png")))
    return 0


if __name__ == "__main__":
    sys.exit(main())