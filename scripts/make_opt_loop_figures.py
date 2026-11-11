"""Figures for the tau/Re recovery demo (docs/OPT_LOOP_TAU_RECOVERY.md).

Panel A  TG decay: simulated peak |u_x| vs the exact exp(-2 nu k^2 t) curve
         (physics anchor - the rate the inversion has to identify).
Panel B  convergence: relative error of the recovered nu vs iteration, all
         tracks and seeds (log scale), with the 3% gate drawn.
Panel C  bar chart of max|err| per track against the gate.

Usage::

    python scripts/make_opt_loop_figures.py --json artifacts/opt_loop_tau_recovery.json \
        --out figures/opt_loop_tau_recovery.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from tensorlbm.d3q19 import macroscopic3d  # noqa: E402
from tensorlbm.opt_loop_tau_recovery import TaylorGreenCase, simulate  # noqa: E402


def decay_curve(n: int = 16, tau: float = 0.55, steps: int = 260, stride: int = 10):
    case = TaylorGreenCase(n=n)
    nu = case.nu_from_tau(tau)
    obs_times = tuple(range(stride, steps + 1, stride))
    sim = simulate(case, tau, obs_times)
    t = list(obs_times)
    measured = [float(ux.abs().max()) for ux, _ in sim]
    exact = [case.u0 * float(case.analytic(float(s), nu)[0].abs().max()) for s in obs_times]
    continuous = [case.u0 * math.exp(-2.0 * nu * case.k**2 * s) for s in obs_times]
    return t, measured, exact, continuous, nu


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=Path("artifacts/opt_loop_tau_recovery.json"))
    ap.add_argument("--out", type=Path, default=Path("figures/opt_loop_tau_recovery.png"))
    args = ap.parse_args()
    payload = json.loads(args.json.read_text())

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
    t, measured, exact, continuous, nu = decay_curve()
    ax = axes[0]
    ax.semilogy(t, measured, "o", ms=3, label="LBM (differentiable solver)")
    ax.semilogy(t, continuous, "-", lw=1.5, label=r"exact NS: $u_0e^{-2\nu k^2 t}$")
    ax.semilogy(t, exact, "--", lw=1.0, label="exact sampled on the lattice")
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\max|u_x|$")
    ax.set_title("A. Taylor-Green decay, n=16, tau=0.55")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    colors = plt.cm.tab10.colors
    for i, track in enumerate(payload["tracks"]):
        for run in track["runs"]:
            hist = run["history"]
            it = [h["iter"] for h in hist]
            err = [max(abs(h["rel_err_pct"]), 1e-9) for h in hist]
            ax.semilogy(it, err, "-", lw=1.1, color=colors[i % 10], alpha=0.85,
                        label=f"{track['track']} s{run['seed']}")
    ax.axhline(3.0, color="tab:red", ls="--", lw=1.2, label="3% gate")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$|\nu_{hat}-\nu_{true}|/\nu_{true}$  [%]")
    ax.set_title("B. optimiser convergence (all tracks/seeds)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    names = [t["track"] for t in payload["tracks"]]
    vals = [t["summary"]["max_abs_rel_err_pct"] for t in payload["tracks"]]
    bars = ax.bar(range(len(names)), vals, color=[colors[i % 10] for i in range(len(names))])
    ax.axhline(3.0, color="tab:red", ls="--", lw=1.2, label="3% gate")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3g}%", ha="center", va="bottom", fontsize=7)
    ax.set_yscale("log")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("max |err| [%]")
    ax.set_title("C. worst-seed error per track")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y", which="both")

    fig.suptitle(
        "tau / Re recovery through the differentiable LBM (Taylor-Green decay), "
        f"nu_true={nu:.6f} (tau=0.55), torch {payload.get('torch_version')}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())