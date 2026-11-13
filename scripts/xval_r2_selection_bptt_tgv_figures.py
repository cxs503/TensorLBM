#!/usr/bin/env python3
"""Figures for XVAL-LBM-R2-SELECTION-BPTT-TGV-ANCHOR (from results.json only)."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "xval_r2_selection_bptt_tgv"
RES = json.loads((ART / "results.json").read_text())
FIG = {}

CB = {"smagorinsky": "#1f77b4", "oss_proxy": "#d62728"}
LS = {"field": "-", "energy": "--", "enstrophy": ":"}


def fig_loss_curves():
    """Training loss vs theta for the three objectives (identifiability probe)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, c in zip(axes, ("smagorinsky", "oss_proxy")):
        g = RES["curves"][c]["grid"]
        th = [p["theta"] for p in g]
        for o in ("field", "energy", "enstrophy"):
            j = [p[f"loss_{o}"] for p in g]
            ax.plot(th, j, LS[o], color=CB[c], marker="o", ms=2.5, label=o)
        for k, v in RES["verdict"]["combos"].items():
            if v["closure"] != c:
                continue
            ax.axvline(v["theta_selected"], color="k", lw=0.8, alpha=0.35, ls="--")
        ax.set_yscale("log")
        ax.set_xlabel(r"sub-grid constant $\theta$ ($C_s$ or $S_c$)")
        ax.set_ylabel(r"training loss $J$ (t = 50/100/150)")
        ax.set_title(f"{c}: objectives vs $\\theta$  (dashed = BPTT-selected $\\theta$)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = ART / "fig1_objective_landscapes.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out, "Training loss vs the sub-grid constant for both closures and all three objectives; vertical dashed lines are the BPTT-selected constants."


def fig_ke_curves():
    """Held-out KE error vs theta, with no-model and OSS-as-shipped markers."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for c in ("smagorinsky", "oss_proxy"):
        g = RES["curves"][c]["grid"]
        ax.semilogx([max(p["theta"], 1e-3) for p in g], [p["ke_err"] * 100 for p in g],
                    "-o", ms=3, color=CB[c], label=f"{c}: held-out $|\\Delta K|/K$")
    v = RES["verdict"]
    ax.axhline(15.0, color="k", ls=":", lw=1.2, label="owner bar 15 %")
    ax.axhline(v["no_model"]["ke_err"] * 100, color="0.4", ls="--", lw=1,
               label="no sub-grid model")
    ax.plot([0.17], [v["oss_as_shipped"]["ke_err"] * 100], "s", color="crimson", ms=8,
            label="OSS as shipped ($S_c=0.17$)")
    ax.plot([max(v["selected_theta"], 1e-3)], [v["selected_ke_err"] * 100], "*",
            color="green", ms=14, label="selected combo")
    ax.set_xlabel(r"sub-grid constant $\theta$")
    ax.set_ylabel("held-out KE error at t = 200  [%]")
    ax.set_title("Selection judge: energy integral (held-out snapshot)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    out = ART / "fig2_holdout_ke_vs_theta.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out, "Held-out kinetic-energy error at t=200 vs the sub-grid constant for both closures, with the no-model baseline, the owner's 15 % bar and the as-shipped OSS constant."


def fig_bptt_trajectories():
    """BPTT trajectories: theta vs iteration for every (closure, objective, start)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, c in zip(axes, ("smagorinsky", "oss_proxy")):
        for k, a in RES["arms"].items():
            if a["closure"] != c:
                continue
            h = a["bptt"]["history"]
            ys = [max(p[0], 1e-4) for p in h]
            ax.semilogy(range(1, len(ys) + 1), ys, LS[a["objective"]],
                        color=CB[c], alpha=0.85 if a["theta_init"] == 0.02 else 0.55,
                        lw=1.6 if a["theta_init"] == 0.02 else 1.0,
                        label=f"{a['objective']} from {a['theta_init']:g}")
        ax.axhline(1e-3, color="k", ls=":", lw=1, label="lower clamp")
        ax.set_xlabel("Adam step (BPTT)")
        ax.set_ylabel(r"$\theta$")
        ax.set_title(f"{c}: BPTT selection trajectories")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    out = ART / "fig3_bptt_trajectories.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out, "BPTT trajectories of the sub-grid constant for all 12 arms (solid: start 0.02, faded: start 0.17); the integral objectives pin the constant to the lower clamp."


def fig_verdict_bars():
    """Per-combo held-out metrics bar chart."""
    combos = RES["verdict"]["combos"]
    keys = sorted(combos)
    labels = [k.replace("|", "\n") for k in keys]
    ke = [combos[k]["holdout"]["ke_err"] * 100 for k in keys]
    ze = [combos[k]["holdout"]["z_err"] * 100 for k in keys]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.bar(x - 0.2, ke, 0.4, label=r"$|\Delta K|/K$ @ t=200")
    ax.bar(x + 0.2, ze, 0.4, label=r"$|\Delta Z|/Z$ @ t=200")
    ax.axhline(RES["verdict"]["no_model"]["ke_err"] * 100, color="0.3", ls="--", lw=1,
               label="no-model $|\\Delta K|/K$")
    ax.axhline(15.0, color="k", ls=":", lw=1.2, label="owner bar 15 %")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("held-out relative error [%]")
    ax.set_title("Selected model vs held-out reference (selection judged out of sample)")
    ax.legend(fontsize=7.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = ART / "fig4_holdout_metrics.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out, "Held-out energy and enstrophy errors of the six (closure, objective) selections; every selection is far inside the 15 % bar, and the no-model baseline is the best of all."


def main():
    out_json = {}
    for fn in (fig_loss_curves, fig_ke_curves, fig_bptt_trajectories, fig_verdict_bars):
        path, caption = fn()
        out_json[path.name] = {
            "caption": caption,
            "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
            "png_base64": base64.b64encode(path.read_bytes()).decode(),
        }
        print(path, path.stat().st_size)
    (ART / "figures.json").write_text(json.dumps(out_json, indent=2))
    print("wrote", ART / "figures.json")


if __name__ == "__main__":
    main()