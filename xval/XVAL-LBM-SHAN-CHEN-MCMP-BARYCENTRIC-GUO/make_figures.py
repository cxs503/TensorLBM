#!/usr/bin/env python3
"""Figures for anchor ``XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO``.

Reads ``artifacts/xval_lbm_shan_chen_mcmp/results.json`` (+ the OSS reference
JSON) and writes PNGs next to it.  No numbers are typed in here — every curve
comes from the artifacts, so a re-run of ``run_xval.py`` refreshes the figures.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "xval_lbm_shan_chen_mcmp"
RES = json.loads((ART / "results.json").read_text())
REF = json.loads((ART / "oss_ref_shan_chen_two_component.json").read_text())
THRESH = RES["threshold_rel_l2"]

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def fig_profiles() -> Path:
    ref_a = np.asarray(REF["ref_a"])
    ref_b = np.asarray(REF["ref_b"])
    arms = RES["arms"]
    prim = arms["lib_barycentric_guo_exp"]["profiles"]
    lit = arms["oss_literal_barycentric"]["profiles"]
    x = np.arange(len(ref_a))
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.2), sharex=True)
    for ax, which, ref in ((axes[0], "a", ref_a), (axes[1], "b", ref_b)):
        ax.plot(x, ref, "o", ms=3.2, mfc="none", color="k", label="OSS ref (lssfau/lbmpy)")
        ax.plot(x, np.asarray(prim[which]), "-", lw=1.6, color="C0",
                label="TensorLBM barycentric-Guo, psi_exp")
        ax.plot(x, np.asarray(lit[which]), "--", lw=1.1, color="C3",
                label="OSS literal (u_bary in Guo source)")
        ax.set_ylabel(rf"$\rho_{which}$ [l.u.]")
        ax.legend(frameon=False, fontsize=8)
    axes[1].set_xlabel("x [lattice]")
    axes[0].set_title("Shan-Chen MCMP, 64x64 periodic, G=6, psi_exp, 1000 steps")
    fig.tight_layout()
    out = ART / "fig1_profiles_vs_oss.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_arms() -> Path:
    names = list(RES["arms"])
    vals, colors, labels = [], [], []
    for n in names:
        a = RES["arms"][n]
        v = a.get("worst_rel_l2")
        vals.append(np.nan if v is None else max(v, 1e-6))
        colors.append("C2" if a["passes_15pct"] else "C3")
        d = a.get("diverged_at_step")
        labels.append(n + (f"\n(diverge @{d})" if d else ""))
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    bars = ax.bar(range(len(vals)), vals, color=colors)
    ax.set_yscale("log")
    ax.axhline(THRESH, color="k", ls="--", lw=1.0, label=f"acceptance {THRESH:.0%}")
    for i, (b, v) in enumerate(zip(bars, vals)):
        txt = "diverged" if not np.isfinite(v) or RES["arms"][names[i]].get("worst_rel_l2") is None \
            else f"{RES['arms'][names[i]]['worst_rel_l2']:.2e}"
        ax.text(b.get_x() + b.get_width() / 2, max(v, 2e-6) * 1.35, txt,
                ha="center", fontsize=7, rotation=0)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=7)
    ax.set_ylabel(r"worst relative $L_2$ error vs OSS ref")
    ax.set_title("One-factor-at-a-time ablation of the velocity coupling / force form / psi")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = ART / "fig2_arm_errors.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_steps() -> Path:
    sweep = RES["sensitivity"]["steps"]
    xs = sorted((int(k), v["component_a"]["rel_l2"]) for k, v in sweep.items())
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.semilogy([a for a, _ in xs], [b for _, b in xs], "o-", color="C0")
    ax.axvline(REF["parameters"]["steps"], color="k", ls=":", lw=1.0, label="OSS step count")
    ax.axhline(THRESH, color="C3", ls="--", lw=1.0, label=f"acceptance {THRESH:.0%}")
    ax.set_xlabel("time steps")
    ax.set_ylabel(r"relative $L_2$ error of $\rho_a$")
    ax.set_title("Time alignment: the residual is minimal exactly at 1000 steps")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = ART / "fig3_step_sensitivity.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_field() -> Path:
    prim = RES["arms"]["lib_barycentric_guo_exp"]
    field_a = np.asarray(RES["arms"]["lib_barycentric_guo_exp"].get("field_a", []))
    # field_a is stored only for the primary arm at the time of writing; fall back
    # to the profile if a stage-restricted run dropped it.
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    ref_a = np.asarray(REF["ref_a"])
    prof = np.asarray(prim["profiles"]["a"])
    axes[0].plot(np.arange(len(ref_a)), ref_a, "k-o", ms=2.5, mfc="none", lw=0.8, label="OSS ref")
    axes[0].plot(np.arange(len(prof)), prof, "C0-", lw=1.4, label="TensorLBM")
    axes[0].set_xlabel("x [lattice]")
    axes[0].set_ylabel(r"$\rho_a$")
    axes[0].set_title("profile (row 0)")
    axes[0].legend(frameon=False, fontsize=8)
    d = prof - ref_a
    axes[1].plot(np.arange(len(d)), d * 1e3, "C1-", lw=1.4)
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_xlabel("x [lattice]")
    axes[1].set_ylabel(r"$(\rho_a^{\rm TL}-\rho_a^{\rm OSS})\times10^{3}$")
    axes[1].set_title(f"pointwise difference (max {np.abs(d).max():.2e})")
    fig.tight_layout()
    out = ART / "fig4_difference.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_sens() -> Path:
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    tau = RES["sensitivity"]["tau"]
    grid = RES["sensitivity"]["grid"]
    dtype = RES["sensitivity"]["dtype"]
    tp = sorted((float(k), v["component_a"]["rel_l2"]) for k, v in tau.items())
    ax.semilogy([a for a, _ in tp], [b for _, b in tp], "s-", color="C4", label=r"$\tau$ sweep")
    for k, v in grid.items():
        ax.axhline(v["component_a"]["rel_l2"], color="C5", ls="--", lw=1.0,
                   label=f"N={k} (block-avg)" if k != "64" else None)
    for k, v in dtype.items():
        ax.plot([], [], " ", label=f"{k}: {v['component_a']['rel_l2']:.2e}")
    ax.axhline(THRESH, color="C3", ls=":", lw=1.0, label=f"acceptance {THRESH:.0%}")
    ax.set_xlabel(r"relaxation time $\tau$")
    ax.set_ylabel(r"relative $L_2$ error of $\rho_a$")
    ax.set_title("Parameter / resolution / precision sensitivity")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    out = ART / "fig5_sensitivity.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


if __name__ == "__main__":
    outs = [fig_profiles(), fig_arms(), fig_steps(), fig_field(), fig_sens()]
    for o in outs:
        print(o.relative_to(ROOT), o.stat().st_size)