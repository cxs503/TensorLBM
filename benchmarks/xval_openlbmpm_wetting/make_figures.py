#!/usr/bin/env python3
"""Figures for XVAL-LBM-GEOMETRIC-WETTING-CONTACT-ANGLE (matplotlib, Agg)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent


def fig_operator(out: Path) -> str:
    d = json.loads((HERE / "operator_study.json").read_text())
    cases = d["cases"]
    angles = sorted({c["theta_prescribed"] for c in cases})
    variants = ["V1_closed2pt", "V2_cand2pt", "V3_candIso9", "V4_candIso9_diff"]
    colors = {"V1_closed2pt": "tab:blue", "V2_cand2pt": "tab:cyan",
              "V3_candIso9": "tab:green", "V4_candIso9_diff": "tab:red"}
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for v in variants:
        f = [np.mean([c["fidelity_mean_abs_dphi"] for c in cases
                      if c["variant"] == v and c["theta_prescribed"] == a]) for a in angles]
        axes[0].plot(angles, f, "o-", color=colors[v], label=v)
    axes[0].set_xlabel("prescribed contact angle θ (deg)")
    axes[0].set_ylabel(r"M2 fidelity  $\langle|\phi_{ghost}-\phi_{exact}|\rangle$")
    axes[0].set_yscale("log")
    axes[0].set_title("wall-row fidelity to the analytic continuation")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    for v in variants:
        e = [max(c["rel_err_pct"] for c in cases
                 if c["variant"] == v and c["theta_prescribed"] == a) for a in angles]
        e = [min(max(x, 1e-16), 1e3) for x in e]
        axes[1].plot(angles, e, "s-", color=colors[v], label=v)
    axes[1].axhline(15.0, color="k", ls="--", lw=1.0, label="15 % gate")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("prescribed contact angle θ (deg)")
    axes[1].set_ylabel("M1 relative read-back error (%)")
    axes[1].set_title("operator self-consistency (gate ≤ 15 %)")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.suptitle("openLBMPM geometric-wetting XVAL — operator level (27 pairs, deterministic)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = out / "fig_operator_xval.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)


def fig_solver(out: Path) -> str:
    res = json.loads((HERE / "results.json").read_text())["results"]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for name, r in res.items():
        h = r["history"]
        steps = [x["step"] for x in h]
        ax.plot(steps, [x["theta_capgeom"] for x in h], "-",
                label=f"{name} (BC {'on' if r['wetting'] else 'off'}), cap-geometry")
        ax.axhline(r["theta_prescribed"], ls=":", lw=0.8, color="grey")
    ax.set_xlabel("step")
    ax.set_ylabel("measured θ (deg), openLBMPM cap-geometry recipe")
    ax.set_title("solver level: measured contact angle vs prescribed (A=0.005)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    p = out / "fig_solver_angle.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)


def fig_fields(out: Path) -> str:
    files = sorted(HERE.glob("phi_*.npy"))
    if not files:
        return ""
    fig, axes = plt.subplots(1, len(files), figsize=(4.6 * len(files), 3.4), squeeze=False)
    for ax, f in zip(axes[0], files):
        phi = np.load(f)
        im = ax.imshow(phi, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.contour(phi, levels=[0.0], colors="k", linewidths=0.6)
        ax.set_title(f.stem.replace("phi_", ""), fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("phase field φ = (ρ_r-ρ_b)/ρ at the end of each arm", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = out / "fig_phase_fields.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)


if __name__ == "__main__":
    out = HERE
    made = [fig_operator(out), fig_solver(out), fig_fields(out)]
    (out / "figures.json").write_text(json.dumps([m for m in made if m], indent=1))
    print("\n".join(m for m in made if m))