#!/usr/bin/env python
"""Figures for the single-phase Poiseuille baseline skeleton case report.

Reads the JSON written by ``scripts/run_poiseuille_baseline_study.py``; no number
in the figures is hand-copied.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm.verification.xval.poiseuille_skeleton import run_baseline_case  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "xval_poiseuille_baseline_skeleton"
FIG = ART / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def load(name: str) -> dict:
    return json.loads((ART / f"study_{name}.json").read_text())


def fig_profiles() -> Path:
    oss = load("oss")
    runs = {r["resolution"]: run_baseline_case(r["resolution"], max_steps=600_000)
            for r in oss["rows"] if r["resolution"] in (16, 64, 128)}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    for res, r in runs.items():
        y = r.y / r.params.height_lu
        u = r.u_num / np.max(r.u_exact)
        ax1.plot(u, y, "o", ms=3.2, label=f"TensorLBM D2Q9, H={int(r.params.height_lu)}")
    ref = runs[16]
    ax1.plot(ref.u_exact / np.max(ref.u_exact), ref.y / ref.params.height_lu, "k-", lw=1.6,
             label=r"analytic $g\,y(H-y)/(2\nu)$")
    ax1.set_xlabel(r"$u/u_{peak}$")
    ax1.set_ylabel(r"$y/H$")
    ax1.set_title("(a) plane Poiseuille profile")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(alpha=0.3)

    for res, r in runs.items():
        resid = (r.u_num - r.u_exact) / np.max(r.u_exact)
        ax2.semilogy(np.abs(resid), r.y / r.params.height_lu, "-",
                     label=fr"$|u-u_{{exact}}|/u_{{peak}}$, H={int(r.params.height_lu)}")
        shape = r.slip_kpis["eps_inf_shape"]
        ax2.axvline(shape, ls=":", lw=1.0)
    ax2.set_xlabel(r"$|u-u_{exact}|/u_{peak}$")
    ax2.set_ylabel(r"$y/H$")
    ax2.set_title("(b) raw residual (flat offset) vs slip-corrected level (dotted)")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    out = FIG / "fig1_profile_and_residual.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_convergence() -> Path:
    oss = load("oss")
    fixed = load("fixed")
    fig, ax = plt.subplots(figsize=(6.2, 4.4))

    for name, blob, style in (("OSS convention ($Re$, $Ma$ fixed)", oss, "o-"),
                              (r"fixed $\nu$, $g$ (textbook refinement)", fixed, "s-")):
        H = np.array([r["height_lu"] for r in blob["rows"]])
        e = np.array([r["kpis"]["eps_inf"] for r in blob["rows"]])
        ax.loglog(H, e, style, label=name)
    H = np.array([r["height_lu"] for r in fixed["rows"]])
    e = np.array([r["kpis"]["eps_inf"] for r in fixed["rows"]])
    guide = e[0] * (H / H[0]) ** -2.0
    ax.loglog(H, guide, "k--", lw=1.0, label=r"$\propto H^{-2}$")
    shape = [r["slip_kpis"]["eps_inf_shape"] for r in oss["rows"]]
    H_oss = [r["height_lu"] for r in oss["rows"]]
    ax.loglog(H_oss, shape, ":", color="0.4", lw=1.2, label="slip-corrected shape error")
    ax.set_xlabel(r"$H$ [lattice units]")
    ax.set_ylabel(r"$\varepsilon_\infty$ vs parabola")
    ax.set_title(r"grid refinement: OSS convention plateaus, fixed $\nu$ is $O(H^{-2})$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG / "fig2_convergence.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_slip_law() -> Path:
    tau_blob = load("tau")
    rows = tau_blob["rows"]
    tau = np.array([r["tau"] for r in rows])
    f = np.array([r["slip_over_g"] for r in rows])
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(tau, f, "o", label=r"measured $f(\tau)=\mathrm{slip}/g$ (H=30)")
    tt = np.linspace(tau.min(), tau.max(), 200)
    a = tau_blob["asymptote"]
    ax.plot(tt, a["slope"] * tt + a["intercept"], "k--", lw=1.0,
            label=fr"asymptote ${a['slope']:.2f}\tau {a['intercept']:+.2f}$ ($\tau\geq2$)")
    ax.axhline(0.0, color="0.6", lw=0.8)
    if tau_blob["slip_zero_tau"]:
        ax.axvline(tau_blob["slip_zero_tau"], color="C3", ls=":", lw=1.2,
                   label=fr"slip-free $\tau\approx{tau_blob['slip_zero_tau']:.3f}$")
    ax.set_xlabel(r"relaxation time $\tau$")
    ax.set_ylabel(r"wall slip per unit body force, $\mathrm{slip}/g$")
    ax.set_title("uniform wall slip of half-way bounce-back + Guo forcing")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG / "fig3_slip_law.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> int:
    paths = [fig_profiles(), fig_convergence(), fig_slip_law()]
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())