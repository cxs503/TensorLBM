#!/usr/bin/env python3
"""Figures for benchmark B14 (2-D body-force plane Poiseuille, D2Q9 BGK).

All numbers are read from ``result.json`` (produced by ``run.py``), never typed
by hand: every panel is a projection of the archived artefacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "figs"


def load() -> dict:
    return json.loads((HERE / "result.json").read_text())


def fig_profile(res: dict) -> Path:
    ref = res["reference_case"]
    y = np.array(ref["y"])
    un = np.array(ref["u_numerical"])
    ua = np.array(ref["u_analytic"])
    H = ref["H"]
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax[0].plot(ua, y / H, "k-", lw=2.0, label="analytic  $g/(2\\nu)\\,y(H-y)$")
    ax[0].plot(un, y / H, "o", ms=4.5, mfc="none", mec="C0",
               label="D2Q9 BGK ($H=%d$, $\\tau=%.1f$)" % (H, ref["tau"]))
    ax[0].set_xlabel("$u_x$ [lattice]")
    ax[0].set_ylabel("$y/H$")
    ax[0].set_title("steady profile: " + ("PASS" if ref["pass"] else "FAIL"))
    ax[0].legend(loc="lower right", fontsize=8)
    ax[1].semilogy(y / H, np.abs(un - ua) / ua.max(), "s-", ms=3.5, color="C3")
    ax[1].axhline(0.02, color="gray", ls="--", lw=1.0, label="pre-declared 2 % gate")
    ax[1].set_xlabel("$y/H$")
    ax[1].set_ylabel("$|u_{num}-u_{ana}|/u_{max}$")
    ax[1].set_title("pointwise error (max %.3f %%)" % (100 * ref["eps_inf"]))
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    out = FIG / "profile_vs_analytic.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_convergence(res: dict) -> Path:
    cases = res["resolution_study"]
    Hs = np.array([c["H"] for c in cases], float)
    ei = np.array([c["eps_inf"] for c in cases])
    e2 = np.array([c["eps_2"] for c in cases])
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    ax.loglog(Hs, ei, "o-", label=r"$\varepsilon_\infty$ (max)")
    ax.loglog(Hs, e2, "s--", label=r"$\varepsilon_2$ (L2)")
    ref = ei[-1] * (Hs / Hs[-1]) ** -2.0
    ax.loglog(Hs, ref, "k:", lw=1.0, label="slope $-2$")
    ax.axhline(0.02, color="gray", ls="--", lw=1.0)
    ax.text(Hs[0], 0.021, "2 % gate", fontsize=7, color="gray")
    for h, e in zip(Hs, ei):
        ax.annotate("%.2e" % e, (h, e), textcoords="offset points", xytext=(2, -10), fontsize=7)
    ax.set_xlabel("channel height $H$ [lattice nodes]")
    ax.set_ylabel("relative error vs analytic parabola")
    ax.set_title("grid refinement (fitted order %.2f)" % abs(res["convergence"]["fitted_order"]))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG / "convergence.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_defect(res: dict) -> Path:
    sens = res["sensitivity"]
    leg = sens.get("legacy", [])
    fix = sens.get("steps", [])
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.7))
    if leg:
        st = [c["steps"] for c in leg]
        ax[0].plot(st, [c["mass_drift"] for c in leg], "o-", color="C3",
                   label="pre-fix Guo source (control)")
        if fix:
            st2 = [c["steps"] for c in fix]
            ax[0].plot(st2, [c["mass_drift"] for c in fix], "s-", color="C0",
                       label="zero-sum Guo source (fixed)")
        ax[0].axhline(-1e-6, color="gray", ls="--", lw=1.0, label=r"gate $-10^{-6}$")
        ax[0].set_xlabel("steps")
        ax[0].set_ylabel(r"$(\rho_{end}-\rho_0)/\rho_0$")
        ax[0].set_title("mass drift: the defect signature")
        ax[0].legend(fontsize=7.5)
        ax[0].grid(alpha=0.3)
        ax[1].plot(st, [1e2 * c["amp_err"] for c in leg], "o-", color="C3",
                   label="pre-fix (no steady state)")
        if fix:
            ax[1].plot(st2, [1e2 * c["amp_err"] for c in fix], "s-", color="C0",
                       label="fixed (stationary)")
        ax[1].axhline(0, color="k", lw=0.8)
        ax[1].set_xlabel("steps")
        ax[1].set_ylabel(r"$u_{max}^{num}/u_{max}^{ana}-1$  [%]")
        ax[1].set_title("amplitude vs step count")
        ax[1].legend(fontsize=7.5)
        ax[1].grid(alpha=0.3)
    fig.tight_layout()
    out = FIG / "defect_control.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_tau(res: dict) -> Path:
    sens = res["sensitivity"]
    tau = sens.get("tau", [])
    if not tau:
        return FIG / "none.png"
    t = np.array([c["tau"] for c in tau])
    e = np.array([c["eps_inf"] for c in tau])
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.semilogy(t, e, "o-")
    ax.axhline(0.02, color="gray", ls="--", lw=1.0, label="2 % gate")
    for x, y in zip(t, e):
        ax.annotate("%.2e" % y, (x, y), textcoords="offset points", xytext=(3, 5), fontsize=7)
    ax.set_xlabel(r"relaxation time $\tau$   ($\nu=(\tau-1/2)/3$)")
    ax.set_ylabel(r"$\varepsilon_\infty$  at $H=32$")
    ax.set_title("relaxation-time sensitivity")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG / "tau_sensitivity.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    FIG.mkdir(exist_ok=True)
    res = load()
    for f in (fig_profile, fig_convergence, fig_defect, fig_tau):
        print("wrote", f(res))


if __name__ == "__main__":
    main()