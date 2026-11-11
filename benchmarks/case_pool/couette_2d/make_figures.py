#!/usr/bin/env python3
"""Figures for CASE-COUETTE-2D (reads artifacts/summary.json, no recompute).

fig1_couette_profile.png   u(y)/U0 vs y/H — numérics at H = 16/32/64 vs the
                           closed-form straight line, plus the pointwise error.
fig2_couette_errors.png    (a) L2 and max-pointwise error vs H (log-log);
                           (b) sensitivity of the L2 error to Re (tau sweep)
                           and to Ma (U0 sweep).

Usage:
    PYTHONPATH=src:benchmarks python3 benchmarks/case_pool/couette_2d/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"


def _grid_points(summary: dict) -> list[dict]:
    pts = [
        r
        for r in summary["route_a"]
        if r["axis"] == "grid" and r["collision"] == "bgk" and r["tau"] == 0.8 and r["U0"] == 0.05
    ]
    return sorted(pts, key=lambda r: r["H"])


def fig_profile(summary: dict) -> Path:
    pts = _grid_points(summary)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
    colors = {16: "tab:blue", 32: "tab:green", 64: "tab:red"}
    ref = None
    for r in pts:
        y = np.array([p["y"] for p in r["profile"]])
        u = np.array([p["u_num"] for p in r["profile"]])
        ur = np.array([p["u_ref"] for p in r["profile"]])
        H = r["H"]
        ref = ur / r["U0"]
        ax.plot(u / r["U0"], (y - 0.5) / H, "o", ms=3.2, mfc="none",
                color=colors.get(H, "k"), label=f"H={H} (n_steps={r['n_steps']})")
        ax2.semilogy(y / H, np.abs(u - ur) / r["U0"] + 1e-18, "o-", ms=2.5, lw=0.9,
                     color=colors.get(H, "k"), label=f"H={H}")
    ax.plot(ref, np.arange(1, len(ref) + 1) - 0.5, "-", color="k", lw=1.2,
            label="analytic $u/U_0 = y/H$")
    ax.set_xlabel("$u/u_{wall}$")
    ax.set_ylabel("$y/H$")
    ax.set_title("Plane Couette: numerical profile vs closed form")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    ax2.set_xlabel("$y/H$")
    ax2.set_ylabel(r"$|u_{num}-u_{ref}|/u_{wall}$")
    ax2.set_title("pointwise profile error")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = ART / "fig1_couette_profile.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_errors(summary: dict) -> Path:
    pts = _grid_points(summary)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    H = np.array([r["H"] for r in pts], dtype=float)
    l2 = np.array([r["errors_pct"]["profile_l2_rel_pct"] for r in pts])
    mx = np.array([r["errors_pct"]["profile_max_rel_pct"] for r in pts])
    tau = np.array([r["errors_pct"]["wall_shear_rel_pct"] for r in pts])
    ax.loglog(H, l2, "o-", label="profile $L_2$ rel. err.")
    ax.loglog(H, mx, "s-", label="profile max pointwise")
    ax.loglog(H, tau, "^-", label=r"wall shear $\tau_w$ vs $\mu U/H$")
    ax.axhline(summary["tol_policy_pct"], color="r", ls="--", lw=1.0, label="tol 2%")
    ax.set_xlabel("H (lattice nodes across the gap)")
    ax.set_ylabel("relative error [%]")
    ax.set_title("(a) grid convergence (tau=0.8, U0=0.05, BGK)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    for axis, marker, lab in (("tau", "o", "tau sweep (Re = U0 H / nu)"),
                              ("U0", "s", "U0 sweep (Ma = U0 / c_s)")):
        rows = [r for r in summary["route_a"] if r["axis"] == axis]
        if not rows:
            continue
        x = np.array([r["Re"] if axis == "tau" else r["Ma"] for r in rows])
        y = np.array([r["errors_pct"]["profile_l2_rel_pct"] for r in rows])
        ax2.semilogy(x, y, marker + "-", label=lab)
    ax2.axhline(summary["tol_policy_pct"], color="r", ls="--", lw=1.0, label="tol 2%")
    ax2.set_xlabel("Re (tau sweep)   /   Ma (U0 sweep)")
    ax2.set_ylabel("profile $L_2$ rel. err. [%]")
    ax2.set_title("(b) sensitivity: viscosity and compressibility")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    out = ART / "fig2_couette_errors.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> None:
    summary = json.loads((ART / "summary.json").read_text())
    for p in (fig_profile(summary), fig_errors(summary)):
        print(p.relative_to(HERE.parents[2]))


if __name__ == "__main__":
    main()