#!/usr/bin/env python3
"""Figures for XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTION.

Reads ``artifacts/zou_he_channel_results.json`` (written by ``run_xval.py``) and
emits PNGs into ``figures/``.  No number is typed by hand.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)


def poiseuille(y, h, u_max):
    s = [yy / h for yy in y]
    return [u_max * 4.0 * s_i * (1.0 - s_i) for s_i in s]


def fig_profiles(res: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    panels = [
        ("C1_coarse_col60", "velocity inlet + pressure outlet\n(C1: 120x18, u_in=0.05)"),
        ("C3_coarse_col60", "pressure inlet + pressure outlet\n(C3: 120x18, d rho=0.004)"),
    ]
    for ax, (key, title) in zip(axes, panels, strict=True):
        if key not in res:
            ax.set_visible(False)
            continue
        r = res[key]
        ny = len(r["ux_from_lbm"]) + 2
        y = list(range(1, ny - 1))
        ax.plot(r["ux_analytic"], y, "k-", lw=2, label="analytic Poiseuille")
        ax.plot(r["ux_from_lbm"], y, "o", ms=4, mfc="none", color="C3", label="TensorLBM Zou/He")
        ax.set_xlabel("$u_x$ [lattice]")
        ax.set_ylabel("y [lattice]")
        ax.set_title(f"{title}\n$dp_{{max}}$ = {r['dp_max'] * 100:.2f} %", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Zou/He channel inlet/outlet reconstruction vs analytic Poiseuille", fontsize=11)
    fig.tight_layout()
    out = FIG / "fig1_profiles.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_error(res: dict) -> Path:
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    keys = [
        ("C1_coarse_col60", "C1 coarse 120x18"),
        ("C2_fine_col110", "C2 fine 220x34"),
        ("C3_coarse_col60", "C3 coarse 120x18"),
        ("C4_fine_col110", "C4 fine 220x34"),
    ]
    for key, label in keys:
        if key not in res:
            continue
        r = res[key]
        ny = len(r["ux_from_lbm"]) + 2
        y = list(range(1, ny - 1))
        err = [
            abs(a - b) / r["u_max_ana"] for a, b in zip(r["ux_from_lbm"], r["ux_analytic"], strict=True)
        ]
        ax.semilogy(y, [max(e, 1e-12) for e in err], "o-", ms=3, lw=1, label=label)
    ax.axhline(0.15, color="r", ls="--", lw=1, label="owner standard 15 %")
    ax.axhline(0.03, color="g", ls=":", lw=1, label="stretch 3 %")
    ax.set_xlabel("y [lattice]")
    ax.set_ylabel(r"$|u_x^{LBM} - u_x^{ana}| / u_{max}^{ana}$")
    ax.set_title("pointwise profile error (predeclared primary metric)", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG / "fig2_error.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_sweep(res: dict) -> Path:
    sweep = res.get("C5_drho_sweep")
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    if not isinstance(sweep, list):
        ax.text(0.5, 0.5, "C5 sweep not run", ha="center")
    else:
        dr = [s["d_rho"] for s in sweep]
        ax.plot(dr, [s["dp_max"] * 100 for s in sweep], "o-", label="$dp_{max}$ [%]")
        ax.plot(
            dr,
            [abs(s["amplitude_err"]) * 100 for s in sweep],
            "s--",
            label="|amplitude error| [%]",
        )
        ax.set_xscale("log")
        ax.set_xlabel(r"$\Delta\rho = \rho_{in} - \rho_{out}$")
        ax.set_ylabel("error [%]")
        ax.set_title("driving-level sensitivity (C5: 120x18, mid-column)", fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG / "fig3_sweep.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_history(res: dict) -> Path:
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for key, label in (("C1_coarse_run", "C1 velocity inlet"), ("C4_fine_run", "C4 pressure driven")):
        run = res.get(key)
        if not isinstance(run, dict) or not run.get("history"):
            continue
        h = run["history"]
        ax.plot([x["step"] for x in h], [x["umax"] for x in h], lw=1.2, label=label)
    ax.set_xlabel("time step")
    ax.set_ylabel(r"$\max_y u_x$ at the measurement column")
    ax.set_title("march to steady state", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG / "fig4_convergence.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    data = json.loads((HERE / "artifacts" / "zou_he_channel_results.json").read_text())
    res = data["results"]
    paths = [fig_profiles(res), fig_error(res), fig_sweep(res), fig_history(res)]
    for p in paths:
        print("wrote", p)
    (FIG / "figures_manifest.json").write_text(
        json.dumps([p.name for p in paths], indent=1)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())