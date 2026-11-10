#!/usr/bin/env python
"""Generate the figures of the lattice-unit bootstrap gate report.

Reads ``artifacts/units_bootstrap_gate/gate_report.json`` plus the anchor
ladder and writes PNGs into the same directory.  All numbers come from the
gate report (no hand-typed values): the script re-runs the gate only if the
JSON is missing.

Usage::

    PYTHONPATH=src python scripts/units_bootstrap_gate_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorlbm.units import ANCHORS, GRID_SPREAD_TOL, TOLERANCES  # noqa: E402
from tensorlbm.units import anchors as anchor_mod  # noqa: E402
from tensorlbm.units import gate as gate_mod  # noqa: E402

OUT = Path("artifacts/units_bootstrap_gate")
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]


def load_report() -> dict:
    path = OUT / "gate_report.json"
    if path.exists():
        return json.loads(path.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    report = gate_mod.run_gate()
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=float))
    return report


def fig_anchor_errors(report: dict) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, name in enumerate(ANCHORS):
        rows = report["anchors"][name]
        grids = [r["grid"] for r in rows]
        errs = [max(abs(r["worst_rel_error"]), 1e-17) for r in rows]
        ax.semilogy(grids, errs, "o-", color=PALETTE[i], label=f"{name}")
        ax.axhline(TOLERANCES[name], color=PALETTE[i], ls=":", lw=1, alpha=0.7)
    ax.set_xlabel("lattice resolution of the anchor case [cells]")
    ax.set_ylabel("worst |relative error| vs analytic law")
    ax.set_title("Four analytic anchors: error vs pre-declared tolerance\n"
                 "(dotted line = tolerance)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = OUT / "fig_anchor_errors.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_mutation_matrix(report: dict) -> Path:
    matrix = report["mutation_battery"]["matrix_worst_rel_error"]
    muts = list(matrix)
    anchors = list(ANCHORS)
    data = np.array([[max(matrix[m][a], 1e-18) for a in anchors] for m in muts])
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    im = ax.imshow(np.log10(data), cmap="magma", aspect="auto", vmin=-16, vmax=1.5)
    ax.set_xticks(range(len(anchors)))
    ax.set_xticklabels(anchors, rotation=15)
    ax.set_yticks(range(len(muts)))
    ax.set_yticklabels(muts)
    for i, m in enumerate(muts):
        for j, a in enumerate(anchors):
            caught = data[i, j] > TOLERANCES[a]
            ax.text(j, i, ("X" if caught else "-"),
                    ha="center", va="center", fontsize=9,
                    color="white" if caught else "#cccccc")
            ax.text(j, i - 0.28, f"{data[i, j]:.1e}", ha="center", va="center",
                    fontsize=6, color="#dddddd")
    fig.colorbar(im, ax=ax, label="log10 worst |relative error|")
    ax.set_title("Mutation battery: single-site coefficient defects\n"
                 "X = caught by that anchor")
    fig.tight_layout()
    path = OUT / "fig_mutation_battery.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_grid_invariance(report: dict) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, (name, gi) in enumerate(report["grid_invariance"].items()):
        grids = np.array(gi["grids"], dtype=float)
        vals = np.array(gi["values"])
        ax.plot(grids, (vals - 1.0) * 1e4, "o-", color=PALETTE[i],
                label=f"{name} (spread {gi['spread']*1e4:.2f}e-4)")
    ax.axhline(0.0, color="k", lw=1)
    ax.set_xlabel("lattice resolution of the three (dx, dt) groups")
    ax.set_ylabel("dimensionless anchor number - 1  [1e-4]")
    ax.set_title("Grid invariance over three (dx, dt) groups\n"
                 f"required spread <= {GRID_SPREAD_TOL:.0%}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = OUT / "fig_grid_invariance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_poiseuille_profile() -> Path:
    case = anchor_mod.build_case("poiseuille", 0, 48)
    from tensorlbm.units.lattice import channel_poiseuille

    lat = case.lattice
    sim = channel_poiseuille(int(case.grid), 1, lat["tau"], lat["force_lb"],
                             tol=1e-8, check_every=100)
    bridge = case.bridge
    y_lb = np.arange(int(case.grid)) + 0.5
    y_phys = bridge.length_to_phys(y_lb) * 1e3          # mm
    u_phys = bridge.velocity_to_phys(sim["ux"]) * 1e3    # mm/s
    h_lb = lat["h_lb"]
    exact_lb = lat["force_lb"] / (2.0 * lat["nu_lb"]) * y_lb * (2 * h_lb - y_lb)
    u_exact = bridge.velocity_to_phys(exact_lb) * 1e3

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    ax = axes[0]
    ax.plot(u_phys, y_phys, "o", ms=4, label="LBM (lattice run, dimensionalised)")
    ax.plot(u_exact, y_phys, "-", lw=1.4, label="analytic parabola $f h^2/(8\\rho\\nu)$")
    ax.set_xlabel("u [mm/s]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"Poiseuille profile (Ny={case.grid}, "
                 f"tau={lat['tau']:.2f}, Ma={bridge.mach(anchor_mod.U_LB):.3f})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.semilogy(y_phys, np.maximum(np.abs(u_phys - u_exact) / u_exact.max(), 1e-18),
                "s-", ms=3, color="#d62728")
    ax.set_xlabel("y [mm]")
    ax.set_ylabel("|u_LBM - u_exact| / u_max")
    ax.set_title("Normalised profile residual")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = OUT / "fig_poiseuille_profile.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = load_report()
    paths = [fig_anchor_errors(report), fig_mutation_matrix(report),
             fig_grid_invariance(report), fig_poiseuille_profile()]
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()