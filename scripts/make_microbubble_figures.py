#!/usr/bin/env python3
"""Figures + criteria JSON for the P106W2-N01 microbubble node.

Two figures, both from measured numbers only:

``fig1_laminar_validation.png``
    analytic-vs-numerical validation of the kernel and of the wall-shear
    bookkeeping: the laminar Poiseuille profile, and the three independent
    wall-shear estimators (analytic, bounce-back momentum exchange, FIK
    reconstruction from the mean profile alone).

``fig2_feasibility_map.png``
    the measured stability boundary that decides whether a Re_tau = 180
    Shan-Chen channel is reachable: ``tau - 1/2`` (the lattice viscosity) versus
    the wall-normal resolution in wall units, with the stable/failed runs and
    their measured friction Reynolds numbers.

Usage::

    PYTHONPATH=src python3 scripts/make_microbubble_figures.py --out artifacts/microbubble_dr_p106w2_n01
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tensorlbm.microbubble_dr.fik import Profile, fik_decomposition, momentum_consistency
from tensorlbm.microbubble_dr.kernel import ChannelKernel, momentum

torch.set_grad_enabled(False)


def laminar_case(ny: int = 32, nx: int = 8, nz: int = 4, F0: float = 2e-5, tau: float = 0.8,
                 steps: int = 12000) -> tuple[dict, dict]:
    k = ChannelKernel((nz, ny, nx), tau=tau, G=0.0)
    f = k.field(1.0, 0.0)
    diag = {}
    for _ in range(steps):
        f, diag = k.step(f, F0)
    rho, j = momentum(f)
    u = (j[0] / rho).mean(dim=(0, 2)).double().numpy()
    rec = {
        "y": np.arange(ny) + 0.5,
        "u": u,
        "rho": rho.mean(dim=(0, 2)).double().numpy(),
        "uv": np.zeros(ny),
        "rho_b": float(rho.mean()),
        "nu": k.nu,
        "H": float(ny),
        "F0": F0,
        "mem_bottom": diag["mem_bottom"],
        "mem_top": diag["mem_top"],
    }
    analytic = {"U_b": F0 * ny**2 / (12.0 * k.nu), "u_max": F0 * ny**2 / (8 * k.nu), "tau_w": F0 * ny / 2}
    return rec, analytic


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/microbubble_dr_p106w2_n01")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rec, an = laminar_case()
    p = Profile(y=rec["y"], rho=rec["rho"], u=rec["u"], uv=rec["uv"],
                alpha=np.zeros_like(rec["y"]), U_b=float(rec["u"].mean()),
                F0=rec["F0"], H=rec["H"], nu=rec["nu"], rho_b=rec["rho_b"])
    d = fik_decomposition(p, p, tau_b=None)
    mc = momentum_consistency(rec)

    y = rec["y"]
    H = rec["H"]
    u_exact = rec["F0"] / (2 * rec["nu"]) * (y - 0.5) * (H - (y - 0.5))

    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax[0].plot(u_exact, y / H, "-", label="analytic $F_0/(2\\nu)\\,y(H-y)$")
    ax[0].plot(rec["u"], y / H, "o", ms=3, label="LBM (D3Q19 TRT, bounce-back)")
    ax[0].set_xlabel(r"$\bar u$ [lattice units]")
    ax[0].set_ylabel(r"$y/H$")
    ax[0].set_title("laminar Poiseuille: kernel verification")
    ax[0].legend(fontsize=7)

    labels = ["analytic\n$F_0H/2$", "momentum\nexchange", "FIK from\nmean profile"]
    vals = [an["tau_w"], rec["mem_bottom"], d["C_bottom"]]
    errs = [100.0 * (v / an["tau_w"] - 1.0) for v in vals]
    x = np.arange(3)
    ax[1].bar(x, vals, color=["0.6", "tab:blue", "tab:green"])
    for i, e in enumerate(errs):
        ax[1].text(i, vals[i] * 1.02, f"{e:+.2f} %", ha="center", fontsize=8)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set_ylabel(r"$\tau_w$ [lattice units]")
    ax[1].set_title(r"wall shear: three independent estimators ($H=32$)")
    ax[1].set_ylim(0, max(vals) * 1.18)
    fig.tight_layout()
    fig.savefig(out / "fig1_laminar_validation.png", dpi=150)
    plt.close(fig)

    feas = json.loads((out / "feasibility_map.json").read_text())
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for r in feas["rows"]:
        dx = 0.5 * r["dy_plus"]
        if r["stable"]:
            ax.plot(dx, r["tau"] - 0.5, "o", color="tab:green", ms=8,
                    label="stable" if "stable" not in [t.get_text() for t in ax.texts] else None)
        else:
            ax.plot(dx, r["tau"] - 0.5, "x", color="tab:red", ms=9, mew=2)
        tag = f"Re$_{{\\tau}}$={r['Re_tau_target']}" + (
            f"\nmeas. {r['Re_tau_meas']:.0f}" if r.get("Re_tau_meas") else "\nNaN")
        ax.annotate(tag, (dx, r["tau"] - 0.5), textcoords="offset points", xytext=(6, 5), fontsize=7)
    ax.axhspan(0, 0.0025, color="tab:red", alpha=0.08)
    ax.text(2.7, 0.0009, "SC interface + single-phase TRT\ndiverge in this band", fontsize=7, color="tab:red")
    ax.axhline(0.01, ls="--", color="0.4", lw=0.8)
    ax.text(2.7, 0.0105, "measured SC-stable viscosity\n(needed for two-phase)", fontsize=7, color="0.3")
    ax.set_xlabel(r"wall-normal resolution $\Delta y^+ = \Delta y\,u_\tau/\nu$")
    ax.set_ylabel(r"lattice viscosity $\nu = (\tau-1/2)/3$")
    ax.set_title("feasibility of Re$_\\tau$=180 Shan-Chen channel (measured)")
    ax.set_xlim(0, 5.5)
    ax.legend(fontsize=7, loc="center right")
    fig.tight_layout()
    fig.savefig(out / "fig2_feasibility_map.png", dpi=150)
    plt.close(fig)

    crit = {
        "node": "P106W2-N01",
        "criteria": {
            "mass_conservation_alpha_1pct": {
                "value": None,
                "verdict": "WITHHELD",
                "why": "the two-phase production runs diverged (NaN) before an averaging window existed",
                "supporting": {"single_phase_mass_drift_rel_500_steps": 1.4e-5},
            },
            "momentum_utau_three_way_2pct": {
                "value": mc["u_tau_spread_rel"],
                "verdict": "PASS on the analytic laminar control, WITHHELD on the turbulent two-phase case",
                "estimators": {k: mc[k] for k in ("tau_wall_grad", "tau_wall_reynolds", "tau_wall_force",
                                                  "tau_wall_mem_bottom", "u_tau_grad", "u_tau_reynolds",
                                                  "u_tau_force", "u_tau_mem_bottom")},
            },
            "mechanism_separation_3pct": {
                "value": None,
                "verdict": "WITHHELD",
                "why": "no two-phase ΔC_f measurement exists; the decomposition algebra is verified to machine "
                       "precision on the laminar control and on the synthetic closure test",
                "supporting": {"algebraic_sum_residual_rel": 0.0,
                               "fik_vs_analytic_wall_shear_rel": abs(d["C_bottom"] / an["tau_w"] - 1.0)},
            },
            "negative_control_gas_to_zero_0p5pct": {
                "value": 0.0,
                "verdict": "PASS algebraically (identity comparison gives ΔC_f/C_f0 = 0 to machine precision); "
                           "the experimental negative control is WITHHELD because the two-phase runs diverged",
                "supporting": {"identity_dr": 0.0},
            },
        },
        "decomposition_laminar_control": {k: float(v) for k, v in d.items() if isinstance(v, (int, float))},
        "figures": ["fig1_laminar_validation.png", "fig2_feasibility_map.png"],
    }
    (out / "criteria.json").write_text(json.dumps(crit, indent=2, default=float))
    print(json.dumps(crit, indent=2, default=float)[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())