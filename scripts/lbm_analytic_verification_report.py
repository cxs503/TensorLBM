#!/usr/bin/env python3
"""Generate the evidence bundle for the D3Q19 analytic verification suite.

Runs the four analytic cases of ``tests/test_lbm_verification.py`` plus two
sensitivity sweeps (tau for the shear-wave viscosity, channel height for the
Poiseuille wall-location artefact) and writes:

* ``artifacts/verification/lbm_analytic_suite.json``  — all numbers
* ``artifacts/verification/lbm_analytic_suite.md``    — markdown tables
      (numbers are written by this script only; nothing is hand-copied)
* ``artifacts/verification/fig_*.png``                — report figures

Usage::

    PYTHONPATH=src python scripts/lbm_analytic_verification_report.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import test_lbm_verification as tv  # noqa: E402

OUT = os.path.join(ROOT, "artifacts", "verification")
os.makedirs(OUT, exist_ok=True)


def _pct(x: float) -> str:
    return f"{100.0 * x:.4f} %"


def main() -> int:
    t0 = time.time()
    bundle: dict = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # --- four analytic cases ------------------------------------------------
    bundle["A_shear_wave"] = {f"tau_{t}": tv.run_shear_wave(tau=t) for t in (0.7, 0.8, 1.0)}
    bundle["B_couette_H32"] = tv.run_couette(ny=34)
    bundle["B_couette_H16"] = tv.run_couette(ny=18)
    bundle["C_poiseuille"] = tv._poiseuille_cached(ny=18)
    bundle["C_resolution"] = tv.run_poiseuille_resolution_study(ny_list=(10, 18, 34))
    bundle["D_friction"] = tv.run_wall_friction()

    # --- shear-wave viscosity sensitivity to tau ----------------------------
    taus = [0.55, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0, 2.5]
    sweep = [tv.run_shear_wave(tau=t) for t in taus]
    bundle["A_tau_sweep"] = {"tau": taus,
                             "nu_exact": [m["nu_exact"] for m in sweep],
                             "nu_measured": [m["nu_measured"] for m in sweep],
                             "rel_err": [m["rel_err"] for m in sweep]}
    bundle["runtime_s"] = time.time() - t0

    with open(os.path.join(OUT, "lbm_analytic_suite.json"), "w") as fh:
        json.dump(bundle, fh, indent=2, default=float)

    # --- figure 1: shear-wave decay + nu(tau) -------------------------------
    m0 = bundle["A_shear_wave"]["tau_1.0"]
    k = 2.0 * np.pi / 32
    t = np.arange(0, m0["steps"] + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].semilogy(t, np.abs(m0["amplitude_0"]) * np.exp(-m0["nu_exact"] * k * k * t),
                   "k-", label=r"analytic $A_0e^{-\nu k^2 t}$")
    ax[0].plot([0, m0["steps"]], [m0["amplitude_0"], abs(m0["amplitude_t"])],
               "ro", label="LBM (mode projection)")
    ax[0].set_xlabel("steps"); ax[0].set_ylabel(r"$|\hat u_y(k)|$")
    ax[0].set_title("A: shear-wave decay, nx=32, tau=1.0"); ax[0].legend(); ax[0].grid(alpha=.3)

    sw = bundle["A_tau_sweep"]
    ax[1].plot(sw["tau"], np.array(sw["rel_err"]) * 100, "o-")
    ax[1].axhline(1.0, color="r", ls="--", label="1 % gate")
    ax[1].set_xlabel("tau"); ax[1].set_ylabel(r"$|\nu_{meas}-\nu_{exact}|/\nu_{exact}$ [%]")
    ax[1].set_title("A': viscosity error vs relaxation time"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_A_shear_wave.png"), dpi=130)
    plt.close(fig)

    # --- figure 2: Poiseuille profiles + convergence ------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for H, style in (("16.0", "o"), ("32.0", "s")):
        p = bundle["C_resolution"]["profiles"][H]
        num = np.array(p["num"])[1:-1]; ex = np.array(p["exact"]); yc = np.array(p["y"])[1:-1] - 0.5
        ax[0].plot(yc / (yc.max() * 2), num / ex.max(), style, ms=4, label=f"LBM H={H}")
        ax[0].plot(yc / (yc.max() * 2), ex / ex.max(), "-", lw=1,
                   label=f"analytic H={H}" if style == "o" else None)
    ax[0].set_xlabel(r"$y_c/H$"); ax[0].set_ylabel(r"$u_x/u_{x,max}$")
    ax[0].set_title("C: Poiseuille profile vs analytic"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)

    Hs = bundle["C_resolution"]["H"]
    ax[1].loglog(Hs, bundle["C_resolution"]["max_rel_err"], "o-", label="max profile error")
    hh = np.array(Hs, float)
    ax[1].loglog(hh, bundle["C_resolution"]["max_rel_err"][0] * (hh[0] / hh) ** 1.0, "k--",
                 label=r"$\propto H^{-1}$ reference")
    ax[1].set_xlabel("channel height H"); ax[1].set_ylabel("relative error [-]")
    ax[1].set_title("C': wall-location error vs H"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_C_poiseuille.png"), dpi=130)
    plt.close(fig)

    # --- figure 3: Couette + wall drag -------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for key, style in (("B_couette_H16", "o"), ("B_couette_H32", "s")):
        m = bundle[key]
        y = np.array(m["y"]); u = np.array(m["profile"])
        ax[0].plot(y / m["H"], u / m["u_lid"], style, ms=4, label=f"LBM H={m['H']:.1f}")
        ax[0].plot(y / m["H"], m["slope_exact"] * (y - 0.5) / m["u_lid"], "-", lw=1,
                   label="analytic" if style == "o" else None)
    ax[0].set_xlabel(r"$y/H$"); ax[0].set_ylabel(r"$u_x/u_{lid}$")
    ax[0].set_title("B: Couette profile"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)

    d = bundle["D_friction"]
    lab = ["exact\n$G\\,N_{fluid}$", "wet-node MEM", "Ladd sum form"]
    val = [d["drag_exact_both_walls"], d["drag_measured_both_walls"], d["drag_standard_sum_form"]]
    bars = ax[1].bar(lab, val, color=["k", "tab:green", "tab:red"], alpha=.8)
    for b, v in zip(bars, val):
        ax[1].text(b.get_x() + b.get_width() / 2, v * 1.01, f"{v:.5f}", ha="center", fontsize=8)
    ax[1].set_ylabel("total wall drag [lattice]")
    ax[1].set_title(f"D: momentum balance (wet-node err {100*d['drag_rel_err']:.4f} %)")
    ax[1].grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_BD_couette_friction.png"), dpi=130)
    plt.close(fig)

    # --- markdown tables ----------------------------------------------------
    md = ["# D3Q19 analytic verification suite — generated tables",
          "", f"generated: {bundle['generated_utc']}  |  wall time {bundle['runtime_s']:.1f} s", ""]
    md += ["## S1 shear-wave viscosity (Step A)", "",
           "| tau | nu_exact | nu_measured | rel. error | mass drift |",
           "|---|---|---|---|---|"]
    for i, tau in enumerate(sw["tau"]):
        md.append(f"| {tau} | {sw['nu_exact'][i]:.6f} | {sw['nu_measured'][i]:.6f} | "
                  f"{_pct(sw['rel_err'][i])} | — |")
    md += ["", "## S2 Poiseuille profile (Step C) and refinement", "",
           "| H | max profile error | peak error | steps to steady | implied wall offset |",
           "|---|---|---|---|---|"]
    for j, H in enumerate(bundle["C_resolution"]["H"]):
        md.append(f"| {H:.0f} | {_pct(bundle['C_resolution']['max_rel_err'][j])} | "
                  f"{_pct(bundle['C_resolution']['peak_rel_err'][j])} | "
                  f"{bundle['C_resolution']['steps'][j]} | "
                  f"{bundle['C_resolution']['implied_wall_offset'][j]:.3f} |")
    md += ["", "## S3 Couette interior shear rate (Step B)", "",
           "| H | slope fit | slope exact | rel. error | linearity residual | steps |",
           "|---|---|---|---|---|---|"]
    for key in ("B_couette_H16", "B_couette_H32"):
        m = bundle[key]
        md.append(f"| {m['H']:.1f} | {m['slope_fit']:.8f} | {m['slope_exact']:.8f} | "
                  f"{_pct(m['slope_rel_err'])} | {m['linearity_residual']:.2e} | {m['steps']} |")
    md += ["", "## S4 wall drag: observers vs exact balance (Step D)", "",
           "| quantity | value |", "|---|---|",
           f"| exact $G N_{{fluid}}$ (both walls) | {d['drag_exact_both_walls']:.6e} |",
           f"| wet-node MEM | {d['drag_measured_both_walls']:.6e} (err {_pct(d['drag_rel_err'])}) |",
           f"| Ladd sum form | {d['drag_standard_sum_form']:.6e} "
           f"(ratio {d['standard_over_wet_ratio']:.3f}) |",
           f"| wall shear (wet-node) | {d['wall_shear_measured']:.6e} vs exact {d['wall_shear_exact']:.6e} |",
           f"| transverse leakage fy, fz | {d['fy_leak']:.2e}, {d['fz_leak']:.2e} |"]
    with open(os.path.join(OUT, "lbm_analytic_suite.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")

    print("\n".join(md))
    print(f"\nwrote {OUT}/lbm_analytic_suite.json/.md + 3 png in {bundle['runtime_s']:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())