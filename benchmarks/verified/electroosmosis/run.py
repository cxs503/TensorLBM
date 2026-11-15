#!/usr/bin/env python
"""Benchmark: electro-osmotic (EOF) plug flow in a slit microchannel.

Capability under test
---------------------
Lattice-Boltzmann solution of the steady Stokes problem driven by the
electro-osmotic body force ``rho_e(y) * E`` originating in the electrical
double layer (EDL), with no-slip walls -- i.e. the flow side of
electrokinetics (Lab-on-chip / microfluidic transport), not just the
concentration/PNP side.

Reference
---------
Symmetric slit, both walls at the same zeta potential.  In the Debye-Huckel
limit the EDL potential obeys ``psi'' = psi/lambda_D**2`` with
``psi(0) = psi(H) = zeta``, hence

    psi(y) = zeta cosh((y - H/2)/lambda_D) / cosh(H/(2 lambda_D))
    rho_e  = -eps psi''

and because ``mu u'' = -rho_e E = eps E psi''`` is a total derivative, the
solution with two no-slip walls is obtained for *any* EDL structure:

    u(y) = u_hs [1 - psi(y)/zeta],    u_hs = -eps zeta E / mu

``u_hs`` is the Helmholtz-Smoluchowski velocity, so the exact profile is a
scaled copy of the potential defect.  ``kappa_H = H/lambda_D`` controls how
plug-like the flow is (the core approaches ``u_hs`` as kappa_H grows).

What this benchmark demonstrates
--------------------------------
1. The LBM kernel reproduces the exact profile to O(dy^2); grid convergence
   over ny and over kappa_H (fixed ny) is reported.
2. Two independent discretisations (D2Q9 LBM vs direct FD/Thomas solve) agree.
3. Negative controls (0.8x force, reversed field, no field) are rejected, so
   the pass criterion has discriminating power.
4. Defect record: the previous ``run_electroosmosis``/gate C3 was circular
   (it compared the analytical identity against itself) - see the README.

Runs in lattice units; ``u_hs_lat = 0.02`` keeps Ma ~ 0.03.

Usage:  python benchmarks/verified/electroosmosis/run.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.verification.cdr_pnp.eof_lbm import (  # noqa: E402
    run_eof_benchmark,
)

TOLERANCE = 0.03  # 3%: repo-wide acceptance line
CASES = [(5.0, 65), (5.0, 129), (10.0, 65), (10.0, 129), (20.0, 65), (20.0, 129)]
MAX_STEPS = 300_000


def run_case(kappa_H: float, ny: int, tol: float = 1e-12) -> dict:
    out = {"kappa_H": kappa_H, "ny": ny, "nodes_per_debye": (ny - 2) / kappa_H}
    lbm = run_eof_benchmark(ny=ny, kappa_H=kappa_H, solver="lbm",
                            max_steps=MAX_STEPS, tol=tol)
    fd = run_eof_benchmark(ny=ny, kappa_H=kappa_H, solver="fd")
    out.update(
        {
            "lbm_max_rel": lbm["errors"]["max_rel"],
            "lbm_l2_rel": lbm["errors"]["l2_rel"],
            "lbm_bias_rel": lbm["errors"]["bias_rel"],
            "fd_max_rel": fd["errors"]["max_rel"],
            "lbm_vs_fd_max_rel": float(
                np.max(np.abs(lbm["u"] - fd["u"])) / lbm["u_hs"]),
            "flatness_core": lbm["flatness"],
            "steps": lbm["steps"],
            "converged": bool(lbm["converged"]),
            "residual": float(lbm["residual"]),
            "passed": bool(lbm["errors"]["max_rel"] < TOLERANCE
                           and lbm["converged"]),
        }
    )
    return out, lbm, fd


def negative_controls(ny: int = 65, kappa_H: float = 20.0) -> list:
    rows = []
    for name, kw in (("physical", dict()),
                     ("force_0.8x", dict(force_scale=0.8)),
                     ("field_reversed", dict(force_sign=-1.0)),
                     ("no_field", dict(force_scale=0.0))):
        r = run_eof_benchmark(ny=ny, kappa_H=kappa_H, solver="lbm",
                              max_steps=120_000, tol=1e-11, **kw)
        err = r["errors"]["max_rel"]
        rows.append({"control": name, "max_rel_error": err,
                     "passes_3pct": bool(err < TOLERANCE)})
    return rows


def make_figures(cases_data, neg_rows, fd_sweep, outdir: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    figs = []

    # (1) normalised velocity profiles vs the exact solution, kappa_H sweep
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    ax = axes[0]
    colors = {5.0: "C0", 10.0: "C1", 20.0: "C2"}
    for kappa_H in (5.0, 10.0, 20.0):
        _, lbm, _ = [c for c in cases_data if c[0]["kappa_H"] == kappa_H
                     and c[0]["ny"] == 129][0]
        xi = lbm["y_phys"] / lbm["H_eff"]
        ax.plot(xi, lbm["u"] / lbm["u_hs"], "-", color=colors[kappa_H],
                label=f"LBM  κH={kappa_H:g}")
        ax.plot(xi, lbm["u_exact"] / lbm["u_hs"], "k--", lw=0.8)
    ax.plot([], [], "k--", lw=0.8, label="exact  $u_{hs}[1-\\psi/\\zeta]$")
    ax.set_xlabel("y / H")
    ax.set_ylabel("$u / u_{hs}$")
    ax.set_title("(a) EOF plug-flow profiles, ny = 129")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (2) discretisation error vs EDL resolution
    ax = axes[1]
    for kappa_H in (5.0, 10.0, 20.0):
        pts = [(c[0]["nodes_per_debye"], c[0]["lbm_max_rel"])
               for c in cases_data if c[0]["kappa_H"] == kappa_H]
        pts.sort()
        ax.loglog([p[0] for p in pts], [p[1] for p in pts], "o-",
                  color=colors[kappa_H], label=f"LBM κH={kappa_H:g}")
        pts = [(c[0]["nodes_per_debye"], c[0]["fd_max_rel"])
               for c in cases_data if c[0]["kappa_H"] == kappa_H]
        pts.sort()
        ax.loglog([p[0] for p in pts], [p[1] for p in pts], "s--",
                  color=colors[kappa_H], alpha=0.6, label=f"FD κH={kappa_H:g}")
    ax.axhline(TOLERANCE, color="r", ls=":", label="3% acceptance")
    ax.set_xlabel("nodes per Debye length  $(H/\\kappa_H^{-1})/(ny-2)^{-1}$")
    ax.set_ylabel("max profile error / $u_{hs}$")
    ax.set_title("(b) grid convergence, LBM vs independent FD")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig.savefig(outdir / "eof_profiles_convergence.png", dpi=150)
    figs.append("eof_profiles_convergence.png")
    plt.close(fig)

    # (3) EDL structure + negative controls
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), constrained_layout=True)
    ax = axes[0]
    _, lbm, _ = [c for c in cases_data if c[0]["kappa_H"] == 20.0
                 and c[0]["ny"] == 129][0]
    xi = lbm["y_phys"] / lbm["H_eff"]
    ax.plot(xi, lbm["psi_norm"], color="C3", label="$\\psi/\\zeta$ (EDL potential)")
    ax.plot(xi, lbm["u"] / lbm["u_hs"], color="C0", label="$u/u_{hs}$ (LBM)")
    ax.set_xlabel("y / H")
    ax.set_title("(c) EDL structure and flow, κH = 20")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    names = [r["control"] for r in neg_rows]
    errs = [r["max_rel_error"] for r in neg_rows]
    bars = ax.bar(names, errs, color=["C2", "C1", "C1", "C1"])
    ax.set_yscale("log")
    ax.axhline(TOLERANCE, color="r", ls=":", label="3% acceptance")
    ax.set_ylabel("max profile error / $u_{hs}$")
    ax.set_title("(d) negative controls must be rejected")
    for b, e in zip(bars, errs):
        ax.text(b.get_x() + b.get_width() / 2, e * 1.15, f"{100*e:.1f}%",
                ha="center", fontsize=8)
    ax.legend(fontsize=8)
    fig.savefig(outdir / "eof_edl_and_controls.png", dpi=150)
    figs.append("eof_edl_and_controls.png")
    plt.close(fig)
    return figs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="ny=65 only")
    args = ap.parse_args()

    cases = [(k, ny) for (k, ny) in CASES if (not args.quick or ny == 65)]
    cases_data = []
    rows = []
    for kappa_H, ny in cases:
        row, lbm, fd = run_case(kappa_H, ny)
        rows.append(row)
        cases_data.append((row, lbm, fd))
        print(f"kappa_H={kappa_H:5.1f} ny={ny:4d} "
              f"nodes/lam={(ny-2)/kappa_H:5.2f}  "
              f"LBM={100*row['lbm_max_rel']:7.3f}%  "
              f"FD={100*row['fd_max_rel']:7.3f}%  "
              f"LBM-vs-FD={100*row['lbm_vs_fd_max_rel']:6.3f}%  "
              f"flat={100*row['flatness_core']:.3f}%  "
              f"steps={row['steps']}")

    neg = negative_controls()
    for r in neg:
        print(f"control {r['control']:16s} err={100*r['max_rel_error']:8.2f}%  "
              f"passes_3pct={r['passes_3pct']}")

    figs = make_figures(cases_data, neg, None, HERE / "figures")

    worst = max(r["lbm_max_rel"] for r in rows)
    result = {
        "benchmark": "electroosmosis",
        "capability": "LBM electro-osmotic (EDL-driven) flow in a slit",
        "reference": "exact Poisson-Boltzmann profile u = u_hs(1 - psi/zeta)",
        "tolerance": TOLERANCE,
        "units": "lattice (u_hs_lat = 0.02, tau = 1.0, nu = 1/6)",
        "cases": rows,
        "negative_controls": neg,
        "worst_case_error": worst,
        "all_passed": bool(all(r["passed"] for r in rows)
                           and all(not r["passes_3pct"] or r["control"] == "physical"
                                   for r in neg)),
        "verdict": "VERIFIED" if worst < TOLERANCE else "NOT_VERIFIED",
        "figures": figs,
    }
    (HERE / "result.json").write_text(json.dumps(result, indent=2))
    print(f"\nworst case error = {100*worst:.3f}%  -> {result['verdict']}")
    print(f"wrote {HERE/'result.json'}")
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())