#!/usr/bin/env python
"""Figures + criteria table for node IN31-N12 (RB discrete stability + HR).

Reads the JSON artifacts produced by ``run_bench.py`` / ``run_hr_bench.py`` and
writes PNG figures plus a machine-readable criteria summary.  No number in the
figures is typed by hand.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RA_CRIT = 1707.7597
A_CRIT = 3.11632


def load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def fig_rb_growth(rb_runs, out_png: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for run in rb_runs:
        arms = sorted(run["arms"], key=lambda a: a["Ra"])
        ra = [a["Ra"] for a in arms]
        gl = [a["gamma_hat_lattice"] for a in arms]
        gc = [a["gamma_hat_continuum"] for a in arms]
        ny = run["params"]["ny"]
        axes[0].plot(ra, gl, "o-", label=f"lattice ny={ny}")
        axes[0].plot(ra, gc, "s--", label=f"continuum ny={ny}")
        rel = [100 * a["rel_err_lattice_vs_cont"] for a in arms]
        axes[1].plot(ra, rel, "o-", label=f"ny={ny}")
    axes[0].axhline(0.0, color="k", lw=0.8)
    axes[0].axvline(RA_CRIT, color="grey", lw=0.8, ls=":")
    axes[0].set_xlabel("Ra")
    axes[0].set_ylabel(r"$\hat\gamma = \gamma H^2/\nu$")
    axes[0].set_title("growth rate: lattice operator vs continuum")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].set_xscale("log")
    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[1].set_xlabel("Ra")
    axes[1].set_ylabel(r"relative difference of $\hat\gamma$ [%]")
    axes[1].set_title("lattice - continuum (equal Ra, equal a)")
    axes[1].legend(fontsize=8)
    axes[1].set_xscale("log")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def fig_ra_c_convergence(rb_runs, out_png: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    nys = sorted(r["params"]["ny"] for r in rb_runs)
    ra_c = {r["params"]["ny"]: r["Ra_c_lattice"] for r in rb_runs}
    ny_arr = np.array([ra_c[n] for n in nys])
    axes[0].plot(nys, ny_arr, "o-", label=r"$Ra_c$ (benchmark $H=ny$)")
    real = np.array(nys, dtype=float)
    axes[0].plot(nys, RA_CRIT * (real / (real - 1.0)) ** 3, "k:", label=r"$1707.76\,(ny/(ny-1))^3$")
    axes[0].plot(nys, RA_CRIT * (real / (real - 1.0)) ** 3 * 1.0, "k:", label=None)
    axes[0].axhline(RA_CRIT, color="grey", ls="--", label="Chandrasekhar 1707.76")
    shifted = np.array([ra_c[n] * ((n - 1.0) / n) ** 3 for n in nys])
    axes[1].plot(nys, 100 * (shifted / RA_CRIT - 1.0), "o-", label="after H_eff = ny-1 remap")
    axes[1].plot(nys, 100 * (ny_arr / RA_CRIT - 1.0), "s--", label="as measured (H = ny)")
    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[1].set_xlabel("ny (gap in lattice units)")
    axes[1].set_ylabel(r"$Ra_c$ shift vs 1707.76 [%]")
    axes[1].set_title("convergence of the lattice critical Rayleigh number")
    axes[1].legend(fontsize=8)
    axes[0].set_xlabel("ny")
    axes[0].set_ylabel(r"$Ra_c$")
    axes[0].set_title(r"$Ra_c$ with the measured effective gap")
    axes[0].legend(fontsize=8)
    axes[0].set_xscale("log")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def fig_eig_vs_sim(rb_runs, out_png: Path):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    allx, ally = [], []
    for run in rb_runs:
        for c in run.get("sim_checks", []):
            allx.append(c["gamma_step_sim"])
            ally.append(c["gamma_step_eig"])
            ax.plot(c["gamma_step_sim"], c["gamma_step_eig"], "o", ms=6,
                    label=f"ny={run['params']['ny']}")
    if allx:
        xs = np.array([0.0, 1.25 * max(abs(v) for v in allx + ally)])
        ax.plot(xs, xs, "k-", lw=0.8, label="identity")
        ax.plot(xs, xs * 1.01, "k--", lw=0.6)
        ax.plot(xs, xs * 0.99, "k--", lw=0.6)
        ax.set_xlabel(r"$\gamma$ from time-marching the non-linear solver")
        ax.set_ylabel(r"$\gamma$ from the Jacobian eigenvalue")
        ax.set_title("eigenvalue vs simulation (dashed: $\\pm1\\%$)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def fig_base_profile(prof, out_png: Path):
    rows = [r for r in prof if r["Ra"] == min(p["Ra"] for p in prof if p["ny"] == r["ny"])]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    x = np.array([1.0 / r["ny"] for r in sorted(rows, key=lambda r: r["ny"])])
    y = np.array([r["H_eff"] for r in sorted(rows, key=lambda r: r["ny"])])
    ax.plot(x, y, "o-", label="measured $H_{eff}$")
    ax.plot(x, 1.0 / x - 1.0, "k:", label="$ny-1$")
    ax.plot(x, 1.0 / x, "k--", label="$ny$ (half-way gap)")
    ax.set_xlabel("1/ny")
    ax.set_ylabel("effective gap $H_{eff}$ [lattice units]")
    ax.set_title("discrete conduction profile pins the walls at the wall nodes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def fig_hr(dec, settle, out_png: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    if dec:
        rows = dec["decoupling"]
        lams = sorted({r["lam"] for r in rows})
        for dr in sorted({r["density_ratio"] for r in rows}):
            sel = sorted([r for r in rows if r["density_ratio"] == dr], key=lambda r: r["lam"])
            axes[0].plot([r["lam"] for r in sel], [r["lam_measured"] for r in sel], "o-",
                         label=f"$\\rho_d/\\rho_c$={dr}")
        xs = np.array([min(lams) * 0.8, max(lams) * 1.2])
        axes[0].plot(xs, xs, "k--", lw=0.8, label="prescribed")
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].set_xlabel(r"prescribed $\lambda=\mu_d/\mu_c$")
        axes[0].set_ylabel(r"measured $\lambda$ (shear-wave)")
        axes[0].set_title("viscosity ratio is decoupled from density ratio")
        axes[0].legend(fontsize=8)
    if settle:
        runs = [r for r in settle.get("settling", []) if r.get("status") == "ok"]
        if runs:
            lams = [r["setup"]["lam_nominal"] for r in runs]
            axes[1].plot(lams, [r["U_rel"] for r in runs], "o-", label="measured LBM")
            axes[1].plot(lams, [r["U_hr"] for r in runs], "s--", label="Hadamard-Rybczynski")
            for r, x in zip(runs, lams):
                axes[1].errorbar(x, r["U_rel"], yerr=r["U_rel_spread"], fmt="none", capsize=3)
            axes[1].set_xscale("log")
            axes[1].set_xlabel(r"$\lambda$")
            axes[1].set_ylabel("terminal relative velocity")
            axes[1].set_title("drop settling vs HR law")
            axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rb-dir", default="artifacts/in31_n12")
    ap.add_argument("--extra-rb", default="")
    ap.add_argument("--figdir", default="figures/in31_n12")
    ap.add_argument("--out", default="artifacts/in31_n12/criteria.json")
    args = ap.parse_args()
    rb_dir = Path(args.rb_dir)
    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)

    runs = []
    extra = [Path(args.extra_rb)] if args.extra_rb else []
    for p in extra + sorted(rb_dir.glob("rb_lattice_ny*.json")):
        d = load(p)
        if d:
            runs.append(d)
    runs = sorted({r["params"]["ny"]: r for r in runs}.values(), key=lambda r: r["params"]["ny"])
    prof = load(rb_dir / "base_profile.json") or []
    dec = load(rb_dir / "hr_decoupling.json")
    settle = load(rb_dir / "hr_settling.json")

    if runs:
        fig_rb_growth(runs, figdir / "rb_growth_rate.png")
        fig_ra_c_convergence(runs, figdir / "rb_ra_c_convergence.png")
        fig_eig_vs_sim(runs, figdir / "rb_eig_vs_sim.png")
    if prof:
        fig_base_profile(prof, figdir / "rb_base_profile.png")
    if dec or settle:
        fig_hr(dec, settle, figdir / "hr_decoupling_settling.png")

    summary = {"node": "IN31-N12", "figures": [str(p) for p in sorted(figdir.glob("*.png"))]}
    if runs:
        summary["rb"] = {
            "grids": [
                {
                    "ny": r["params"]["ny"],
                    "Ra_c_lattice": r["Ra_c_lattice"],
                    "Ra_c_rel_shift_pct": 100 * r["Ra_c_rel_shift"],
                    "Ra_c_after_Heff_remap": r["Ra_c_lattice"]
                    * ((r["params"]["ny"] - 1.0) / r["params"]["ny"]) ** 3,
                    "arm_max_abs_rel_err_pct": max(
                        abs(100 * a["rel_err_lattice_vs_cont"]) for a in r["arms"]
                    ),
                    "sim_vs_eig_max_rel_err_pct": max(
                        [abs(100 * c["rel_err_sim_vs_eig"]) for c in r.get("sim_checks", [])] or [float("nan")]
                    ),
                    "base_residual": r.get("base_residual"),
                    "elapsed_s": r.get("elapsed_s"),
                }
                for r in runs
            ]
        }
    if dec:
        summary["decoupling"] = {
            "max_abs_lam_rel_err_pct": 100 * max(abs(r["lam_rel_err"]) for r in dec["decoupling"]),
            "density_slope": dec.get("decoupling_density_slope"),
        }
    if settle and settle.get("settling_summary"):
        summary["settling"] = settle["settling_summary"]
    Path(args.out).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())