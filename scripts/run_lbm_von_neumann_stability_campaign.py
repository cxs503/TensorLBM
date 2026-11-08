#!/usr/bin/env python3
"""Run the P155-N08 von Neumann stability / equivalent-equation campaign.

Produces (under ``--out``, default ``artifacts/lbm_von_neumann_p155_n08``):

* ``results.json``            full campaign record (every point, every number)
* ``summary.json``            criterion verdicts and headline numbers
* ``fig1_dispersion.png``     shear/acoustic dispersion-dissipation vs measurement
* ``fig2_stability_map.png``  (tau, Ma) stability phase diagram + divergence probes
* ``fig3_c1_parity.png``      measured vs predicted modal growth (C1)
* ``fig4_viscosity.png``      nu_num(theta), quadratic truncation, 1% band (C3/C4)
* ``fig5_bgk_vs_mrt.png``     BGK/MRT boundary difference (C5)

Usage
-----
    python scripts/run_lbm_von_neumann_stability_campaign.py [--out DIR] [--quick]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorlbm.stability import von_neumann as vn
from tensorlbm.stability.campaign import (
    PREDECLARED,
    c1_modal_agreement,
    c1_unstable_branch_check,
    c2_stability_boundary,
    c3_equivalent_equation,
    c4_viscosity_ratio_table,
    c5_collision_comparison,
    ma_to_u,
)


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, complex):
        return [obj.real, obj.imag]
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def fig_dispersion(out: Path) -> dict:
    """Shear attenuation and acoustic phase speed: theory vs measurement."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    thetas = np.linspace(0.01, 1.6, 64)
    colls = [("BGK", 0.55), ("BGK", 0.70), ("BGK", 1.00)]
    colors = ["#1f77b4", "#2ca02c", "#9467bd"]
    for (name, tau), col in zip(colls, colors):
        coll = vn.Collision("bgk", tau)
        row = vn.numerical_viscosity(1.0, coll)  # cheap sanity call
        assert row["nu_ce"] == coll.viscosity
        curve = vn.shear_branch_curve(thetas, coll, (0.0, 0.0))
        sig = np.array([c["sigma"] for c in curve])
        axes[0].plot(thetas, -sig / thetas**2, color=col, label=f"BGK tau={tau}")
        sp = [vn.spectrum(t, 0.0, coll, (0.0, 0.0)) for t in thetas]
        c_plus = np.array([s.phase_speed[s.index_of("sound+")] for s in sp])
        axes[1].plot(thetas, c_plus / vn.CS, color=col, label=f"BGK tau={tau}")
    axes[0].axhline(1.0 / 6.0, color="k", ls=":", lw=1)
    axes[1].axhline(1.0, color="k", ls=":", lw=1)
    axes[0].set_xlabel(r"$\theta$ [lattice]")
    axes[0].set_ylabel(r"$\nu_{num}(\theta)=-\sigma_{shear}/\theta^2$")
    axes[0].set_title("Numerical viscosity of the shear branch")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel(r"$\theta$ [lattice]")
    axes[1].set_ylabel(r"$c_{phase}/c_s$")
    axes[1].set_title("Acoustic branch phase speed")
    axes[1].legend(fontsize=8)

    # measurement overlay (shear branch, tau=0.7 at several Ma)
    k = 2.0 * math.pi * 2.0 / 32.0
    marks = []
    for ma in (0.0, 0.2, 0.4):
        run = vn.run_modal_box(
            k, 0.0, vn.Collision("bgk", 0.7), ma_to_u(ma), n=32, n_steps=3000, seed="shear"
        )
        marks.append(
            {
                "ma": ma,
                "theta": k,
                "growth_pred": run.predicted_growth,
                "growth_meas": run.measured_growth,
                "branch": run.notes["target_branch"],
            }
        )
    axes[2].plot(
        [m["growth_pred"] for m in marks],
        [m["growth_meas"] for m in marks],
        "o",
        ms=7,
        label=r"BGK $\tau=0.7$, shear seed",
    )
    lim = [min(m["growth_pred"] for m in marks) * 1.05, 1e-3]
    axes[2].plot(lim, lim, "k--", lw=1, label="1:1")
    axes[2].set_xlabel(r"von Neumann $\sigma=\ln|\mu|$ [1/step]")
    axes[2].set_ylabel(r"measured $\sigma$ [1/step]")
    axes[2].set_title("Dispersion-dissipation: prediction vs box experiment")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    path = out / "fig1_dispersion.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"figure": path.name, "marks": marks}


def fig_stability_map(out: Path, c2: dict, c5: dict) -> dict:
    """(tau, Ma) phase diagram: lambda_max, boundary, probe verdicts."""
    taus = np.linspace(0.44, 1.10, 40)
    mas = np.linspace(0.0, 0.8, 33)
    lam = np.zeros((mas.size, taus.size))
    for j, tau in enumerate(taus):
        coll = vn.Collision("bgk", float(tau))
        for i, ma in enumerate(mas):
            lam[i, j] = vn.spectral_radius(coll, ma_to_u(float(ma)), n_theta=13)["lambda_max"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    cs = axes[0].contourf(taus, mas, lam, levels=np.linspace(0.98, 1.25, 28), cmap="inferno")
    c1 = axes[0].contour(taus, mas, lam, levels=[1.0], colors="cyan", linewidths=2)
    axes[0].clabel(c1, fmt={1.0: r"$\lambda_{max}=1$"})
    fig.colorbar(cs, ax=axes[0], label=r"$\lambda_{max}$")
    axes[0].set_xlabel(r"BGK $\tau$")
    axes[0].set_ylabel("Ma")
    axes[0].set_title("von Neumann amplification (fine $\\theta$ grid)")
    for p in c2["points"]:
        if p["collision"] != "bgk":
            continue
        marker = "x" if p["diverged"] else "o"
        color = "red" if p["diverged"] else "lime"
        axes[0].plot(p["tau"], p["ma"], marker, color=color, ms=7, mew=1.4)
    axes[0].plot([], [], "o", color="lime", label="probe: no divergence")
    axes[0].plot([], [], "x", color="red", label="probe: diverged")
    axes[0].legend(loc="upper right", fontsize=8)

    # BGK vs MRT delta map from the C5 grid
    grid_tau = sorted({r["tau"] for r in c5["rows"]})
    grid_ma = sorted({r["ma"] for r in c5["rows"]})
    delta = np.zeros((len(grid_ma), len(grid_tau)))
    for r in c5["rows"]:
        delta[grid_ma.index(r["ma"]), grid_tau.index(r["tau"])] = r["delta"]
    vmax = max(abs(delta).max(), 1e-6)
    cs2 = axes[1].pcolormesh(
        grid_tau, grid_ma, delta, cmap="coolwarm", vmin=-vmax, vmax=vmax, shading="nearest"
    )
    fig.colorbar(cs2, ax=axes[1], label=r"$\lambda_{max}^{BGK}-\lambda_{max}^{MRT}$")
    axes[1].set_xlabel(r"$\tau$")
    axes[1].set_ylabel("Ma")
    axes[1].set_title("MRT minus BGK stability margin (>0: MRT better)")
    fig.tight_layout()
    path = out / "fig2_stability_map.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"figure": path.name}


def fig_c1_parity(out: Path, c1: dict) -> dict:
    pts = [p for p in c1["points"] if p["valid"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for proto, marker, color in (
        ("A-physical", "o", "#1f77b4"),
        ("B-branch-selective", "s", "#d62728"),
    ):
        sel = [p for p in pts if p["protocol"] == proto]
        if not sel:
            continue
        x = [p["growth_pred"] for p in sel]
        y = [p["growth_meas"] for p in sel]
        axes[0].plot(x, y, marker, color=color, ms=6, label=f"{proto} (n={len(sel)})")
    xs = np.array([p["growth_pred"] for p in pts])
    ys = np.array([p["growth_meas"] for p in pts])
    lo = min(xs.min(), ys.min()) * 1.02
    hi = max(xs.max(), ys.max()) * 1.02
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
    axes[0].set_xlabel(r"von Neumann prediction $\ln|\mu|$ [1/step]")
    axes[0].set_ylabel(r"measured growth rate [1/step]")
    axes[0].set_title("C1 parity: predicted vs measured modal growth")
    axes[0].legend(fontsize=8)

    errs = np.array([p["growth_rel_err"] for p in pts])
    axes[1].hist(np.log10(np.maximum(errs, 1e-12)), bins=24, color="#1f77b4")
    axes[1].axvline(
        math.log10(PREDECLARED["c1"]["growth_rel_err_max"]),
        color="r",
        ls="--",
        label="5% criterion",
    )
    axes[1].set_xlabel(r"$\log_{10}$(relative growth-rate error)")
    axes[1].set_ylabel("count")
    axes[1].set_title(f"C1 error distribution (median {np.median(errs):.2e})")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = out / "fig3_c1_parity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"figure": path.name}


def fig_viscosity(out: Path, c3: dict, c4: dict) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for c in c3["curves"]:
        th = np.array(c["theta"])
        nu = np.array(c["nu_num"])
        axes[0].plot(th, nu, "-", label=f"{c['collision']} Ma={c['ma']}")
        axes[0].axhline(c["nu_phys"], ls=":", lw=0.8, color="k")
        fit = c["fit"]
        axes[0].plot(th, fit["a"] + fit["b"] * th**2, "--", lw=1)
    axes[0].set_xlabel(r"$\theta$ [lattice]")
    axes[0].set_ylabel(r"$\nu_{num}(\theta)$")
    axes[0].set_title(r"C3: $\nu_{num}$ and the $a+b\theta^2$ truncation (dashed)")
    axes[0].legend(fontsize=7)

    for ma in sorted({r["ma"] for r in c4["rows"]}):
        sel = [r for r in c4["rows"] if abs(r["ma"] - ma) < 1e-12]
        th = np.array(sel[0]["theta"])
        ratio = np.median(np.array([r["ratio"] for r in sel]), axis=0)
        axes[1].loglog(th, np.maximum(ratio, 1e-12), label=f"Ma={ma}")
    axes[1].axhline(PREDECLARED["c4"]["ratio_threshold"], color="r", ls="--", lw=1, label="1% gate")
    axes[1].set_xlabel(r"$\theta$ [lattice]")
    axes[1].set_ylabel(r"$|\nu_{num}-\nu_{phys}|/\nu_{phys}$")
    axes[1].set_title("C4: numerical-viscosity contamination (median over $\\tau$)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = out / "fig4_viscosity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"figure": path.name}


def fig_bgk_mrt(out: Path, c5: dict) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    grid_tau = sorted({r["tau"] for r in c5["rows"]})
    grid_ma = sorted({r["ma"] for r in c5["rows"]})
    lb = np.zeros((len(grid_ma), len(grid_tau)))
    lm = np.zeros_like(lb)
    for r in c5["rows"]:
        i, j = grid_ma.index(r["ma"]), grid_tau.index(r["tau"])
        lb[i, j] = r["lambda_bgk"]
        lm[i, j] = r["lambda_mrt"]
    for arr, lab, ls in ((lb, "BGK", "-"), (lm, "MRT", "--")):
        axes[0].plot(
            grid_tau, arr[grid_ma.index(min(grid_ma))], ls, label=f"{lab} Ma={min(grid_ma)}"
        )
    md = grid_ma[len(grid_ma) // 2]
    axes[0].plot(grid_tau, lb[grid_ma.index(md)], "-o", ms=4, label=f"BGK Ma={md}")
    axes[0].plot(grid_tau, lm[grid_ma.index(md)], "--s", ms=4, label=f"MRT Ma={md}")
    axes[0].axhline(1.0, color="k", ls=":", lw=1)
    axes[0].set_xlabel(r"$\tau$")
    axes[0].set_ylabel(r"$\lambda_{max}$")
    axes[0].set_title("Stability margin along $\\tau$")
    axes[0].legend(fontsize=8)

    mc = c5["ma_crit"]
    x = np.arange(len(mc))
    axes[1].bar(x - 0.2, [e["ma_crit_bgk"] for e in mc], width=0.4, label="BGK")
    axes[1].bar(x + 0.2, [e["ma_crit_mrt"] for e in mc], width=0.4, label="MRT")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{e['tau']:.3f}" for e in mc], rotation=45, fontsize=8)
    axes[1].set_xlabel(r"$\tau$")
    axes[1].set_ylabel(r"$Ma_{crit}$")
    axes[1].set_title("C5: critical Mach number (von Neumann boundary)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = out / "fig5_bgk_vs_mrt.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"figure": path.name}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/lbm_von_neumann_p155_n08")
    ap.add_argument("--quick", action="store_true", help="small grids (smoke run)")
    ap.add_argument(
        "--supplement-only",
        action="store_true",
        help="run only the supplementary unstable-branch check (fast)",
    )
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.supplement_only:
        sup = {"node": "P155-N08", "criterion_block": {"C1-supplement": c1_unstable_branch_check()}}
        (out / "results_supplement.json").write_text(
            json.dumps(sup, indent=2, default=_json_default)
        )
        print(
            json.dumps(sup["criterion_block"]["C1-supplement"], indent=2, default=_json_default)[
                :1200
            ]
        )
        return 0

    t0 = time.time()
    if args.quick:
        c1 = c1_modal_agreement(
            taus=(0.60, 0.80),
            mas=(0.0, 0.2),
            modes=(1, 2),
            acoustic_taus=(0.60,),
            acoustic_mas=(0.0,),
            acoustic_modes=(2,),
            n_steps=1500,
        )
        c2 = c2_stability_boundary(taus=(0.45, 0.52, 0.9), mas=(0.0, 0.4))
        c5 = c5_collision_comparison(
            taus=(0.505, 0.6, 0.9), mas=(0.0, 0.3, 0.5), n_theta=17, search_rates=False
        )
    else:
        c1 = c1_modal_agreement()
        c2 = c2_stability_boundary()
        c5 = c5_collision_comparison()
    c3 = c3_equivalent_equation()
    c4 = c4_viscosity_ratio_table()
    wall = time.time() - t0

    figs = []
    figs.append(fig_dispersion(out))
    figs.append(fig_stability_map(out, c2, c5))
    figs.append(fig_c1_parity(out, c1))
    figs.append(fig_viscosity(out, c3, c4))
    figs.append(fig_bgk_mrt(out, c5))

    results = {
        "node": "P155-N08",
        "domain": "numerical/LBM_stability",
        "repo": "TensorLBM",
        "predeclared": PREDECLARED,
        "wall_seconds": wall,
        "criteria": {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5},
        "figures": [f["figure"] for f in figs],
        "all_pass": bool(all([c1["pass"], c2["pass"], c3["pass"], c4["pass"], c5["pass"]])),
    }
    (out / "results.json").write_text(json.dumps(results, indent=2, default=_json_default))

    summary = {
        "node": "P155-N08",
        "wall_seconds": wall,
        "C1": {
            "pass": c1["pass"],
            "n_points_counted": c1["n_points_counted"],
            "n_points_passed": c1["n_points_passed"],
            "min_points_required": PREDECLARED["c1"]["min_points"],
            "max_rel_err": c1["max_rel_err"],
            "median_rel_err": c1["median_rel_err"],
            "max_phase_speed_rel_err": c1["max_phase_speed_rel_err"],
            "groups": c1["groups"],
        },
        "C2": {
            "pass": c2["pass"],
            "tpr": c2["tpr"],
            "fpr": c2["fpr"],
            "n_unstable_guarded": c2["unstable_guarded"]["n"],
            "n_stable": c2["stable"]["n"],
            "weakly_unstable_below_guard": c2["weakly_unstable_below_guard"],
        },
        "C3": {
            "pass": c3["pass"],
            "max_rel_dev": c3["max_rel_dev"],
            "max_intercept_rel_err": c3["max_intercept_rel_err"],
            "curves": [
                {
                    "collision": c["collision"],
                    "ma": c["ma"],
                    "a": c["fit"]["a"],
                    "b": c["fit"]["b"],
                    "nu_phys": c["nu_phys"],
                    "max_rel_dev": c["fit"]["max_rel_dev"],
                    "tracking_overlap_min": c["tracking_overlap_min"],
                }
                for c in c3["curves"]
            ],
        },
        "C4": {
            "pass": c4["pass"],
            "bands_by_ma": c4["bands_by_ma"],
            "intercept_shift_by_ma": c4["intercept_shift_by_ma"],
            "intercept_shift_vs_ma2": c4["intercept_shift_vs_ma2"],
            "ma_threshold_from_shift": c4["ma_threshold_from_shift"],
        },
        "C5": {
            "pass": c5["pass"],
            "dominance": c5["dominance"],
            "n_mrt_only_stable": c5["n_mrt_only_stable"],
            "n_bgk_only_stable": c5["n_bgk_only_stable"],
            "ma_crit": c5["ma_crit"],
            "rate_search_best": (c5.get("rate_search") or {}).get("best"),
            "rate_search_any_global": (c5.get("rate_search") or {}).get("any_global_dominance"),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default))
    print(json.dumps(summary, indent=2, default=_json_default))
    print(f"\nwall: {wall:.1f} s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
