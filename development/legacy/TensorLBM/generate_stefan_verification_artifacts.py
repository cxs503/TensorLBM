#!/usr/bin/env python3
"""Generate Stefan phase-change verification artifacts for node p262w1-N01.

Produces:
  - interface_position.png   : front position s(t) vs Neumann analytical
  - temperature_profile.png  : T(x) at final time vs analytical
  - convergence_study.json   : grid convergence data with observed order

All criteria F1-F6 are checked and reported.
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm.phase_change.stefan_verification import (
    EnthalpyThermalLBM,
    NeumannSimilarity,
    run_planar_stefan,
    l2_profile_error,
)

OUTDIR = Path(__file__).resolve().parents[1] / "artifacts" / "stefan_stefan_verification"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Physical parameters (lattice units, synthetic domain)
STEFAN_NUMBER = 0.5
ALPHA = 1.0 / 6.0  # = cs2*(tau-0.5) with tau=1, D2Q5
TAU = 1.0
CP = 1.0
LATENT = 1.0
T_FREEZE = 0.0


def generate_interface_position_plot(res):
    """F1: Interface position s(t) vs analytical — error ≤3% at 5 time points."""
    exact = NeumannSimilarity(
        stefan_number=STEFAN_NUMBER, alpha=ALPHA, side="solid",
        cp=CP, latent=LATENT, t_freeze=T_FREEZE,
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    series = res["series"]
    t_vals = [e["time"] for e in series]
    s_lbm = [e["front_integral"] for e in series]
    s_local = [e["front_local"] for e in series]

    t_fine = np.linspace(0.1, max(t_vals), 200)
    s_ana = [exact.front(t) for t in t_fine]

    ax1.plot(t_fine, s_ana, "k-", lw=2, label="Neumann $s=2\\lambda\\sqrt{\\alpha t}$")
    ax1.plot(t_vals, s_lbm, "b-s", ms=6, label="LBM (phase integral)")
    ax1.plot(t_vals, s_local, "r-^", ms=5, label="LBM ($f_l=0.5$)")
    ax1.set_xlabel("Time $t$ (lattice units)", fontsize=12)
    ax1.set_ylabel("Interface position $s(t)$", fontsize=12)
    ax1.set_title("Stefan Solidification: Front Position vs Time", fontsize=13)
    ax1.legend(fontsize=10); ax1.grid(alpha=0.3)

    errors = [e["front_rel_error"] * 100 for e in series]
    colors = ["green" if e <= 3.0 else "red" for e in errors]
    ax2.bar(range(len(series)), errors, color=colors)
    ax2.axhline(3.0, color="r", ls="--", lw=1.5, label="3% threshold (F1)")
    ax2.set_xlabel("Sample index (increasing time)", fontsize=12)
    ax2.set_ylabel("Relative error [%]", fontsize=12)
    ax2.set_title("Interface Position Error at 5 Time Points", fontsize=13)
    ax2.set_xticks(range(len(series)))
    ax2.set_xticklabels([f"t={e['time']:.0f}" for e in series], rotation=30, fontsize=9)
    ax2.legend(fontsize=10); ax2.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"Stefan Problem: Ste={STEFAN_NUMBER}, α={ALPHA}, λ={exact.lam:.4f}, nx={res['grid']['nx']}",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    path = OUTDIR / "interface_position.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [OK] {path}")
    return path


def generate_temperature_profile_plot(res):
    """F2: Temperature T(x,t) vs analytical — L2 error ≤2%."""
    exact = NeumannSimilarity(
        stefan_number=STEFAN_NUMBER, alpha=ALPHA, side="solid",
        cp=CP, latent=LATENT, t_freeze=T_FREEZE,
    )
    profile = res["profile"]
    x = np.array(profile["x"])
    T_num = np.array(profile["T_num"])
    T_exact = np.array(profile["T_exact"])
    f_l = np.array(profile["f_l"])
    s_exact = profile["front_exact"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5))

    ax1.plot(x, T_num, "b-", lw=2, label="LBM $T(x,t)$")
    ax1.plot(x, T_exact, "r--", lw=1.5, label="Neumann analytical")
    ax1.axvline(s_exact, color="k", ls=":", alpha=0.5, label=f"Front $s={s_exact:.1f}$")
    ax1.set_xlabel("$x$ (lattice cells)", fontsize=12)
    ax1.set_ylabel("$T$", fontsize=12)
    ax1.set_title("Temperature Profile", fontsize=13)
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    ax2.plot(x, f_l, "g-", lw=2, label="Liquid fraction $f_l$")
    ax2.axhline(0.5, color="gray", ls="--", alpha=0.5, label="$f_l=0.5$")
    ax2.axvline(s_exact, color="k", ls=":", alpha=0.5)
    ax2.set_xlabel("$x$ (lattice cells)", fontsize=12)
    ax2.set_ylabel("$f_l$", fontsize=12)
    ax2.set_title("Phase Field", fontsize=13)
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    error = np.abs(T_num - T_exact) / exact.dT * 100
    ax3.semilogy(x, error, "m-", lw=1.5)
    ax3.axvline(s_exact, color="k", ls=":", alpha=0.5)
    ax3.axhline(2.0, color="r", ls="--", lw=1.5, label="2% (F2)")
    band = np.abs(x - s_exact) <= 4.0
    ax3.fill_between(x, 1e-3, 100, where=band, alpha=0.1, color="gray", label="excluded")
    ax3.set_xlabel("$x$ (lattice cells)", fontsize=12)
    ax3.set_ylabel("Relative error [%]", fontsize=12)
    ax3.set_title("Pointwise Temperature Error", fontsize=13)
    ax3.set_ylim(1e-3, 100); ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    perr = res["final"]["profile_error"]
    fig.suptitle(
        f"L2 error = {perr['l2_rel']*100:.3f}%, Linf = {perr['linf_rel']*100:.3f}% (interface band excluded)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    path = OUTDIR / "temperature_profile.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [OK] {path}")
    return path


def run_convergence_study():
    """F4: Grid convergence — 3 levels."""
    resolutions = (25, 50, 100)
    runs = []
    for n_front in resolutions:
        nx = 4 * n_front
        res = run_planar_stefan(
            mode="freezing", nx=nx, stefan_number=STEFAN_NUMBER, tau=TAU,
            front_fraction=0.25, sample_fractions=(1.0,), energy_audit=False,
        )
        runs.append({
            "n_front": n_front, "nx": nx, "steps": res["parameters"]["steps"],
            "front_rel_error": res["final"]["front_rel_error"],
            "front_local_rel_error": res["final"]["front_local_rel_error"],
            "profile_l2_rel": res["final"]["profile_error"]["l2_rel"],
            "profile_linf_rel": res["final"]["profile_error"]["linf_rel"],
        })
        print(f"  nx={nx}: front_err={res['final']['front_rel_error']*100:.2f}%, "
              f"profile_l2={res['final']['profile_error']['l2_rel']*100:.3f}%")

    def fit_order(xs, ys):
        x = np.log(1.0 / np.asarray(xs, dtype=float))
        y = np.log(np.asarray(ys, dtype=float))
        slope, _ = np.polyfit(x, y, 1)
        return float(slope)

    hs = [r["n_front"] for r in runs]
    orders = {}
    for key in ("front_rel_error", "front_local_rel_error", "profile_l2_rel"):
        vals = [r[key] for r in runs]
        orders[key] = fit_order(hs, vals) if all(v > 0 for v in vals) else float("nan")

    study = {
        "node": "p262w1-N01", "mission_id": "m9873",
        "module": "tensorlbm.phase_change.stefan_verification",
        "description": "1-D Stefan solidification grid convergence study",
        "parameters": {
            "stefan_number": STEFAN_NUMBER, "alpha": ALPHA, "tau": TAU,
            "cp": CP, "latent": LATENT, "t_freeze": T_FREEZE,
            "front_fraction": 0.25, "lattice": "D2Q5",
        },
        "grid_levels": [
            {
                "n_front_cells": r["n_front"], "nx_total": r["nx"],
                "dx_effective": f"L/{r['n_front']}", "steps": r["steps"],
                "front_rel_error_pct": round(r["front_rel_error"] * 100, 4),
                "front_local_rel_error_pct": round(r["front_local_rel_error"] * 100, 4),
                "profile_l2_rel_pct": round(r["profile_l2_rel"] * 100, 4),
                "profile_linf_rel_pct": round(r["profile_linf_rel"] * 100, 4),
            }
            for r in runs
        ],
        "observed_orders": {k: round(v, 4) for k, v in orders.items()},
        "criteria_check": {
            "F4_convergence_order_front": {
                "value": round(orders["front_rel_error"], 4),
                "threshold": 1.0,
                "pass": orders["front_rel_error"] >= 1.0,
            },
        },
    }
    path = OUTDIR / "convergence_study.json"
    path.write_text(json.dumps(study, indent=2, default=float))
    print(f"  [OK] {path}")
    return study, orders


def main():
    print("=" * 70)
    print("  STEFAN PHASE-CHANGE VERIFICATION — ARTIFACT GENERATION")
    print("  Node: p262w1-N01 | Mission: m9873")
    print("=" * 70)

    print("\n[1/4] Planar freezing (nx=400, 5 time samples)...")
    res_freeze = run_planar_stefan(
        mode="freezing", nx=400, stefan_number=STEFAN_NUMBER, tau=TAU,
        front_fraction=0.25, sample_fractions=(0.2, 0.4, 0.6, 0.8, 1.0),
        energy_audit=True,
    )
    print(f"  Final: s_exact={res_freeze['final']['front_exact']:.2f}, "
          f"s_lbm={res_freeze['final']['front_integral']:.2f}, "
          f"err={res_freeze['final']['front_rel_error']*100:.2f}%")

    print("\n[2/4] Planar melting (nx=400, energy audit)...")
    res_melt = run_planar_stefan(
        mode="melting", nx=400, stefan_number=STEFAN_NUMBER, tau=TAU,
        front_fraction=0.25, sample_fractions=(1.0,), energy_audit=True,
    )
    print(f"  Energy closure (2nd order): {res_melt['energy']['closure_2nd_order']*100:.2f}%")

    print("\n[3/4] Generating figures...")
    generate_interface_position_plot(res_freeze)
    generate_temperature_profile_plot(res_freeze)

    print("\n[4/4] Grid convergence (3 levels: nx=100,200,400)...")
    conv, orders = run_convergence_study()

    # Summary
    print("\n" + "=" * 70)
    print("  CRITERIA CHECK")
    print("=" * 70)
    f1_max = max(e["front_rel_error"] for e in res_freeze["series"]) * 100
    f2_val = res_freeze["final"]["profile_error"]["l2_rel"] * 100
    f3_val = res_melt["energy"]["closure_2nd_order"] * 100
    f4_val = orders["front_rel_error"]

    checks = [
        ("F1: front error ≤3% (5 time points)", f1_max, 3.0, f1_max <= 3.0),
        ("F2: T profile L2 ≤2%", f2_val, 2.0, f2_val <= 2.0),
        ("F3: energy closure ≤3%", f3_val, 3.0, f3_val <= 3.0),
        ("F4: convergence order ≥1", f4_val, 1.0, f4_val >= 1.0),
    ]
    all_pass = True
    for name, val, thr, ok in checks:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}  {name}: {val:.4f} (threshold: {thr})")
        if not ok:
            all_pass = False

    print(f"\n  Overall: {'ALL PASS ✓' if all_pass else 'SOME FAILED ✗'}")

    # Save criteria summary
    criteria = {
        "F1_front_error_max_pct": round(f1_max, 4),
        "F2_temperature_L2_pct": round(f2_val, 4),
        "F3_energy_closure_pct": round(f3_val, 4),
        "F4_convergence_order": round(f4_val, 4),
        "all_pass": all_pass,
    }
    (OUTDIR / "criteria.json").write_text(json.dumps(criteria, indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
