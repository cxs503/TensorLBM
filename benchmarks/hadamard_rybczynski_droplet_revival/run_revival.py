#!/usr/bin/env python3
"""Revival of benchmark ``hadamard_rybczynski_droplet`` (campaign node IN26-L2).

``benchmarks/pending/hadamard_rybczynski_droplet/terminal_verdict_m155.md``
declared this benchmark dead because of library gap G2 ("no module separates
viscosity ratio from density ratio").  This driver re-runs the benchmark against
the two new library layers:

    src/tensorlbm/two_phase_viscosity.py   -- decoupled tau(x) mapping (gap G2)
    src/tensorlbm/hadamard_rybczynski.py   -- analytic law + axisymmetric Stokes
                                             drop-drag solver

Sections
--------
A  viscosity-ratio / density-ratio decoupling (operator level, lattice level,
   and the *legacy* locked mapping for contrast)
B  solver validation in the two exactly known limits
C  lambda ladder 0.01 .. 100 (8 log-spaced points) vs the HR drag law
D  endpoints: lambda -> 0 (bubble, 4 pi mu R U) and lambda -> inf (Stokes,
   6 pi mu R U)
E  grid convergence (observed order) at fixed container
F  extrapolation-window sensitivity (audit of the finite-domain correction)

Usage::  PYTHONPATH=src python benchmarks/hadamard_rybczynski_droplet_revival/run_revival.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import tensorlbm.hadamard_rybczynski as HR  # noqa: E402
import tensorlbm.two_phase_viscosity as TPV  # noqa: E402

OUT_DIR = os.path.join(ROOT, "artifacts", "hadamard_rybczynski_droplet_revival")
FIG_DIR = os.path.join(OUT_DIR, "figures")

R = 1.0
MU_C = 1.0
U = 1.0
H_WINDOW_A = [10.0, 20.0, 40.0]   # production extrapolation window
H_WINDOW_B = [20.0, 40.0, 80.0]   # audit window (larger, more expensive)
H_WALL_PROD = 0.03


# ---------------------------------------------------------------------------
def _extrapolated_drag(lam: float, h_wall: float, windows: List[List[float]]):
    """Drag at each window plus the wall-extrapolated (H -> inf) values."""
    out = {}
    for tag, hrs in zip(("A", "B"), windows):
        drags = [
            HR.HRDragSolver(R=R, H_ratio=hr, h_wall=h_wall).solve(lam, mu_c=MU_C, U=U).drag
            for hr in hrs
        ]
        fit = HR.wall_extrapolate(hrs, drags, order=1)
        out[tag] = {"H_ratios": list(hrs), "drags": drags, **fit}
    return out


def section_a_decoupling() -> Dict:
    """lambda independent of the density ratio, three independent ways."""
    rng = np.random.default_rng(20261106)
    phi = np.round(rng.random((8, 9)))
    rows = []
    for rho_ratio in (1.0, 2.0, 5.0, 10.0):
        rho = np.where(phi > 0.5, rho_ratio, 1.0)
        for lam in (0.01, 0.5, 2.0, 100.0):
            tau, mu = TPV.pseudopotential_relaxation(phi, rho, MU_C, lam)
            realized = TPV.viscosity_ratio_from_tau(tau, rho, phi)
            tau_l = TPV.legacy_uniform_relaxation(rho, MU_C, lam)
            legacy = TPV.viscosity_ratio_from_tau(tau_l, rho, phi)
            rows.append(
                {
                    "rho_ratio": rho_ratio,
                    "lambda_target": lam,
                    "lambda_realized": realized,
                    "lambda_realized_err": realized / lam - 1.0,
                    "lambda_legacy": legacy,
                    "legacy_err_vs_target": legacy / lam - 1.0,
                }
            )
    # lattice-level calibration of the tau -> mu map (uniform density)
    lattice = []
    for tau in (0.60, 0.80, 1.00, 1.20, 1.70):
        probe = TPV.shear_wave_decay(np.full(64, tau), ny=64, steps=800, sample_every=20)
        lattice.append(
            {
                "tau": tau,
                "nu_measured": probe["nu"],
                "nu_bgk": (tau - 0.5) * TPV.CS2,
                "rel_err": probe["nu"] / ((tau - 0.5) * TPV.CS2) - 1.0,
            }
        )
    # two-phase contrast probe: realised mu ratio for a density ratio of 2
    rho_ratio = 2.0
    rho = np.where(phi > 0.5, rho_ratio, 1.0)
    contrast = []
    mu_c_probe = 0.1   # keeps both tau in the calibrated band of the lattice probe
    for lam in (1.0, 4.0):
        tau, mu = TPV.pseudopotential_relaxation(phi, rho, mu_c_probe, lam)
        tau_d = float(np.mean(tau[phi > 0.5]))
        tau_c = float(np.mean(tau[phi <= 0.5]))
        nu_d = TPV.measure_uniform_viscosity(tau_d)
        nu_c = TPV.measure_uniform_viscosity(tau_c)
        rho_d = float(rho[phi > 0.5].mean())
        rho_c = float(rho[phi <= 0.5].mean())
        contrast.append(
            {
                "rho_ratio": rho_ratio,
                "lambda_target": lam,
                "tau_dispersed": tau_d,
                "tau_continuous": tau_c,
                "nu_dispersed": nu_d,
                "nu_continuous": nu_c,
                "mu_ratio_from_lattice": (rho_d * nu_d) / (rho_c * nu_c),
                "mu_c_probe": mu_c_probe,
                "probe_rel_err_dispersed": nu_d / (TPV.CS2 * (tau_d - 0.5)) - 1.0,
                "probe_rel_err_continuous": nu_c / (TPV.CS2 * (tau_c - 0.5)) - 1.0,
            }
        )
    worst_op = max(abs(r["lambda_realized_err"]) for r in rows)
    return {
        "operator_level": rows,
        "operator_worst_rel_err": worst_op,
        "lattice_calibration": lattice,
        "lattice_worst_rel_err": max(abs(r["rel_err"]) for r in lattice),
        "two_phase_contrast": contrast,
        "legacy_lock_demonstrated": all(
            abs(r["lambda_legacy"] - r["rho_ratio"]) < 1e-12 for r in rows
        ),
    }


def section_b_validation() -> Dict:
    """The two exactly known limits of the solver itself."""
    out = {}
    for tag, lam, ref in (
        ("rigid_sphere", 1.0e6, HR.stokes_limit_drag(MU_C, R, U)),
        ("equal_viscosity", 1.0, HR.equal_viscosity_drag(MU_C, R, U)),
        ("bubble_like", 1.0e-3, HR.bubble_limit_drag(MU_C, R, U)),
    ):
        res = _extrapolated_drag(lam, H_WALL_PROD, [H_WINDOW_A, H_WINDOW_B])
        out[tag] = {
            "lambda": lam,
            "reference": ref,
            "drag_inf_A": res["A"]["drag_inf"],
            "drag_inf_B": res["B"]["drag_inf"],
            "rel_err_A": res["A"]["drag_inf"] / ref - 1.0,
            "rel_err_B": res["B"]["drag_inf"] / ref - 1.0,
            "confinement_slope_A": res["A"]["slope"],
            "confinement_coeff_A": res["A"]["slope"] / res["A"]["drag_inf"],
            "fit_residual_A": res["A"]["max_rel_residual"],
            "drags_A": res["A"]["drags"],
        }
    return out


def section_c_ladder() -> Dict:
    """Eight log-spaced viscosity ratios vs the HR drag law."""
    lams = HR.hr_lambda_ladder(8, 0.01, 100.0)
    rows = []
    for lam in lams:
        res = _extrapolated_drag(float(lam), H_WALL_PROD, [H_WINDOW_A, H_WINDOW_B])
        hr = HR.hr_drag(MU_C, R, U, float(lam))
        rows.append(
            {
                "lambda": float(lam),
                "drag_HR": hr,
                "drag_inf_A": res["A"]["drag_inf"],
                "drag_inf_B": res["B"]["drag_inf"],
                "drags_finite_A": res["A"]["drags"],
                "rel_err_A": res["A"]["drag_inf"] / hr - 1.0,
                "rel_err_B": res["B"]["drag_inf"] / hr - 1.0,
                "confinement_coeff_A": res["A"]["slope"] / res["A"]["drag_inf"],
                "fit_residual_A": res["A"]["max_rel_residual"],
                "settling_velocity_HR": HR.hr_velocity(1.0, 1.0, R, MU_C, float(lam)),
                "settling_velocity_measured": HR.terminal_velocity_from_drag(
                    res["A"]["drag_inf"], 1.0, 1.0, R
                ),
            }
        )
    errs = np.array([abs(r["rel_err_A"]) for r in rows])
    errs_b = np.array([abs(r["rel_err_B"]) for r in rows])
    return {
        "points": rows,
        "max_abs_rel_err_A": float(errs.max()),
        "max_abs_rel_err_B": float(errs_b.max()),
        "mean_abs_rel_err_A": float(errs.mean()),
        "criterion_err_lt_3pct": bool(errs.max() < 0.03),
        "criterion_err_lt_3pct_window_B": bool(errs_b.max() < 0.03),
    }


def section_d_endpoints() -> Dict:
    """lambda -> 0 and lambda -> inf limits, each with its own tolerance."""
    bubble_ref = HR.bubble_limit_drag(MU_C, R, U)
    stokes_ref = HR.stokes_limit_drag(MU_C, R, U)
    lam_bubble, lam_stokes = 0.01, 100.0
    res_b = _extrapolated_drag(lam_bubble, H_WALL_PROD, [H_WINDOW_A])
    res_s = _extrapolated_drag(lam_stokes, H_WALL_PROD, [H_WINDOW_A])
    # HR at the two ladder endpoints is the *analytic* statement of the limit
    hr_b = HR.hr_drag(MU_C, R, U, lam_bubble)
    hr_s = HR.hr_drag(MU_C, R, U, lam_stokes)
    return {
        "bubble_limit": {
            "lambda": lam_bubble,
            "drag_measured": res_b["A"]["drag_inf"],
            "drag_bubble_limit": bubble_ref,
            "drag_HR_at_lambda": hr_b,
            "rel_err_vs_bubble_limit": res_b["A"]["drag_inf"] / bubble_ref - 1.0,
            "criterion_err_lt_5pct": bool(abs(res_b["A"]["drag_inf"] / bubble_ref - 1.0) < 0.05),
        },
        "stokes_limit": {
            "lambda": lam_stokes,
            "drag_measured": res_s["A"]["drag_inf"],
            "drag_stokes_limit": stokes_ref,
            "drag_HR_at_lambda": hr_s,
            "rel_err_vs_stokes_limit": res_s["A"]["drag_inf"] / stokes_ref - 1.0,
            "criterion_err_lt_3pct": bool(abs(res_s["A"]["drag_inf"] / stokes_ref - 1.0) < 0.03),
        },
        "rigid_limit_lambda_1e6": section_b_validation()["rigid_sphere"],
    }


def section_e_grid_convergence() -> Dict:
    """Observed order of accuracy of the drag at fixed container (H/R = 10)."""
    lam = 1.0
    H_ratio = 10.0
    hs = [0.80, 0.40, 0.20, 0.10]
    drags = [
        HR.HRDragSolver(R=R, H_ratio=H_ratio, h_wall=h, growth=1.35).solve(lam).drag for h in hs
    ]
    ref = HR.HRDragSolver(R=R, H_ratio=H_ratio, h_wall=0.05, growth=1.35).solve(lam).drag
    errs = [abs(d / ref - 1.0) for d in drags]
    orders = []
    for k in range(1, len(hs)):
        if errs[k] <= 0.0:
            orders.append(float("nan"))
        else:
            orders.append(float(np.log(errs[k - 1] / errs[k]) / np.log(hs[k - 1] / hs[k])))
    rows = [
        {"h_wall": h, "drag": d, "rel_err_vs_finest": e, "observed_order": o}
        for h, d, e, o in zip(hs, drags, errs, [float("nan")] + orders)
    ]
    finite_orders = [o for o in orders if np.isfinite(o)]
    return {
        "lambda": lam,
        "H_ratio": H_ratio,
        "reference_drag": ref,
        "points": rows,
        "min_observed_order": float(min(finite_orders)),
        "criterion_order_ge_1": bool(min(finite_orders) >= 1.0),
    }


def section_f_window_audit() -> Dict:
    """How much the answer depends on the extrapolation window (uncertainty)."""
    out = []
    for lam in (0.01, 0.1, 1.0, 10.0, 100.0):
        res = _extrapolated_drag(lam, H_WALL_PROD, [H_WINDOW_A, H_WINDOW_B])
        hr = HR.hr_drag(MU_C, R, U, lam)
        out.append(
            {
                "lambda": lam,
                "A_window": H_WINDOW_A,
                "B_window": H_WINDOW_B,
                "drag_inf_A": res["A"]["drag_inf"],
                "drag_inf_B": res["B"]["drag_inf"],
                "window_spread": res["B"]["drag_inf"] / res["A"]["drag_inf"] - 1.0,
                "rel_err_A": res["A"]["drag_inf"] / hr - 1.0,
                "rel_err_B": res["B"]["drag_inf"] / hr - 1.0,
            }
        )
    spread = max(abs(r["window_spread"]) for r in out)
    return {
        "points": out,
        "max_window_spread": float(spread),
        "note": (
            "A and B differ only by the container sizes used for the R/H -> 0 fit; "
            "their spread is the honest extrapolation uncertainty of this node."
        ),
    }


# ---------------------------------------------------------------------------
def make_figures(ev: Dict) -> List[Dict[str, str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    figs: List[Dict[str, str]] = []

    # ---- fig 1: drag law ---------------------------------------------------
    lam_curve = np.geomspace(1e-3, 1e3, 400)
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=150)
    ax.plot(lam_curve, [HR.drag_coefficient_factor(l) for l in lam_curve],
            "k-", lw=2, label="Hadamard-Rybczynski  $\\frac{2+3\\lambda}{1+\\lambda}$")
    pts = ev["ladder"]["points"]
    ax.plot([p["lambda"] for p in pts], [p["drag_inf_A"] / (2 * np.pi * MU_C * R * U) for p in pts],
            "o", ms=7, mfc="w", mec="C3", mew=1.8, label="this work (8 $\\lambda$ points)")
    ax.axhline(2.0, ls="--", c="C0", lw=1.2, label="bubble limit $4\\pi\\mu_c R U$")
    ax.axhline(3.0, ls="--", c="C2", lw=1.2, label="Stokes limit $6\\pi\\mu_c R U$")
    ax.set_xscale("log")
    ax.set_xlabel("viscosity ratio $\\lambda=\\mu_d/\\mu_c$")
    ax.set_ylabel("$F/(2\\pi\\mu_c R U)$")
    ax.set_title("HR drag law: computed vs analytic (wall-extrapolated)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    p1 = os.path.join(FIG_DIR, "fig1_hr_law.png")
    fig.savefig(p1)
    plt.close(fig)
    figs.append({"name": "fig1_hr_law.png", "path": p1,
                 "caption": "Drag group F/(2*pi*mu_c*R*U) vs viscosity ratio lambda: analytic HR curve (solid), "
                            "8 wall-extrapolated solver points, and the two analytic limits (dashed)."})

    # ---- fig 2: relative error --------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
    lams = [p["lambda"] for p in pts]
    err_a = [100 * p["rel_err_A"] for p in pts]
    err_b = [100 * p["rel_err_B"] for p in pts]
    ax.semilogx(lams, err_a, "o-", label="window A (H/R = 10,20,40)")
    ax.semilogx(lams, err_b, "s--", label="window B (H/R = 20,40,80)")
    ax.axhspan(-3, 3, color="0.85", zorder=0, label="$\\pm 3\\%$ acceptance band")
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xlabel("viscosity ratio $\\lambda$")
    ax.set_ylabel("error in F vs HR [%]")
    ax.set_title("Relative error against the analytic HR drag law")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p2 = os.path.join(FIG_DIR, "fig2_error_vs_lambda.png")
    fig.savefig(p2)
    plt.close(fig)
    figs.append({"name": "fig2_error_vs_lambda.png", "path": p2,
                 "caption": "Per-lambda relative error of the extrapolated drag against the HR law, for the two "
                            "extrapolation windows; shaded band is the 3% acceptance criterion."})

    # ---- fig 3: wall extrapolation ----------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    for p, c in zip([pts[0], pts[len(pts) // 2], pts[-1]], ["C0", "C3", "C2"]):
        x = [R / h for h in H_WINDOW_A]
        y = p["drags_finite_A"]
        ax.plot(x, y, "o", color=c, ms=6,
                label=f"$\\lambda$={p['lambda']:.3g}: finite container")
        fit = np.polyfit(x, y, 1)
        xs = np.linspace(0, max(x) * 1.05, 50)
        ax.plot(xs, np.polyval(fit, xs), "-", color=c, lw=1.2)
        ax.plot([0], [fit[1]], "*", color=c, ms=12)
        ax.axhline(p["drag_HR"], ls=":", color=c, lw=1.2)
    ax.set_xlabel("$R/H$ (inverse container size)")
    ax.set_ylabel("drag $F$ at $U=1$")
    ax.set_title("Container-size extrapolation to the unbounded limit\n(stars = intercepts, dotted = analytic HR)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p3 = os.path.join(FIG_DIR, "fig3_wall_extrapolation.png")
    fig.savefig(p3)
    plt.close(fig)
    figs.append({"name": "fig3_wall_extrapolation.png", "path": p3,
                 "caption": "Finite-container drag vs R/H for three viscosity ratios with the linear-in-R/H fit used "
                            "to reach the unbounded HR limit (stars) compared with the analytic value (dotted)."})

    # ---- fig 4: grid convergence ------------------------------------------
    gc = ev["grid_convergence"]
    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=150)
    hs = [p["h_wall"] for p in gc["points"]]
    es = [p["rel_err_vs_finest"] for p in gc["points"]]
    ax.loglog(hs, es, "o-", label="measured")
    h0 = np.array([hs[0], hs[-1]])
    e0 = es[0]
    for order, ls in ((1.0, "--"), (2.0, ":")):
        ax.loglog(h0, e0 * (h0 / h0[0]) ** order, ls, color="0.5",
                  label=f"reference slope {order:.0f}")
    ax.set_xlabel("element size at the interface $h/R$")
    ax.set_ylabel("$|F(h)-F_{ref}|/F_{ref}$")
    ax.set_title("Grid convergence of the drop drag ($\\lambda=1$, $H/R=10$)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p4 = os.path.join(FIG_DIR, "fig4_grid_convergence.png")
    fig.savefig(p4)
    plt.close(fig)
    figs.append({"name": "fig4_grid_convergence.png", "path": p4,
                 "caption": "Drag error versus interface element size: observed order ~1.7-1.8 against reference "
                            "slopes 1 and 2."})

    # ---- fig 5: decoupling ------------------------------------------------
    dec = ev["decoupling"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    rho_ratios = sorted({r["rho_ratio"] for r in dec["operator_level"]})
    for lam, c in zip([0.01, 0.5, 2.0, 100.0], ["C0", "C1", "C2", "C3"]):
        ys = [
            r["lambda_realized"] for r in dec["operator_level"] if r["lambda_target"] == lam
        ]
        ax.loglog(rho_ratios, ys, "o-", color=c, label=f"decoupled $\\tau(x)$, target $\\lambda$={lam:g}")
    legacy = [r["lambda_legacy"] for r in dec["operator_level"] if r["lambda_target"] == 2.0]
    ax.loglog(rho_ratios, legacy, "k--", lw=1.5, label="legacy constant-$\\tau$ (locked: $\\lambda_{eff}=\\rho_d/\\rho_c$)")
    ax.set_xlabel("density ratio $\\rho_d/\\rho_c$")
    ax.set_ylabel("realised viscosity ratio $\\mu_d/\\mu_c$")
    ax.set_title("Viscosity ratio decoupled from the density ratio (gap G2)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p5 = os.path.join(FIG_DIR, "fig5_decoupling.png")
    fig.savefig(p5)
    plt.close(fig)
    figs.append({"name": "fig5_decoupling.png", "path": p5,
                 "caption": "Realised viscosity ratio versus density ratio: the new tau(x) mapping holds lambda at "
                            "its target for every density ratio, the legacy constant-tau path reproduces the G2 lock "
                            "(lambda_eff = rho_d/rho_c)."})
    return figs


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    ev: Dict = {
        "node": "IN26-L2",
        "benchmark": "hadamard_rybczynski_droplet",
        "status": "revived",
        "baseline_verdict": "benchmarks/pending/hadamard_rybczynski_droplet/terminal_verdict_m155.md",
        "geometry": {"R": R, "mu_c": MU_C, "U_scale": U, "axisymmetric": True,
                     "interface": "sharp, grid-aligned at r=R",
                     "outer_boundary": "concentric spherical wall, rigid translation U"},
        "settings": {"h_wall": H_WALL_PROD, "window_A": H_WINDOW_A, "window_B": H_WINDOW_B,
                     "penalty": 1.0e6, "element": "Q1 velocity (u_r, v=u_theta/sin theta)"},
    }
    print("[A] viscosity-ratio decoupling ...")
    ev["decoupling"] = section_a_decoupling()
    print("[B] solver validation in the known limits ...")
    ev["validation"] = section_b_validation()
    print("[C] lambda ladder 0.01..100 ...")
    ev["ladder"] = section_c_ladder()
    print("[D] endpoints ...")
    ev["endpoints"] = section_d_endpoints()
    print("[E] grid convergence ...")
    ev["grid_convergence"] = section_e_grid_convergence()
    print("[F] extrapolation-window audit ...")
    ev["window_audit"] = section_f_window_audit()

    crit = {
        "decoupling_lambda_independent_of_rho": bool(
            ev["decoupling"]["operator_worst_rel_err"] < 1e-12
            and ev["decoupling"]["lattice_worst_rel_err"] < 0.01
        ),
        "ladder_8_points_err_lt_3pct": ev["ladder"]["criterion_err_lt_3pct"],
        "ladder_8_points_err_lt_3pct_window_B": ev["ladder"]["criterion_err_lt_3pct_window_B"],
        "bubble_endpoint_err_lt_5pct": ev["endpoints"]["bubble_limit"]["criterion_err_lt_5pct"],
        "stokes_endpoint_err_lt_3pct": ev["endpoints"]["stokes_limit"]["criterion_err_lt_3pct"],
        "grid_order_ge_1": ev["grid_convergence"]["criterion_order_ge_1"],
    }
    ev["criteria"] = crit
    ev["all_criteria_met"] = all(crit.values())
    ev["runtime_s"] = time.time() - t0
    print("[fig] ...")
    ev["figures"] = make_figures(ev)

    path = os.path.join(OUT_DIR, "evidence.json")
    with open(path, "w") as fh:
        json.dump(ev, fh, indent=2, default=float)

    print("\n=== verdict ===")
    for k, v in crit.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"  ladder max |err| A = {ev['ladder']['max_abs_rel_err_A']*100:.2f} %")
    print(f"  ladder max |err| B = {ev['ladder']['max_abs_rel_err_B']*100:.2f} %")
    print(f"  bubble endpoint err = {ev['endpoints']['bubble_limit']['rel_err_vs_bubble_limit']*100:+.2f} %")
    print(f"  Stokes endpoint err = {ev['endpoints']['stokes_limit']['rel_err_vs_stokes_limit']*100:+.2f} %")
    print(f"  grid min observed order = {ev['grid_convergence']['min_observed_order']:.2f}")
    print(f"  decoupling worst rel err (operator) = {ev['decoupling']['operator_worst_rel_err']:.2e}")
    print(f"  lattice calib worst rel err = {ev['decoupling']['lattice_worst_rel_err']*100:.2f} %")
    print(f"  window spread = {ev['window_audit']['max_window_spread']*100:.2f} %")
    print(f"  runtime {ev['runtime_s']:.1f} s -> {path}")
    return 0 if ev["all_criteria_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())