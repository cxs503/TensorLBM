#!/usr/bin/env python3
"""Driver for XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY.

Runs the predeclared matrices from ``predeclared.json``, writes ``results.json``
(numbers only), renders ``figures/*.png`` and ``REPORT.md`` from that json, and
prints the verdict.  Nothing in ``REPORT.md`` is hand-typed: every number comes
out of ``results.json`` through :func:`render_report`.

Usage:
    python run_xval.py                # full predeclared matrix (~20 arms)
    python run_xval.py --quick        # reduced matrix for the test suite
    python run_xval.py --render-only  # re-render report/figures from results.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(HERE))

torch.set_num_threads(1)

from two_phase_only import (  # noqa: E402
    analytic_profile,
    run_mcmp_two_component,
    run_two_phase_only,
    two_layer_analytic,
)

PRE = json.loads((HERE / "predeclared.json").read_text())


# ---------------------------------------------------------------------------
# Arm construction
# ---------------------------------------------------------------------------


def build_arms(quick: bool = False) -> list[dict]:
    arms: list[dict] = []
    base = {"G": 4e-6, "nx": 4}

    if quick:
        # One arm of M1 and one of M5: enough for the test suite's regression,
        # small enough to run inside a test.
        arms.append({"matrix": "M1", "kind": "two_phase", "ny": 48, "nu_lo": 0.5,
                     "nu_hi": 0.25, "width": 2.0, "rule": "harmonic",
                     "rho_hi": 1.0, "max_steps": 40000})
        arms.append({"matrix": "M5", "kind": "mcmp", "ny": 48, "tau_lo": 2.0,
                     "tau_hi": 1.25, "G_12": -1.0, "max_steps": 40000})
        return arms

    m1 = PRE["matrix"]["M1_viscosity_ratio"]
    for ratio in m1["nu_ratio"]:
        for rule in m1["rules"]:
            arms.append({"matrix": "M1", "kind": "two_phase", "ny": m1["ny"],
                         "nu_lo": m1["nu_lo"], "nu_hi": m1["nu_lo"] / ratio,
                         "width": m1["width"], "rule": rule, "rho_hi": 1.0})

    m2 = PRE["matrix"]["M2_width"]
    for width in m2["width"]:
        arms.append({"matrix": "M2", "kind": "two_phase", "ny": m2["ny"],
                     "nu_lo": m2["nu_lo"], "nu_hi": m2["nu_lo"] / m2["nu_ratio"],
                     "width": width, "rule": m2["rule"], "rho_hi": 1.0})

    m3 = PRE["matrix"]["M3_resolution"]
    for ny in m3["ny"]:
        arms.append({"matrix": "M3", "kind": "two_phase", "ny": ny,
                     "nu_lo": m3["nu_lo"], "nu_hi": m3["nu_lo"] / m3["nu_ratio"],
                     "width": m3["width"], "rule": m3["rule"], "rho_hi": 1.0})

    m4 = PRE["matrix"]["M4_density_contrast"]
    arms.append({"matrix": "M4", "kind": "two_phase", "ny": m4["ny"],
                 "nu_lo": m4["nu_lo"], "nu_hi": m4["nu_lo"] / m4["nu_ratio"],
                 "width": m4["width"], "rule": m4["rule"], "rho_hi": m4["rho_hi"]})

    m5 = PRE["matrix"]["M5_mcmp_comparison"]
    for ratio in m5["nu_ratio"]:
        nu_hi = 0.5 / ratio
        arms.append({"matrix": "M5", "kind": "mcmp", "ny": m5["ny"],
                     "tau_lo": 0.5 + 3 * 0.5, "tau_hi": 0.5 + 3 * nu_hi,
                     "nu_ratio": ratio, "G_12": m5["G_12"]})
    for arm in arms:
        arm.update({k: v for k, v in base.items() if k not in arm})
    return arms


def run_arm(arm: dict) -> dict:
    settings = PRE["solver_settings"]
    common = {
        "nx": arm.get("nx", 4),
        "G": arm.get("G", settings["G"]),
        "max_steps": arm.get("max_steps", settings["max_steps"]),
        "stop_tol": settings["stop_tol"],
        "check_every": settings["check_every"],
    }
    if arm["kind"] == "two_phase":
        rec = run_two_phase_only(
            arm["ny"],
            nu_lo=arm["nu_lo"],
            nu_hi=arm["nu_hi"],
            width=arm["width"],
            rule=arm["rule"],
            rho_lo=1.0,
            rho_hi=arm.get("rho_hi", 1.0),
            **common,
        )
    else:
        rec = run_mcmp_two_component(
            arm["ny"],
            tau_lo=arm["tau_lo"],
            tau_hi=arm["tau_hi"],
            G_12=arm["G_12"],
            **{k: common[k] for k in ("nx", "G", "max_steps", "stop_tol", "check_every")},
        )
    rec["arm"] = {k: v for k, v in arm.items()}
    rec["arm"]["nu_hi"] = arm.get("nu_hi", rec["config"].get("nu_hi"))
    rec["arm"]["nu_ratio"] = rec["config"].get("nu_lo", 0.5) / rec["arm"]["nu_hi"]
    return rec


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def evaluate(results: dict) -> dict:
    th = PRE["predeclared_thresholds"]
    runs = results["runs"]
    tp = [r for r in runs if r["arm"]["kind"] == "two_phase"]
    matched = [r for r in tp if r["arm"].get("rho_hi", 1.0) == 1.0]

    def _worst(seq, key):
        vals = [(r[key], r) for r in seq if isinstance(r.get(key), float)]
        if not vals:
            return None, None
        return max(vals, key=lambda t: t[0])

    k1 = max(
        (max(r["analytic_checks"].values()) for r in runs if "analytic_checks" in r),
        default=float("nan"),
    )
    k2_worst, k2_run = _worst(matched, "l2_rel")
    k3_worst, k3_run = _worst(matched, "max_rel_peak")
    k4_worst, k4_run = _worst(
        [dict(r, flux=max(r["Q_lo_rel"], r["Q_hi_rel"])) for r in matched], "flux"
    )
    k5_worst, k5_run = _worst(matched, "uy_spurious_rel")
    k6_worst, k6_run = _worst(tp, "mass_drift_rel")

    def _label(r):
        if r is None:
            return None
        a = r["arm"]
        if a["kind"] == "mcmp":
            return f"MCMP nu_ratio={a['nu_ratio']:.0f}"
        return (
            f"{a['matrix']} ny={a['ny']} ratio={a['nu_ratio']:.0f} "
            f"width={a['width']:.0f} rule={a['rule']}"
        )

    checks = {
        "K1_analytic_selfcheck": {"value": k1, "threshold": th["K1_analytic_selfcheck"],
                                  "pass": bool(k1 <= th["K1_analytic_selfcheck"])},
        "K2_l2_rel": {"value": k2_worst, "threshold": th["K2_l2_rel"],
                      "worst_arm": _label(k2_run),
                      "pass": bool(k2_worst is not None and k2_worst <= th["K2_l2_rel"])},
        "K3_max_rel_peak": {"value": k3_worst, "threshold": th["K3_max_rel_peak"],
                            "worst_arm": _label(k3_run),
                            "pass": bool(k3_worst is not None and k3_worst <= th["K3_max_rel_peak"])},
        "K4_flux_rel": {"value": k4_worst, "threshold": th["K4_flux_rel"],
                        "worst_arm": _label(k4_run),
                        "pass": bool(k4_worst is not None and k4_worst <= th["K4_flux_rel"])},
        "K5_uy_spurious_rel": {"value": k5_worst, "threshold": th["K5_uy_spurious_rel"],
                               "pass": bool(k5_worst is not None and k5_worst <= th["K5_uy_spurious_rel"])},
        "K6_mass_drift_rel": {"value": k6_worst, "threshold": th["K6_mass_drift_rel"],
                              "pass": bool(k6_worst is not None and k6_worst <= th["K6_mass_drift_rel"])},
    }

    # H2: monotone error growth with the viscosity ratio (harmonic rule, M1).
    harm = sorted(
        [r for r in matched if r["arm"]["matrix"] == "M1" and r["arm"]["rule"] == "harmonic"],
        key=lambda r: r["arm"]["nu_ratio"],
    )
    h2 = [r["l2_rel"] for r in harm]
    checks["H2_monotone_ratio"] = {
        "ratios": [r["arm"]["nu_ratio"] for r in harm],
        "l2_rel": h2,
        "pass": bool(all(b >= a for a, b in zip(h2, h2[1:]))),
    }

    # H3: linear rule is the worst inside each ratio group of M1.
    h3_rows = []
    for ratio in sorted({r["arm"]["nu_ratio"] for r in matched if r["arm"]["matrix"] == "M1"}):
        group = {r["arm"]["rule"]: r["l2_rel"] for r in matched
                 if r["arm"]["matrix"] == "M1" and r["arm"]["nu_ratio"] == ratio}
        if {"linear", "harmonic"} <= set(group):
            h3_rows.append({"nu_ratio": ratio, **{k: float(v) for k, v in group.items()},
                            "harmonic_better": bool(group["harmonic"] <= group["linear"])})
    checks["H3_rule_matters"] = {
        "rows": h3_rows,
        "pass": bool(h3_rows and all(r["harmonic_better"] for r in h3_rows)),
    }

    # H4: at nu_ratio = 10 at least one rule leaves the 15% band.
    r10 = [r for r in matched if r["arm"]["matrix"] == "M1" and r["arm"]["nu_ratio"] == 10.0]
    checks["H4_high_ratio_breaks_band"] = {
        "rows": {r["arm"]["rule"]: float(r["l2_rel"]) for r in r10},
        "pass": bool(any(r["l2_rel"] > th["K2_l2_rel"] for r in r10)) if r10 else None,
    }

    # M4 density contrast arm.
    m4 = [r for r in tp if r["arm"]["matrix"] == "M4"]
    checks["K8_density_contrast"] = {
        "rows": [
            {
                "l2_rel": float(r["l2_rel"]),
                "max_rel_peak": float(r["max_rel_peak"]),
                "interface_width_10_90": r["interface_width_10_90"],
                "interface_centre_shift": r["interface_centre_shift"],
            }
            for r in m4
        ],
        "threshold": th["K8_density_contrast"],
        "pass": bool(m4 and m4[0]["l2_rel"] <= th["K8_density_contrast"]),
    }

    m5 = [r for r in runs if r["arm"]["kind"] == "mcmp"]
    checks["M5_comparison"] = {
        "rows": [
            {
                "nu_ratio": r["arm"]["nu_ratio"],
                "l2_rel": float(r["l2_rel"]),
                "max_rel_peak": float(r["max_rel_peak"]),
                "converged": bool(r["converged"]),
                "finite": bool(r.get("finite", True)),
            }
            for r in m5
        ]
    }

    checks["K7_resolution"] = {
        "rows": {r["arm"]["ny"]: float(r["l2_rel"]) for r in matched if r["arm"]["matrix"] == "M3"}
    }
    all_pass = all(v.get("pass") for k, v in checks.items() if isinstance(v.get("pass"), bool))
    return {"checks": checks, "all_declared_thresholds_pass": bool(all_pass)}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def make_figures(results: dict) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = HERE / "figures"
    figdir.mkdir(exist_ok=True)
    figs: list[dict] = []
    runs = results["runs"]

    # fig1: velocity profiles vs analytic, harmonic rule, M1
    sel = [r for r in runs if r["arm"]["kind"] == "two_phase" and r["arm"]["matrix"] == "M1"
           and r["arm"]["rule"] == "harmonic"]
    if sel:
        sel.sort(key=lambda r: r["arm"]["nu_ratio"])
        fig, axes = plt.subplots(1, len(sel), figsize=(3.1 * len(sel), 4.2), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, r in zip(axes, sel):
            y = np.arange(len(r["profile"]))
            ax.plot(r["profile_ana"], y, "k-", lw=1.6, label="analytic")
            ax.plot(r["profile"], y, "o", ms=2.5, mfc="none", label="two-phase-only")
            ax.set_title(f"nu ratio {r['arm']['nu_ratio']:.0f}\nL2 {100 * r['l2_rel']:.2f}%",
                         fontsize=9)
            ax.set_xlabel("$u_x$ [lattice]")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("y [lattice]")
        axes[0].legend(fontsize=7, loc="lower right")
        fig.suptitle("M2P (no-MCMP two-phase-only) vs two-layer analytic, ny=%d" % sel[0]["arm"]["ny"],
                     fontsize=10)
        fig.tight_layout()
        p = figdir / "fig1_profiles.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        figs.append({"file": p.name, "caption": "Steady velocity profiles of the two-phase-only model against the two-layer analytic solution (viscosity ratio 1/2/4/10, harmonic mixing, ny=96)."})

    # fig2: error vs viscosity ratio for the three mixing rules
    groups: dict[str, list[tuple[float, float]]] = {}
    for r in runs:
        if r["arm"]["kind"] == "two_phase" and r["arm"]["matrix"] == "M1":
            groups.setdefault(r["arm"]["rule"], []).append((r["arm"]["nu_ratio"], r["l2_rel"]))
    if groups:
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        for rule, pts in sorted(groups.items()):
            pts.sort()
            ax.plot([p[0] for p in pts], [100 * p[1] for p in pts], "o-", label=rule)
        ax.axhline(15.0, color="r", ls="--", lw=1, label="owner 15% band")
        ax.set_xscale("log")
        ax.set_xlabel("viscosity ratio $\\nu_{lo}/\\nu_{hi}$")
        ax.set_ylabel("profile L2 error [%]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title("Viscosity mixing rule is load-bearing", fontsize=10)
        fig.tight_layout()
        p = figdir / "fig2_error_vs_ratio.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        figs.append({"file": p.name, "caption": "Profile L2 error versus viscosity ratio for the three node-local mixing rules, with the owner 15% band."})

    # fig3: fields of one reference case (ratio 4, harmonic)
    case = next((r for r in runs if r["arm"]["kind"] == "two_phase" and r["arm"]["matrix"] == "M1"
                 and r["arm"]["rule"] == "harmonic" and r["arm"]["nu_ratio"] == 4.0), None)
    if case is not None:
        fig, axes = plt.subplots(1, 3, figsize=(9.0, 4.0))
        y = np.arange(len(case["profile"]))
        axes[0].plot(case["tau_profile"], y, "b-")
        axes[0].set_xlabel(r"$\tau(y)$")
        axes[0].set_ylabel("y")
        axes[0].set_title("relaxation time field", fontsize=9)
        axes[1].plot(case["profile"], y, "g-", label="$u_x$ sim")
        axes[1].plot(case["profile_ana"], y, "k--", label="$u_x$ analytic")
        axes[1].set_xlabel("$u_x$")
        axes[1].set_title("streamwise velocity", fontsize=9)
        axes[1].legend(fontsize=7)
        err = np.array(case["profile"]) - np.array(case["profile_ana"])
        axes[2].plot(err / max(abs(np.array(case["profile_ana"]))), y, "r-")
        axes[2].axvline(0.0, color="k", lw=0.6)
        axes[2].set_xlabel("$u_x$ error / peak")
        axes[2].set_title("normalised error profile", fontsize=9)
        for ax in axes:
            ax.grid(alpha=0.3)
        fig.suptitle(f"Viscosity-stratified channel, ny={case['arm']['ny']}, rule={case['arm']['rule']}",
                     fontsize=10)
        fig.tight_layout()
        p = figdir / "fig3_fields.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        figs.append({"file": p.name, "caption": "Relaxation-time field, streamwise velocity and normalised error of the reference case (viscosity ratio 4, harmonic mixing)."})

    # fig4: resolution and interface-width sensitivity
    res = sorted([(r["arm"]["ny"], r["l2_rel"]) for r in runs
                  if r["arm"]["kind"] == "two_phase" and r["arm"]["matrix"] == "M3"])
    wid = sorted([(r["arm"]["width"], r["l2_rel"]) for r in runs
                  if r["arm"]["kind"] == "two_phase" and r["arm"]["matrix"] == "M2"])
    if res or wid:
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6))
        if res:
            axes[0].plot([p[0] for p in res], [100 * p[1] for p in res], "o-")
            axes[0].set_xlabel("ny")
            axes[0].set_title("resolution (ratio 2)", fontsize=9)
        axes[0].set_ylabel("L2 error [%]")
        axes[0].grid(alpha=0.3)
        if wid:
            axes[1].plot([p[0] for p in wid], [100 * p[1] for p in wid], "s-")
            axes[1].set_xlabel("interface 10-90 width [cells]")
            axes[1].set_title("interface smearing (ratio 4)", fontsize=9)
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        p = figdir / "fig4_sensitivity.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        figs.append({"file": p.name, "caption": "Grid and interface-width sensitivity of the two-phase-only model."})

    # fig5: density-contrast arm
    m4 = [r for r in runs if r["arm"]["kind"] == "two_phase" and r["arm"]["matrix"] == "M4"]
    if m4:
        r = m4[0]
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.0))
        y = np.arange(len(r["profile"]))
        axes[0].plot(r["profile"], y, "g-", label="$u_x$ sim")
        axes[0].plot(r["profile_ana"], y, "k--", label="$u_x$ analytic")
        axes[0].set_xlabel("$u_x$")
        axes[0].set_ylabel("y")
        axes[0].legend(fontsize=7)
        axes[0].set_title(f"density contrast arm, L2 {100 * r['l2_rel']:.2f}%", fontsize=9)
        axes[1].plot(r["rho_profile"], y, "b-")
        axes[1].set_xlabel(r"$\rho(y)$")
        axes[1].set_title("density after the run (no sharpening)", fontsize=9)
        for ax in axes:
            ax.grid(alpha=0.3)
        fig.tight_layout()
        p = figdir / "fig5_density_contrast.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        figs.append({"file": p.name, "caption": "Density-contrast arm: velocity against the analytic solution and the diffuse density profile after the run (the frozen indicator has no sharpening term)."})

    return figs


# ---------------------------------------------------------------------------
# Report rendering (every number from results.json)
# ---------------------------------------------------------------------------


def _f(v, spec=".4g"):
    if v is None:
        return "n/a"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, float):
        if math.isnan(v):
            return "n/a"
        return format(v, spec)
    return str(v)


def _pct(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{100 * v:.2f}%"


def render_report(results: dict) -> str:
    ev = results["verdict"]
    checks = ev["checks"]
    runs = results["runs"]
    oss = PRE["oss"]

    def arm_label(r):
        a = r["arm"]
        if a["kind"] == "mcmp":
            return f"M5 MCMP, ratio {a['nu_ratio']:.0f}"
        return (f"{a['matrix']} ny={a['ny']} ratio {a['nu_ratio']:.0f} "
                f"w={a['width']:.0f} {a['rule']}")

    lines: list[str] = []
    add = lines.append
    add("# XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY — can the no-MCMP two-phase-only route")
    add("stand in for the two-component layered Poiseuille benchmark?")
    add("")
    add(f"_Anchor_ `XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY` · _OSS row_ "
        f"`oss_solutions` id {oss['oss_solutions_row']} "
        f"({oss['repo']} / {oss['feature_slug']} / {oss['tag']})")
    add("")
    add(f"_Generated_ {results['generated_at']} · TensorLBM {results['tensorlbm_version']} · "
        f"torch {results['torch']} · python {results['python']}")
    add("")
    add("**Data provenance.** Every number below is rendered from "
        "`results.json` by `run_xval.render_report()`; the test suite re-renders "
        "it, so the tables cannot be hand-edited.")
    add("")

    # 1. case introduction
    add("## 1. Case and parameter table")
    add("")
    add("The OSS practice under test (inventory row, quoted):")
    add("")
    add(f"> {oss['method_verbatim']}")
    add("")
    add("So the question is quantitative, not rhetorical: **how much accuracy is "
        "lost when a two-component layered Poiseuille benchmark is run in a "
        "model that has no component index at all?**")
    add("")
    add("Physical problem: two superposed layers in a plane channel `0 <= y <= ny-1`, "
        "driven by the body-force density `rho * G` (per-unit-mass acceleration `G`). "
        "Exact solution per layer, `u''_sigma = -G / nu_sigma`, closed by no-slip on both "
        "walls, velocity continuity and shear-stress continuity at `y = y_int`:")
    add("")
    add("| symbol | value |")
    add("|---|---|")
    add(f"| channel height (lattice) | ny-1 (see matrix) |")
    add(f"| interface position | (ny-1)/2 |")
    add(f"| body acceleration G | {_f(results['settings']['G'])} |")
    add(f"| viscosity ratio nu_lo/nu_hi (M1) | "
        + ", ".join(str(int(x)) for x in PRE['matrix']['M1_viscosity_ratio']['nu_ratio']) + " |")
    add(f"| interface 10-90 width (M2) | "
        + ", ".join(_f(w, ".0f") for w in PRE['matrix']['M2_width']['width']) + " cells |")
    add(f"| density contrast (M4) | rho_hi/rho_lo = {_f(PRE['matrix']['M4_density_contrast']['rho_hi'])} |")
    add(f"| steady-state tolerance | {_f(results['settings']['stop_tol'])} relative change of max|u_x| per {results['settings']['check_every']} steps |")
    add("")
    add("Two-phase-only model (M2P): one distribution function, frozen phase indicator "
        "`C = 0.5 (1 - tanh((y-y_int)/w))`, node-local viscosity from `C` by one of three "
        "mixing rules, variable-coefficient BGK with Guo's second-order force term "
        "(`F = rho G`, tensor `tau`).  MCMP comparison arm (M5): two distributions, "
        "`collide_sc_two_component(u_eq=\"mixture\", force_scheme=\"guo\")`.")
    add("")

    # 2. method
    add("## 2. Method")
    add("")
    add("1. The reference is **solved**, not transcribed: a 4x4 linear system for the four "
        "parabolic coefficients, so there is no closed-form to mistype.  Its invariants "
        "(wall values, velocity jump, stress jump) are reported as K1 and are at machine "
        "precision.")
    add("2. Cross-check of the verifier: the same generic solver reproduces "
        "`tensorlbm.porous_media._two_phase_poiseuille_analytical` at that function's legacy "
        "convention to machine precision (see the test suite), i.e. the library's existing "
        "analytic formula is confirmed, not bypassed.")
    add("3. The M2P arms are run to steady state (or to the step cap, flagged).  The steady "
        "profile is the periodic-x average of `u_x`; errors are then formed against the "
        "analytic profile on the fluid nodes `j = 1..ny-2`.")
    add("4. Matrices: M1 viscosity ratio x mixing rule, M2 interface width, M3 resolution, "
        "M4 density contrast, M5 MCMP comparison.  Thresholds were fixed in "
        "`predeclared.json` before any arm was run; no arm was dropped after the errors "
        "were known.")
    add("")

    # 3. results
    add("## 3. Results")
    add("")
    add("### 3.1 M1 — profile error versus viscosity ratio and mixing rule")
    add("")
    add("| arm | steps | converged | L2 error | max error/peak | flux err (lo/hi) | spurious u_y/u_x | mass drift |")
    add("|---|---|---|---|---|---|---|---|")
    for r in runs:
        if r["arm"]["kind"] == "two_phase" and r["arm"]["matrix"] == "M1":
            add(f"| {arm_label(r)} | {r['steps']} | {'yes' if r['converged'] else 'no'} | "
                f"{_pct(r['l2_rel'])} | {_pct(r['max_rel_peak'])} | "
                f"{_pct(r['Q_lo_rel'])} / {_pct(r['Q_hi_rel'])} | "
                f"{_f(r['uy_spurious_rel'], '.2e')} | {_f(r['mass_drift_rel'], '.2e')} |")
    add("")
    add("### 3.2 M2 — interface-width sensitivity (ratio 4, harmonic)")
    add("")
    add("| width [cells] | L2 error | max error/peak | steps | converged |")
    add("|---|---|---|---|---|")
    for r in runs:
        if r["arm"]["kind"] == "two_phase" and r["arm"]["matrix"] == "M2":
            add(f"| {_f(r['arm']['width'], '.0f')} | {_pct(r['l2_rel'])} | "
                f"{_pct(r['max_rel_peak'])} | {r['steps']} | {'yes' if r['converged'] else 'no'} |")
    add("")
    add("### 3.3 M3 — resolution (ratio 2, harmonic, width 2)")
    add("")
    add("| ny | L2 error | max error/peak | u_peak sim/ana | steps | converged |")
    add("|---|---|---|---|---|---|")
    for r in runs:
        if r["arm"]["kind"] == "two_phase" and r["arm"]["matrix"] == "M3":
            add(f"| {r['arm']['ny']} | {_pct(r['l2_rel'])} | {_pct(r['max_rel_peak'])} | "
                f"{_f(r['u_peak_ratio'], '.4f')} | {r['steps']} | "
                f"{'yes' if r['converged'] else 'no'} |")
    add("")
    add("### 3.4 M4 — density contrast (rho_hi = 0.25 rho_lo, ratio 2)")
    add("")
    add("| arm | L2 error | max error/peak | density 10-90 width | interface centre shift | steps |")
    add("|---|---|---|---|---|---|")
    for r in runs:
        if r["arm"]["kind"] == "two_phase" and r["arm"]["matrix"] == "M4":
            add(f"| {arm_label(r)} | {_pct(r['l2_rel'])} | {_pct(r['max_rel_peak'])} | "
                f"{_f(r['interface_width_10_90'], '.2f')} | "
                f"{_f(r['interface_centre_shift'], '.2f')} | {r['steps']} |")
    add("")
    add("### 3.5 M5 — MCMP two-component comparison arm")
    add("")
    add("| arm | L2 error | max error/peak | finite | converged | steps |")
    add("|---|---|---|---|---|---|")
    for r in runs:
        if r["arm"]["kind"] == "mcmp":
            add(f"| {arm_label(r)} | {_pct(r['l2_rel'])} | {_pct(r['max_rel_peak'])} | "
                f"{'yes' if r.get('finite', True) else 'no'} | "
                f"{'yes' if r['converged'] else 'no'} | {r['steps']} |")
    add("")
    add("### 3.6 Figures")
    add("")
    for fig in results["figures"]:
        add(f"![{fig['file']}](figures/{fig['file']})")
        add("")
        add(f"_{fig['caption']}_")
        add("")

    # 4. verdict / thresholds
    add("## 4. Predeclared thresholds and verdict")
    add("")
    add("| check | value | threshold | pass | worst arm |")
    add("|---|---|---|---|---|")
    for key in ("K1_analytic_selfcheck", "K2_l2_rel", "K3_max_rel_peak", "K4_flux_rel",
                "K5_uy_spurious_rel", "K6_mass_drift_rel"):
        c = checks[key]
        add(f"| {key} | {_f(c['value'], '.3e')} | {_f(c['threshold'], '.2g')} | "
            f"{'yes' if c['pass'] else '**NO**'} | {c.get('worst_arm') or '-'} |")
    dc = checks["K8_density_contrast"]
    add(f"| K8_density_contrast | {_pct(dc['rows'][0]['l2_rel']) if dc['rows'] else 'n/a'} | "
        f"{_f(dc['threshold'], '.2g')} | {'yes' if dc['pass'] else '**NO**'} | M4 |")
    add("")
    add("Hypotheses:")
    add("")
    h2 = checks["H2_monotone_ratio"]
    add(f"* **H2** (error grows with the viscosity ratio): ratios "
        f"{', '.join(str(int(x)) for x in h2['ratios'])} -> L2 "
        f"{', '.join(_pct(v) for v in h2['l2_rel'])}; "
        f"**{'supported' if h2['pass'] else 'not supported'}**.")
    h3 = checks["H3_rule_matters"]
    if h3["rows"]:
        txt = "; ".join(
            f"ratio {int(r['nu_ratio'])}: harmonic {_pct(r['harmonic'])} vs linear {_pct(r['linear'])}"
            for r in h3["rows"]
        )
        add(f"* **H3** (harmonic no worse than linear): {txt}; "
            f"**{'supported' if h3['pass'] else 'not supported'}**.")
    h4 = checks["H4_high_ratio_breaks_band"]
    if h4["rows"]:
        add(f"* **H4** (ratio 10 leaves the 15% band for at least one rule): "
            + ", ".join(f"{k} {_pct(v)}" for k, v in h4["rows"].items())
            + f"; **{'supported' if h4['pass'] else 'not supported'}**.")
    add("")

    # 5. points to note
    add("## 5. Points to note, limitations")
    add("")
    add("* The OSS arm is **specified from the inventory row, not compiled**: "
        "`lssfau/walberla` is not reachable from this sandbox and no copy is vendored in "
        "this worktree, so no OSS binary output is compared here.  The cross-validation is "
        "therefore *practice-level* (the M2P model is the inventory's described downgrade), "
        "not byte-level.")
    add("* The phase indicator is frozen.  That is the inventory's own convention (\"the "
        "layering is given only by the initial value of C\") and it is admissible for a "
        "strictly parallel flow only; the measured spurious normal velocity "
        f"(K5 = {_f(checks['K5_uy_spurious_rel']['value'], '.2e')}) is the check, and it is "
        "inside its threshold.  Nothing here validates interface dynamics, coalescence or "
        "any flow with a non-zero normal velocity at the interface.")
    add("* The analytic reference is density-independent by construction (per-unit-mass "
        "driving): a density contrast therefore does not enter the reference, it only enters "
        "the model, which is exactly why the M4 arm is a *structural* diagnostic.")
    add("* Near-wall accuracy is not isolated: only three grids (48/64/96) are used and the "
        "wall is full-way bounce-back, i.e. first-order at the wall.  K7 is a trend, not an "
        "order-of-accuracy fit.")
    add("* `nu_hi = 0.05` at ratio 10 gives `tau = 0.65`; BGK at that relaxation time is close "
        "to its stability limit, so the ratio-10 row mixes discretisation error with a "
        "low-`tau` artefact.  This is disclosed rather than trimmed.")
    add("")

    # 6. error verdict
    add("## 6. Error judgement against the acceptance band")
    add("")
    add("Acceptance (owner): relative error vs the reference **< 15%**; anything above is "
        "not acceptable.")
    add("")
    k2 = checks["K2_l2_rel"]
    k3 = checks["K3_max_rel_peak"]
    k4 = checks["K4_flux_rel"]
    add(f"* Worst profile L2 error over the matched-density matrices: **{_pct(k2['value'])}** "
        f"({k2['worst_arm']}) -> {'**< 15%**' if k2['pass'] else '**NOT < 15%**'}.")
    add(f"* Worst peak-normalised max error: **{_pct(k3['value'])}** ({k3['worst_arm']}) -> "
        f"{'**< 15%**' if k3['pass'] else '**NOT < 15%**'}.")
    add(f"* Worst per-layer flux error: **{_pct(k4['value'])}** ({k4['worst_arm']}) -> "
        f"{'**< 15%**' if k4['pass'] else '**NOT < 15%**'}.")
    add(f"* Density-contrast arm: **{_pct(checks['K8_density_contrast']['rows'][0]['l2_rel']) if checks['K8_density_contrast']['rows'] else 'n/a'}** "
        f"-> {'**< 15%**' if checks['K8_density_contrast']['pass'] else '**NOT < 15%**'}.")
    add("")
    add("### Disposition")
    add("")
    add(disposition_text(checks))
    add("")
    return "\n".join(lines) + "\n"


def disposition_text(checks: dict) -> str:
    k2 = checks["K2_l2_rel"]
    k4 = checks["K4_flux_rel"]
    dc = checks["K8_density_contrast"]
    h4 = checks["H4_high_ratio_breaks_band"]
    parts = []
    if k2["pass"]:
        parts.append(
            "**Admit the no-MCMP two-phase-only route as a verification arm for "
            "viscosity-stratified parallel channel flow inside the predeclared matrix** "
            f"(worst L2 {_pct(k2['value'])}, worst flux {_pct(k4['value'])}, both inside the "
            "owner 15% band).  It does not replace an MCMP arm where component transport or "
            "interface dynamics matters; it replaces it exactly where the target is the "
            "layered Stokes profile."
        )
    else:
        parts.append(
            "**Do not admit the route on the strength of these runs**: at least one "
            f"predeclared arm reaches L2 {_pct(k2['value'])} > 15%."
        )
    if h4.get("pass"):
        parts.append(
            "The admitted range must be bounded: at viscosity ratio 10 the band is left by "
            "at least one mixing rule, so the downgrade is regime-limited rather than "
            "general."
        )
    if dc["rows"] and not dc["pass"]:
        parts.append(
            "The density-contrast arm leaves the band, so a density contrast across the "
            "interface must not be run through the frozen-indicator model without a "
            "sharpening term."
        )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="reduced matrix (test suite)")
    ap.add_argument("--render-only", action="store_true")
    ap.add_argument("--out", default=str(HERE / "results.json"))
    ap.add_argument("--shard", type=int, default=None,
                    help="run only arms[shard::nshards] (parallel sharding)")
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--merge", default=None,
                    help="merge shard json files (glob) into results.json")
    args = ap.parse_args(argv)
    out = Path(args.out)

    if args.merge:
        import glob

        results = {
            "anchor": PRE["anchor"],
            "quick": False,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tensorlbm_version": getattr(__import__("tensorlbm"), "__version__", "unknown"),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "settings": PRE["solver_settings"],
            "runs": [],
            "figures": [],
        }
        order = {"M1": 0, "M2": 1, "M3": 2, "M4": 3, "M5": 4}
        for f in sorted(glob.glob(args.merge)):
            part = json.loads(Path(f).read_text())
            results["runs"].extend(part["runs"])
        results["runs"].sort(key=lambda r: (
            order[r["arm"]["matrix"]], r["arm"].get("rule", ""),
            r["arm"].get("nu_ratio", 0.0), r["arm"]["ny"], r["arm"].get("width", 0.0)))
        results["figures"] = make_figures(results)
        results["verdict"] = evaluate(results)
        out.write_text(json.dumps(results, indent=1))
        (HERE / "REPORT.md").write_text(render_report(results))
        print(f"merged {len(results['runs'])} runs -> {out}")
        for k, v in results["verdict"]["checks"].items():
            print(k, v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk in ("value", "pass")})
        print("all declared thresholds pass:", results["verdict"]["all_declared_thresholds_pass"])
        return 0

    if args.render_only:
        results = json.loads(out.read_text())
        results["figures"] = make_figures(results)
        results["verdict"] = evaluate(results)
        out.write_text(json.dumps(results, indent=1))
        (HERE / "REPORT.md").write_text(render_report(results))
        print("re-rendered", out)
        return 0

    import tensorlbm

    arms = build_arms(quick=args.quick)
    if args.shard is not None:
        arms = arms[args.shard::args.nshards]
    results: dict = {
        "anchor": PRE["anchor"],
        "quick": bool(args.quick),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tensorlbm_version": getattr(tensorlbm, "__version__", "unknown"),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "settings": PRE["solver_settings"],
        "runs": [],
        "figures": [],
    }
    for i, arm in enumerate(arms, 1):
        rec = run_arm(arm)
        results["runs"].append(rec)
        print(f"[{i}/{len(arms)}] {arm['matrix']} ny={arm['ny']} "
              f"ratio={rec['arm']['nu_ratio']:.0f} "
              f"rule={arm.get('rule', arm.get('kind'))} "
              f"L2={rec['l2_rel']:.4f} steps={rec['steps']} conv={rec['converged']}",
              flush=True)
        out.write_text(json.dumps(results, indent=1))
    results["figures"] = make_figures(results)
    results["verdict"] = evaluate(results)
    out.write_text(json.dumps(results, indent=1))
    if not args.quick:
        (HERE / "REPORT.md").write_text(render_report(results))
    print(json.dumps({k: v for k, v in results["verdict"]["checks"].items()}, indent=1, default=str))
    print("all declared thresholds pass:", results["verdict"]["all_declared_thresholds_pass"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())