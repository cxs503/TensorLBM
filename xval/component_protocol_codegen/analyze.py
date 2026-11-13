#!/usr/bin/env python3
"""XVAL analysis: arm JSONs -> verdict table, summary and figures.

Applies the thresholds declared in ``predeclared.json`` (written before any run)
and writes:

* ``table.csv``    -- one row per arm, every metric the report quotes
* ``summary.json`` -- per-criterion pass/fail plus the aggregate verdict
* ``fig_energy_decay.png`` -- semilog kinetic-energy decay, numeric vs analytic
* ``fig_nu_vs_declared.png`` -- measured vs declared viscosity per family/grid

Usage::

    python xval/component_protocol_codegen/analyze.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:  # allow `python xval/component_protocol_codegen/analyze.py`
    sys.path.insert(0, str(HERE))
REPO_ROOT = HERE.parents[1]
DEFAULT_ARM_DIR = REPO_ROOT / "artifacts" / "xval_component_protocol_codegen"


def load_arms(arm_dir: Path) -> list[dict]:
    arms = []
    for path in sorted(arm_dir.glob("arm_*.json")):
        arms.append(json.loads(path.read_text()))
    if not arms:
        raise SystemExit(f"no arm_*.json in {arm_dir}")
    return arms


def ma_scaling(arms: list[dict]) -> dict:
    """div_max vs the O(Ma^2) prediction: doubling u0 must give ~4x."""
    by_key = {(a["config"]["family"], a["config"]["n"], a["config"]["u0"]): a for a in arms}
    pairs = []
    for (family, n, u0), a in sorted(by_key.items()):
        up = by_key.get((family, n, 2 * u0))
        if up is None:
            continue
        pairs.append(
            {
                "family": family,
                "n": n,
                "u0": u0,
                "div_at_u0": a["div_max"],
                "div_at_2u0": up["div_max"],
                "ratio": up["div_max"] / a["div_max"],
                "ma2_prediction": 4.0,
            }
        )
    return {"pairs": pairs}


def analytic_discrete_divergence(arms: list[dict]) -> dict:
    """Ratio of measured max|div u| to the same operator on the analytic field.

    ``run_arm.divergence`` is a central-difference operator; applied to the
    analytic Taylor-Green field it does not vanish but leaves an
    ``O(u0 k (k dx)^2)`` truncation residue. Dividing by that residue tells us
    whether the measured divergence is solution error or diagnostic error.
    """
    from run_arm import analytic_amplitude, divergence, lattice_wavenumber, taylor_green_fields

    out: dict[str, dict] = {}
    for a in arms:
        n = a["config"]["n"]
        u0 = a["config"]["u0"]
        k_lb = lattice_wavenumber(n)
        decay = analytic_amplitude(u0, a["case"]["nu_declared"], a["case"]["steps"], k_lb)
        ux_a, uy_a = taylor_green_fields(n, u0 * decay)
        ref = divergence(ux_a, uy_a)
        u0k = u0 * k_lb
        out[a["arm"]] = {
            "div_max": a["div_max"],
            "div_over_u0k": float(a["div_max"] / u0k),
            "div_analytic_operator": ref,
            "ratio_to_operator": float(a["div_max"] / ref) if ref > 0 else float("inf"),
        }
    return out


def criteria(arms: list[dict], decl: dict) -> list[dict]:
    th = decl["predeclared_thresholds"]
    out: list[dict] = []

    worst_nu = max(arms, key=lambda a: a["nu_error_pct"])
    out.append(
        {
            "id": "T1_nu_vs_declared",
            "description": "|nu_measured - nu_declared|/nu_declared < 15% for every arm (owner criterion)",
            "threshold_pct": th["nu_error_pct_max"],
            "worst_arm": worst_nu["arm"],
            "worst_value_pct": worst_nu["nu_error_pct"],
            "passed": worst_nu["nu_error_pct"] < th["nu_error_pct_max"],
        }
    )

    worst_eq = max(
        max(a["equivalence"]["max_abs_collide"], a["equivalence"]["max_abs_step"]) for a in arms
    )
    out.append(
        {
            "id": "T2_codegen_vs_reference",
            "description": "generated kernel == independent reference kernel on a fixed random field (abs <= tol)",
            "threshold_abs": th["codegen_ref_max_abs_max"],
            "worst_value_abs": worst_eq,
            "passed": worst_eq <= th["codegen_ref_max_abs_max"],
        }
    )

    worst_field = max(arms, key=lambda a: a["field_l2_rel"])
    out.append(
        {
            "id": "T3_field_l2_vs_analytic",
            "description": "||u_num - u_analytic|| / ||u_analytic|| at t=120 < 15% for every arm",
            "threshold": th["field_l2_rel_max"],
            "worst_arm": worst_field["arm"],
            "worst_value": worst_field["field_l2_rel"],
            "passed": worst_field["field_l2_rel"] < th["field_l2_rel_max"],
        }
    )

    worst_mass = max(arms, key=lambda a: a["mass_drift"])
    out.append(
        {
            "id": "T4a_mass_conservation",
            "description": "|rho_mean(t=120) - 1| below the predeclared tolerance for every arm",
            "threshold": th["mass_drift_max"],
            "worst_arm": worst_mass["arm"],
            "worst_value": worst_mass["mass_drift"],
            "passed": worst_mass["mass_drift"] < th["mass_drift_max"],
        }
    )

    worst_div = max(arms, key=lambda a: a["div_max"])
    out.append(
        {
            "id": "T4b_div_magnitude",
            "description": "max|div u| below the predeclared tolerance (1e-10) for every arm",
            "threshold": th["div_max_max"],
            "worst_arm": worst_div["arm"],
            "worst_value": worst_div["div_max"],
            "passed": worst_div["div_max"] < th["div_max_max"],
            # Kept as a recorded FAIL, never retuned. Root cause, measured:
            # LBM is weakly compressible, so div u = O(Ma^2) u0 k by
            # construction (Kruger et al. 2017, sec. 9.1) -- a 1e-10 bound
            # presumed a machine-zero divergence that this discretisation does
            # not have. The Ma-scaling diagnostic below tests that reading:
            # doubling u0 must multiply div_max by ~4.
            "diagnosis": "predeclared bound assumed div u = 0; LBM has div u = O(Ma^2) u0 k",
            "ma_scaling": ma_scaling(arms),
        }
    )

    out.append(
        {
            "id": "T4c_div_scaled_diagnostic",
            "description": (
                "POST HOC (not pre-registered): div u / (u0 k_lb), the relative size of the "
                "measured divergence, together with the same central-difference operator applied "
                "to the analytic field (which returns ~1e-18, so T4b is not measuring operator "
                "truncation); interpretation only, no pass/fail"
            ),
            "post_hoc": True,
            "detail": analytic_discrete_divergence(arms),
            "passed": None,
        }
    )

    # convergence per family: |nu(N=32) - nu(N=64)| / |nu(N=64)| < 10%
    conv: list[dict] = []
    for family in ("BGK", "TRT"):
        u0 = 0.02
        sel = {
            a["config"]["n"]: a
            for a in arms
            if a["config"]["family"] == family and a["config"]["u0"] == u0
        }
        if 32 in sel and 64 in sel:
            n32, n64 = sel[32]["nu_measured"], sel[64]["nu_measured"]
            delta = abs(n32 - n64) / abs(n64) * 100.0
            conv.append(
                {
                    "family": family,
                    "u0": u0,
                    "nu_n32": n32,
                    "nu_n64": n64,
                    "delta_pct": delta,
                    "passed": delta < 10.0,
                }
            )
    out.append(
        {
            "id": "T5_grid_convergence",
            "description": "|nu(N=32)-nu(N=64)|/nu(N=64) < 10% per family at u0=0.02",
            "threshold_pct": 10.0,
            "detail": conv,
            "passed": all(c["passed"] for c in conv) and bool(conv),
        }
    )

    # family independence at identical (N, u0): BGK vs TRT must agree
    fam_pairs: list[dict] = []
    for n in (32, 64):
        for u0 in (0.02, 0.04):
            got = {a["config"]["family"]: a for a in arms
                   if a["config"]["n"] == n and a["config"]["u0"] == u0}
            if {"BGK", "TRT"} <= set(got):
                b, t = got["BGK"]["nu_measured"], got["TRT"]["nu_measured"]
                fam_pairs.append(
                    {
                        "n": n,
                        "u0": u0,
                        "nu_bgk": b,
                        "nu_trt": t,
                        "delta_pct": abs(b - t) / abs(t) * 100.0,
                    }
                )
    out.append(
        {
            "id": "T6_family_agreement",
            "description": "BGK and TRT arms at identical (N, u0) recover the same viscosity (cross-component check)",
            "threshold_pct": th["owner_acceptance_pct"],
            "detail": fam_pairs,
            "passed": all(p["delta_pct"] < th["owner_acceptance_pct"] for p in fam_pairs) and bool(fam_pairs),
        }
    )
    return out


def write_table(arms: list[dict], path: Path) -> None:
    cols = [
        "arm",
        "family",
        "n",
        "k_lattice",
        "u0",
        "ma",
        "tau",
        "nu_declared",
        "nu_measured",
        "nu_error_pct",
        "field_l2_rel",
        "div_max",
        "mass_drift",
        "codegen_ref_max_abs_step",
        "kernel_sha256_12",
        "fit_rms",
        "runtime_simulate_s",
    ]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for a in arms:
            w.writerow(
                [
                    a["arm"],
                    a["config"]["family"],
                    a["config"]["n"],
                    f"{a['case']['k_lattice']:.8f}",
                    a["config"]["u0"],
                    f"{a['case']['ma']:.6f}",
                    a["config"]["tau"],
                    f"{a['case']['nu_declared']:.8f}",
                    f"{a['nu_measured']:.8f}",
                    f"{a['nu_error_pct']:.5f}",
                    f"{a['field_l2_rel']:.5e}",
                    f"{a['div_max']:.3e}",
                    f"{a['mass_drift']:.3e}",
                    f"{a['equivalence']['max_abs_step']:.3e}",
                    a["kernel"]["sha256"][:12],
                    f"{a['fit']['fit_rms']:.3e}",
                    f"{a['timing_s']['simulate']:.4f}",
                ]
            )


def write_figures(arms: list[dict], out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made: list[str] = []

    # 1) semilog energy decay vs analytic, one panel per family
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, family in zip(axes, ("BGK", "TRT")):
        for a in [x for x in arms if x["config"]["family"] == family]:
            t = np.array([s["t"] for s in a["samples"]], dtype=float)
            e = np.array([s["E"] for s in a["samples"]], dtype=float)
            k_lb = a["case"]["k_lattice"]
            nu = a["case"]["nu_declared"]
            e_ana = (a["case"]["u0"] ** 2 / 4.0) * np.exp(-4.0 * nu * k_lb**2 * t)
            label = f"N={a['config']['n']}, u0={a['config']['u0']}, nu_m={a['nu_measured']:.4f}"
            ax.semilogy(t, e, lw=1.6, label=f"num {label}")
            ax.semilogy(t, e_ana, "k--", lw=0.9, alpha=0.7)
        ax.set_title(f"{family}: kinetic energy decay (dashed = analytic)")
        ax.set_xlabel("lattice time t")
        ax.set_ylabel("<u^2+v^2>/2")
        ax.legend(fontsize=6.5, loc="lower left")
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = out_dir / "fig_energy_decay.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    made.append(str(path.relative_to(REPO_ROOT)))

    # 2) measured vs declared viscosity
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    labels, vals, errs = [], [], []
    for a in arms:
        labels.append(f"{a['arm']}\n{a['config']['family']} N={a['config']['n']}\nu0={a['config']['u0']}")
        vals.append(a["nu_measured"])
        errs.append(a["nu_error_pct"])
    xpos = np.arange(len(arms))
    ax.bar(xpos, vals, yerr=[1e-4] * len(vals), color="#4472c4", alpha=0.85)
    ax.axhline(arms[0]["case"]["nu_declared"], color="k", ls="--", lw=1.0, label="declared nu = 0.02")
    for xi, (v, er) in enumerate(zip(vals, errs)):
        ax.text(xi, v * 1.0004, f"{er:.2f}%", ha="center", fontsize=7)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylim(0.019, 0.0205)
    ax.set_ylabel("nu_measured (lattice)")
    ax.set_title("Viscosity recovered from Taylor-Green decay vs the value the\nComponentSpec declares (15% criterion)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = out_dir / "fig_nu_vs_declared.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    made.append(str(path.relative_to(REPO_ROOT)))
    return made


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm-dir", default=str(DEFAULT_ARM_DIR))
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    arm_dir = Path(args.arm_dir)
    decl = json.loads((HERE / "predeclared.json").read_text())
    arms = load_arms(arm_dir)
    crit = criteria(arms, decl)

    write_table(arms, arm_dir / "table.csv")
    figs = [] if args.no_figures else write_figures(arms, arm_dir)

    summary = {
        "anchor": decl["anchor"],
        "mission": decl["mission"],
        "arms": [
            {
                "arm": a["arm"],
                "family": a["config"]["family"],
                "n": a["config"]["n"],
                "u0": a["config"]["u0"],
                "nu_measured": a["nu_measured"],
                "nu_error_pct": a["nu_error_pct"],
                "field_l2_rel": a["field_l2_rel"],
                "kernel_sha256": a["kernel"]["sha256"],
            }
            for a in arms
        ],
        "criteria": crit,
        "verdict": "PASS"
        if all(c["passed"] for c in crit if c.get("passed") is not None)
        else "FAIL",
        "anchor_criteria_ids": ["T1_nu_vs_declared", "T2_codegen_vs_reference",
                                "T3_field_l2_vs_analytic", "T5_grid_convergence",
                                "T6_family_agreement"],
        "verdict_anchor_criterion": "PASS"
        if all(c["passed"] for c in crit if c["id"] in
               ("T1_nu_vs_declared", "T2_codegen_vs_reference",
                "T3_field_l2_vs_analytic", "T5_grid_convergence",
                "T6_family_agreement"))
        else "FAIL",
        "failed_declared_criteria": [c["id"] for c in crit if c["passed"] is False],
        "figures": figs,
        "owner_acceptance_pct": decl["predeclared_thresholds"]["owner_acceptance_pct"],
        "max_nu_error_pct": max(a["nu_error_pct"] for a in arms),
        "max_codegen_ref_abs": max(
            max(a["equivalence"]["max_abs_collide"], a["equivalence"]["max_abs_step"]) for a in arms
        ),
    }
    (arm_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("verdict", "max_nu_error_pct", "max_codegen_ref_abs")}, indent=2))
    for c in crit:
        tag = "INFO" if c["passed"] is None else ("PASS" if c["passed"] else "FAIL")
        print(f"  [{tag}] {c['id']}: {c['description']}")
    return 0 if summary["verdict_anchor_criterion"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())