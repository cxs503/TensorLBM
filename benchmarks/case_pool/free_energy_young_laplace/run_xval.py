#!/usr/bin/env python3
"""XVAL-LBM-FREE-ENERGY-LBM-ANALYTIC-YOUNG-LAPLACE-D -- run entry.

Grades the curved-interface (2-D cylinder) Young-Laplace law of the free-energy
binary-fluid LBM transport kernel (``tensorlbm.phasefield.free_energy_pressure``)
against two references of the *same* free-energy functional:

* the analytic surface tension ``sigma = (2/3) sqrt(2 kappa A)`` with the
  sharp-interface law ``dp = sigma / R`` (headline KPI, owner band 15 %);
* the exact diffuse-interface pressure jump of the analytic ``tanh`` profile,
  ``dp = integral kappa (d phi/dr)^2 / r dr`` (finite-``xi`` reference);
* the same-pipeline discrete reference for the pressure-tensor routes
  (``tensorlbm.phasefield.laplace_droplet_calibration.discrete_reference_tension``),
  which cancels the nine-point-stencil bias of the post-processing.

Criteria live in ``predeclared.json`` next to this file and are never tuned after
looking at the numbers.

    PYTHONPATH=src python benchmarks/case_pool/free_energy_young_laplace/run_xval.py
    PYTHONPATH=src python .../run_xval.py --smoke --out /tmp/yl_smoke

Artifacts: ``summary.json`` (every measured row, per-criterion table, noise
diagnostics), ``ref_vs_got.json`` (benchmark rows: ref / got / error / band),
``fig_laplace_convergence.png``, ``fig_profile_and_pressure_anisotropy.png``.
Exit code 0 on PASS, 1 on FAIL (the table is printed either way).

Repro (full campaign, CPU, ~4 min on one thread):
    PYTHONPATH=src python benchmarks/case_pool/free_energy_young_laplace/run_xval.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.phasefield.laplace_droplet_calibration import (  # noqa: E402
    LaplaceCriteria,
    default_cases,
    diffuse_interface_correction,
    grade_report,
    run_suite,
    sigma_analytic,
    sigma_from_gradient_energy,
    smoke_cases,
)

ANCHOR = "XVAL-LBM-FREE-ENERGY-LBM-ANALYTIC-YOUNG-LAPLACE-D"


def _criteria(cfg: dict) -> LaplaceCriteria:
    return LaplaceCriteria(
        owner_band=cfg["owner_band"],
        stretch_band=cfg["stretch_band"],
        sigma_integral_band=cfg["sigma_integral_band"],
        dp_tensor_band=cfg["dp_tensor_band"],
        xi_band=cfg["xi_band"],
        contour_consistency_cells=cfg["contour_consistency_cells"],
        spurious_velocity_over_cs=cfg["spurious_velocity_over_cs"],
        conservation_drift=cfg["conservation_drift"],
        noise_band=cfg.get("noise_band", 0.02),
    )


def _benchmark_rows(report: dict) -> list[dict]:
    """Flatten the campaign into ref/got/tolerance rows (the L4 table)."""
    rows = []
    for row in report["rows"]:
        cid = row["case"]["case_id"]
        m = row["measures"]
        e = m["errors"]
        A, kappa = row["case"]["A"], row["case"]["kappa"]
        sigma = sigma_analytic(A, kappa)
        ref_dp = sigma / m["R_fit"]
        ref_dp_diffuse = report["diffuse_interface_reference"][cid]["dp_diffuse_exact"]
        rows += [
            {
                "case": cid,
                "metric": "sigma_from_dp_times_R_vs_analytic",
                "ref": sigma,
                "got": m["sigma_capillary_fit"],
                "err_pct": 100.0 * e["sigma_capillary_fit"],
                "tol_pct": 100.0 * row["checks"]["C1_sigma_capillary_vs_analytic"]["band"],
                "pass": row["checks"]["C1_sigma_capillary_vs_analytic"]["passed"],
            },
            {
                "case": cid,
                "metric": "dp_bulk_vs_sigma_over_R",
                "ref": ref_dp,
                "got": m["dp_bulk_mean"],
                "err_pct": 100.0 * (m["dp_bulk_mean"] - ref_dp) / ref_dp,
                "tol_pct": 100.0 * row["checks"]["C1_sigma_capillary_vs_analytic"]["band"],
                "pass": row["checks"]["C1_sigma_capillary_vs_analytic"]["passed"],
            },
            {
                "case": cid,
                "metric": "dp_bulk_vs_diffuse_interface_exact",
                "ref": ref_dp_diffuse,
                "got": m["dp_bulk_mean"],
                "err_pct": 100.0 * (m["dp_bulk_mean"] - ref_dp_diffuse) / ref_dp_diffuse,
                "tol_pct": 100.0 * row["checks"]["C1_sigma_capillary_vs_analytic"]["band"],
                "pass": abs(m["dp_bulk_mean"] - ref_dp_diffuse) / ref_dp_diffuse
                <= row["checks"]["C1_sigma_capillary_vs_analytic"]["band"],
            },
            {
                "case": cid,
                "metric": "sigma_integral_vs_same_pipeline_reference",
                "ref": m["sigma_integral_ref"],
                "got": m["sigma_integral"],
                "err_pct": 100.0 * e["sigma_integral_vs_discrete_ref"],
                "tol_pct": 100.0 * row["checks"]["C2_sigma_integral_vs_discrete_reference"]["band"],
                "pass": row["checks"]["C2_sigma_integral_vs_discrete_reference"]["passed"],
            },
            {
                "case": cid,
                "metric": "dp_tensor_vs_same_pipeline_reference",
                "ref": m["dp_tensor_ref"],
                "got": m["dp_tensor"],
                "err_pct": 100.0 * e["dp_tensor_vs_discrete_ref"],
                "tol_pct": 100.0 * row["checks"]["C3_tensor_capillary_vs_discrete_reference"]["band"],
                "pass": row["checks"]["C3_tensor_capillary_vs_discrete_reference"]["passed"],
            },
            {
                "case": cid,
                "metric": "interface_width_xi_fit_vs_analytic",
                "ref": m["xi_analytic"],
                "got": m["xi_fit"],
                "err_pct": 100.0 * e["xi_fit"],
                "tol_pct": 100.0 * row["checks"]["C4_interface_width_recovered"]["band"],
                "pass": row["checks"]["C4_interface_width_recovered"]["passed"],
            },
            {
                "case": cid,
                "metric": "steady_window_relative_standard_error",
                "ref": 0.0,
                "got": e["dp_relative_standard_error"],
                "err_pct": 100.0 * e["dp_relative_standard_error"],
                "tol_pct": 100.0 * row["checks"]["C8_steady_window_noise"]["band"],
                "pass": row["checks"]["C8_steady_window_noise"]["passed"],
            },
        ]
    return rows


def _figures(rows: list[dict], out: Path) -> list[str]:
    made: list[str] = []
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
    for row in rows:
        hist = row["measures_history"]
        steps = [h["step"] for h in hist]
        ax[0].plot(steps, [h["dp_bulk"] for h in hist], lw=0.9, label=row["case"]["case_id"])
        ax[1].plot(steps, [h["u_max"] for h in hist], lw=0.9, label=row["case"]["case_id"])
    ax[0].set_xlabel("step")
    ax[0].set_ylabel(r"$\Delta p$ (lattice units)")
    ax[0].set_title("snapshot bulk pressure jump (scatter = why time averaging)")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("step")
    ax[1].set_ylabel(r"$\max|u|$")
    ax[1].set_title("spurious current")
    for a in ax:
        a.legend(fontsize=7)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_pressure_jump_history.png", dpi=130)
    plt.close(fig)
    made.append("fig_pressure_jump_history.png")

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
    for row in rows:
        prof = row["measures_profile"]
        r = [p["r"] for p in prof]
        ax[0].plot(r, [p["phi"] for p in prof], label=row["case"]["case_id"])
        ax[1].plot(r, [p["anis"] for p in prof], label=row["case"]["case_id"])
    ax[0].axhline(0.0, color="k", lw=0.6)
    ax[0].set_xlabel("r (lattice units)")
    ax[0].set_ylabel(r"$\bar\phi(r)$")
    ax[0].set_title("azimuthally averaged order parameter")
    ax[1].set_xlabel("r (lattice units)")
    ax[1].set_ylabel(r"$\overline{P_{rr}-P_{tt}}$")
    ax[1].set_title("radial pressure anisotropy (steady-window mean)")
    for a in ax:
        a.legend(fontsize=7)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_profile_and_pressure_anisotropy.png", dpi=130)
    plt.close(fig)
    made.append("fig_profile_and_pressure_anisotropy.png")

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for row in rows:
        m = row["measures"]
        cid = row["case"]["case_id"]
        ax.plot(m["radius_over_xi"], 100.0 * m["errors"]["sigma_capillary_fit"], "o",
                label=f"{cid}: dp*R_fit vs sigma")
        ax.plot(m["radius_over_xi"], 100.0 * m["errors"]["sigma_integral_vs_discrete_ref"], "s",
                label=f"{cid}: tensor integral vs ref")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.axhspan(-15.0, 15.0, color="g", alpha=0.08)
    ax.axhspan(-3.0, 3.0, color="g", alpha=0.15)
    ax.set_xlabel(r"$R/\xi$")
    ax.set_ylabel(r"relative error of recovered $\sigma$ (%)")
    ax.set_title("Young-Laplace calibration (bands: 3 % stretch, 15 % owner)")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_laplace_convergence.png", dpi=130)
    plt.close(fig)
    made.append("fig_laplace_convergence.png")
    return made


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "artifacts"))
    ap.add_argument("--criteria", default=str(HERE / "predeclared.json"))
    ap.add_argument("--smoke", action="store_true", help="reduced case set (CI)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    declared = json.loads(Path(args.criteria).read_text())
    criteria = _criteria(declared["criteria"])

    analytic = []
    for row in declared["analytic_checks"]:
        num = sigma_from_gradient_energy(row["A"], row["kappa"])
        analytic.append(
            {
                **row,
                "sigma_from_gradient_energy": num,
                "rel_error_gradient_integral": (num - row["sigma"]) / row["sigma"],
                "passed": abs(num - row["sigma"]) / row["sigma"] <= row["band"],
            }
        )
        print(f"[analytic] A={row['A']} kappa={row['kappa']} sigma={row['sigma']:.8f} "
              f"integral={num:.8f} rel={analytic[-1]['rel_error_gradient_integral']:+.2e}",
              flush=True)

    cases = smoke_cases() if args.smoke else default_cases()
    t0 = time.time()
    rows = run_suite(cases, criteria)
    for row in rows:
        cid = row["case"]["case_id"]
        m, e = row["measures"], row["measures"]["errors"]
        print(f"[{cid}] R_fit={m['R_fit']:.3f} xi_fit={m['xi_fit']:.3f} "
              f"dp_mean={m['dp_bulk_mean']:.6e} sd={m['dp_bulk_std']:.2e} n={m['n_samples']} "
              f"sigma_cap={m['sigma_capillary_fit']:.6e} err_cap={100*e['sigma_capillary_fit']:+.2f}% "
              f"err_int_ref={100*e['sigma_integral_vs_discrete_ref']:+.2f}% "
              f"err_dpT_ref={100*e['dp_tensor_vs_discrete_ref']:+.2f}% "
              f"u_max={m['u_max']:.2e} passed={row['passed']}", flush=True)

    report = grade_report(rows, criteria)
    report.update(
        {
            "anchor": ANCHOR,
            "repo": "TensorLBM",
            "model": "D2Q9 free-energy binary fluid, pressure tensor in the equilibrium "
                     "(tensorlbm.phasefield.free_energy_pressure kernel)",
            "reference": {
                "laplace_2d": "dp = sigma/R (cylinder, one principal curvature)",
                "sigma": "sigma = (2/3) sqrt(2 kappa A) = (4/3) kappa / xi",
                "xi": "xi = sqrt(2 kappa / A)",
                "diffuse_interface_exact": "dp = integral kappa (d phi/dr)^2 / r dr on the "
                                           "analytic tanh profile (finite-xi reference)",
                "radius_definitions": ["R_contour (phi=0 level set)",
                                       "R_fit (atanh tanh-fit)",
                                       "R_sigma (surface of tension, first moment of P_rr-P_tt)"],
            },
            "protocol": {
                "measurement": "steady-window time average of the bulk pressure jump "
                               "(single-snapshot readings scatter +-10..30 % step-to-step)",
                "cases": [c.as_dict() for c in cases],
                "diffuse_correction_formula": "dp_cyl*R/sigma = "
                                              + str(round(diffuse_interface_correction(
                                                  cases[0].R, cases[0].A, cases[0].kappa), 5)),
            },
            "analytic_checks": analytic,
            "analytic_checks_passed": all(a["passed"] for a in analytic),
            "wall_seconds": round(time.time() - t0, 1),
            "rows": rows,
        }
    )
    report["figures"] = _figures(rows, out)
    (out / "summary.json").write_text(json.dumps(report, indent=1))

    bench = {
        "anchor": ANCHOR,
        "criteria_source": str(Path(args.criteria).name),
        "smoke": bool(args.smoke),
        "verdict": report["verdict"],
        "worst_abs_rel_error_sigma_capillary": report["worst_abs_rel_error_sigma_capillary"],
        "worst_abs_rel_error_vs_diffuse_exact": report["worst_abs_rel_error_vs_diffuse_exact"],
        "rows": _benchmark_rows(report),
    }
    bench["n_rows"] = len(bench["rows"])
    bench["n_pass"] = sum(1 for r in bench["rows"] if r["pass"])
    (out / "ref_vs_got.json").write_text(json.dumps(bench, indent=1))

    print("\nverdict:", report["verdict"],
          "| worst |err| sigma_capillary =",
          f"{100*report['worst_abs_rel_error_sigma_capillary']:.2f}%",
          "| worst |err| vs diffuse exact =",
          f"{100*report['worst_abs_rel_error_vs_diffuse_exact']:.2f}%")
    if report["failing_checks"]:
        print("failing:", report["failing_checks"])
    ok = report["verdict"] == "PASS" and report["analytic_checks_passed"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())