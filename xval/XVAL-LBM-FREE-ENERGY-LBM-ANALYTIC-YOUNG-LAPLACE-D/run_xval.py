#!/usr/bin/env python3
"""XVAL-LBM-FREE-ENERGY-LBM-ANALYTIC-YOUNG-LAPLACE-D -- run entry.

Grades the curved-interface Laplace law of the free-energy binary-fluid LBM
(``tensorlbm.phasefield.free_energy_pressure`` transport kernel) against the
*analytic* surface tension ``sigma = (2/3) sqrt(2 kappa A)`` of the same free energy
functional.  Criteria come from ``predeclared.json`` next to this file and are never
tuned after looking at the numbers.

    python xval/XVAL-LBM-FREE-ENERGY-LBM-ANALYTIC-YOUNG-LAPLACE-D/run_xval.py \
        --out xval/XVAL-LBM-FREE-ENERGY-LBM-ANALYTIC-YOUNG-LAPLACE-D/artifacts

Artifacts: ``summary.json`` (every measured row + per-criterion table),
``fig_laplace_convergence.png``, ``fig_profile_and_pressure.png``.
Exit code 0 on PASS, 1 on FAIL (the table is printed either way).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.phasefield.laplace_droplet_calibration import (  # noqa: E402
    DropletCase,
    LaplaceCriteria,
    grade_case,
    grade_report,
    run_droplet_case,
    sigma_from_gradient_energy,
)

# case set: two radii at one resolution + one radius at a second resolution, so both
# the finite-xi curvature trend and the grid-refinement trend are visible.
CASES = [
    dict(case_id="N64_R10", N=64, R=10.0, steps=800),
    dict(case_id="N64_R12", N=64, R=12.0, steps=800),
    dict(case_id="N80_R16", N=80, R=16.0, steps=800),
]

BASE = dict(A=0.1, kappa=0.2, Gamma=0.05, tau_f=1.0, tau_phi=1.0, sample_every=100)


def _figures(rows: list[dict], out: Path) -> list[str]:
    made: list[str] = []
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for row in rows:
        hist = row["measures_history"]
        steps = [h["step"] for h in hist]
        ax[0].plot(steps, [h["dp_bulk"] for h in hist], marker="o", ms=3, label=row["case"]["case_id"])
        ax[1].plot(steps, [h["u_max"] for h in hist], marker="o", ms=3, label=row["case"]["case_id"])
    ax[0].set_xlabel("step")
    ax[0].set_ylabel(r"$\Delta p$ (lattice units)")
    ax[0].set_title("bulk pressure jump vs time")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("step")
    ax[1].set_ylabel(r"$\max|u|$")
    ax[1].set_title("spurious current vs time")
    for a in ax:
        a.legend(fontsize=7)
        a.grid(alpha=0.3)
    p1 = out / "fig_pressure_jump_history.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    made.append(p1.name)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
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
    ax[1].set_title("radial pressure anisotropy")
    for a in ax:
        a.legend(fontsize=7)
        a.grid(alpha=0.3)
    p2 = out / "fig_profile_and_pressure_anisotropy.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=120)
    plt.close(fig)
    made.append(p2.name)

    fig, ax = plt.subplots(figsize=(5.2, 4))
    for row in rows:
        m = row["measures"]
        ax.plot(m["radius_over_xi"], 100.0 * m["errors"]["sigma_capillary_fit"], "o",
                label=f"{row['case']['case_id']} dp*R")
        ax.plot(m["radius_over_xi"], 100.0 * m["errors"]["sigma_integral"], "s",
                label=f"{row['case']['case_id']} integral")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.axhspan(-15.0, 15.0, color="g", alpha=0.08)
    ax.axhspan(-3.0, 3.0, color="g", alpha=0.12)
    ax.set_xlabel(r"$R/\xi$")
    ax.set_ylabel(r"relative error of recovered $\sigma$ (%)")
    ax.set_title("Young-Laplace calibration vs analytic tension")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    p3 = out / "fig_laplace_convergence.png"
    fig.tight_layout()
    fig.savefig(p3, dpi=120)
    plt.close(fig)
    made.append(p3.name)
    return made


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "artifacts"))
    ap.add_argument("--criteria", default=str(HERE / "predeclared.json"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    declared = json.loads(Path(args.criteria).read_text())
    crit_cfg = declared["criteria"]
    criteria = LaplaceCriteria(
        owner_band=crit_cfg["owner_band"],
        stretch_band=crit_cfg["stretch_band"],
        sigma_integral_band=crit_cfg["sigma_integral_band"],
        dp_tensor_band=crit_cfg["dp_tensor_band"],
        xi_band=crit_cfg["xi_band"],
        contour_consistency_cells=crit_cfg["contour_consistency_cells"],
        spurious_velocity_over_cs=crit_cfg["spurious_velocity_over_cs"],
        conservation_drift=crit_cfg["conservation_drift"],
    )

    analytic = []
    for row in declared["analytic_checks"]:
        A, kappa = row["A"], row["kappa"]
        num = sigma_from_gradient_energy(A, kappa)
        analytic.append(
            {
                **row,
                "sigma_from_gradient_energy": num,
                "rel_error_gradient_integral": (num - row["sigma"]) / row["sigma"],
                "passed": abs(num - row["sigma"]) / row["sigma"] <= row["band"],
            }
        )
        print(f"[C1] A={A} kappa={kappa} sigma={row['sigma']:.8f} integral={num:.8f} "
              f"rel={analytic[-1]['rel_error_gradient_integral']:+.2e}", flush=True)

    rows: list[dict] = []
    for cfg in CASES:
        case = DropletCase(**{**BASE, **cfg})
        t0 = time.time()
        measures = run_droplet_case(case)
        graded = grade_case(case, measures, criteria)
        graded["case"]["case_id"] = cfg["case_id"]
        graded["wall_seconds"] = round(time.time() - t0, 1)
        graded["measures_history"] = measures.history
        graded["measures_profile"] = measures.profile
        rows.append(graded)
        err = graded["measures"]["errors"]
        print(
            f"[{cfg['case_id']}] R_fit={graded['measures']['R_fit']:.3f} "
            f"xi_fit={graded['measures']['xi_fit']:.3f} "
            f"dp={graded['measures']['dp_bulk']:.6e} "
            f"sigma_cap={graded['measures']['sigma_capillary_fit']:.6e} "
            f"sigma_int={graded['measures']['sigma_integral']:.6e} "
            f"err_cap={100*err['sigma_capillary_fit']:+.2f}% "
            f"err_int={100*err['sigma_integral']:+.2f}% "
            f"u_max={graded['measures']['u_max']:.2e}",
            flush=True,
        )

    report = grade_report(rows, criteria)
    report.update(
        {
            "anchor": "XVAL-LBM-FREE-ENERGY-LBM-ANALYTIC-YOUNG-LAPLACE-D",
            "repo": "TensorLBM",
            "model": "D2Q9 free-energy binary fluid with the pressure tensor in the "
                     "equilibrium (tensorlbm.phasefield.free_energy_pressure kernel)",
            "reference": {
                "laplace_2d": "dp = sigma/R (cylinder, one principal curvature)",
                "sigma": "sigma = (2/3) sqrt(2 kappa A) = (4/3) kappa / xi",
                "xi": "xi = sqrt(2 kappa / A)",
                "radius_definitions": ["R_contour (phi=0 level set)",
                                       "R_fit (atanh tanh-fit)",
                                       "R_sigma (surface of tension, first moment of P_rr-P_tt)"],
            },
            "analytic_checks": analytic,
            "analytic_checks_passed": all(a["passed"] for a in analytic),
            "rows": rows,
        }
    )
    report["figures"] = _figures(rows, out)
    (out / "summary.json").write_text(json.dumps(report, indent=1))

    print("\nverdict:", report["verdict"],
          "worst |err| sigma_capillary =", f"{100*report['worst_abs_rel_error_sigma_capillary']:.2f}%",
          "| worst |err| sigma_integral =", f"{100*report['worst_abs_rel_error_sigma_integral']:.2f}%")
    if report["failing_checks"]:
        print("failing:", report["failing_checks"])
    ok = report["verdict"] == "PASS" and report["analytic_checks_passed"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())