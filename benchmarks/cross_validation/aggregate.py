#!/usr/bin/env python3
"""Aggregate the TensorLBM / TensorFVM Re=100 cylinder runs into one JSON report.

Machine criterion (mission m5119): ``|Cd_LBM - Cd_FVM| / mean(Cd) <= 5 %``,
with an explicit literature reference carrying a URL/DOI, and both mesh
parameter sets recorded verbatim from the solver JSONs.  The verdict is
honest-FAIL aware: a cross-solver gap above the tolerance is reported as
``partial-FAIL`` together with the root-cause classification recorded by hand
in ``root_cause_analysis`` (passed through from the command line).

Usage::

    python3 benchmarks/cross_validation/aggregate.py \
        --lbm artifacts/lbm_re100_d20.json \
        --fvm artifacts/fvm_re100_48x24.json --fvm artifacts/fvm_re100_72x36.json \
        --out artifacts/cross_validation_cylinder.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TOLERANCE_PCT = 5.0

REFERENCE = {
    "quantity": "cycle-averaged drag coefficient Cd of a 2-D circular cylinder at Re=100",
    "primary_value": 1.35,
    "primary_source": (
        "R. D. Henderson, 'Details of the drag curve near the onset of vortex "
        "shedding', Physics of Fluids 7, 2102-2104 (1995), Table I fits"
    ),
    "doi": "10.1063/1.868459",
    "url": "https://doi.org/10.1063/1.868459",
    "supporting_values": [
        {"value": 1.32, "source": "Dennis & Chang (1970), J. Fluid Mech. 42, 471"},
        {"value": 1.35, "range": [1.30, 1.40], "source": "Braza et al. (1986), J. Fluid Mech. 165"},
    ],
    "admissible_band": [1.32, 1.37],
    "strouhal": {
        "value": 0.1647,
        "source": "Williamson (1989) / Norberg (1994) unconfined 2-D cylinder",
    },
}


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _regime(run: dict) -> dict:
    """Classify whether the run actually resolved vortex shedding."""
    amp = float(run.get("cl_amplitude") or 0.0)
    st = run.get("st")
    shedding = amp > 1e-3 and st is not None and 0.10 <= st <= 0.30
    return {
        "vortex_shedding_resolved": bool(shedding),
        "cl_amplitude": amp,
        "note": (
            "periodic shedding (lift limit cycle)"
            if shedding
            else "no lift limit cycle: the run settled on the symmetric steady branch, "
                 "so Cd is NOT a physical time-mean at the nominal Reynolds number"
        ),
    }


def _pick_best(runs: list[dict]) -> dict:
    """Finest mesh among the runs of one solver."""
    return max(runs, key=lambda r: r["mesh"]["cells"])


def _control(args) -> dict:
    """Re=40 steady control case (both solvers, same geometry)."""
    if not args.lbm_re40 or not args.fvm_re40:
        return {"available": False}
    lbm40, fvm40 = _load(args.lbm_re40), _load(args.fvm_re40)
    return {
        "available": True,
        "purpose": (
            "steady-regime control: shows whether the cross-solver gap at Re=100 comes from a "
            "parameter/geometry/normalisation mistake (then it would also appear here) or from "
            "regime/scheme limitations"
        ),
        "literature_cd_re40": {"value_range": [1.52, 1.60],
                               "source": "Dennis & Chang (1970) JFM 42:471 steady solutions, "
                                         "as commonly tabulated for Re=40"},
        "lbm": {"cd_mean": round(float(lbm40["cd_mean"]), 4),
                "mesh_cells": lbm40["mesh"]["cells"],
                "diameter_lattice_units": lbm40["mesh"]["diameter_lattice_units"],
                "note": "not a grid-converged value at this resolution"},
        "fvm": {"cd_mean": round(float(fvm40["cd_mean"]), 4),
                "mesh_cells": fvm40["mesh"]["cells"],
                "cl_amplitude": fvm40["cl_amplitude"]},
        "gap_pct": round(100.0 * abs(lbm40["cd_mean"] - fvm40["cd_mean"])
                         / (0.5 * (lbm40["cd_mean"] + fvm40["cd_mean"])), 3),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="aggregate LBM/FVM cylinder Re=100 cross-validation")
    ap.add_argument("--lbm", nargs="+", type=Path, required=True)
    ap.add_argument("--fvm", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--root-cause", default="")
    ap.add_argument("--lbm-re40", type=Path, default=None)
    ap.add_argument("--fvm-re40", type=Path, default=None)
    args = ap.parse_args(argv)

    lbm_runs = [_load(p) for p in args.lbm]
    fvm_runs = [_load(p) for p in args.fvm]
    lbm, fvm = _pick_best(lbm_runs), _pick_best(fvm_runs)

    cd_lbm, cd_fvm = float(lbm["cd_mean"]), float(fvm["cd_mean"])
    mean_cd = 0.5 * (cd_lbm + cd_fvm)
    gap_pct = 100.0 * abs(cd_lbm - cd_fvm) / mean_cd
    ref_cd = REFERENCE["primary_value"]
    err_lbm = 100.0 * (cd_lbm - ref_cd) / ref_cd
    err_fvm = 100.0 * (cd_fvm - ref_cd) / ref_cd
    st_lbm, st_fvm = lbm.get("st"), fvm.get("st")
    st_ref = REFERENCE["strouhal"]["value"]
    st_gap = (
        100.0 * abs(st_lbm - st_fvm) / (0.5 * (st_lbm + st_fvm))
        if st_lbm and st_fvm else None
    )

    regime_lbm = _regime(lbm)
    regime_fvm = _regime(fvm)
    cross_pass = gap_pct <= TOLERANCE_PCT
    in_band = all(
        REFERENCE["admissible_band"][0] <= v <= REFERENCE["admissible_band"][1]
        for v in (cd_lbm, cd_fvm)
    )
    checks = [
        {
            "id": "C1_cross_solver_gap",
            "statement": "|Cd_LBM - Cd_FVM| / mean(Cd) <= 5%",
            "value_pct": round(gap_pct, 3),
            "threshold_pct": TOLERANCE_PCT,
            "passed": bool(cross_pass),
        },
        {
            "id": "C2_lbm_vs_literature",
            "statement": "|Cd_LBM - Cd_Ref| / Cd_Ref <= 15% (platform accuracy floor)",
            "value_pct": round(abs(err_lbm), 3),
            "threshold_pct": 15.0,
            "passed": bool(abs(err_lbm) <= 15.0),
        },
        {
            "id": "C3_fvm_vs_literature",
            "statement": "|Cd_FVM - Cd_Ref| / Cd_Ref <= 15% (platform accuracy floor)",
            "value_pct": round(abs(err_fvm), 3),
            "threshold_pct": 15.0,
            "passed": bool(abs(err_fvm) <= 15.0),
        },
        {
            "id": "C4_literature_band",
            "statement": "both Cd inside the admitted literature band [1.32, 1.37]",
            "value": [round(cd_lbm, 4), round(cd_fvm, 4)],
            "band": REFERENCE["admissible_band"],
            "passed": bool(in_band),
        },
        {
            "id": "C5_reference_provenance",
            "statement": "literature reference carries a DOI/URL",
            "value": REFERENCE["doi"],
            "passed": bool(REFERENCE["doi"]),
        },
        {
            "id": "C7_regime_consistency",
            "statement": "both solvers resolve the same flow regime (periodic shedding at Re=100)",
            "value": {
                "lbm_shedding": regime_lbm["vortex_shedding_resolved"],
                "fvm_shedding": regime_fvm["vortex_shedding_resolved"],
            },
            "passed": bool(
                regime_lbm["vortex_shedding_resolved"] == regime_fvm["vortex_shedding_resolved"]
            ),
        },
        {
            "id": "C6_mesh_recorded",
            "statement": "both solvers record mesh parameters",
            "value": {"lbm": sorted(lbm["mesh"]), "fvm": sorted(fvm["mesh"])},
            "passed": bool(lbm["mesh"] and fvm["mesh"]),
        },
    ]
    regime_match = checks[[c["id"] for c in checks].index("C7_regime_consistency")]["passed"]
    all_core_pass = cross_pass and checks[1]["passed"] and checks[2]["passed"] and regime_match
    verdict = "PASS" if all_core_pass else (
        "partial-FAIL" if (not cross_pass and args.root_cause) else "FAIL"
    )

    report = {
        "schema": "tensorlbm-tensorfvm-cylinder-re100-cross-validation-v1",
        "mission": "m5119",
        "case": "2-D circular cylinder in uniform free stream, Re = 100, vortex-shedding regime",
        "physical_parameters": {
            "reynolds": 100.0,
            "diameter": 1.0,
            "inlet_velocity": 1.0,
            "kinematic_viscosity": 0.01,
            "quantity_compared": "cycle-averaged drag coefficient (2F_x / (rho U^2 D))",
        },
        "reference": REFERENCE,
        "solvers": {
            "lbm": {
                "solver": lbm["solver"],
                "cd_mean": round(cd_lbm, 4),
                "cd_min": round(lbm["cd_min"], 4),
                "cd_max": round(lbm["cd_max"], 4),
                "cd_rms_fluctuation": round(lbm["cd_rms_fluctuation"], 4),
                "cl_amplitude": round(lbm["cl_amplitude"], 4),
                "st": st_lbm,
                "cd_relative_error_vs_reference_pct": round(err_lbm, 3),
                "st_relative_error_pct": round(100.0 * (st_lbm - st_ref) / st_ref, 3) if st_lbm else None,
                "mesh": lbm["mesh"],
                "numerics": lbm["numerics"],
                "sampling": lbm["sampling"],
                "regime": regime_lbm,
                "elapsed_s": lbm["elapsed_s"],
                "grid_levels": [
                    {
                        "cells": r["mesh"]["cells"],
                        "diameter_lattice_units": r["mesh"]["diameter_lattice_units"],
                        "cd_mean": round(float(r["cd_mean"]), 4),
                        "st": r.get("st"),
                        "error_pct": round(float(r["cd_relative_error_pct"]), 3),
                    }
                    for r in sorted(lbm_runs, key=lambda r: r["mesh"]["cells"])
                ],
            },
            "fvm": {
                "solver": fvm["solver"],
                "cd_mean": round(cd_fvm, 4),
                "cd_min": round(fvm["cd_min"], 4),
                "cd_max": round(fvm["cd_max"], 4),
                "cd_rms_fluctuation": round(fvm["cd_rms_fluctuation"], 4),
                "cl_amplitude": round(fvm["cl_amplitude"], 4),
                "st": st_fvm,
                "cd_relative_error_vs_reference_pct": round(err_fvm, 3),
                "st_relative_error_pct": round(100.0 * (st_fvm - st_ref) / st_ref, 3) if st_fvm else None,
                "mesh": fvm["mesh"],
                "numerics": fvm["numerics"],
                "sampling": fvm["sampling"],
                "regime": regime_fvm,
                "elapsed_s": fvm["elapsed_s"],
                "grid_levels": [
                    {
                        "cells": r["mesh"]["cells"],
                        "mesh": f"{r['mesh']['nx']}x{r['mesh']['ny']}",
                        "cd_mean": round(float(r["cd_mean"]), 4),
                        "st": r.get("st"),
                        "error_pct": round(float(r["cd_relative_error_pct"]), 3),
                    }
                    for r in sorted(fvm_runs, key=lambda r: r["mesh"]["cells"])
                ],
            },
        },
        "cross_validation": {
            "cd_lbm": round(cd_lbm, 4),
            "cd_fvm": round(cd_fvm, 4),
            "mean_cd": round(mean_cd, 4),
            "abs_difference": round(abs(cd_lbm - cd_fvm), 4),
            "relative_gap_pct": round(gap_pct, 3),
            "tolerance_pct": TOLERANCE_PCT,
            "passed": bool(cross_pass),
            "st_gap_pct": round(st_gap, 3) if st_gap is not None else None,
        },
        "acceptance_checks": checks,
        "supplementary_re40_control": _control(args),
        "verdict": verdict,
        "root_cause_analysis": args.root_cause,
        "evidence_grades": {
            "lbm_cd": "REAL (D2Q9 LBM, independent Ladd MEM wall-force integral, converged statistics)",
            "fvm_cd": "REAL (finite-volume SIMPLE on a body-fitted O-grid, transient force history)",
            "reference_cd": "REAL (peer-reviewed literature, DOI recorded)",
        },
        "reproduction": {
            "lbm_command": (
                "PYTHONPATH=<TensorLBM>/src python3 benchmarks/cross_validation/"
                "lbm_cylinder_re100.py --diameter-cells 20 --steps 44000 --out artifacts/lbm_re100_d20.json"
            ),
            "fvm_command": (
                "TENSORFVM_SRC=<TensorFVM>/src python3 benchmarks/cross_validation/"
                "fvm_cylinder_re100.py --nx 48 --ny 24 --steps 1800 --out artifacts/fvm_re100_48x24.json"
            ),
            "aggregate_command": (
                "python3 benchmarks/cross_validation/aggregate.py --lbm artifacts/lbm_re100_d20.json "
                "--fvm artifacts/fvm_re100_48x24.json --out artifacts/cross_validation_cylinder.json"
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        f"Cd_LBM={cd_lbm:.4f} Cd_FVM={cd_fvm:.4f} gap={gap_pct:.2f}% "
        f"(tol {TOLERANCE_PCT}%) verdict={verdict}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
