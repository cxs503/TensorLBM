#!/usr/bin/env python3
"""Run the XVAL-LBM-VON-NEUMANN-AUTO cross-validation benchmark.

Produces ``result.json`` with every check, every number, programmatically
(no hand-typed values).  The script is self-contained: it imports from
``tensorlbm.stability`` and runs both the analytic and measured arms.

Usage
-----
    PYTHONPATH=src python3 benchmarks/xval_von_neumann_auto/run_xval.py [--quick]

The ``--quick`` flag runs a reduced set (3 tau values, 2 Ma values, n=16 box)
for fast validation (~30 s).  Without it, the full campaign runs (~8 min).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

import sys
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.stability import von_neumann as vn
from tensorlbm.stability.campaign import (
    PREDECLARED,
    c1_modal_agreement,
    c2_stability_boundary,
    c3_equivalent_equation,
    c4_viscosity_ratio_table,
    ma_to_u,
)


def fmt(x: float, sig: int = 6) -> str:
    """Format a float for JSON-friendly output."""
    if x != x:
        return "NaN"
    if x == 0.0:
        return "0"
    if abs(x) >= 1e-3 and abs(x) < 1e5:
        return f"{x:.{sig}g}"
    return f"{x:.{sig}e}"


def run_c1_checks(quick: bool = False) -> list[dict]:
    """C1: measured vs predicted modal growth for physical shear + acoustic branches."""
    checks = []
    taus = [0.55, 0.60, 0.70, 0.80, 1.00] if not quick else [0.60, 0.80]
    mas = [0.0, 0.2, 0.4] if not quick else [0.0, 0.2]
    n_box = 16 if quick else 32
    n_steps = 600 if quick else 1200

    for tau in taus:
        for ma in mas:
            for m in [2, 4]:
                k = 2.0 * math.pi * m / n_box
                coll = vn.Collision("bgk", tau)
                u0 = ma_to_u(ma)

                # Analytic prediction
                sp = vn.spectrum(k, 0.0, coll, u0)
                shear_idx = sp.shear_index
                sigma_pred = sp.growth[shear_idx]

                # Measured
                try:
                    run = vn.run_modal_box(
                        k, 0.0, coll, u0,
                        n=n_box, n_steps=n_steps,
                        seed="shear", amplitude=1e-6,
                    )
                    sigma_meas = run.measured_growth
                    rel_err = abs(sigma_meas - sigma_pred) / max(abs(sigma_pred), 1e-300)
                    passed = rel_err < PREDECLARED["c1"]["growth_rel_err_max"]

                    checks.append({
                        "id": f"C1-shear-tau{tau}-ma{ma}-m{m}",
                        "quantity": "shear growth rate",
                        "ref": float(sigma_pred),
                        "got": float(sigma_meas),
                        "err_pct": float(rel_err * 100),
                        "abs_err_pct": float(rel_err * 100),
                        "tol_pct": PREDECLARED["c1"]["growth_rel_err_max"] * 100,
                        "pass": bool(passed),
                        "fit_r2": float(run.fit_r2),
                        "purity": float(run.purity),
                        "ref_source": "von_neumann.spectrum (analytic eigenvalue)",
                    })
                except Exception as e:
                    checks.append({
                        "id": f"C1-shear-tau{tau}-ma{ma}-m{m}",
                        "quantity": "shear growth rate",
                        "ref": float(sigma_pred),
                        "got": None,
                        "err_pct": None,
                        "abs_err_pct": None,
                        "tol_pct": PREDECLARED["c1"]["growth_rel_err_max"] * 100,
                        "pass": False,
                        "fit_r2": None,
                        "purity": None,
                        "ref_source": "von_neumann.spectrum (analytic eigenvalue)",
                        "error": str(e),
                    })

    # Acoustic branch checks
    for tau in [0.60, 0.80] if not quick else [0.60]:
        for ma in [0.0, 0.2] if not quick else [0.0]:
            for m in [2, 4]:
                k = 2.0 * math.pi * m / n_box
                coll = vn.Collision("bgk", tau)
                u0 = ma_to_u(ma)

                sp = vn.spectrum(k, 0.0, coll, u0)
                sound_idx = sp.index_of("sound+")
                sigma_pred = sp.growth[sound_idx]
                cs_pred = sp.phase_speed[sound_idx]

                try:
                    run = vn.run_modal_box(
                        k, 0.0, coll, u0,
                        n=n_box, n_steps=n_steps // 2,
                        seed="eigen", target="sound+", amplitude=1e-6,
                    )
                    sigma_meas = run.measured_growth
                    cs_meas = run.measured_phase_speed
                    rel_err_growth = abs(sigma_meas - sigma_pred) / max(abs(sigma_pred), 1e-300)
                    rel_err_cs = abs(cs_meas - cs_pred) / max(abs(cs_pred), 1e-300) if abs(cs_pred) > 1e-10 else abs(cs_meas - cs_pred)
                    passed = rel_err_growth < PREDECLARED["c1"]["growth_rel_err_max"]

                    checks.append({
                        "id": f"C1-acoustic-tau{tau}-ma{ma}-m{m}",
                        "quantity": "acoustic growth rate",
                        "ref": float(sigma_pred),
                        "got": float(sigma_meas),
                        "err_pct": float(rel_err_growth * 100),
                        "abs_err_pct": float(rel_err_growth * 100),
                        "tol_pct": PREDECLARED["c1"]["growth_rel_err_max"] * 100,
                        "pass": bool(passed),
                        "fit_r2": float(run.fit_r2),
                        "purity": float(run.purity),
                        "phase_speed_ref": float(cs_pred),
                        "phase_speed_got": float(cs_meas),
                        "phase_speed_err_pct": float(rel_err_cs * 100),
                        "ref_source": "von_neumann.spectrum (analytic eigenvalue)",
                    })
                except Exception as e:
                    checks.append({
                        "id": f"C1-acoustic-tau{tau}-ma{ma}-m{m}",
                        "quantity": "acoustic growth rate",
                        "ref": float(sigma_pred),
                        "got": None,
                        "err_pct": None,
                        "abs_err_pct": None,
                        "tol_pct": PREDECLARED["c1"]["growth_rel_err_max"] * 100,
                        "pass": False,
                        "fit_r2": None,
                        "purity": None,
                        "ref_source": "von_neumann.spectrum (analytic eigenvalue)",
                        "error": str(e),
                    })

    return checks


def run_c2_checks(quick: bool = False) -> dict:
    """C2: stability boundary -- predicted unstable points must diverge, stable must not."""
    taus = [0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 1.00] if not quick else [0.45, 0.55, 0.80]
    mas = [0.0, 0.2, 0.4, 0.6] if not quick else [0.0, 0.2]
    n_box = 16 if quick else 32
    n_steps = 2000 if quick else 5000

    results = c2_stability_boundary(taus=taus, mas=mas, n=n_box, n_steps=n_steps)
    return results


def run_c3_checks(quick: bool = False) -> dict:
    """C3: equivalent equation -- nu_num(theta) quadratic fit."""
    taus = [0.60, 0.80, 1.00] if not quick else [0.60, 0.80]
    mas = [0.0, 0.2] if not quick else [0.0]

    results = c3_equivalent_equation(taus=taus, mas=mas)
    return results


def run_c4_checks(quick: bool = False) -> dict:
    """C4: physical-viscosity contamination band."""
    taus = [0.52, 0.60, 0.70, 0.80, 1.00, 1.20] if not quick else [0.60, 0.80]
    mas = [0.0, 0.1, 0.2, 0.3, 0.4] if not quick else [0.0, 0.2]
    thetas = np.linspace(0.002, 1.20, 40) if not quick else np.linspace(0.02, 0.60, 10)

    results = c4_viscosity_ratio_table(taus=taus, mas=mas, thetas=thetas)
    return results


def run_analytic_limits() -> list[dict]:
    """Additional analytic limit checks (Chapman-Enskog viscosity, sound speed)."""
    checks = []

    # Chapman-Enskog viscosity recovery at theta->0
    for tau in [0.55, 0.60, 0.70, 0.80, 1.00]:
        coll = vn.Collision("bgk", tau)
        nu_ce = coll.viscosity
        curve = vn.shear_branch_curve([1e-3, 1e-2], coll)
        for row in curve:
            nu_num = row["nu_num"]
            rel_err = abs(nu_num - nu_ce) / nu_ce
            checks.append({
                "id": f"CE-nu-tau{tau}-theta{row['k']:.4f}",
                "quantity": "nu at theta->0 (Chapman-Enskog)",
                "ref": float(nu_ce),
                "got": float(nu_num),
                "err_pct": float(rel_err * 100),
                "abs_err_pct": float(rel_err * 100),
                "tol_pct": 2.0,
                "pass": bool(rel_err < 0.02),
                "ref_source": "Chapman-Enskog: nu = (tau - 0.5)/3",
            })

    # Sound speed at small theta
    for tau in [0.60, 0.80, 1.00]:
        coll = vn.Collision("bgk", tau)
        sp = vn.spectrum(0.01, 0.0, coll)
        cs_theory = vn.CS
        c_plus = sp.phase_speed[sp.index_of("sound+")]
        rel_err = abs(c_plus - cs_theory) / cs_theory
        checks.append({
            "id": f"sound-speed-tau{tau}",
            "quantity": "sound speed at theta->0",
            "ref": float(cs_theory),
            "got": float(c_plus),
            "err_pct": float(rel_err * 100),
            "abs_err_pct": float(rel_err * 100),
            "tol_pct": 0.1,
            "pass": bool(rel_err < 0.001),
            "ref_source": "Lattice theory: cs = 1/sqrt(3)",
        })

    # Ma^2 viscosity shift
    for tau in [0.70]:
        coll = vn.Collision("bgk", tau)
        for ma in [0.1, 0.2, 0.3]:
            row = vn.numerical_viscosity(0.002, coll, ma_to_u(ma))
            shift = row["nu_num"] / coll.viscosity - 1.0
            expected_shift = -ma * ma
            rel_err = abs(shift - expected_shift) / max(abs(expected_shift), 1e-300)
            checks.append({
                "id": f"Ma2-shift-tau{tau}-ma{ma}",
                "quantity": "Ma^2 viscosity shift",
                "ref": float(expected_shift),
                "got": float(shift),
                "err_pct": float(rel_err * 100),
                "abs_err_pct": float(abs(shift - expected_shift) * 100),
                "tol_pct": 5.0,
                "pass": bool(abs(shift - expected_shift) < 2e-4),
                "ref_source": "nu_num(0, Ma) = nu_phys * (1 - Ma^2)",
            })

    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="XVAL-LBM-VON-NEUMANN-AUTO runner")
    parser.add_argument("--quick", action="store_true", help="Reduced set for fast validation")
    parser.add_argument("--out", type=str, default=str(HERE / "result.json"), help="Output JSON path")
    args = parser.parse_args()

    t0 = time.time()
    result = {
        "anchor": "XVAL-LBM-VON-NEUMANN-AUTO",
        "mission": 18789,
        "quick": args.quick,
        "provenance": {
            "repo": "TensorLBM",
            "branch": "agent/m18789",
            "external_anchor": "Sterling & Chen (1996) JCP 123:196-206; Lallemand & Luo (2000) PRE 61:6546",
        },
        "gate_pct": 15.0,
        "checks": [],
        "convergence": [],
        "summary": {},
    }

    # C1: modal agreement
    print("Running C1: modal agreement (shear + acoustic)...")
    c1_checks = run_c1_checks(quick=args.quick)
    result["checks"].extend(c1_checks)
    c1_passed = sum(1 for c in c1_checks if c.get("pass"))
    c1_total = len(c1_checks)
    print(f"  C1: {c1_passed}/{c1_total} passed")

    # C2: stability boundary
    print("Running C2: stability boundary...")
    c2_result = run_c2_checks(quick=args.quick)
    result["c2_stability"] = c2_result
    c2_pass = c2_result.get("pass", False)
    print(f"  C2: pass={c2_pass}, TPR={c2_result.get('tpr', 'n/a')}, FPR={c2_result.get('fpr', 'n/a')}")

    # C3: equivalent equation
    print("Running C3: equivalent equation...")
    c3_result = run_c3_checks(quick=args.quick)
    result["c3_equivalent_equation"] = c3_result
    c3_pass = c3_result.get("pass", False)
    print(f"  C3: pass={c3_pass}")

    # C4: viscosity band
    print("Running C4: viscosity contamination band...")
    c4_result = run_c4_checks(quick=args.quick)
    result["c4_viscosity_band"] = c4_result
    c4_pass = c4_result.get("pass", False)
    print(f"  C4: pass={c4_pass}")

    # Analytic limits
    print("Running analytic limit checks...")
    limit_checks = run_analytic_limits()
    result["checks"].extend(limit_checks)
    limit_passed = sum(1 for c in limit_checks if c.get("pass"))
    limit_total = len(limit_checks)
    print(f"  Limits: {limit_passed}/{limit_total} passed")

    # Convergence data (for the convergence plot)
    for tau in [0.60, 0.80]:
        for n in [16, 32, 64]:
            coll = vn.Collision("bgk", tau)
            nu_ce = coll.viscosity
            k = 2.0 * math.pi * 2.0 / n
            sp = vn.spectrum(k, 0.0, coll)
            shear_idx = sp.shear_index
            sigma_pred = sp.growth[shear_idx]
            nu_from_sigma = -sigma_pred / (k * k) if abs(sigma_pred) > 1e-300 else nu_ce
            err_pct = abs(nu_from_sigma - nu_ce) / nu_ce * 100
            result["convergence"].append({
                "arm": f"BGK-tau{tau}",
                "nx": n,
                "nu_got": float(nu_from_sigma),
                "nu_ref": float(nu_ce),
                "err_pct": float(err_pct),
            })

    # Summary
    all_checks = result["checks"]
    checks_passed = sum(1 for c in all_checks if c.get("pass"))
    checks_total = len(all_checks)

    result["summary"] = {
        "checks_passed": checks_passed,
        "checks_total": checks_total,
        "c1_passed": c1_passed,
        "c1_total": c1_total,
        "c2_pass": c2_pass,
        "c3_pass": c3_pass,
        "c4_pass": c4_pass,
        "limits_passed": limit_passed,
        "limits_total": limit_total,
        "all_pass": bool(
            c1_passed == c1_total and c2_pass and c3_pass and c4_pass and limit_passed == limit_total
        ),
        "wall_time_s": round(time.time() - t0, 1),
    }

    # Write result
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nResult written to {out_path}")
    print(f"Wall time: {result['summary']['wall_time_s']} s")
    print(f"Overall: {checks_passed}/{checks_total} checks passed")
    print(f"C1: {c1_passed}/{c1_total}, C2: {c2_pass}, C3: {c3_pass}, C4: {c4_pass}")
    print(f"Limits: {limit_passed}/{limit_total}")
    print(f"ALL PASS: {result['summary']['all_pass']}")

    return 0 if result["summary"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
