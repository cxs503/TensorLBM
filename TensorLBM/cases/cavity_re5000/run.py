#!/usr/bin/env python3
"""CASE-CAVITY-RE5000: Lid-driven square cavity at Re=5000.

Reference: Ghia, U., Ghia, K.N., & Shin, C.T. (1982).
"High-Re solutions for incompressible flow using the Navier-Stokes equations
and a multigrid method." Journal of Computational Physics, 48(3), 387-411.
- Table III: u/U along vertical centerline x/L=0.5, Re=5000
- Table IV: v/U along horizontal centerline y/L=0.5, Re=5000

Tolerance: <8% pointwise error on Ghia table rows.

Usage:
    python cases/cavity_re5000/run.py --n 129 --steps 50000
    python cases/cavity_re5000/run.py --n 129 257 --steps 100000 --out result.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Ensure we use this worktree's tensorlbm
# run.py is at TensorLBM/cases/cavity_re5000/run.py
# tensorlbm src is at TensorLBM/src/
CASE_DIR = Path(__file__).resolve().parent
TENSORLBM_SRC = CASE_DIR.parents[1] / "src"
sys.path.insert(0, str(TENSORLBM_SRC))

import tensorlbm
assert str(Path(tensorlbm.__file__).resolve()).startswith(str(TENSORLBM_SRC)), (
    f"imported tensorlbm from {tensorlbm.__file__}, expected {TENSORLBM_SRC}"
)

from tensorlbm.verification.lid_driven_cavity.solver import CavityConfig, CavitySolver
from tensorlbm.verification.lid_driven_cavity.reference_data import (
    get_ghia_reference, GHIA_RE5000, interpolate_ghia
)
from tensorlbm.verification.lid_driven_cavity.verification import compute_errors

TOL_PCT = 8.0
RE = 5000.0
U_LID = 0.1


def run_single(n, re=RE, u_lid=U_LID, max_steps=100000, verbose=True):
    """Run a single cavity simulation and return results."""
    cfg = CavityConfig(n=n, re=re, u_lid=u_lid, max_steps=max_steps)
    solver = CavitySolver(cfg)

    if verbose:
        print(f"[run] n={n} re={re} u_lid={u_lid} tau={cfg.tau:.5f}", flush=True)

    t0 = time.time()
    for step in range(max_steps):
        solver.step()
        if verbose and step % 10000 == 0 and step > 0:
            rho, ux, uy = solver._macroscopic(solver.f)
            resid = float((ux**2 + uy**2).max().sqrt())
            print(f"  step={step} resid={resid:.2e}", flush=True)

    elapsed = time.time() - t0
    y, u, x, v = solver.get_profiles()

    # Find extrema
    u_min_idx = np.argmin(u)
    v_max_idx = np.argmax(v)

    return {
        "n": n,
        "re": re,
        "u_lid": u_lid,
        "tau": cfg.tau,
        "steps": max_steps,
        "elapsed_s": round(elapsed, 1),
        "y_centerline": y.tolist(),
        "u_centerline": u.tolist(),
        "x_centerline": x.tolist(),
        "v_centerline": v.tolist(),
        "u_min": float(u[u_min_idx]),
        "u_min_y": float(y[u_min_idx]),
        "v_max": float(v[v_max_idx]),
        "v_max_x": float(x[v_max_idx]),
    }


def compare_with_ghia(result):
    """Compare simulation results with Ghia Re=5000 reference."""
    ref = get_ghia_reference(5000)
    y_sim = np.array(result["y_centerline"])
    u_sim = np.array(result["u_centerline"])
    x_sim = np.array(result["x_centerline"])
    v_sim = np.array(result["v_centerline"])

    errors = compute_errors(y_sim, u_sim, x_sim, v_sim, ref)

    # Pointwise comparison on Ghia table rows
    u_ref = interpolate_ghia(ref, y_sim, axis='u')
    v_ref = interpolate_ghia(ref, x_sim, axis='v')

    du_pct = np.abs(u_sim - u_ref) * 100.0  # already normalized by U_lid
    dv_pct = np.abs(v_sim - v_ref) * 100.0

    max_err = float(max(du_pct.max(), dv_pct.max()))
    passed = max_err <= TOL_PCT

    return {
        "l2_u": errors["l2_u"],
        "l2_v": errors["l2_v"],
        "max_u": errors["max_u"],
        "max_v": errors["max_v"],
        "max_abs_err_pct": max_err,
        "tolerance_pct": TOL_PCT,
        "passed": passed,
        "ref_source": "Ghia et al. (1982) JCP 48(3), Table III/IV, Re=5000",
    }


def main():
    ap = argparse.ArgumentParser(description="CASE-CAVITY-RE5000: Lid-driven cavity Re=5000")
    ap.add_argument("--n", type=int, nargs="+", default=[129, 257],
                    help="Grid resolutions (odd numbers for centerline)")
    ap.add_argument("--steps", type=int, default=100000, help="Max simulation steps")
    ap.add_argument("--u-lid", type=float, default=U_LID, help="Lid velocity in lattice units")
    ap.add_argument("--out", type=str, default="result.json", help="Output JSON file")
    ap.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = ap.parse_args()

    payload = {
        "case": "cavity_re5000_ghia",
        "description": "顶盖方腔 Re=5000，D2Q9 BGK，Ghia(1982) Table III/IV 验证",
        "lattice": "D2Q9",
        "collision": "BGK",
        "boundary": "bounce-back (walls) + equilibrium (lid)",
        "reference": "Ghia et al. (1982) JCP 48(3), Table III/IV, Re=5000",
        "tolerance_pct": TOL_PCT,
        "environment": {
            "tensorlbm_src": str(TENSORLBM_SRC),
            "tensorlbm_file": tensorlbm.__file__,
        },
    }

    grids = {}
    t0 = time.time()
    for n in args.n:
        result = run_single(n, re=RE, u_lid=args.u_lid, max_steps=args.steps, verbose=not args.quiet)
        comparison = compare_with_ghia(result)
        grids[str(n)] = {**result, "comparison": comparison}
        status = "PASS" if comparison["passed"] else "FAIL"
        print(f"[done] n={n} err={comparison['max_abs_err_pct']:.2f}% {status} t={result['elapsed_s']:.0f}s", flush=True)

    payload["grids"] = grids
    payload["wall_clock_s"] = round(time.time() - t0, 1)

    # Determine overall pass
    all_passed = all(g["comparison"]["passed"] for g in grids.values())
    payload["overall_pass"] = all_passed

    out_path = Path(args.out)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"DONE -> {out_path}", flush=True)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
