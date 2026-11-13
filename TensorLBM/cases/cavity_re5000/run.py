#!/usr/bin/env python3
"""CASE-CAVITY-RE5000: Lid-driven square cavity at Re=5000.

Reference: Ghia, U., Ghia, K.N., & Shin, C.T. (1982).
"High-Re solutions for incompressible flow using the Navier-Stokes equations
and a multigrid method." Journal of Computational Physics, 48(3), 387-411.
- Table III: u/U along vertical centerline x/L=0.5, Re=5000
- Table IV: v/U along horizontal centerline y/L=0.5, Re=5000

Tolerance: <8% pointwise error on Ghia table rows.

Key fix: Uses regularized BGK (RLBM, Latt & Chopard 2006) collision instead of
plain BGK. At Re=5000, tau ~ 0.502 is near the BGK stability limit (tau > 0.5).
RLBM filters non-hydrodynamic modes, enabling stable convergence.

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


def run_single(n, re=RE, u_lid=U_LID, max_steps=100000, use_rlbm=True, verbose=True):
    """Run a single cavity simulation and return results."""
    cfg = CavityConfig(n=n, re=re, u_lid=u_lid, max_steps=max_steps, use_rlbm=use_rlbm)
    solver = CavitySolver(cfg)

    if verbose:
        method = "RLBM (regularized BGK)" if use_rlbm else "BGK"
        print(f"[run] n={n} re={re} u_lid={u_lid} tau={cfg.tau:.5f} method={method}", flush=True)

    t0 = time.time()
    ux_prev, uy_prev = None, None

    for step in range(max_steps):
        solver.step()
        if verbose and step % 10000 == 0 and step > 0:
            rho, ux, uy = solver._macroscopic(solver.f)
            if ux_prev is not None:
                fluid = ~solver.solid
                resid = float(torch.max(
                    torch.abs(ux[fluid] - ux_prev[fluid]).max(),
                    torch.abs(uy[fluid] - uy_prev[fluid]).max(),
                ))
                print(f"  step={step} resid={resid:.2e}", flush=True)
            ux_prev = ux.clone()
            uy_prev = uy.clone()

    elapsed = time.time() - t0
    y, u, x, v = solver.get_profiles()

    # Find extrema
    u_min_idx = np.argmin(u)
    v_max_idx = np.argmax(v)

    # Final residual
    rho, ux_final, uy_final = solver._macroscopic(solver.f)
    if ux_prev is not None:
        fluid = ~solver.solid
        final_resid = float(torch.max(
            torch.abs(ux_final[fluid] - ux_prev[fluid]).max(),
            torch.abs(uy_final[fluid] - uy_prev[fluid]).max(),
        ))
    else:
        final_resid = None

    return {
        "n": n,
        "re": re,
        "u_lid": u_lid,
        "tau": cfg.tau,
        "use_rlbm": use_rlbm,
        "steps": max_steps,
        "elapsed_s": round(elapsed, 1),
        "final_resid": final_resid,
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

    du_pct = np.abs(u_sim - u_ref) * 100.0
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
    import torch
    ap = argparse.ArgumentParser(description="CASE-CAVITY-RE5000: Lid-driven cavity Re=5000")
    ap.add_argument("--n", type=int, nargs="+", default=[129],
                    help="Grid resolutions (odd numbers for centerline)")
    ap.add_argument("--steps", type=int, default=100000, help="Max simulation steps")
    ap.add_argument("--u-lid", type=float, default=U_LID, help="Lid velocity in lattice units")
    ap.add_argument("--out", type=str, default="result.json", help="Output JSON file")
    ap.add_argument("--quiet", action="store_true", help="Suppress progress output")
    ap.add_argument("--no-rlbm", action="store_true", help="Use plain BGK instead of RLBM")
    args = ap.parse_args()

    use_rlbm = not args.no_rlbm

    payload = {
        "case": "cavity_re5000_ghia",
        "description": "顶盖方腔 Re=5000，D2Q9 RLBM（正则化BGK），Ghia(1982) Table III/IV 验证",
        "lattice": "D2Q9",
        "collision": "RLBM (regularized BGK, Latt & Chopard 2006)" if use_rlbm else "BGK",
        "boundary": "bounce-back (stationary walls) + Zou-He (moving lid)",
        "reference": "Ghia et al. (1982) JCP 48(3), Table III/IV, Re=5000",
        "tolerance_pct": TOL_PCT,
        "key_fix": "RLBM collision filters non-hydrodynamic modes, enabling stability at tau~0.502",
        "environment": {
            "tensorlbm_src": str(TENSORLBM_SRC),
            "tensorlbm_file": tensorlbm.__file__,
            "torch_version": torch.__version__,
            "dtype": "float64",
        },
    }

    grids = {}
    t0 = time.time()
    all_finite = True
    for n in args.n:
        try:
            result = run_single(n, re=RE, u_lid=args.u_lid, max_steps=args.steps,
                               use_rlbm=use_rlbm, verbose=not args.quiet)
            comparison = compare_with_ghia(result)
            grids[str(n)] = {**result, "comparison": comparison}
            status = "PASS" if comparison["passed"] else "FAIL"
            print(f"[done] n={n} err={comparison['max_abs_err_pct']:.2f}% {status} t={result['elapsed_s']:.0f}s", flush=True)
        except Exception as e:
            print(f"[error] n={n}: {e}", flush=True)
            grids[str(n)] = {"n": n, "status": "error", "error": str(e)}
            all_finite = False

    payload["grids"] = grids
    payload["wall_clock_s"] = round(time.time() - t0, 1)
    payload["all_finite"] = all_finite

    # Determine overall pass
    all_passed = all(
        g.get("comparison", {}).get("passed", False)
        for g in grids.values()
        if "comparison" in g
    )
    payload["overall_pass"] = all_passed and all_finite

    out_path = Path(args.out)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"DONE -> {out_path}", flush=True)

    return 0 if (all_passed and all_finite) else 1


if __name__ == "__main__":
    sys.exit(main())
