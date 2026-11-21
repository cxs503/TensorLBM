#!/usr/bin/env python3
"""CASE-CAVITY-RE5000: lid-driven square cavity at Re = 5000 (D2Q9, RLBM).

Physics
-------
Square cavity ``L x L``, no-slip bottom/left/right walls, top lid moving in
+x with ``U_lid = 0.1`` lattice units.  ``Re = U_lid * L / nu`` with ``L =
n - 1`` lattice units, hence ``tau = 3 nu + 1/2`` (0.5019 at n = 129).

Collision is regularized BGK (RLBM, Latt & Chopard 2006, Phys. Rev. E 71,
016708): the non-equilibrium part is projected onto the second-order
Hermite subspace, which damps the ghost modes that make plain BGK marginal at
``tau -> 0.5``.  Walls use half-way link bounce-back with the Ladd (1994)
momentum-transfer correction on the moving lid.

Reference data discipline
-------------------------
The pointwise gate needs a provenance-bearing transcription of Ghia, Ghia &
Shin (1982) JCP 48, 387-411, Table III/IV (Re = 5000).  That table cannot be
obtained in this (offline) environment -- the earlier ``GHIA_RE5000`` constant
shipped in ``reference_data.py`` was shown to be fabricated, see
``docs/audit_m15543.md``.  Therefore:

* with ``--ref-json PATH`` the run performs the pointwise ref-vs-got gate and
  records the file's sha256 in the artefact (provenance bound);
* without it the run reports ``run_status = "withheld"`` and **never** claims a
  pass.  A converged, finite solution plus its convergence history is still
  reported, which is what this script can honestly prove offline.

Exit codes: 0 pass, 2 divergence/non-finite, 3 withheld (no reference),
4 reference supplied but out of tolerance.

Usage
-----
    python cases/cavity_re5000/run.py --n 65 --steps 60000
    python cases/cavity_re5000/run.py --n 129 --steps 150000 \\
        --ref-json /path/to/ghia_re5000_tableIII_IV.json --out result.json
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

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
    load_reference_json,
    GhiaReference,
)
from tensorlbm.verification.lid_driven_cavity.verification import compute_errors

TOL_PCT = 8.0
RE = 5000.0
U_LID = 0.1


def run_single(n, re=RE, u_lid=U_LID, max_steps=150000, collision="trt", verbose=True,
               monitor_every=5000, resid_tol=1e-7, min_steps=20000, ramp_steps=2000):
    """Run one cavity simulation to (near) steady state.

    Returns a dict with the centreline profiles, the convergence history and a
    machine-readable ``status`` (``"ok"`` / ``"diverged"``).
    """
    cfg = CavityConfig(n=n, re=re, u_lid=u_lid, max_steps=max_steps,
                       use_trt=(collision == "trt"),
                       use_rlbm=(collision != "bgk"),
                       ramp_steps=ramp_steps)
    solver = CavitySolver(cfg)
    fluid = solver.fluid

    if verbose:
        extra = f" tau_minus={cfg.tau_minus:.3f}" if collision == "trt" else ""
        print(f"[run] n={n} re={re} u_lid={u_lid} tau={cfg.tau:.5f} "
              f"collision={collision}{extra}", flush=True)

    t0 = time.time()
    _, ux_prev, uy_prev = solver._macroscopic(solver.f)
    history = []
    converged = False
    status = "ok"
    steps_run = 0

    for step in range(1, max_steps + 1):
        solver.step()
        steps_run = step
        if step % monitor_every == 0:
            if not bool(torch.all(torch.isfinite(solver.f))):
                status = "diverged"
                break
            _, ux, uy = solver._macroscopic(solver.f)
            resid = float(torch.max(torch.abs(ux[fluid] - ux_prev[fluid]).max(),
                                    torch.abs(uy[fluid] - uy_prev[fluid]).max()))
            history.append({"step": step, "resid": resid})
            if verbose:
                print(f"  step={step} resid={resid:.3e}", flush=True)
            ux_prev, uy_prev = ux.clone(), uy.clone()
            if resid < resid_tol and step >= min_steps:
                converged = True
                break

    elapsed = time.time() - t0
    if status == "ok" and not bool(torch.all(torch.isfinite(solver.f))):
        status = "diverged"

    y, u, x, v = solver.get_profiles()
    if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
        status = "diverged"

    if status == "diverged":
        return {
            "n": n, "re": re, "u_lid": u_lid, "tau": cfg.tau,
            "collision": collision, "status": "diverged",
            "steps_run": steps_run, "elapsed_s": round(elapsed, 1),
            "resid_history": history,
        }

    u_min_idx = int(np.argmin(u))
    v_max_idx = int(np.argmax(v))
    rho, _, _ = solver._macroscopic(solver.f)

    return {
        "n": n,
        "re": re,
        "u_lid": u_lid,
        "tau": cfg.tau,
        "collision": collision,
        "ramp_steps": cfg.ramp_steps,
        "status": "ok",
        "steps_run": steps_run,
        "max_steps": max_steps,
        "elapsed_s": round(elapsed, 1),
        "converged": converged,
        "resid_tol": resid_tol,
        "monitor_every": monitor_every,
        "resid_history": history,
        "final_resid": history[-1]["resid"] if history else None,
        "y_centerline": y.tolist(),
        "u_centerline": u.tolist(),
        "x_centerline": x.tolist(),
        "v_centerline": v.tolist(),
        "u_min": float(u[u_min_idx]),
        "u_min_y": float(y[u_min_idx]),
        "v_max": float(v[v_max_idx]),
        "v_max_x": float(x[v_max_idx]),
        "rho_min": float(rho.min()),
        "rho_max": float(rho.max()),
    }


def compare_with_reference(result, ref: GhiaReference, tol_pct=TOL_PCT):
    """Row-by-row comparison against a provenance-bearing Ghia reference.

    Both ``u`` and ``v`` are normalised by the lid velocity, so the reported
    ``abs_err_pct`` is ``100 * |sim - ref|`` in percent of ``U_lid`` (i.e.
    percentage points of the dimensionless velocity), and ``rel_err_pct`` is
    the error relative to the local reference magnitude.
    """
    y_sim = np.asarray(result["y_centerline"])
    u_sim = np.asarray(result["u_centerline"])
    x_sim = np.asarray(result["x_centerline"])
    v_sim = np.asarray(result["v_centerline"])

    errors = compute_errors(y_sim, u_sim, x_sim, v_sim, ref)
    u_ref = np.interp(y_sim, ref.y, ref.u)
    v_ref = np.interp(x_sim, ref.x, ref.v)

    du = np.abs(u_sim - u_ref) * 100.0
    dv = np.abs(v_sim - v_ref) * 100.0
    max_err = float(max(du.max(), dv.max()))

    rows = []
    for i in range(len(y_sim)):
        rows.append({
            "coord_u": float(y_sim[i]),
            "u_sim": float(u_sim[i]),
            "u_ref": float(u_ref[i]),
            "u_abs_err_pct": float(du[i]),
        })
    for i in range(len(x_sim)):
        rows.append({
            "coord_v": float(x_sim[i]),
            "v_sim": float(v_sim[i]),
            "v_ref": float(v_ref[i]),
            "v_abs_err_pct": float(dv[i]),
        })

    return {
        "status": "PASS" if max_err <= tol_pct else "FAIL",
        "l2_u": errors["l2_u"],
        "l2_v": errors["l2_v"],
        "max_u": errors["max_u"],
        "max_v": errors["max_v"],
        "max_abs_err_pct": max_err,
        "tolerance_pct": tol_pct,
        "error_unit": "percent of U_lid (dimensionless velocity points)",
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser(description="CASE-CAVITY-RE5000: lid-driven cavity")
    ap.add_argument("--n", type=int, nargs="+", default=[129],
                    help="Grid resolutions (odd numbers hit the centreline node)")
    ap.add_argument("--re", type=float, default=RE, help="Reynolds number")
    ap.add_argument("--steps", type=int, default=150000, help="Max simulation steps")
    ap.add_argument("--u-lid", type=float, default=U_LID, help="Lid velocity (lattice units)")
    ap.add_argument("--out", type=str, default="result.json", help="Output JSON file")
    ap.add_argument("--ref-json", type=str, default=None,
                    help="Provenance-bearing Ghia reference table (JSON); "
                         "without it the pointwise gate is WITHHELD")
    ap.add_argument("--monitor-every", type=int, default=5000)
    ap.add_argument("--resid-tol", type=float, default=1e-7)
    ap.add_argument("--min-steps", type=int, default=20000)
    ap.add_argument("--nproc", type=int, default=2, help="torch CPU threads")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--ramp-steps", type=int, default=2000,
                    help="Linear lid-velocity ramp (0 = impulsive start, which "
                         "is unstable at Re=5000)")
    ap.add_argument("--collision", choices=["bgk", "rlbm", "trt"], default="trt",
                    help="Collision model. trt (default) uses the magic "
                         "parameter 3/16; it is the only model that survives "
                         "tau -> 1/2 here (Re=5000, n=129 -> tau=0.5077)")
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.nproc))
    collision = args.collision

    ref = None
    ref_provenance = {"status": "WITHHELD",
                      "reason": "no --ref-json supplied; Ghia (1982) Table III/IV "
                                "(Re=5000) has no provenance-bearing transcription "
                                "in this repository, see docs/audit_m15543.md"}
    if args.ref_json:
        ref = load_reference_json(args.ref_json)
        raw = Path(args.ref_json).read_bytes()
        ref_provenance = {
            "status": "BOUND",
            "path": str(Path(args.ref_json).resolve()),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "re": ref.re,
            "n_points_u": int(len(ref.u)),
            "n_points_v": int(len(ref.v)),
            "source": ref.source,
        }

    collision_desc = {
        "trt": "TRT (two-relaxation-time, magic parameter Lambda = 3/16)",
        "rlbm": "RLBM (regularized BGK, Latt & Chopard 2006)",
        "bgk": "BGK (single relaxation time)",
    }[collision]
    payload = {
        "case": "cavity_re5000",
        "description": "顶盖方腔 Re=5000，D2Q9，半程链接反弹 + Ladd 顶盖动量修正",
        "lattice": "D2Q9",
        "collision": collision_desc,
        "boundary": "half-way link bounce-back (walls) + Ladd momentum transfer (lid)",
        "reference": "Ghia et al. (1982) JCP 48(3) 387-411, Table III/IV (Re=5000)",
        "reference_provenance": ref_provenance,
        "tolerance_pct": TOL_PCT,
        "environment": {
            "tensorlbm_src": str(TENSORLBM_SRC),
            "tensorlbm_file": tensorlbm.__file__,
            "torch_version": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "dtype": "float64",
        },
    }

    grids = {}
    t0 = time.time()
    all_ok = True
    for n in args.n:
        result = run_single(n, re=args.re, u_lid=args.u_lid, max_steps=args.steps,
                            verbose=not args.quiet,
                            monitor_every=args.monitor_every,
                            resid_tol=args.resid_tol, min_steps=args.min_steps,
                            ramp_steps=args.ramp_steps)
        if result["status"] != "ok":
            all_ok = False
            grids[str(n)] = result
            print(f"[done] n={n} DIVERGED at step {result['steps_run']}", flush=True)
            continue
        if ref is not None:
            result["comparison"] = compare_with_reference(result, ref)
            status = result["comparison"]["status"]
            err = result["comparison"]["max_abs_err_pct"]
            print(f"[done] n={n} err={err:.2f}% {status} t={result['elapsed_s']:.0f}s",
                  flush=True)
        else:
            result["comparison"] = {"status": "WITHHELD",
                                    "reason": ref_provenance["reason"]}
            status = "WITHHELD"
            print(f"[done] n={n} u_min={result['u_min']:.4f}@{result['u_min_y']:.3f} "
                  f"t={result['elapsed_s']:.0f}s comparison=WITHHELD", flush=True)
        grids[str(n)] = result

    payload["grids"] = grids
    payload["wall_clock_s"] = round(time.time() - t0, 1)

    comparisons = [g.get("comparison", {}) for g in grids.values()]
    if not all_ok or any(c.get("status") == "FAIL" for c in comparisons):
        payload["run_status"] = "failed"
        rc = 2 if not all_ok else 4
    elif ref is None:
        payload["run_status"] = "withheld"
        rc = 3
    else:
        payload["run_status"] = "ok"
        rc = 0
    payload["overall_pass"] = (payload["run_status"] == "ok")

    out_path = Path(args.out)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"DONE -> {out_path} run_status={payload['run_status']}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())