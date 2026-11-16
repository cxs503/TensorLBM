"""Round 7: single-tree A/B of the body-force normalisation for the NEBB driver.

Anchor: ``XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S``.

Round 6 diagnosed two definition defects and fixed them in
``tensorlbm.gpu_core_path`` (closure-aware driving width; physical velocity in
the diagnostics), but the evidence was a *two-tree* comparison (pre-fix
snapshot vs post-fix worktree).  A two-tree comparison cannot exclude tree
drift, so this harness re-establishes the claim inside one process:

=========================  ===========================================
arm                        body-force normalisation
=========================  ===========================================
``closure_width``          ``a = 8 nu u_max / H_wall**2``,
                           ``H_wall = ny-1`` (nebb) / ``ny-2`` (halfway)
``legacy_halfway_width``   ``a = 8 nu u_max / (ny-2)**2`` for either closure
                           (the pre-fix formula; emulated exactly on the
                           fixed tree by ``u_max_target *= (H_wall/(ny-2))**2``)
=========================  ===========================================

Both arms use the identical solver, grid, viscosity and declared target
``u_max = 0.05``; the only changed variable is the normalisation.  The metric
is ``ratio_vs_declared = u_max_phys / 0.05``, where ``u_max_phys`` is the raw
moment shifted by ``a/2`` outside the solver (Guo, Zheng & Shi 2002), so it does
not presuppose the diagnostic fix.

Predictions were declared in ``predeclared_r7.json`` *before* the run:

* P1 the fixed normalisation gives ``ratio_vs_declared == 1`` on both closures;
* P2 the legacy normalisation gives ``((ny-1)/(ny-2))**2`` on the nebb closure
  (1.26531 at ny=10, 1.12886 at ny=18) -- i.e. the metric *can* see the defect;
* P3 on the halfway closure the two normalisations coincide (width control);
* P4 the physical-velocity error against the closed-form parabola is round-off;
* P5 the public diagnostic reports the physical velocity (shift ``a/2``).

Usage
-----
    python run_xval_r7_drive_width_ab.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_xval_r6_driver_consistency import rel_err_pointwise, run_core_path  # noqa: E402

U_MAX = 0.05            # declared centre-line target, identical in every arm
NU = 0.2                # tau = 1.1
TOL_STEADY = 1e-12
CHECK_EVERY = 200
MAX_STEPS = 20_000   # 1200-5000 steps reach 1e-12; the cap keeps a non-converging arm bounded


def run_arm(ny: int, wall_bc: str, drive: str, *, u_max: float = U_MAX) -> dict:
    h_wall = float(ny - 1) if wall_bc == "nebb" else float(ny - 2)
    h_legacy = float(ny - 2)
    scale = (h_wall / h_legacy) ** 2
    cfg_target = u_max if drive == "closure_width" else u_max * scale
    row = run_core_path(ny, wall_bc, u_max=cfg_target, max_steps=MAX_STEPS)
    a = row["body_force"]
    row.update({
        "drive": drive,
        "declared_u_max": u_max,
        "cfg_u_max_target": cfg_target,
        "drive_width": h_wall if drive == "closure_width" else h_legacy,
        "ratio_vs_declared": row["u_max_phys"] / u_max,
        # the solver's own steady-state residual records how converged the cell is
        "converged": bool(row["steady_residual"] < TOL_STEADY),
        # P5: the public diagnostic must be the physical velocity, i.e. shifted
        # by a/2 from the raw moment
        "diagnostic_shift_ratio": (row["public_profile_equals_raw"] / (0.5 * a)
                                   if a else float("nan")),
    })
    return row


def check(checks: list, cid: str, claim: str, metric: str, value, ok: bool,
          predicted=None) -> None:
    checks.append({
        "id": cid, "claim": claim, "metric": metric, "value": value,
        "predicted": predicted, "pass": bool(ok),
    })


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "artifacts", "r7"))
    ap.add_argument("--nys", default="10,18")
    ap.add_argument("--arms", default="",
                    help="comma separated subset 'wall_bc:ny:drive'; default = all. "
                         "Arms already present in <out>/drive_width_ab.json are kept, so "
                         "a partially-queued campaign can be completed incrementally.")
    ap.add_argument("--checks-only", action="store_true",
                    help="recompute the predeclared checks from the stored arms")
    args = ap.parse_args(argv)
    nys = [int(s) for s in args.nys.split(",")]
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "drive_width_ab.json")

    arms = []
    if os.path.exists(path):
        arms = json.load(open(path))["arms"]

    def key(a):
        return (a["wall_bc"], int(a["ny"]), a["drive"])

    want = []
    if args.arms:
        for spec in args.arms.split(","):
            wall_bc, ny, drive = spec.split(":")
            want.append((wall_bc, int(ny), drive))
    else:
        for wall_bc in ("nebb", "halfway"):
            for ny in nys:
                for drive in ("closure_width", "legacy_halfway_width"):
                    want.append((wall_bc, ny, drive))

    if not args.checks_only:
        have = {key(a) for a in arms}
        for wall_bc, ny, drive in want:
            if (wall_bc, ny, drive) in have:
                print(f"[skip] {wall_bc}/{ny}/{drive} already in {path}", flush=True)
                continue
            row = run_arm(ny, wall_bc, drive)
            arms.append(row)
            print(f"[{wall_bc:8s} ny={ny:3d} {drive:20s}] "
                  f"ratio_vs_declared={row['ratio_vs_declared']:.6f} "
                  f"err_phys={row['err_phys_vs_analytic']:.3e} "
                  f"err_pt={row['err_phys_pointwise']:.3e} "
                  f"diag_shift/a2={row['diagnostic_shift_ratio']:.6f} "
                  f"steps={row['steps_used']} res={row['steady_residual']:.1e}",
                  flush=True)
            # checkpoint after every arm: a queued campaign can be resumed
            json.dump({"anchor": "XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S",
                       "round": 7, "declared_u_max": U_MAX, "nu": NU,
                       "tol_steady": TOL_STEADY, "check_every": CHECK_EVERY,
                       "predeclared": "predeclared_r7.json",
                       "arms": arms, "checks": [], "ok": None, "complete": False},
                      open(path, "w"), indent=2)

    for a in arms:
        print(f"[arm] {a['wall_bc']:8s} ny={a['ny']:3d} {a['drive']:20s} "
              f"ratio={a['ratio_vs_declared']:.6f}", flush=True)

    def get(wall_bc, ny, drive):
        return next(a for a in arms
                    if a["wall_bc"] == wall_bc and a["ny"] == ny and a["drive"] == drive)

    present_nys = sorted({int(a["ny"]) for a in arms})
    complete = all(any(a["wall_bc"] == w and int(a["ny"]) == ny and a["drive"] == d
                       for a in arms)
                   for w in ("nebb", "halfway") for ny in nys
                   for d in ("closure_width", "legacy_halfway_width"))

    checks: list = []
    ok_all = True

    for wall_bc in ("nebb", "halfway"):
        for ny in present_nys:
            r = get(wall_bc, ny, "closure_width")
            ok = abs(r["ratio_vs_declared"] - 1.0) < 0.01
            ok_all &= ok
            check(checks, "P1_fix", "fixed normalisation delivers the declared u_max",
                  f"ratio_vs_declared[{wall_bc},ny={ny}]", r["ratio_vs_declared"], ok, 1.0)
    for ny in present_nys:
        r = get("nebb", ny, "legacy_halfway_width")
        pred = ((ny - 1) / (ny - 2)) ** 2
        ok = abs(r["ratio_vs_declared"] - pred) < 0.01 * pred
        ok_all &= ok
        check(checks, "P2_defect_detectable",
              "legacy width recovers the pre-fix overshoot (metric is sensitive)",
              f"ratio_vs_declared[nebb,legacy,ny={ny}]", r["ratio_vs_declared"], ok, pred)
    for ny in present_nys:
        a = get("halfway", ny, "closure_width")["ratio_vs_declared"]
        b = get("halfway", ny, "legacy_halfway_width")["ratio_vs_declared"]
        ok = abs(a - b) < 1e-9
        ok_all &= ok
        check(checks, "P3_width_control",
              "halfway: legacy and closure normalisation coincide",
              f"|ratio_closure - ratio_legacy|[halfway,ny={ny}]", abs(a - b), ok, 0.0)
    for wall_bc in ("nebb", "halfway"):
        for ny in present_nys:
            r = get(wall_bc, ny, "closure_width")
            ok = r["err_phys_pointwise"] < 1e-10
            ok_all &= ok
            check(checks, "P4_exactness",
                  "physical velocity reproduces the closed-form parabola to round-off",
                  f"err_phys_pointwise[{wall_bc},ny={ny}]", r["err_phys_pointwise"], ok)
    for wall_bc in ("nebb", "halfway"):
        for ny in present_nys:
            r = get(wall_bc, ny, "closure_width")
            ok = 0.99 <= r["diagnostic_shift_ratio"] <= 1.01
            ok_all &= ok
            check(checks, "P5_diagnostic",
                  "public profile == raw moment + a/2",
                  f"diagnostic_shift_ratio[{wall_bc},ny={ny}]",
                  r["diagnostic_shift_ratio"], ok, 1.0)

    payload = {
        "anchor": "XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S",
        "round": 7,
        "declared_u_max": U_MAX,
        "nu": NU,
        "tol_steady": TOL_STEADY,
        "check_every": CHECK_EVERY,
        "predeclared": "predeclared_r7.json",
        "arms": arms,
        "checks": checks,
        "complete": bool(complete),
        "requested_nys": nys,
        "ok": bool(ok_all and complete) if complete else None,
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    failed = [c["id"] for c in checks if not c["pass"]]
    print(("ALL PREDECLARED CHECKS PASS" if (ok_all and complete)
           else f"FAILED: {sorted(set(failed))} complete={complete}"), "->", path)
    return 0 if (ok_all and complete) else 1


if __name__ == "__main__":
    sys.exit(main())
