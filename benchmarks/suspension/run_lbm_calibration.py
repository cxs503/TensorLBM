"""P46W3-N03 — LBM calibration driver for the suspension effective-viscosity closure.

Two measurement groups (see ``P46W3_N03_PREDECLARED_DESIGN.md``):

``drag``  Stokes drag on the *same discretised sphere* in the *same cell* that the
          viscosity sweep uses, from which the hydrodynamic radius is inverted,
          ``a_h = F / (6 pi mu U K_HS)``.
``shear`` the moving-wall Couette shear cell of
          :mod:`tensorlbm.ice_suspension_rheology.couette`, at a resolution ladder
          ``r in {3.5, 5.0, 6.5}`` x a volume-fraction ladder, 2 microstructure
          seeds each, plus the pure-fluid negative control.

Everything is executed through the existing, already-verified case runner
(``python -m tensorlbm.ice_suspension_rheology.case_runner``) in one subprocess
per case; this script only declares the design and collects the raw records.

Usage (on the compute node / cluster)::

    python benchmarks/suspension/run_lbm_calibration.py --jobs 12 \
        --out benchmarks/suspension/artifacts/raw_cases.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

# ---------------------------------------------------------------------------
# predeclared design (fixed before any case was run)
# ---------------------------------------------------------------------------
CELL = dict(nx=48, ny=48, nz=57, nu=0.1, u_wall=0.02)
STEPS = dict(n_steps=2600, sample_every=25, wall_margin=7)

RADII = (3.5, 5.0, 6.5)                       # discretisation ladder (D = 7, 10, 13)
DILUTE_PHI = (0.005, 0.01, 0.02, 0.05)        # dilute sweep, all three radii
DENSE_PHI = (0.15, 0.25, 0.35, 0.45)          # concentrated sweep, r = 5.0
DENSE_RADIUS = 5.0
SEEDS = (1, 2)


def build_cases(overrides: dict | None = None) -> list[dict]:
    """Case table exactly as predeclared; ``overrides`` only for smoke tests."""
    ov = overrides or {}
    radii = tuple(ov.get("radii", RADII))
    seeds = tuple(ov.get("seeds", SEEDS))
    dilute_phi = tuple(ov.get("dilute_phi", DILUTE_PHI))
    dense_phi = tuple(ov.get("dense_phi", DENSE_PHI))
    dense_radius = float(ov.get("dense_radius", DENSE_RADIUS))
    cell = dict(CELL, **{k: v for k, v in ov.items() if k in CELL})
    steps = dict(STEPS, **{k: v for k, v in ov.items() if k in STEPS})
    cases: list[dict] = []
    # --- group 1: drag ladder (a_h / a_eff) -------------------------------
    for r in radii:
        cases.append({
            "mode": "drag",
            "tag": f"drag_r{r:g}",
            "spec": dict(box=cell["nx"], nz=cell["nz"], radius=r, re_p=0.1,
                         nu=cell["nu"], n_steps=steps["n_steps"]),
        })
    # --- group 2: pure-fluid negative control -----------------------------
    cases.append({
        "mode": "shear",
        "tag": "pure_control",
        "spec": dict(phi=0.0, seed=1, radius=dense_radius, **cell, **steps),
    })
    # --- group 3: dilute sweep, resolution ladder -------------------------
    for r in radii:
        for phi in dilute_phi:
            for s in seeds:
                cases.append({
                    "mode": "shear",
                    "tag": f"dilute_r{r:g}_phi{phi:g}_s{s}",
                    "spec": dict(phi=phi, seed=s, radius=r, **cell, **steps),
                })
    # --- group 4: concentrated sweep -------------------------------------
    for phi in dense_phi:
        for s in seeds:
            cases.append({
                "mode": "shear",
                "tag": f"dense_r{dense_radius:g}_phi{phi:g}_s{s}",
                "spec": dict(phi=phi, seed=s, radius=dense_radius, **cell, **steps),
            })
    return cases


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "artifacts", "raw_cases.json"))
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--only", default=None,
                    help="comma separated tag prefixes to run (default: all)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny cell/fewer steps -- pipeline check only, NOT the campaign")
    ap.add_argument("--steps", type=int, default=None, help="override n_steps")
    ap.add_argument("--radii", default=None, help="comma separated particle radii")
    ap.add_argument("--dilute-phi", default=None, help="comma separated dilute fractions")
    ap.add_argument("--dense-phi", default=None, help="comma separated dense fractions")
    ap.add_argument("--seeds", default=None, help="comma separated microstructure seeds")
    args = ap.parse_args(argv)

    from tensorlbm.ice_suspension_rheology.calibrate import _dispatch

    overrides = {}
    if args.steps:
        overrides["n_steps"] = int(args.steps)
    if args.radii:
        overrides["radii"] = tuple(float(x) for x in args.radii.split(","))
    if args.dilute_phi:
        overrides["dilute_phi"] = tuple(float(x) for x in args.dilute_phi.split(","))
    if args.dense_phi:
        overrides["dense_phi"] = tuple(float(x) for x in args.dense_phi.split(","))
    if args.seeds:
        overrides["seeds"] = tuple(int(x) for x in args.seeds.split(","))
    if args.smoke:
        overrides = dict(nx=24, ny=24, nz=25, n_steps=400, radii=(3.0, 3.5),
                         dilute_phi=(0.005, 0.01), dense_phi=(0.2,),
                         dense_radius=3.0)
        print("[smoke] reduced cell -- not a campaign result", file=sys.stderr)
    cases = build_cases(overrides)
    if args.only:
        keep = tuple(s.strip() for s in args.only.split(","))
        cases = [c for c in cases if c["tag"].startswith(keep)]
    if not cases:
        print("no cases selected", file=sys.stderr)
        return 2

    threads = args.threads or max(1, (os.cpu_count() or 4) // max(1, args.jobs))
    t0 = time.time()
    raw: list[dict] = []
    for mode in ("drag", "shear"):
        group = [c for c in cases if c["mode"] == mode]
        if not group:
            continue
        print(f"[run] {len(group)} {mode} cases, jobs={args.jobs}, threads={threads}",
              flush=True)
        recs = _dispatch([dict(c["spec"]) for c in group], mode=mode,
                         jobs=args.jobs, threads=threads)
        for c, r in zip(group, recs):
            r = dict(r)
            r["tag"] = c["tag"]
            r["mode"] = mode
            raw.append(r)
            print(f"[done] {c['tag']}: "
                  + ("error=" + str(r.get("error")) if "error" in r
                     else f"phi_lattice={r.get('phi_lattice'):.4f} "
                          f"profile_r2={r.get('profile_r2'):.5f} "
                          f"eta_eff={r.get('eta_eff'):.5f}"), flush=True)
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w") as fh:
                json.dump({"design": {"cell": {k: v for k, v in
                                 (dict(CELL, **{kk: vv for kk, vv in overrides.items() if kk in CELL})).items()},
                                      "steps": {k: v for k, v in
                                 (dict(STEPS, **{kk: vv for kk, vv in overrides.items() if kk in STEPS})).items()},
                                      "radii": list(overrides.get("radii", RADII)),
                                      "dilute_phi": list(overrides.get("dilute_phi", DILUTE_PHI)),
                                      "dense_phi": list(overrides.get("dense_phi", DENSE_PHI)),
                                      "dense_radius": float(overrides.get("dense_radius", DENSE_RADIUS)),
                                      "seeds": list(overrides.get("seeds", SEEDS)),
                                      "smoke": bool(args.smoke),
                                      "jobs": args.jobs, "threads": threads},
                           "elapsed_s": time.time() - t0,
                           "cases": raw}, fh, default=float, indent=1)
    print(f"[all done] {len(raw)} records in {time.time() - t0:.0f} s -> {args.out}")
    return 0 if all("error" not in r for r in raw) else 1


if __name__ == "__main__":
    raise SystemExit(main())