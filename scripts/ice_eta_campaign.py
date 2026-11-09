"""IN13-W4-N02 campaign driver: effective viscosity of ice-crystal suspensions.

Runs, in order

1. the single-sphere drag pre-gate (Stokes + GCI, three grids),
2. the concentration sweep (dilute -> semi-dilute -> packed) with the LBM
   shear cell,
3. the strain-rate sweep at fixed ``phi`` (linearity + inertial crossover),
4. the shape / polydispersity sensitivity cases of the applicability map,
5. the closure fit + contract assembly + figures + ``calibration.json``.

Everything is JSON-serialised under ``--out``; nothing is hand-copied into the
report.  The bash-level invocation is::

    python scripts/ice_eta_campaign.py --out runs/ice_eta_campaign --jobs 20

The module-level defaults are sized for a 128-core node (20 concurrent cells,
6 torch threads each).  All cases are independent, so the campaign is
embarrassingly parallel.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from tensorlbm.ice_suspension_rheology import calibrate, closures, shapes
from tensorlbm.ice_suspension_rheology.calibrate import (
    _dispatch,
    analyse,
    drag_gate,
    make_figures,
    write_report,
)
from tensorlbm.ice_suspension_rheology.couette import ShearCell, ShearCellConfig


def case_set(full: bool):
    """The (name, kwargs) case list of the campaign."""
    base = dict(
        nx=32, ny=32, nz=45, radius=4.0, nu=0.1, u_wall=0.02,
        n_steps=1700, wall_margin=7,
    )
    phi = ([0.01, 0.02, 0.03, 0.05] if not full else
           [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30])
    seeds = [1, 2] if not full else [1, 2]
    cases = [dict(**base, phi=0.0, seed=0, tag="pure")]
    for p in phi:
        for s in seeds:
            if p > 0.10 and s != 1:      # one seed for the expensive dense cases
                continue
            cases.append(dict(**base, phi=p, seed=s, tag=f"phi{p:.3f}_s{s}"))
    return cases


def run_rate_sweep(base: dict, jobs: int, threads=None):
    u_walls = [0.005, 0.01, 0.02, 0.035, 0.05]
    cases = [dict(**{**base, "u_wall": u}, phi=0.05, seed=1, tag=f"rate_u{u:.3f}")
             for u in u_walls]
    return u_walls, _dispatch(cases, jobs=jobs, threads=threads)


def shape_cases(base: dict, jobs: int, threads=None):
    """Applicability map: faceted near-spheres and polydisperse populations."""
    cases = [
        dict(**base, phi=0.05, seed=3, shape="faceted", tag="faceted_phi0.05",
             shape_kwargs={"n_faces": 80}),
        dict(**base, phi=0.05, seed=4, shape="sphere", polydisperse=0.25,
             tag="polydisperse_phi0.05"),
    ]
    return _dispatch(cases, jobs=jobs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/ice_eta_campaign")
    ap.add_argument("--jobs", type=int, default=20)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--skip-gate", action="store_true")
    ap.add_argument("--threads", type=int, default=None,
                    help="torch threads per case (default: cpus / jobs)")
    ap.add_argument("--gate-only", action="store_true")
    ap.add_argument("--small", action="store_true",
                    help="tiny cell for a pipeline smoke test")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "campaign.log"), "a")

    def say(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    t0 = time.time()
    gate = None
    if not args.skip_gate:
        say("drag pre-gate (D=8,12,16, Re=0.1)")
        gate = drag_gate(diameters=(8, 12, 16), re_p=0.1, steps=3000,
                         domain_factor=4.0, jobs=3)
        say("gate: " + json.dumps({k: gate[k] for k in
                                   ("max_err_pct", "gci", "pass_3pct", "pass_gci_5pct")}))
        json.dump(gate, open(os.path.join(args.out, "drag_gate.json"), "w"), indent=2)
    if args.gate_only:
        return 0

    cases = case_set(args.full)
    if args.small:
        small = dict(nx=16, ny=16, nz=21, radius=3.0, nu=0.1, u_wall=0.01,
                     n_steps=400, wall_margin=5)
        for c in cases:
            c.update(small)
        cases = [c for c in cases if c["phi"] in (0.0, 0.02, 0.05, 0.10)]
    base = {k: v for k, v in cases[0].items() if k not in ("phi", "seed", "tag")}
    say(f"concentration sweep: {len(cases)} cases")
    raw = _dispatch(cases, jobs=args.jobs, threads=args.threads,
                    workdir=os.path.join(args.out, "cases"))
    for c, r in zip(cases, raw):
        r["tag"] = c["tag"]
    json.dump(raw, open(os.path.join(args.out, "sweep_raw.json"), "w"),
              indent=2, default=float)
    pure = next(r for r in raw if r.get("phi_target", 1) == 0)
    sweep = [r for r in raw if r.get("phi_target", 0) > 0]
    say("sweep done, pure eta_eff=%.6f r2=%.5f"
        % (pure["eta_eff"], pure["profile_r2"]))

    say("strain-rate sweep at phi=0.05")
    u_walls, rate = run_rate_sweep(base, args.jobs, args.threads)
    json.dump(rate, open(os.path.join(args.out, "rate_raw.json"), "w"),
              indent=2, default=float)

    say("shape / polydispersity cases")
    shape_raw = shape_cases(base, args.jobs, args.threads)
    json.dump(shape_raw, open(os.path.join(args.out, "shape_raw.json"), "w"),
              indent=2, default=float)

    ana = analyse(sweep, pure, gate=gate, rate=rate)
    # shape cases measured against the same pure control
    tau0, g0 = pure["tau_corrected"], pure["gamma_dot_measured"]
    ana["shape_sensitivity"] = [
        {
            "tag": r.get("tag"),
            "shape": r.get("shape"),
            "polydisperse": r.get("polydisperse"),
            "phi": r["phi_lattice"],
            "eta_r": (r["tau_corrected"] / tau0) * (g0 / r["gamma_dot_measured"]),
            "profile_r2": r["profile_r2"],
        }
        for r in shape_raw if "error" not in r
    ]
    ana["meta"] = {
        "campaign": "IN13-W4-N02",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cases": len(cases) + len(rate) + len(shape_raw) + (3 if gate else 0),
        "base_config": base,
        "rate_u_walls": u_walls,
        "wall_clock_s": time.time() - t0,
        "full": args.full,
    }
    figs = make_figures(ana, os.path.join(args.out, "figures"))
    path = write_report(ana, args.out, figs, ana["meta"])
    say("report written: " + path)
    summary = {
        "dilute_fit": ana.get("dilute_fit"),
        "batchelor": {k: v for k, v in (ana.get("batchelor") or {}).items() if k != "rows"},
        "kd": {k: v for k, v in (ana.get("krieger_dougherty") or {}).items() if k != "fit"},
        "pure_control": ana.get("pure_control"),
        "rate": {k: v for k, v in (ana.get("rate_sweep") or {}).items() if k != "points"},
        "contract": ana.get("contract"),
        "shape": ana.get("shape_sensitivity"),
    }
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"),
              indent=2, default=float)
    say(json.dumps(summary, indent=1, default=float)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())