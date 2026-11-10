#!/usr/bin/env python3
"""Driver for the stand-off resolvability study (see ``compressible_shock.resolvability``).

Two subcommands so that the expensive part is one grid per cluster job::

    # one grid of the ladder (parallel y-slabs inside the node)
    python scripts/run_resolvability_study.py grid --mach 3 --nx 1200 --ny 400 \
        --t-end 2 --backend parallel --nprocs 24 --out runs/resolvability/ma3_G4.json

    # merge the per-grid records: Richardson, GCI, three-state verdict, budget
    python scripts/run_resolvability_study.py collect --mach 3 \
        --inputs runs/resolvability/ma3_G1.json ... --out runs/resolvability/ma3_summary.json

The ``grid`` records are self-describing (geometry, t_end, backend, wall time,
positivity fallbacks, mass audit), so a summary can always be traced back to the
exact run that produced it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensorlbm.compressible_shock import resolvability as R  # noqa: E402


def _load(paths):
    out = []
    for p in paths:
        with open(p) as fh:
            out.append(json.load(fh))
    return out


def cmd_grid(args):
    rec = R.run_grid(
        mach=args.mach, nx=args.nx, ny=args.ny, t_end=args.t_end,
        step_x=args.step_x, step_h=args.step_h,
        backend=args.backend, nprocs=args.nprocs,
        label=args.label or f"G_{args.nx}x{args.ny}",
        audit=not args.no_audit, sample_dt=args.sample_dt,
        threads=args.threads, save_fields=args.save_fields, verbose=args.verbose,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    R.dump(rec, args.out)
    print(json.dumps({k: rec[k] for k in
                      ("label", "nx", "ny", "R", "standoff", "standoff_mass",
                       "shock_thickness_cells", "wall_time_s",
                       "positivity_fallbacks")}, indent=2))
    return 0


def cmd_collect(args):
    grids = _load(args.inputs)
    summary = R.collect_study(grids, mach=args.mach, qoi=args.qoi)
    # multi-Mach budget law when several summaries are supplied
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    R.dump(summary, args.out)
    print(json.dumps({"mach": summary["mach"],
                      "qoi": summary["qoi"],
                      "verdicts": summary["verdict_counts"],
                      "standoff_extrapolated": summary["standoff_extrapolated"],
                      "A1_pass": summary["A1"]["pass"],
                      "A2_pass": summary["A2"]["pass"],
                      "A2": {k: summary["A2"][k] for k in
                             ("p", "gci_21", "cross_triplet_spread", "reasons")}}, indent=2))
    return 0


def cmd_budget(args):
    per_mach = {}
    for mach, path in zip(args.machs, args.summaries):
        with open(path) as fh:
            s = json.load(fh)
        per_mach[float(mach)] = {
            "standoff_extrapolated": s["standoff_extrapolated"],
            "grids": {k: {"ok": v.get("ok", True), "standoff": v.get("standoff"),
                          "dx": v.get("dx")} for k, v in s["grids"].items()},
        }
    law = R.grid_budget_law(per_mach, lx=args.lx, ly=args.ly,
                            target_err=args.target_err, r_cert=R.R_CERT)
    R.dump(law, args.out)
    print(json.dumps(law, indent=2))
    return 0


def cmd_mach_scan(args):
    per_mach, summaries = {}, {}
    for mach, path in zip(args.machs, args.summaries):
        with open(path) as fh:
            summaries[float(mach)] = json.load(fh)
        per_mach[float(mach)] = {
            "standoff_extrapolated": summaries[float(mach)]["standoff_extrapolated"],
            "grids": {k: {"ok": v.get("ok", True), "standoff": v.get(args.qoi_key),
                          "dx": v.get("dx")}
                      for k, v in summaries[float(mach)]["grids"].items()},
        }
    out = {
        "A4": R.mach_scan_summary(per_mach, step_h=args.step_h),
        "budget": R.grid_budget_law(per_mach, lx=args.lx, ly=args.ly,
                                    target_err=args.target_err, r_cert=R.R_CERT,
                                    p_ref=args.p_ref),
        # sensitivity: the same budget under the *design* order.  The bracket
        # between the measured order and 2 is what the study quotes, because the
        # measured order of this QoI is below the design order (see A2).
        "budget_design_order": R.grid_budget_law(per_mach, lx=args.lx, ly=args.ly,
                                                 target_err=args.target_err,
                                                 r_cert=R.R_CERT, p_ref=2.0),
        "per_mach": {str(m): {"standoff_extrapolated": s["standoff_extrapolated"],
                              "A1": s["A1"]["pass"], "A2": s["A2"]["pass"]}
                     for m, s in summaries.items()},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    R.dump(out, args.out)
    print(json.dumps({"A4": out["A4"]["pass"], "direction_ok": out["A4"]["direction_ok"],
                      "correlation_states": out["A4"]["correlation_states"],
                      "budget_rows": [{k: r[k] for k in
                                       ("mach", "standoff_extrapolated", "N_min",
                                        "limiting_requirement")}
                                      for r in out["budget"]["rows"]]}, indent=2))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("grid", help="run one grid of the ladder")
    g.add_argument("--mach", type=float, default=3.0)
    g.add_argument("--nx", type=int, required=True)
    g.add_argument("--ny", type=int, required=True)
    g.add_argument("--t-end", type=float, default=2.0)
    g.add_argument("--step-x", type=float, default=0.6)
    g.add_argument("--step-h", type=float, default=0.2)
    g.add_argument("--backend", choices=["serial", "parallel", "torch"], default="parallel")
    g.add_argument("--nprocs", type=int, default=8)
    g.add_argument("--threads", type=int, default=0,
                   help="torch backend threads (0 = torch default = cores)")
    g.add_argument("--save-fields", default=None,
                   help="write primitives to this .npz for figures / re-analysis")
    g.add_argument("--verbose", action="store_true")
    g.add_argument("--label", default=None)
    g.add_argument("--sample-dt", type=float, default=0.1)
    g.add_argument("--no-audit", action="store_true")
    g.add_argument("--out", required=True)
    g.set_defaults(func=cmd_grid)

    c = sub.add_parser("collect", help="merge per-grid records")
    c.add_argument("--mach", type=float, default=3.0)
    c.add_argument("--qoi", choices=sorted(R.QOI_KEYS), default="mass")
    c.add_argument("--inputs", nargs="+", required=True)
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_collect)

    b = sub.add_parser("budget", help="grid budget law across Mach numbers")
    b.add_argument("--machs", nargs="+", type=float, required=True)
    b.add_argument("--summaries", nargs="+", required=True)
    b.add_argument("--target-err", type=float, default=0.05)
    b.add_argument("--lx", type=float, default=3.0)
    b.add_argument("--ly", type=float, default=1.0)
    b.add_argument("--out", required=True)
    b.set_defaults(func=cmd_budget)

    m = sub.add_parser("mach-scan", help="A4 Mach trend + grid budget law")
    m.add_argument("--machs", nargs="+", type=float, required=True)
    m.add_argument("--summaries", nargs="+", required=True)
    m.add_argument("--step-h", type=float, default=0.2)
    m.add_argument("--qoi-key", default="standoff_mass")
    m.add_argument("--p-ref", type=float, default=2.0,
                   help="observed order used to normalise the truncation estimate "
                        "when the scan cannot identify it (take it from the four-grid ladder)")
    m.add_argument("--target-err", type=float, default=0.05)
    m.add_argument("--lx", type=float, default=3.0)
    m.add_argument("--ly", type=float, default=1.0)
    m.add_argument("--out", required=True)
    m.set_defaults(func=cmd_mach_scan)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())