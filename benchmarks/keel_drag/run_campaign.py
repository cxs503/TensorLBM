#!/usr/bin/env python
"""Ice-keel drag campaign (mission P46W3-N01): run the case table, store records.

Usage
-----
    python benchmarks/keel_drag/run_campaign.py --out benchmarks/keel_drag/campaign
    python benchmarks/keel_drag/run_campaign.py --shard 0 --nshards 4 --out .../shard0

The case table is built by :mod:`tensorlbm.keel_drag` (one-at-a-time sweeps of
``h_k``, ``b_k``, ``s_k`` plus the Reynolds sweep with its level-ice partners and
the two structural controls). One JSON file per case is written so that a shard
that dies can be resumed without losing the rest; ``merge_shards.py`` folds them
into ``cases.json``.

Everything that could make the numbers look better than they are is *off*: no
case is filtered after the fact, a diverged case is stored with its error string
and stays in the table, and the acceptance verdicts are computed by
``analyse.py`` from the stored JSON only (no hand-entered numbers anywhere).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from tensorlbm import keel_drag as kd  # noqa: E402


def build_cases(*, grid: tuple[int, int, int], steps: int, warmup: int, threads: int,
                re_tau: float, u_tau: float) -> list[kd.KeelCase]:
    """The predeclared case table (rationale in the campaign report).

    Aspect ratios follow the observed keel population (``b_k/h_k = 3-6``,
    Wadhams 2000); the depth sweep necessarily runs from a thin rib (16) to a
    near-45 deg ridge (2), which is where the projection split of the wall force
    is structurally weakest, so it is reported rather than excluded.

    The Reynolds sweep varies ``Re_tau`` with ``u_tau = Re_tau/1000``, which holds
    the lattice viscosity (hence the number of steps needed to relax the viscous
    transient) constant while changing ``Re_h = Re_tau^2/3`` by 25x.
    """
    kw = dict(grid=grid, steps=steps, warmup=warmup, threads=threads,
              re_tau=re_tau, u_tau=u_tau)
    cases: list[kd.KeelCase] = []
    cases.append(kd.smooth_case(**kw, case_id=f"smooth_rt{re_tau:04.0f}"))
    cases += kd.keel_depth_sweep(b_k=16.0, s_k=32.0, h_ks=(1.0, 2.0, 3.0, 4.0, 6.0), **kw)
    cases += kd.keel_width_sweep(h_k=4.0, s_k=32.0, b_ks=(8.0, 12.0, 16.0, 20.0, 24.0), **kw)
    cases += kd.keel_spacing_sweep(h_k=2.0, b_k=4.0, s_ks=(8.0, 16.0, 32.0), **kw)
    re_taus = (4.0, 7.0, 13.0, 20.0)
    cases += kd.reynolds_sweep(h_k=4.0, b_k=16.0, s_k=32.0, re_taus=re_taus,
                               u_taus=[rt / 1000.0 for rt in re_taus],
                               grid=grid, steps=steps, warmup=warmup, threads=threads)
    cases.append(kd.keel_case("streamwise_h4_b8_s16", "streamwise",
                              kd.KeelGeometry(h_k=4.0, b_k=8.0, s_k=16.0,
                                              nx=grid[0], ny=grid[1], nz=grid[2],
                                              theta_deg=0.0),
                              re_tau=re_tau, u_tau=u_tau, steps=steps,
                              warmup=warmup, threads=threads))
    seen: dict[str, kd.KeelCase] = {}
    for c in cases:
        seen.setdefault(c.case_id, c)
    return list(seen.values())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(_ROOT / "benchmarks" / "keel_drag" / "campaign"))
    ap.add_argument("--grid", default="32,32,66")
    ap.add_argument("--steps", type=int, default=kd.DEFAULT_STEPS)
    ap.add_argument("--warmup", type=int, default=kd.DEFAULT_WARMUP)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--re-tau", type=float, default=kd.DEFAULT_RE_TAU)
    ap.add_argument("--u-tau", type=float, default=kd.DEFAULT_U_TAU)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--only", default="", help="comma-separated case_id substrings")
    ap.add_argument("--max-passes", type=int, default=3,
                    help="amplitude-iteration passes (last one is the production run)")
    ap.add_argument("--probe-steps", type=int, default=3000,
                    help="steps per amplitude-iteration probe pass")
    ap.add_argument("--no-renorm", action="store_true",
                    help="skip the laminar amplitude renormalisation (diagnostic only)")
    args = ap.parse_args(argv)

    grid = tuple(int(x) for x in args.grid.split(","))
    assert len(grid) == 3
    out = Path(args.out)
    (out / "cases").mkdir(parents=True, exist_ok=True)
    cases = build_cases(grid=grid, steps=args.steps, warmup=args.warmup,
                        threads=args.threads, re_tau=args.re_tau, u_tau=args.u_tau)
    if args.only:
        wants = [w for w in args.only.split(",") if w]
        cases = [c for c in cases if any(w in c.case_id for w in wants)]
    if args.nshards > 1:
        cases = [c for i, c in enumerate(cases) if i % args.nshards == args.shard]

    print(f"[campaign] {len(cases)} cases, grid={grid}, steps={args.steps}, "
          f"threads={args.threads}, renorm={not args.no_renorm}", flush=True)
    manifest = {"grid": list(grid), "steps": args.steps, "warmup": args.warmup,
                "threads": args.threads, "re_tau": args.re_tau, "u_tau": args.u_tau,
                "renorm": not args.no_renorm, "probe_steps": args.probe_steps,
                "max_passes": args.max_passes,
                "shard": args.shard,
                "nshards": args.nshards, "cases": [c.case_id for c in cases],
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / f"manifest_shard{args.shard}.json").write_text(json.dumps(manifest, indent=1))

    t_all = time.time()
    for i, case in enumerate(cases, 1):
        path = out / "cases" / f"{case.case_id}.json"
        t0 = time.time()
        try:
            rec = kd.run_keel_case(case, renorm=not args.no_renorm,
                                   probe_steps=args.probe_steps,
                                   max_passes=args.max_passes)
            status = "ok"
        except Exception as exc:  # honest FAIL: keep the case in the table
            rec = {"case_id": case.case_id, "family": case.family,
                   "tags": dict(case.tags), "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc()[-2000:]}
            status = "error"
        rec["status"] = status
        rec["elapsed_s"] = time.time() - t0
        path.write_text(json.dumps(rec, indent=1))
        if status == "ok":
            print(f"[{i}/{len(cases)}] {case.case_id} ok {rec['elapsed_s']:.0f}s "
                  f"Re_h={rec['re_h_converged']:.1f} Cd={rec['cd_total_balance']:.5f} "
                  f"skin={rec['cd_skin']:.5f} form={rec['cd_form']:.5f} "
                  f"clos={rec['cd_closure_residual']:+.2e}", flush=True)
        else:
            print(f"[{i}/{len(cases)}] {case.case_id} ERROR {rec['error']}", flush=True)
    print(f"[campaign] done in {(time.time() - t_all) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())