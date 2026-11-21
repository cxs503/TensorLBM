#!/usr/bin/env python
"""m23154 sweep driver: sharded Taylor-Green audit of the KBC / Cumulant kernels.

Each task of :data:`TASKS` is one ``(kernel, N, tau)`` point of the
pre-registered protocol in ``docs/m23154/PREREG.json``.  The driver writes one
JSON per task under ``docs/m23154/runs/`` and is shardable
(``--shard i/n``) so the sweep can be spread over the cluster queue.

    python development/experiments/m23154_sweep.py --shard 0/8
    python development/experiments/m23154_sweep.py --list
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import m23154_kbc_cumulant_tg as audit  # noqa: E402

OUTDIR = os.path.join(HERE, "docs", "m23154", "runs")

#: tier A -- tau sweep on a fixed grid (decisive test of nu(tau));
#: tier B -- one finer grid at the canonical Re=24 point (discretisation check);
#: tier C -- the fp32 protocol of benchmarks/pending/collision_kernels_3d_tg for
#:           reconciliation with the earlier published FAIL.
TIER_A_KERNELS = (
    "bgk_d3q19_ctl",
    "kbc_d3q19",
    "kbc_d3q27",
    "cumulant_d3q27",
    "cumulant_geier_d3q27",
    "cascaded_d3q27_ctl",
    "mrt_d3q27_ctl",
    "trt_d3q27_ctl",
    "rlbm_d3q27_ctl",
)
TIER_B_KERNELS = ("kbc_d3q19", "cumulant_d3q27", "cumulant_geier_d3q27")
TIER_C_KERNELS = ("kbc_d3q19",)


def build_tasks() -> list[dict]:
    tasks: list[dict] = []
    for kern in TIER_A_KERNELS:
        for tau in (0.6, 0.8, 1.0, 1.5):
            tasks.append(dict(kernel=kern, n=32, tau=tau, dtype="float64", tier="A"))
    for kern in TIER_B_KERNELS:
        tasks.append(dict(kernel=kern, n=48, tau=0.7, dtype="float64", tier="B"))
    for kern in TIER_C_KERNELS:
        for n in (32, 48):
            tasks.append(dict(kernel=kern, n=n, tau=0.5 + 3.0 * 0.05 * n / 24.0, dtype="float32", tier="C"))
    return tasks


TASKS = build_tasks()


def tag(t: dict) -> str:
    return f"{t['kernel']}_N{t['n']}_tau{t['tau']:.4f}_{t['dtype']}_{t['tier']}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("M23154_THREADS", "8")))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        for i, t in enumerate(TASKS):
            print(i, tag(t))
        return 0

    i, n = (int(x) for x in args.shard.split("/"))
    mine = [t for j, t in enumerate(TASKS) if j % n == i]
    os.makedirs(OUTDIR, exist_ok=True)
    failures = 0
    for t in mine:
        path = os.path.join(OUTDIR, tag(t) + ".json")
        rec = audit.run_case(t["kernel"], t["n"], t["tau"], dtype=t["dtype"], threads=args.threads)
        rec["gates"] = audit.gates(rec)
        rec["tier"] = t["tier"]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=1)
        ok = rec["gates"]["all_pass"]
        failures += 0 if ok else 1
        print(
            f"[shard {args.shard}] {tag(t)}: err={rec['err_pct']:+.4f}% "
            f"nu_err={rec['nu_err_pct']:+.4f}% r2={rec['r2']:.6f} "
            f"gates={'PASS' if ok else 'FAIL'} wall={rec['wall_sec']:.1f}s",
            flush=True,
        )
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())