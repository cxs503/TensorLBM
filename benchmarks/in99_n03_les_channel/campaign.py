#!/usr/bin/env python3
"""IN99-N03 campaign table: LES channel at the MKM-1999 Re_tau anchors.

Grids, Reynolds numbers and step budgets are fixed here so that the gate
evaluation (``gates.py``) and the report generator consume exactly the same
case names.  ``--group`` selects the sub-set of cases a single cluster node
runs; each case is a separate process so the OpenMP threads of a node can be
shared between them.

Design notes
------------
* ``re_tau`` is the Reynolds number MKM actually simulated for the nominal
  180/395/590 anchors (178.12 / 392.24 / 587.19); matching it removes a
  systematic 0.5-1% wall-unit offset from every comparison.
* ``u_tau`` (lattice) is a free scale: it fixes dt.  It is chosen so the
  centreline stays at U_cl ~ 18 u_tau <= 0.16 (Ma <= 0.28) while the number of
  lattice steps per eddy turnover (h / u_tau) keeps the statistics window
  affordable.
* The negative-control arm repeats the h = 32 grid at Re_tau = 178 with the
  sub-grid model switched off (``--cs 0``), so any difference is caused by the
  SGS closure alone.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]

#: name -> run_case.py arguments.  ``group`` is the cluster node sub-set.
CASES: dict[str, dict] = {
    "R180_h24": dict(
        group="a", nx=150, ny=50, nz=75, re_tau=178.12, u_tau=0.006, cs=0.1,
        steps=90000, warmup=15000, threads=40,
    ),
    "R180_h32": dict(
        group="a", nx=200, ny=66, nz=100, re_tau=178.12, u_tau=0.006, cs=0.1,
        steps=90000, warmup=15000, threads=40,
    ),
    "R180_h40": dict(
        group="a", nx=250, ny=82, nz=125, re_tau=178.12, u_tau=0.006, cs=0.1,
        steps=96000, warmup=16000, threads=40,
    ),
    "R180_h32_nosgs": dict(
        group="b", nx=200, ny=66, nz=100, re_tau=178.12, u_tau=0.006, cs=0.0,
        steps=90000, warmup=15000, threads=40,
    ),
    "R395_h40": dict(
        group="b", nx=250, ny=82, nz=125, re_tau=392.24, u_tau=0.008, cs=0.1,
        steps=75000, warmup=13000, threads=40,
    ),
    "R590_h48": dict(
        group="b", nx=300, ny=98, nz=150, re_tau=587.19, u_tau=0.008, cs=0.1,
        steps=76000, warmup=13000, threads=40,
    ),
}


def command(name: str, out_dir: str) -> list[str]:
    spec = dict(CASES[name])
    group = spec.pop("group")
    del group
    args = [sys.executable, str(_HERE / "run_case.py"), "--name", name,
            "--out-dir", out_dir]
    for key, value in spec.items():
        args += [f"--{key.replace('_', '-')}", str(value)]
    return args


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="all", choices=["a", "b", "all"])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out-dir", default="runs/in99_n03")
    ap.add_argument("--sequential", action="store_true")
    args = ap.parse_args()

    names = [n for n, s in CASES.items()
             if (args.only and n in args.only)
             or (not args.only and args.group in ("all", s["group"]))]
    out_abs = Path(args.out_dir)
    if not out_abs.is_absolute():
        out_abs = _REPO / out_abs
    out_abs.mkdir(parents=True, exist_ok=True)
    plan = {"out_dir": str(out_abs), "cases": names,
            "threads": {n: CASES[n]["threads"] for n in names}}
    (_HERE / f"campaign_plan_{args.group}.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=1), flush=True)

    procs = []
    for name in names:
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = str(CASES[name]["threads"])
        log = open(out_abs / f"{name}.log", "w")
        proc = subprocess.Popen(command(name, args.out_dir), env=env,
                                stdout=log, stderr=subprocess.STDOUT)
        procs.append((name, proc, log))
        if args.sequential:
            proc.wait()
    failures = []
    for name, proc, log in procs:
        rc = proc.wait()
        log.close()
        print(f"[campaign] {name} rc={rc}", flush=True)
        if rc != 0:
            failures.append(name)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())