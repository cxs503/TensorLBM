#!/usr/bin/env python3
"""W22 sweep driver: run the 3 grid levels of one tau rung sequentially.

Each case runs in its own subprocess so that one diverged/crashed case cannot
take the remaining rungs down with it; failures are collected and written to
``failures_<tag>.json`` instead of being silently dropped.

Usage:
    python benchmarks/verified/suboff_tau_grid_w22/run_sweep.py \
        --tau 0.5005 --steps 400 --grids 96 144 192 --threads 20 --out runs/w22
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--grids", type=int, nargs="+", default=[96, 144, 192])
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--out", default="runs/w22")
    args = ap.parse_args()

    failures = []
    for N in args.grids:
        tag = f"tau{args.tau:.4f}_N{N}"
        cmd = [
            sys.executable,
            "-u",
            str(_HERE / "run_case.py"),
            "--tau",
            repr(args.tau),
            "--resolution",
            str(N),
            "--steps",
            str(args.steps),
            "--threads",
            str(args.threads),
            "--out",
            args.out,
            "--tag",
            tag,
        ]
        t0 = time.time()
        print(f"[sweep] === {tag} (steps={args.steps}) ===", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.time() - t0
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        print(f"[sweep] {tag} rc={proc.returncode} wall={dt:.0f}s", flush=True)
        if proc.returncode != 0:
            failures.append(
                {
                    "tag": tag,
                    "tau": args.tau,
                    "N": N,
                    "returncode": proc.returncode,
                    "wall_seconds": dt,
                    "stderr_tail": proc.stderr[-2000:],
                    "stdout_tail": proc.stdout[-1000:],
                }
            )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fp = out / f"failures_tau{args.tau:.4f}.json"
    fp.write_text(json.dumps({"tau": args.tau, "failures": failures}, indent=1))
    print(f"[sweep] tau={args.tau} done, {len(failures)} failure(s) -> {fp}", flush=True)


if __name__ == "__main__":
    main()