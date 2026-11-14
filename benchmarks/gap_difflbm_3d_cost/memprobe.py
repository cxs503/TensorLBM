#!/usr/bin/env python3
"""Peak-RSS probe for one configuration (run in a fresh process).

Used by ``run.py`` so the memory number is a clean process high-water mark
(``VmHWM``) rather than a contaminated in-process measurement.

    python3 benchmarks/gap_difflbm_3d_cost/memprobe.py --grid 32 --steps 128 \
        --mode grad --segment 4 --dtype float64

Prints one JSON line: ``{"peak_bytes": ..., "baseline_bytes": ...,
"states_equivalent": ...}`` where ``baseline_bytes`` is the high-water mark of
an otherwise identical process that only builds the initial condition (so host
allocator / interpreter overhead can be subtracted).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "src"))

from tensorlbm import diff3d  # noqa: E402


def _peak():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("no VmHWM")


def body(args):
    dtype = getattr(torch, args.dtype)
    f0 = diff3d.abc_initial(args.grid, 0.05, dtype=dtype)
    if args.mode == "baseline":
        return
    c, w = diff3d.constants(dtype=dtype)
    if args.mode == "forward":
        f = f0
        with torch.no_grad():
            for _ in range(args.steps):
                f = diff3d.step3d(f, args.tau, c, w)
        del f
        return
    if args.mode == "grad":
        res = diff3d.checkpointed_gradient(f0, args.tau, args.steps,
                                           segment=args.segment, dtype=dtype)
        if args.print_grad:
            print(json.dumps({"grad_tau": res.grad["tau"], "value": res.value,
                              "seconds": res.seconds,
                              "n_forward_steps": res.n_forward_steps}),
                  file=sys.stderr)
        return
    raise SystemExit(f"unknown mode {args.mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, required=True)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--segment", type=int, default=4)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    ap.add_argument("--mode", default="grad",
                    choices=["grad", "forward", "baseline"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--print-grad", action="store_true")
    ap.add_argument("--chain", action="store_true",
                    help="re-exec with --mode baseline to report the baseline too")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.chain:
        child = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--grid", str(args.grid),
             "--dtype", args.dtype, "--mode", "baseline", "--threads",
             str(args.threads)], capture_output=True, text=True)
        baseline = json.loads(child.stdout.strip().splitlines()[-1])["peak_bytes"]
    else:
        baseline = 0
    body(args)
    peak = _peak()
    per = diff3d.state_bytes(args.grid, getattr(torch, args.dtype))
    print(json.dumps({"peak_bytes": peak, "baseline_bytes": baseline,
                      "state_bytes": per,
                      "states_equivalent": (peak - baseline) / per
                      if baseline else None}))


if __name__ == "__main__":
    raise SystemExit(main())