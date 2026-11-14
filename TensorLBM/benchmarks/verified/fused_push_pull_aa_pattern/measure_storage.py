#!/usr/bin/env python3
"""Measure per-step peak resident memory of the two fused kernels.

Anchor: ``XVAL-LBM-FUSED-PUSH-PULL-AA-PATTERN``.

The A-A pattern's defining structural claim is "one distribution field alive,
``O(N)`` auxiliary storage".  The fused A-B kernel in
``src/tensorlbm/aa_fused_push_pull.py`` allocates a second ``Q x N`` field per
step; the in-place A-A kernel in ``src/tensorlbm/aa_fused_push_pull_inplace.py``
does not.  This script measures that claim instead of asserting it.

Method
------
Each kernel runs in a *fresh subprocess* (``--kernel ab|aa``), allocates one
``900 x 900`` D2Q9 float64 field, records ``VmHWM`` (peak resident set) right
after that allocation, then advances ``--steps`` steps and records ``VmHWM``
again.  The increment over the post-allocation baseline, divided by the size of
one field, is the number of *additional* fields the step needs.  ``VmHWM`` is a
kernel-provided high-water mark, so allocator caching can only make this an
over-estimate of what the kernel itself touches.

    python measure_storage.py            # runs both kernels, prints a table
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

GRID = 900


def _hwm_mb() -> float:
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM"):
            return int(line.split()[1]) / 1024.0
    raise RuntimeError("VmHWM not available")


def run_one(kernel: str, grid: int, steps: int) -> dict:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "src"))
    import torch

    import tensorlbm.aa_fused_push_pull as AB
    import tensorlbm.aa_fused_push_pull_inplace as AA

    rho = torch.ones(grid, grid, dtype=torch.float64)
    f = torch.zeros(9, grid, grid, dtype=torch.float64)
    for i in range(9):
        f[i] = AB.W9[i] * rho
    field_mb = f.numel() * f.element_size() / 1e6
    solid = AA.channel_masks(grid, grid, 1)
    base = _hwm_mb()
    for _ in range(steps):
        if kernel == "ab":
            f = AB.fused_push_pull_step(f, 0.9, solid=solid)
        else:
            AA.aa_fused_push_pull_step(f, 0.9, solid=solid)
    peak = _hwm_mb()
    return {
        "kernel": kernel,
        "grid": grid,
        "steps": steps,
        "field_mb": round(field_mb, 2),
        "baseline_mb": round(base, 1),
        "peak_mb": round(peak, 1),
        "extra_fields": round((peak - base) / field_mb, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", choices=("ab", "aa"))
    ap.add_argument("--grid", type=int, default=GRID)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.kernel:
        print(json.dumps(run_one(args.kernel, args.grid, args.steps), sort_keys=True))
        return 0

    here = pathlib.Path(__file__).resolve()
    rows = []
    for kernel in ("ab", "aa"):
        out = subprocess.run(
            [sys.executable, str(here), "--kernel", kernel, "--grid", str(args.grid),
             "--steps", str(args.steps)],
            capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()[-1]
        rows.append(json.loads(out))
    print(f"{'kernel':6s} {'field MB':>9s} {'peak above field':>18s}")
    for r in rows:
        print(f"{r['kernel']:6s} {r['field_mb']:9.2f} {r['extra_fields']:18.2f} fields")
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())