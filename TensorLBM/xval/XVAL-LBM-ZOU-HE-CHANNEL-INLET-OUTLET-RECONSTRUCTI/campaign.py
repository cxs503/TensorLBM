#!/usr/bin/env python
"""Campaign driver: one shard = one case of the Zou-He channel cross-validation.

Dispatched through the compute queue (``run_compute`` shards), so this file must
stay dependency-free apart from the repo itself and must write its artifacts
under ``artifacts/xval_zouhe/`` for retrieval.

    PYTHONPATH=src python xval/XVAL-LBM-ZOU-HE-.../campaign.py --index 0
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "..", "..", "artifacts", "xval_zouhe")

# (arm, height, length, steps, burn-in, avg-window, drho, tag)
CASES = [
    ("pressure-pressure", 32, 160, 120000, 20000, 2000, 4.0e-3, "pp_H32"),
    ("pressure-pressure", 16, 160, 120000, 20000, 2000, 4.0e-3, "pp_H16"),
    ("pressure-pressure", 64, 200, 80000, 20000, 2000, 4.0e-3, "pp_H64"),
    ("velocity-pressure", 32, 160, 120000, 20000, 2000, 4.0e-3, "vp_H32"),
    ("equilibrium-pressure", 32, 160, 120000, 20000, 2000, 4.0e-3, "eq_H32"),
    ("pressure-pressure", 32, 160, 120000, 20000, 2000, 8.0e-3, "pp_H32_drho8e-3"),
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True)
    args = ap.parse_args(argv)
    arm, h, nx, steps, burn, win, drho, tag = CASES[args.index % len(CASES)]
    os.makedirs(OUTDIR, exist_ok=True)
    cmd = [
        sys.executable,
        os.path.join(HERE, "case.py"),
        "--arm", arm,
        "--height", str(h),
        "--length", str(nx),
        "--steps", str(steps),
        "--burn-in", str(burn),
        "--avg-window", str(win),
        "--drho", str(drho),
        "--out", os.path.join(OUTDIR, f"case_{tag}.json"),
        "--npz", os.path.join(OUTDIR, f"field_{tag}.npz"),
    ]
    print("running:", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=os.path.join(HERE, "..", ".."))


if __name__ == "__main__":
    raise SystemExit(main())