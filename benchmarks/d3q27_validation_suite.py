#!/usr/bin/env python3
"""D3Q27 activation validation suite driver (gaps G7/G8).

Runs :func:`tensorlbm.d3q27_validation.run_validation_suite` and writes

* ``d3q27_validation_report.json`` — machine-readable report (all judgement
  numbers are produced here, nothing is hand-copied),
* ``figures/*.png`` — field contours, Cd history, isotropy bars, TG
  convergence,

under ``--out``.

Usage::

    python benchmarks/d3q27_validation_suite.py --resolution 16 --steps 3000 \
        --out artifacts/d3q27_validation

The heavy sphere runs (Re = 100/200 × D3Q19/D3Q27) dominate the wall clock;
run this in the background and poll the JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from tensorlbm.d3q27_validation import run_validation_suite  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resolution", type=int, default=16, help="sphere diameter in cells")
    ap.add_argument("--steps", type=int, default=3000, help="steps per sphere case")
    ap.add_argument("--re", type=float, nargs="+", default=[100.0, 200.0])
    ap.add_argument("--lattices", type=int, nargs="+", default=[19, 27])
    ap.add_argument("--tg-sizes", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--isotropy-n", type=int, default=32)
    ap.add_argument(
        "--threads", type=int, default=8, help="torch threads (also per sphere worker)"
    )
    ap.add_argument(
        "--sphere-workers",
        type=int,
        default=1,
        help="concurrent processes for the (Re x lattice) sphere matrix",
    )
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "artifacts" / "d3q27_validation",
        help="output directory",
    )
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    report = run_validation_suite(
        re_values=tuple(args.re),
        lattices=tuple(args.lattices),
        resolution=args.resolution,
        steps=args.steps,
        out_dir=out,
        tg_sizes=tuple(args.tg_sizes),
        isotropy_n=args.isotropy_n,
        make_figures=not args.no_figures,
        sphere_workers=args.sphere_workers,
        threads_per_worker=args.threads,
    )
    wall = time.time() - t0

    print(f"\n=== D3Q27 validation suite finished in {wall / 60:.1f} min ===")
    print(json.dumps(report.get("verdicts", {}), indent=2))
    print("report:", report.get("report_json"))
    for fig in report.get("figures", []):
        print("figure:", fig.get("path"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())