#!/usr/bin/env python3
"""CASE-BACKWARD-STEP-LBM -- Armaly-caliber backward-facing step benchmark (entry point).

Runs one pool case (or one shard of the pool grid) through the library API
:func:`tensorlbm.bfs_armaly.run_armaly_case` and writes real artefacts:

    artifacts/case_backward_step_armaly/<tag>/{result.json,wall_shear.csv,field.npz}

Usage
-----
    PYTHONPATH=src python3 benchmarks/case_pool/backward_step_armaly/run_case.py --preset g2_re100
    PYTHONPATH=src python3 benchmarks/case_pool/backward_step_armaly/run_case.py --shard 3/6
    PYTHONPATH=src python3 benchmarks/case_pool/backward_step_armaly/run_case.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.bfs_armaly import ArmalyConfig, armaly_geometry, run_armaly_case  # noqa: E402

ART = REPO / "artifacts" / "case_backward_step_armaly"

#: pool grid: two resolutions (scale 1 = Armaly ratio 52:49, scale 2 = 104:98) x three Re
GRID = [
    ("g1_re100", 1, 100.0),
    ("g1_re200", 1, 200.0),
    ("g1_re400", 1, 400.0),
    ("g2_re100", 2, 100.0),
    ("g2_re200", 2, 200.0),
    ("g2_re400", 2, 400.0),
]


def build_config(tag: str, scale: int, re: float, steps: int = 90000) -> ArmalyConfig:
    hi, s = armaly_geometry(scale)
    #: u_bar is fixed by tau: keep tau >= 0.55 on every grid/Re pair
    u_bar = 0.1
    return ArmalyConfig(re=re, hi=hi, s=s, u_bar=u_bar, n_steps=steps, tag=tag)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", help="one of " + ", ".join(t for t, _, _ in GRID))
    ap.add_argument("--shard", help="i/n -- run the i-th (1-based) grid entry")
    ap.add_argument("--smoke", action="store_true", help="tiny grid, 400 steps (CI sanity)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    if args.smoke:
        cfg = ArmalyConfig(re=100.0, hi=8, s=7, u_bar=0.1, l_over_h=10.0,
                           up_over_hi=3.0, n_steps=args.steps or 400,
                           avg_from=0.5, avg_stride=50, check_interval=200, tag="smoke")
        out_dir = Path(args.out_dir) if args.out_dir else ART / "smoke"
    else:
        if args.shard:
            i, n = (int(v) for v in args.shard.split("/"))
            if not (1 <= i <= len(GRID)):
                raise SystemExit(f"shard index {i} outside 1..{len(GRID)}")
            tag, scale, re = GRID[i - 1]
        elif args.preset:
            table = {t: (s_, r_) for t, s_, r_ in GRID}
            if args.preset not in table:
                raise SystemExit(f"unknown preset {args.preset!r}")
            tag, (scale, re) = args.preset, table[args.preset]
        else:
            raise SystemExit("need --preset/--shard/--smoke")
        cfg = build_config(tag, scale, re, steps=args.steps or 90000)
        out_dir = Path(args.out_dir) if args.out_dir else ART / tag

    t0 = time.time()
    result = run_armaly_case(cfg, out_dir=out_dir)
    result["entry"] = "benchmarks/case_pool/backward_step_armaly/run_case.py"
    result["tag"] = cfg.tag
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"[{cfg.tag}] Re={cfg.re:g} hi={cfg.hi} s={cfg.s} nx={cfg.nx} ny={cfg.ny} "
          f"tau={cfg.tau:.4f} steps={result['result']['steps_run']} "
          f"X1/s={result['result']['x1_over_s']:.4f} "
          f"({time.time() - t0:.0f}s) -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
