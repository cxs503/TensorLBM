"""Stage C: is the settled angle a property of the wall, or of the initial cap?

Two runs with the **same** wall fraction and grid but different initial caps
(70 deg and 110 deg).  If the wall force had a unique equilibrium angle, both
would settle to the same value.  The measured split quantifies contact-line
pinning / metastability of the SCMP wetting wall -- the reason a *prescribed*
angle has to be read as "the angle that survives relaxation from this cap".

    python run_metastability.py --fw 0.51 --grid coarse --out results/cases/c_*.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import wetting_bench as wb  # noqa: E402

INITS = (70.0, 110.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fw", type=float, default=0.51,
                    help="wall fraction to probe (default sits between the 70/90 deg targets)")
    ap.add_argument("--grid", default="coarse", choices=[g.name for g in wb.GRIDS])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=str(HERE / "results" / "cases"))
    args = ap.parse_args(argv)
    grid = next(g for g in wb.GRIDS if g.name == args.grid)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    records = []
    for init in INITS:
        rec = wb.run_case(init, grid, wall_fraction=args.fw, law="metastability",
                          init_deg=init, threads=args.threads, keep_points=True)
        records.append(rec)
        path = outdir / f"c_{args.grid}_init{int(init)}.json"
        path.write_text(json.dumps(rec, indent=1) + "\n")
        print(f"init={init:5.1f} f_w={args.fw:.3f} -> theta={rec['theta_measured']:.2f} "
              f"status={rec['status']} drift={rec['drift_deg']:.3f}", flush=True)
    finite = [r["theta_measured"] for r in records]
    print(f"split = {abs(finite[0] - finite[1]):.2f} deg between the two initial caps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())