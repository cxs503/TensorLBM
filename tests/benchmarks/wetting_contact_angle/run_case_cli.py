"""Run one contact-angle case and write its record (single OS process).

The sweep is dispatched as independent processes (one per case) because a
forked/threaded pool over torch deadlocks on this box; ``run_all.sh`` fans the
cases out with ``xargs -P``.

    python run_case_cli.py --theta 90 --grid coarse --law young --out results/cases/a_coarse_90.json
    python run_case_cli.py --fw 0.45 --grid coarse --law scan --out results/cases/b1_0.45.json
    python run_case_cli.py --theta 70 --grid fine --law calibrated --fw 0.4123 \
        --out results/cases/b2_fine_70.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import wetting_bench as wb  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--theta", type=float, required=True,
                    help="prescribed angle (physical convention, deg); ignored for --law scan, "
                         "where the initial cap is seeded from the Young angle of --fw")
    ap.add_argument("--grid", required=True, choices=[g.name for g in wb.GRIDS])
    ap.add_argument("--law", default="young", choices=["young", "scan", "calibrated"])
    ap.add_argument("--fw", type=float, default=None, help="wall fraction (overrides the law)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    grid = next(g for g in wb.GRIDS if g.name == args.grid)
    theta, init = args.theta, None
    if args.law == "scan":
        # A scan case prescribes nothing: f_w is the independent variable and the
        # initial cap is seeded at the angle the Young relation *would* predict
        # for that f_w (a relaxation accelerant, not a prescription).  Computing
        # it here keeps the value out of the shell script (no hand-copied numbers).
        psi_l, psi_v = wb._psi_pair()
        init = wb.theta_from_wall_fraction(args.fw, psi_l, psi_v)
        theta = init
    rec = wb.run_case(theta, grid, wall_fraction=args.fw, law=args.law, init_deg=init,
                      threads=args.threads, keep_points=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"[{rec['grid']:>6s}] law={rec['law']:<10s} theta_eq={rec['theta_eq']:6.1f} "
          f"f_w={rec.get('wall_fraction')} theta_meas={rec.get('theta_measured')} "
          f"status={rec['status']} ({rec['runtime_s']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())