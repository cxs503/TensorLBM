#!/usr/bin/env python3
"""IN32-L01 wetting ladder — run one case and emit its JSON artifact.

Examples
--------
    python benchmarks/in32_l01_wetting_ladder/run_case.py sigma_flat
    python benchmarks/in32_l01_wetting_ladder/run_case.py laplace
    python benchmarks/in32_l01_wetting_ladder/run_case.py contact --theta 90
    python benchmarks/in32_l01_wetting_ladder/run_case.py jurin
    python benchmarks/in32_l01_wetting_ladder/run_case.py lamb

Artifacts land in ``artifacts/in32_l01/<case>.json`` (never in the source
tree), and the consolidated report consumes them; no number in
``docs/IN32_L01_WETTING_LADDER.md`` is typed by hand.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.wetting import ladder as L  # noqa: E402

OUT = ROOT / "artifacts" / "in32_l01"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=["sigma_flat", "laplace", "contact", "jurin", "lamb"])
    ap.add_argument("--theta", type=float, default=None)
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    if args.case == "sigma_flat":
        res = L.sigma_flat(steps=args.steps or 2500)
    elif args.case == "laplace":
        res = L.laplace(sigma=args.sigma or 0.02, steps=args.steps or 4000)
    elif args.case == "contact":
        res = L.contact(90.0 if args.theta is None else args.theta, sigma=args.sigma or 0.02,
                        steps=args.steps or 3000)
    elif args.case == "jurin":
        # theta is a required choice for Jurin (a neutral wall has zero drive)
        res = L.jurin(theta_deg=60.0 if args.theta is None else args.theta, steps=args.steps or 6000)
    else:
        res = L.lamb(sigma=args.sigma or 0.06, steps=args.steps or 5000)

    name = args.case if args.case != "contact" else f"contact_{int(round(args.theta))}"
    L.save(Path(args.out) / f"{name}.json", res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())