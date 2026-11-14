"""GAP-DIFFLBM-SOLID-LIMIT campaign driver — one (arm, D, scale) combo per call.

The combo list is deterministic so that the cluster scan template
``python cases/GAP-DIFFLBM-SOLID-LIMIT/campaign.py --index {i} --out ...`` with
``{i}`` in ``0..{n}-1`` covers the whole matrix exactly once:

    per resolution D:  hardwall x scale 1.0  (1 run, scale-free by definition)
                       amplitude x {1.0, 0.75, 0.5}
                       interface x {1.0, 0.75, 0.5}

Usage
-----
``PYTHONPATH=src python cases/GAP-DIFFLBM-SOLID-LIMIT/campaign.py --list``
``PYTHONPATH=src python cases/GAP-DIFFLBM-SOLID-LIMIT/campaign.py --index 3 \
      --out artifacts/GAP-DIFFLBM-SOLID-LIMIT/runs``
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_case import ARMS, SCALES, run_arm  # noqa: E402


def combos(resolutions: Sequence[int], scales: Sequence[float]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for D in resolutions:
        out.append({"arm": "hardwall", "D": int(D), "scale": 1.0})
        for arm, scale in itertools.product(("amplitude", "interface"), scales):
            out.append({"arm": arm, "D": int(D), "scale": float(scale)})
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--resolutions", type=int, nargs="+", default=[20, 40])
    ap.add_argument("--scales", type=float, nargs="+", default=list(SCALES))
    ap.add_argument("--max-steps", type=int, default=80_000)
    ap.add_argument("--min-steps", type=int, default=10_000)
    ap.add_argument("--avg-window", type=int, default=2_000)
    ap.add_argument("--plateau-tol", type=float, default=1.0e-3)
    ap.add_argument("--cv-every", type=int, default=25)
    ap.add_argument("--u-lb", type=float, default=0.1)
    ap.add_argument("--sigma-max", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=Path("artifacts/GAP-DIFFLBM-SOLID-LIMIT/runs"))
    args = ap.parse_args(argv)

    todo = combos(args.resolutions, args.scales)
    if args.list:
        for k, c in enumerate(todo):
            print(k, c)
        return 0
    if args.index is None:
        ap.error("--index or --list is required")
    if not 0 <= args.index < len(todo):
        raise SystemExit(f"index {args.index} outside 0..{len(todo) - 1}")
    c = todo[args.index]
    res = run_arm(
        arm=str(c["arm"]),
        D_cells=int(c["D"]),
        scale=float(c["scale"]),
        max_steps=args.max_steps,
        min_steps=args.min_steps,
        avg_window=args.avg_window,
        plateau_tol=args.plateau_tol,
        cv_every=args.cv_every,
        u_lb=args.u_lb,
        sigma_max=args.sigma_max,
        progress=True,
    )
    res["index"] = args.index
    res["n_combos"] = len(todo)
    args.out.mkdir(parents=True, exist_ok=True)
    name = f"{c['arm']}_D{c['D']}_s{float(c['scale']):g}.json"
    (args.out / name).write_text(json.dumps(res, indent=2))
    print(
        f"[{args.index}/{len(todo)}] {c['arm']} D={c['D']} s={c['scale']}: "
        f"Cd_cv={res['cd_cv']:.4f} Cd_me={res['cd_me']:.4f} "
        f"steps={res['steps']} stop={res['stop_reason']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())