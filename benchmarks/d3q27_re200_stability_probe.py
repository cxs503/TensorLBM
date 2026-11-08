#!/usr/bin/env python3
"""Re = 200 stability probe: D3Q19 vs D3Q27 with MRT (AUTO) and TRT.

The accuracy-ladder suite revealed that the *D3Q27 MRT* relaxation becomes
unstable at Re = 200 / D = 16 (τ = 0.512), while D3Q19 MRT survives the same
case at the same τ.  This script isolates that observation: it reruns the same
Re = 200 case with both lattices under

* ``collision=MRT``  (the engine's AUTO choice) — reproduces the instability,
* ``collision=TRT``  (two-relaxation-time, Λ = 3/16) — the standard remedy for
  low-τ stability,

and writes ``stability_re200_{mrt,trt}.json`` next to the accuracy report so the
documentation can quote measured numbers instead of claims.

Usage::

    python benchmarks/d3q27_re200_stability_probe.py --steps 2000 --resolution 16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from tensorlbm.d3q27_validation import run_sphere_matrix  # noqa: E402
from tensorlbm.general_sim import CollisionModel  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--re", type=float, default=200.0)
    ap.add_argument("--resolution", type=int, default=16)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "artifacts" / "d3q27_validation",
    )
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "re": args.re,
        "resolution": args.resolution,
        "steps": args.steps,
        "note": (
            "Stability probe only: the Cd window mean is reported so that an "
            "unstable case is visible as a collapsing Cd together with a "
            "diverging max|u| (healthy cases keep max|u| ~ 1.2 U)."
        ),
        "collisions": {},
    }
    for name, coll in (("mrt", CollisionModel.MRT), ("trt", CollisionModel.TRT)):
        cases = run_sphere_matrix(
            re_values=(args.re,),
            lattices=(19, 27),
            resolution=args.resolution,
            steps=args.steps,
            out_root=out / f"stability_{name}",
            workers=args.workers,
            threads_per_worker=args.threads,
            sphere_case_kwargs={"collision": coll},
        )
        rows = {}
        for key, per_lat in cases.items():
            for lat, res in per_lat.items():
                rows[lat] = {
                    "cd_mem_wet_node": res["cd_mem_wet_node"],
                    "cd_err_pct_vs_sn": res["cd_err_pct_vs_sn"],
                    "u_max": res["u_max"],
                    "u_max_over_u_in": res["u_max"] / res["u_lb"],
                    "diverged": res["diverged"],
                    "tau": res["tau"],
                    "ma": res["ma"],
                    "stable": bool(res["u_max"] / res["u_lb"] < 3.0 and not res["diverged"]),
                    "cd_history_tail": res["cd_history"][-6:],
                }
        summary["collisions"][name] = rows
        path = out / f"stability_re{int(args.re)}_{name}.json"
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[{name}] {json.dumps(rows, indent=2)}")
        print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())