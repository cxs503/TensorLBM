#!/usr/bin/env python
"""Driver for the NACA0012 aerodynamic-polar verification surface.

Runs one (or several) incidence angles of the polar on the requested grid and
writes one JSON per case.  Designed for cluster submission:

    python scripts/run_naca0012_polar.py --chord 128 --re 1e4 \
        --alphas 0 2 4 6 --steps 40000 --out artifacts/naca0012_polar

The script is intentionally free of any orchestration logic: it is a pure
solver invocation (TensorLBM owns the numerical method, the campaign layer
owns the scheduling).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from tensorlbm.airfoil_polar import AirfoilPolarConfig, run_polar_case  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chord", type=int, default=128)
    ap.add_argument("--re", type=float, default=1.0e4)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 2.0, 4.0, 6.0])
    ap.add_argument("--u", type=float, default=0.12)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--wall", default="interpolated", choices=["interpolated", "staircase"])
    ap.add_argument("--init", default="potential", choices=["potential", "uniform"])
    ap.add_argument("--upstream", type=float, default=3.0)
    ap.add_argument("--downstream", type=float, default=8.0)
    ap.add_argument("--halfheight", type=float, default=4.0)
    ap.add_argument("--sponge", type=float, default=4.0)
    ap.add_argument("--conv-tol", type=float, default=2.0e-3)
    ap.add_argument("--out", default="artifacts/naca0012_polar")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "run_log.jsonl"

    for alpha in args.alphas:
        tag = args.tag or f"c{args.chord}_re{args.re:.0e}_a{alpha:+.1f}"
        cfg = AirfoilPolarConfig(
            chord=args.chord, alpha_deg=alpha, re=args.re, u_in=args.u,
            upstream_c=args.upstream, downstream_c=args.downstream,
            halfheight_c=args.halfheight, sponge_c=args.sponge,
            n_steps=args.steps, wall=args.wall, init=args.init,
            conv_tol=args.conv_tol, tag=tag,
        )
        t0 = time.time()
        res = run_polar_case(cfg)
        res["tag"] = tag
        res["elapsed_s"] = time.time() - t0
        path = out / f"{tag}.json"
        path.write_text(json.dumps(res, indent=1))
        with log.open("a") as fh:
            fh.write(json.dumps({
                "tag": tag, "alpha": alpha, "cl": res["cl"], "cd": res["cd"],
                "steps": res["steps_run"], "converged": res["converged"],
                "s_per_step": res["s_per_step"], "diverged": res["diverged"],
                "elapsed_s": res["elapsed_s"],
            }) + "\n")
        print(f"[{tag}] cl={res['cl']:+.4f} cd={res['cd']:.4f} steps={res['steps_run']} "
              f"conv={res['converged']} {res['s_per_step']:.3f}s/step "
              f"wall={res['elapsed_s']:.0f}s div={res['diverged']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())