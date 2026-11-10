#!/usr/bin/env python
"""Parallel case-list runner for the NACA0012 polar surface.

One process pool, ``--workers`` concurrent cases, ``--threads`` torch threads
each.  A case list can be split across cluster shards with ``--shard/--nshard``
so that a single queue slot does not have to carry the whole campaign.

Case list format (JSON)::

    [{"chord": 64, "alpha": -2.0, "re": 10000, "steps": 25000,
      "tag": "c64_re1e4_a-2.0"}, ...]

By default the list is generated from ``--grids`` x ``--alphas`` x ``--re``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def run_one(spec: dict) -> dict:
    import torch
    from tensorlbm.airfoil_polar import AirfoilPolarConfig, run_polar_case

    if spec.get("threads"):
        torch.set_num_threads(int(spec["threads"]))
        try:  # only settable before any parallel work in the process
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    cfg = AirfoilPolarConfig(
        chord=int(spec["chord"]), alpha_deg=float(spec["alpha"]), re=float(spec["re"]),
        u_in=float(spec.get("u", 0.12)), n_steps=int(spec.get("steps", 25000)),
        warmup_frac=float(spec.get("warmup", 0.4)),
        sample_interval=int(spec.get("sample", 10)),
        conv_tol=float(spec.get("conv_tol", 2e-3)),
        upstream_c=float(spec.get("upstream", 2.5)),
        downstream_c=float(spec.get("downstream", 6.0)),
        halfheight_c=float(spec.get("halfheight", 3.0)),
        sponge_c=float(spec.get("sponge", 3.0)),
        sponge_alpha=float(spec.get("sponge_alpha", 8.0)),
        wall=spec.get("wall", "interpolated"),
        init=spec.get("init", "potential"),
        tag=spec.get("tag", ""),
    )
    t0 = time.time()
    res = run_polar_case(cfg)
    res["tag"] = spec.get("tag", "")
    res["elapsed_s"] = time.time() - t0
    out = Path(spec["out"])
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{res['tag']}.json").write_text(json.dumps(res, indent=1))
    line = (f"[{res['tag']}] cl={res['cl']:+.4f} cd={res['cd']:.4f} "
            f"steps={res['steps_run']} conv={res['converged']} "
            f"{res['s_per_step']:.4f}s/step wall={res['elapsed_s']:.0f}s "
            f"div={res['diverged']}")
    return {"tag": res["tag"], "line": line, "cl": res["cl"], "cd": res["cd"],
            "ok": not res["diverged"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/naca0012_polar")
    ap.add_argument("--cases-file", default=None)
    ap.add_argument("--grids", type=int, nargs="+", default=[64, 96, 144])
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[-2.0, 0.0, 2.0, 4.0, 6.0, 8.0])
    ap.add_argument("--re", type=float, nargs="+", default=[1.0e4])
    ap.add_argument("--steps", type=int, default=25000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--tag-prefix", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.cases_file:
        specs = json.loads(Path(args.cases_file).read_text())
    else:
        specs = []
        for chord, alpha, re_v in itertools.product(args.grids, args.alphas, args.re):
            tag = args.tag_prefix + f"c{chord}_re{re_v:.0e}_a{alpha:+.1f}"
            specs.append({"chord": chord, "alpha": alpha, "re": re_v,
                          "steps": args.steps, "tag": tag})
    for s in specs:
        s.setdefault("out", args.out)
        s["threads"] = args.threads
    if args.nshard > 1:
        specs = specs[args.shard::args.nshard]
    print(f"running {len(specs)} cases (shard {args.shard}/{args.nshard}, "
          f"{args.workers} workers x {args.threads} threads)", flush=True)
    if args.dry_run:
        for s in specs:
            print(" ", s["tag"])
        return 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "run_log.jsonl"
    t0 = time.time()
    with mp.Pool(processes=args.workers) as pool:
        for res in pool.imap_unordered(run_one, specs):
            print(f"{res['line']}  [{time.time() - t0:.0f}s elapsed]", flush=True)
            with log.open("a") as fh:
                fh.write(json.dumps(res) + "\n")
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:  # pragma: no cover - cluster diagnostics
        traceback.print_exc()
        raise SystemExit(1)