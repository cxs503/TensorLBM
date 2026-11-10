#!/usr/bin/env python3
"""P187-W1-N08 单算例驱动：受限域等温加热圆柱 -> Nu(t) 时间序列 JSON。

用于集群分片（每个 shard 跑一个 H/D x 网格级），也可本地单跑。

示例：
    python scripts/run_p187_w1_n08_case.py --h-over-d 8 --diameter 25 \\
        --t-end-d 110 --out runs/p187_w1_n08/hd08_D25.json --threads 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.confined_cylinder_law import (  # noqa: E402
    ConfinedCylinderConfig,
    simulate_confined_cylinder,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h-over-d", type=float, default=8.0)
    ap.add_argument("--diameter", type=int, default=25, help="名义格子直径 2R")
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--pr", type=float, default=0.7)
    ap.add_argument("--u-in", type=float, default=0.1)
    ap.add_argument("--upstream-d", type=float, default=5.0)
    ap.add_argument("--downstream-d", type=float, default=13.0)
    ap.add_argument("--t-end-d", type=float, default=110.0)
    ap.add_argument("--dt-sample-d", type=float, default=0.1)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--save-fields", action="store_true")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--time-budget", type=float, default=0.0, help="秒；到点落盘返回 partial")
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    cfg = ConfinedCylinderConfig(
        h_over_d=args.h_over_d,
        diameter=args.diameter,
        re=args.re,
        pr=args.pr,
        u_in=args.u_in,
        upstream_d=args.upstream_d,
        downstream_d=args.downstream_d,
        t_end_d=args.t_end_d,
        dt_sample_d=args.dt_sample_d,
        dtype=args.dtype,
        threads=args.threads,
        save_fields=args.save_fields,
        label=f"hd{args.h_over_d:g}_D{args.diameter}",
        checkpoint=args.checkpoint,
        time_budget_s=args.time_budget,
        ckpt_every=args.ckpt_every,
    )
    res = simulate_confined_cylinder(cfg)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = res.pop("fields", None)
    res["host"] = os.uname().nodename
    res["machine"] = os.uname().machine
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    if fields is not None:
        np.savez_compressed(
            str(out).replace(".json", "_field.npz"),
            T=fields["T"], ux=fields["ux"], uy=fields["uy"], solid=fields["solid"],
        )
    print(
        f"[p187] {res['status']:7s} t={res['t_reached']:.1f}/{args.t_end_d} {res['label']} "
        f"nodes={res['geometry']['ny']}x{res['geometry']['nx']} "
        f"steps={res['lattice']['n_steps']} elapsed={res['elapsed_s']}s "
        f"Nu_tail={res['final']['nu_local_mean_tail']:.4f} "
        f"Nu_glob_tail={res['final']['nu_balance_mean_tail']:.4f} "
        f"surf_vs_glob={res['final']['surf_vs_global_pct']:.2f}% -> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())