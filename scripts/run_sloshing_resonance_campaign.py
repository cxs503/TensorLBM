#!/usr/bin/env python3
"""Driver for the P87W2-N01 sloshing-resonance acceptance campaign.

Usage::

    # run one shard of the campaign (cases are deterministic, so sharding is safe)
    python scripts/run_sloshing_resonance_campaign.py --out artifacts/p87w2_n01 --shard 0/4
    # judge + write campaign.json once every case JSON is present
    python scripts/run_sloshing_resonance_campaign.py --out artifacts/p87w2_n01 --aggregate

Each case writes ``case_<id>.json``; the aggregation step re-reads them, applies
criteria (a)-(h) and writes ``campaign.json``.  Nothing is recomputed twice, so
a shard can be re-run or extended without losing the finished cases.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(int(os.environ.get("TL_THREADS", "1")))

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.sloshing_resonance.campaign import (  # noqa: E402
    CampaignSpec,
    case_list,
    judge_campaign,
    run_case,
    write_json,
)


def parse_shard(text: str) -> tuple[int, int]:
    i, n = text.split("/")
    return int(i), int(n)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "p87w2_n01")
    ap.add_argument("--shard", default="0/1", help="i/n shard of the case list")
    ap.add_argument("--cells", type=int, default=40, help="base cells across the tank height")
    ap.add_argument("--tau", type=float, default=0.502)
    ap.add_argument("--friction", action="store_true", help="use no-slip walls (sensitivity study)")
    ap.add_argument("--periods", type=float, default=4.0)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated case ids to run")
    args = ap.parse_args(argv)

    spec = CampaignSpec(
        cells_per_height=args.cells,
        tau=args.tau,
        n_periods=args.periods,
        slip_walls=not args.friction,
        convergence_periods=(args.periods, 3.0, 2.5) if args.periods >= 3 else (args.periods,) * 3,
    )
    cases = case_list(spec)
    if args.list:
        for c in cases:
            print(c["id"], c["kind"], c.get("depth"), c.get("mode"), c.get("cells"))
        return 0

    if args.aggregate:
        records = []
        for p in sorted(args.out.glob("case_*.json")):
            records.append(json.loads(p.read_text()))
        judged = judge_campaign(spec, records)
        judged["spec"] = spec.as_dict()
        write_json(args.out / "campaign.json", judged)
        print(json.dumps({k: v for k, v in judged["criteria"].items()}, indent=2)[:4000])
        print(f"criteria pass: {judged['n_criteria_pass']}/{judged['n_criteria_total']}")
        return 0

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    i, n = parse_shard(args.shard)
    todo = [c for c in cases if (not only or c["id"] in only)]
    todo = [c for idx, c in enumerate(todo) if idx % n == i]
    print(f"shard {i}/{n}: {len(todo)} cases -> {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    for case in todo:
        path = args.out / f"case_{case['id']}.json"
        if path.exists() and not only:
            print(f"  [skip] {case['id']} (exists)")
            continue
        t0 = time.time()
        rec = run_case(spec, case)
        write_json(path, rec)
        msg = f"  [done] {case['id']} in {time.time() - t0:.0f}s"
        for key in (
            "f_theory",
            "f_damped",
            "f_natural",
            "zeta",
            "err_natural_pct",
            "phase_error_deg",
            "p_peak_pa",
            "swing_ratio",
        ):
            if key in rec:
                msg += f" {key}={rec[key]:.5g}"
        print(msg, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
