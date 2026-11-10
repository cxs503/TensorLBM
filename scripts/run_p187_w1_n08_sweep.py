#!/usr/bin/env python3
"""P187-W1-N08 扫掠驱动：并行推进 (H/D x 网格级) 全部算例，支持分块续跑。

设计：本机单进程吞吐 ~1.4 Mnodes/s（多线程在 loong64 上反而更慢，实测见报告），
故用 **多进程 x 单线程** 并行；每个子进程带 ``--time-budget``，到点落盘并把
``status=partial`` 写进 JSON，下一次调用从断点续跑，直到 ``status=done``。

用法：
    python scripts/run_p187_w1_n08_sweep.py --time-budget 85 --workers 12
重复调用直到打印 ``ALL DONE``。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "p187_w1_n08"

DEFAULT_HDS = (4.0, 6.0, 8.0, 12.0, 16.0, 24.0)
DEFAULT_DIAMETERS = (11, 17)


def case_paths(hd: float, dia: int):
    stem = f"hd{hd:04.1f}_D{dia}"
    return OUT_DIR / f"{stem}.json", OUT_DIR / f"{stem}.pt", OUT_DIR / f"{stem}.log"


def is_done(jpath: Path) -> bool:
    if not jpath.exists():
        return False
    try:
        with open(jpath) as fh:
            return json.load(fh).get("status") == "done"
    except Exception:
        return False


def launch(hd: float, dia: int, args) -> dict:
    jpath, ckpt, logp = case_paths(hd, dia)
    cmd = [
        sys.executable, str(ROOT / "scripts" / "run_p187_w1_n08_case.py"),
        "--h-over-d", str(hd), "--diameter", str(dia),
        "--re", str(args.re), "--pr", str(args.pr), "--u-in", str(args.u_in),
        "--upstream-d", str(args.upstream_d), "--downstream-d", str(args.downstream_d),
        "--t-end-d", str(args.t_end_d), "--dt-sample-d", str(args.dt_sample_d),
        "--dtype", args.dtype, "--threads", "1",
        "--checkpoint", str(ckpt), "--time-budget", str(args.time_budget),
        "--ckpt-every", str(args.ckpt_every), "--out", str(jpath),
    ]
    if args.save_fields:
        cmd.append("--save-fields")
    t0 = time.time()
    with open(logp, "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
    return {
        "case": f"hd{hd:g}_D{dia}", "rc": proc.returncode,
        "wall_s": round(time.time() - t0, 1),
        "done": is_done(jpath),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hds", type=float, nargs="+", default=list(DEFAULT_HDS))
    ap.add_argument("--diameters", type=int, nargs="+", default=list(DEFAULT_DIAMETERS))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--time-budget", type=float, default=85.0)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--pr", type=float, default=0.7)
    ap.add_argument("--u-in", type=float, default=0.1)
    ap.add_argument("--upstream-d", type=float, default=5.0)
    ap.add_argument("--downstream-d", type=float, default=13.0)
    ap.add_argument("--t-end-d", type=float, default=110.0)
    ap.add_argument("--dt-sample-d", type=float, default=0.1)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--save-fields", action="store_true")
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = [(hd, d) for d in args.diameters for hd in args.hds]
    todo = [(hd, d) for hd, d in cases if not is_done(case_paths(hd, d)[0])]
    if not todo:
        print("ALL DONE")
        return 0
    print(f"{len(todo)}/{len(cases)} 待推进：{[f'hd{hd:g}_D{d}' for hd, d in todo]}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(lambda c: launch(c[0], c[1], args), todo))
    for r in results:
        print(f"  {r['case']:12s} rc={r['rc']} wall={r['wall_s']}s done={r['done']}", flush=True)
    remaining = [c for c in cases if not is_done(case_paths(*c)[0])]
    print("REMAINING" if remaining else "ALL DONE", len(remaining), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())