#!/usr/bin/env python
"""REPRO-LBM-GUO2002 —— 网格阶梯驱动（集群上跑一个子集，逐配置落 JSON）。

配置串格式 ``ra:ny:ma_max``（逗号分隔），例：

    python scripts/repro_guo2002_ladder.py --runs 1e5:192:0.1,1e5:256:0.1 \
        --outdir runs/repro_guo2002_dhc --fields

``--probe-steps N``：只跑 N 步做**计时探针**（不写结果，只报 ms/step 与
按 ``run_to_steady`` 预算公式估的步数），用于决定网格阶梯上不上、要不要拆
作业——判据先于动手，避免盲跑几小时。

与 worker 同源：物理一律走 ``tensorlbm.rb_convection_validation
.run_to_steady``，本文件只做循环编排与落盘。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorlbm import rb_convection_validation as V  # noqa: E402

REF = V.DE_VAHL_DAVIS_NU


def parse_runs(spec: str) -> list[tuple[float, int, float]]:
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        ra, ny, ma = tok.split(":")
        out.append((float(ra), int(ny), float(ma)))
    return out


def probe(cfg: tuple[float, int, float], steps: int, threads: int) -> dict:
    ra, ny, ma = cfg
    torch.set_num_threads(threads)
    sim = V.make_convection(ny=ny, nx=ny, ra=ra, pr=0.71, case="dhc", ma_max=ma)
    sim.step(min(50, steps))
    t0 = time.time()
    sim.step(steps)
    dt = time.time() - t0
    info = sim.info()
    import math
    u_ff = math.sqrt(max(sim.g_beta * sim.dT * sim.H, 1e-30))
    t_turn = sim.H / u_ff
    diff_steps = sim.H ** 2 / sim.alpha_lat
    budget = min(max(20000, 120.0 * t_turn), diff_steps)
    return {
        "ra": ra, "ny": ny, "ma_max": ma,
        "ms_per_step": round(1e3 * dt / steps, 3),
        "us_per_cell": round(1e6 * dt / steps / (ny * ny), 3),
        "t_turn": round(t_turn, 1), "diff_steps": int(diff_steps),
        "budget_120turnovers": int(budget),
        "est_wall_s": round(dt / steps * budget, 1),
        "tau_f": info["tau_f"], "alpha": info["alpha_lat"], "ma_estimate": info["ma_estimate"],
        "threads": threads,
    }


def run_one(cfg: tuple[float, int, float], args, outdir: Path) -> dict:
    ra, ny, ma = cfg
    dtype = torch.float64 if args.dtype == "fp64" else torch.float32
    t0 = time.time()
    out = V.run_to_steady(
        case="dhc", ny=ny, ra=ra, pr=0.71, ma_max=ma, dtype=dtype,
        turnover_target=args.turnover, min_steps=args.min_steps,
        max_steps=args.max_steps, report_every=args.report_every,
        verbose=args.verbose, return_fields=args.fields,
    )
    fields = {k: out.pop(k) for k in ("T_field", "u_field", "v_field") if k in out}
    ref = REF[ra]
    out.update({
        "anchor": "Guo, Shi & Zheng 2002, Int. J. Numer. Meth. Fluids 39:325-342",
        "ref_source": "de Vahl Davis 1983, Int. J. Numer. Meth. Fluids 3:249-264",
        "nu_ref": ref, "err_pct": 100.0 * abs(out["nu"] - ref) / ref,
        "bias_pct": 100.0 * (out["nu"] - ref) / ref,
        "ra_effective_by_gap": ra * ((ny - 2) / ny) ** 3,
        "ma_max_target": ma, "dtype": args.dtype,
        "wall_s": round(time.time() - t0, 1),
    })
    tag = f"ra{ra:g}_ny{ny}_ma{ma:g}"
    (outdir / f"{tag}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    if args.fields and fields:
        import numpy as np
        np.savez_compressed(outdir / f"{tag}.npz",
                            **{k: v.numpy() for k, v in fields.items()})
    print(f"[ladder] Ra={ra:g} ny={ny} ma={ma:g} Nu={out['nu']:.5f} "
          f"ref={ref:.4f} err={out['err_pct']:.2f}% steps={out['steps']} "
          f"turn={out['turnovers']:.1f} Ma={out['ma']:.3f} wall={out['wall_s']}s",
          flush=True)
    return {"ra": ra, "ny": ny, "ma_max": ma, "nu": out["nu"], "err_pct": out["err_pct"],
            "steps": out["steps"], "turnovers": out["turnovers"], "wall_s": out["wall_s"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="ra:ny:ma_max[,ra:ny:ma_max...]")
    ap.add_argument("--outdir", default="runs/repro_guo2002_dhc")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--turnover", type=float, default=120.0)
    ap.add_argument("--min-steps", type=int, default=20000)
    ap.add_argument("--max-steps", type=int, default=1_500_000)
    ap.add_argument("--report-every", type=int, default=2000)
    ap.add_argument("--dtype", choices=["fp32", "fp64"], default="fp64")
    ap.add_argument("--fields", action="store_true")
    ap.add_argument("--probe-steps", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfgs = parse_runs(args.runs)
    if args.probe_steps:
        rows = [probe(c, args.probe_steps, args.threads) for c in cfgs]
        (outdir / "probe.json").write_text(json.dumps(rows, indent=1))
        for r in rows:
            print(json.dumps(r, ensure_ascii=False), flush=True)
        return 0
    summary = []
    for c in cfgs:
        summary.append(run_one(c, args, outdir))
        (outdir / "ladder_summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())