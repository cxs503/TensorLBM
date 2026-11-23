#!/usr/bin/env python3
"""m23015 probe: TG decay stress sweep (low viscosity) for 4 operators.

Runs in background; writes JSON lines to runs/m23015/probe_tg_stress.jsonl.
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/tp-agent/work/m23015/src")

import torch

torch.set_num_threads(8)

from tensorlbm.model_matrix.registry import registry
from tensorlbm.model_matrix.cases import run_taylor_green_decay, Grid

OUT = Path("/root/tp-agent/work/m23015/runs/m23015")
OUT.mkdir(parents=True, exist_ok=True)
out_path = OUT / "probe_tg_stress.jsonl"

reg = registry()
MODELS = ["d3q19/mrt", "d3q19/trt", "d3q19/cumulant", "d3q27/kbc"]
TAUS = [0.51, 0.505]
N = 64

with open(out_path, "w") as fh:
    for tau in TAUS:
        for n in MODELS:
            t0 = time.time()
            try:
                r = run_taylor_green_decay(reg[n], Grid(N, N, 1), tau=tau, u0=0.05)
                m = r["metrics"]
                rec = {
                    "model": n,
                    "tau": tau,
                    "N": N,
                    "status": r["status"],
                    "nu_rel_error_pct": m["nu_rel_error_pct"],
                    "fit_r2": m.get("fit_r2"),
                    "wall_s": round(time.time() - t0, 1),
                }
            except Exception as e:  # noqa: BLE001
                rec = {
                    "model": n,
                    "tau": tau,
                    "N": N,
                    "status": "EXCEPTION",
                    "error": f"{type(e).__name__}: {e}",
                    "wall_s": round(time.time() - t0, 1),
                }
            print(json.dumps(rec), flush=True)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
print("DONE")
