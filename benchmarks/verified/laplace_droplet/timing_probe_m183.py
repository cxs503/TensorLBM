#!/usr/bin/env python
"""Cost probe for the mission-25 failure diagnosis (ms/step per configuration).

Rationale: mission 25 failed as "finish not called" - i.e. a process/budget
failure, not a numerics failure.  To make that claim quantitative instead of
anecdotal, measure the actual per-step cost of every configuration the case
would run with its defaults, so the total serial wall clock can be estimated.

Usage: python timing_probe_m183.py --threads 16 --out artifacts/m183_laplace_droplet/timing_probe.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def load_case():
    spec = importlib.util.spec_from_file_location("laplace_bench_case", HERE / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--nsteps", type=int, default=15)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    case = load_case()
    rows = []
    for dim, L, R in ((2, 60, 15), (2, 100, 25), (2, 160, 40),
                      (3, 60, 15), (3, 100, 25), (3, 160, 40)):
        rho = case.init_rho(dim, L, R, torch.device("cpu"))
        z = torch.zeros_like(rho)
        f = (case.equilibrium(rho, z, z) if dim == 2
             else case.equilibrium3d(rho, z, z, z))

        def step(state, dim=dim):
            if dim == 2:
                return case.stream(case.collide_sc_single_component(
                    state, G=case.G_LIB, tau=case.TAU, psi_fn=case.psi_exp))
            return case.stream3d(case.collide_sc_single_component_3d(
                state, G=case.G_LIB, tau=case.TAU, psi_fn=case.psi_exp))

        step(f)  # warm-up
        t0 = time.perf_counter()
        for _ in range(a.nsteps):
            f = step(f)
        ms = (time.perf_counter() - t0) / a.nsteps * 1000.0
        rows.append({"label": f"D{dim}Q{9 if dim == 2 else 19} R={R} L={L}",
                     "dim": dim, "R": R, "L": L,
                     "n_cells": L ** dim, "ms_per_step": ms,
                     "threads": a.threads,
                     "note": "CPU, eager, measured under concurrent sweep load"})
        print(rows[-1])
    total = sum(r["ms_per_step"] * 6000 / 1000.0 for r in rows)  # min 6000 steps per config
    payload = {"rows": rows, "threads": a.threads, "nsteps": a.nsteps,
               "assumed_steps": 6000,
               "serial_total_s_at_6000_steps": total,
               "torch": torch.__version__,
               "cuda_available": torch.cuda.is_available(),
               "load_note": "measured while the 7 sweep jobs were running (load ~1.0/core)"}
    if a.out:
        Path(a.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print("saved ->", a.out)
    print("serial total at 6000 steps/config: %.0f s (%.1f min)" % (total, total / 60))


if __name__ == "__main__":
    main()
