"""Run the three-confinement rising-bubble sweep (IN57-N02 criterion (1)).

Writes ``artifacts/bubble_dr_in57_n02/bubble_sweep.json``.  Three box sizes are
run in separate processes (the solver is single-process torch, one thread pool
each) because one 88^3 run alone takes minutes.

Usage::

    PYTHONPATH=src python3 scripts/run_bubble3d_sweep.py [--steps 1200] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

BASE = dict(
    radius=8.0,
    G=5.0,
    tau=1.0,
    g_buoy=5.0e-5,
    steps=1200,
    W_int=4.0,
)


def _one(args: tuple[int, int]) -> dict:
    L, threads = args
    import torch

    torch.set_num_threads(threads)
    from tensorlbm.bubble_dr.bubble3d import BubbleConfig, RisingBubble3D

    cfg = BubbleConfig(L=L, dtype=torch.float64, **BASE)
    sim = RisingBubble3D(cfg)
    return sim.run()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--boxes", type=str, default="56,72,88")
    ap.add_argument("--out", type=str, default="artifacts/bubble_dr_in57_n02")
    ap.add_argument("--threads", type=int, default=32)
    args = ap.parse_args()

    BASE["steps"] = args.steps
    boxes = [int(v) for v in args.boxes.split(",") if v]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(len(boxes)) as pool:
        records = pool.map(_one, [(L, args.threads) for L in boxes])
    from tensorlbm.bubble_dr.bubble3d import confinement_extrapolation

    payload = {
        "config": {**BASE, "boxes": boxes},
        "records": records,
        "extrapolation": confinement_extrapolation(records),
        "wall_seconds": time.perf_counter() - t0,
    }
    (out / "bubble_sweep.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["extrapolation"], indent=2))


if __name__ == "__main__":
    main()