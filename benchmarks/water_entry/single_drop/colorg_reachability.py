#!/usr/bin/env python3
"""PF3-COLORG: density-ratio-1000 reachability probes for the two-PDF
partial-density colour-gradient (Latva-Kokko / Rothman-Keller) model.

Runs three probes and writes a JSON summary:
  1. static stratification (heavy below, light above, no body force)
  2. static droplet (heavy droplet in a light bath, quiescent)
  3. coarse-band water entry (falling drop into a pool, 2-D cross-section)

The DE ctrl baseline (free-surface LBM) is 1.435220 D @ 11.0906 ms (abs);
the experimental anchor is 1.857434 D @ 10.394 ms (pc); window 24.005 ms.

Additive: nothing in the existing modules is modified.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root))

from tensorlbm.color_gradient_density_ratio import (
    run_droplet_probe,
    run_stratified_probe,
)
from benchmarks.water_entry.single_drop.colorg_water_entry import run_water_entry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--ratio", type=float, default=1000.0)
    ap.add_argument("--width", type=float, default=8.0)
    ap.add_argument("--nu", type=float, default=5.0)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--A", type=float, default=0.0)
    a = ap.parse_args()

    dev = torch.device("cpu")
    out: dict = {"ratio": a.ratio, "width": a.width, "nu": a.nu, "beta": a.beta, "A": a.A}

    # 1. static stratification
    t0 = time.perf_counter()
    r1 = run_stratified_probe(
        64, 32, a.ratio, 1.0,
        nu_a=a.nu, nu_b=a.nu, A=a.A, beta=a.beta,
        steps=500, width=a.width, dev=dev, sample_every=100,
    )
    r1["wall_s"] = time.perf_counter() - t0
    out["stratified"] = r1

    # 2. static droplet
    t0 = time.perf_counter()
    r2 = run_droplet_probe(
        64, 16.0, a.ratio, 1.0,
        nu_a=a.nu, nu_b=a.nu, A=a.A, beta=a.beta,
        steps=500, width=a.width, dev=dev, sample_every=100,
    )
    r2["wall_s"] = time.perf_counter() - t0
    out["droplet"] = r2

    # 3. coarse-band water entry
    t0 = time.perf_counter()
    r3 = run_water_entry(
        20, a.ratio, 1.0,
        nu_a=a.nu, nu_b=a.nu, A=a.A, beta=a.beta,
        gy=1e-5, gap=1.0, steps=200, width=a.width, dev=dev,
    )
    r3["wall_s"] = time.perf_counter() - t0
    out["water_entry"] = r3

    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1, default=str))
        print(f"wrote {a.out}")
    # print a compact summary
    for name in ("stratified", "droplet", "water_entry"):
        r = out[name]
        print(f"{name}: stable={r['stable']} nan={r.get('nan_step')} "
              f"u_max={r.get('u_max', '-')} "
              f"depth={r.get('max_depth_D', '-')} "
              f"r_drift={r.get('radius_drift_rel', '-')} "
              f"wall={r.get('wall_s', 0):.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
