#!/usr/bin/env python
"""等温 Sod 激波管（D2Q9）— 波系结构可视化演示脚本。

与正式验证档 benchmarks/verified/sod_shock_tube/run.py 使用同一组库入口与
同一步进链（stream → collide_bgk，周期 gather）：
  - tensorlbm.d2q9.equilibrium / macroscopic
  - tensorlbm.solver.collide_bgk / stream

初始条件与正式档完全相同：x=nx/2 处密度间断 ρ_L=1.0 / ρ_R=0.25、u=0、
f=feq（精确单元间断）；τ=0.8（ν=0.1）、ny=4。演示档仅缩短管长与演化时间
（nx=1000、t≤200；正式档 nx=2000/4000、t≤400/800），保留稀疏波/中间态/
激波全结构。定量判据一律取正式档 result.json。

用法：
    python docs/benchmarks/demos/sod_shock_tube_demo.py \
        --nx 1000 --steps 200 --out demo.npz
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_SRC = os.environ.get("TENSORLBM_SRC") or str(
    Path(__file__).resolve().parents[3] / "src"
)
sys.path.insert(0, REPO_SRC)

from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_bgk, stream  # noqa: E402

CS = 1.0 / math.sqrt(3.0)


def main():
    ap = argparse.ArgumentParser(description="等温 Sod 激波管演示")
    ap.add_argument("--nx", type=int, default=1000, help="管长（正式档 2000/4000）")
    ap.add_argument("--ny", type=int, default=4)
    ap.add_argument("--rho-l", type=float, default=1.0)
    ap.add_argument("--rho-r", type=float, default=0.25)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--snaps", type=int, nargs="+", default=[50, 100, 150, 200])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="sod_shock_tube_demo.npz")
    args = ap.parse_args()

    torch.set_num_threads(min(32, os.cpu_count() or 1))
    dev = torch.device(args.device)

    nx, ny = args.nx, args.ny
    tau = args.tau
    xd = nx // 2
    x = torch.arange(nx, dtype=torch.float32)
    rho = torch.where(x < xd, torch.full_like(x, args.rho_l),
                      torch.full_like(x, args.rho_r))
    rho = rho.unsqueeze(0).expand(ny, nx)
    f = equilibrium(rho, torch.zeros_like(rho), torch.zeros_like(rho))

    print(f"tensorlbm src ok; nx={nx} tau={args.tau} snaps={args.snaps}", flush=True)

    t0 = time.time()
    profs = {}
    snaps = set(args.snaps)
    for t in range(args.steps + 1):
        if t in snaps:
            rr, uu, _ = macroscopic(f)
            profs[t] = (rr[0].numpy().copy(), uu[0].numpy().copy())
        f = stream(f)
        f = collide_bgk(f, tau)
    elapsed = time.time() - t0

    rr, uu, _ = macroscopic(f)
    assert torch.isfinite(rr).all() and (rr > 0).all(), "demo run unstable"

    out = {"nx": nx, "ny": ny, "tau": args.tau, "rho_l": args.rho_l,
           "rho_r": args.rho_r, "steps": args.steps, "elapsed_s": elapsed}
    for t in sorted(profs):
        out[f"rho_t{t}"] = profs[t][0]
        out[f"u_t{t}"] = profs[t][1]
    np.savez(args.out, **out)
    print(
        f"saved {args.out}: nx={nx} t_max={args.steps} "
        f"rho_min={float(rr.min()):.4f} t={elapsed:.1f}s（演示档）",
        flush=True,
    )


if __name__ == "__main__":
    main()
