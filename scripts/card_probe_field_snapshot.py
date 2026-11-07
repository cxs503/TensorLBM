#!/usr/bin/env python3
"""Dump a cylinder Re=100 field snapshot for the m5113 probe report.

Produces artifacts/card_probe_field.npz with ux, uy, vorticity and the solid
mask for the exact case used by scripts/card_interference_probe.py, so the
report figure is generated from the same workload that was timed.

Usage:
    SDAA_VISIBLE_DEVICES=14 python scripts/card_probe_field_snapshot.py \
        --steps 2400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import torch  # noqa: E402
import torch_sdaa  # noqa: F401,E402

from card_interference_probe import CASE, build_cylinder_case, sync  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--out", default=str(REPO_ROOT / "artifacts" / "card_probe_field.npz"))
    args = ap.parse_args()

    from tensorlbm import OPPOSITE, collide_bgk, equilibrium, macroscopic, stream

    import os
    if os.environ.get("CARD_PROBE_NY"):   # blockage-confinement sensitivity study
        CASE["ny"] = int(os.environ["CARD_PROBE_NY"])
        CASE["cy"] = CASE["ny"] / 2.0

    dev = torch.device("sdaa:0")
    nx, ny = CASE["nx"], CASE["ny"]
    tau, u_in = CASE["tau"], CASE["u_in"]

    solid_cpu, cyl = build_cylinder_case(nx, ny, CASE["diameter"], CASE["cx"], CASE["cy"])
    solid = solid_cpu.to(dev)
    opp = OPPOSITE.to(dev)

    f = equilibrium(
        torch.ones((ny, nx), dtype=torch.float32, device=dev),
        torch.full((ny, nx), u_in, dtype=torch.float32, device=dev),
        torch.zeros((ny, nx), dtype=torch.float32, device=dev),
        device=dev,
    )
    eq_col = equilibrium(
        torch.ones(ny, dtype=torch.float32, device=dev),
        torch.full((ny,), u_in, dtype=torch.float32, device=dev),
        torch.zeros(ny, dtype=torch.float32, device=dev),
        device=dev,
    ).reshape(9, ny)

    cd_hist = []
    for step in range(args.warmup + args.steps):
        f = collide_bgk(f, tau)
        f = stream(f)
        fs = f[opp]
        f = torch.where(solid.unsqueeze(0), fs, f)
        f[:, :, 0] = eq_col
        f[:, :, -1] = f[:, :, -2]
        if step >= args.warmup:
            fxc = 2.0 * (
                torch.tensor([0, 1, 0, -1, 0, 1, -1, -1, 1],
                             dtype=torch.float32, device=dev).view(9, 1)
                * f[:, cyl.to(dev)]
            ).sum()
            cd_hist.append(float(abs(fxc) / (0.5 * 1.0 * u_in**2 * CASE["diameter"])))
        if step % 500 == 0:
            print(f"  step {step}/{args.warmup + args.steps}", flush=True)
    sync(dev)

    rho, ux, uy = macroscopic(f)
    ux_np = ux.detach().cpu().numpy()
    uy_np = uy.detach().cpu().numpy()

    # vorticity (central differences, interior only)
    wx = np.zeros_like(ux_np)
    wx[1:-1, 1:-1] = (
        (uy_np[2:, 1:-1] - uy_np[:-2, 1:-1]) / 2.0
        - (ux_np[1:-1, 2:] - ux_np[1:-1, :-2]) / 2.0
    )

    cx_grid = CASE["cx"]
    fx = 2.0 * (
        torch.tensor([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=torch.float32, device=dev)
        .view(9, 1) * f[:, cyl.to(dev)]
    ).sum()
    cd = float(abs(fx) / (0.5 * 1.0 * u_in**2 * CASE["diameter"]))

    np.savez_compressed(
        args.out,
        ux=ux_np, uy=uy_np, vorticity=wx, solid=solid_cpu.numpy(),
        cd=cd, cd_mean=float(np.mean(cd_hist[-2000:])),
        cd_min=float(np.min(cd_hist[-2000:])), cd_max=float(np.max(cd_hist[-2000:])),
        cd_series=np.asarray(cd_hist, dtype=np.float32),
        tau=tau, u_in=u_in, diameter=CASE["diameter"],
        cx=cx_grid, cy=CASE["cy"], steps=args.steps,
    )
    h = np.asarray(cd_hist[-2000:])
    print(f"Cd_instant={cd:.4f}  Cd_mean={h.mean():.4f} "
          f"[{h.min():.3f},{h.max():.3f}]  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())