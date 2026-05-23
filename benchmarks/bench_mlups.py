"""MLUPS (Million Lattice-site Updates Per Second) benchmark.

Measures raw throughput of core LBM kernels on the available device.
Run with::

    PYTHONPATH=src python benchmarks/bench_mlups.py [--device cpu|cuda|mps]

Results are printed to stdout in a structured table.
"""
from __future__ import annotations

import argparse
import time

import torch

from tensorlbm import collide_bgk, equilibrium, stream
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.solver3d import collide_bgk3d, stream3d
from tensorlbm.utils import resolve_device


def _mlups_2d(
    ny: int,
    nx: int,
    n_warmup: int,
    n_measure: int,
    tau: float,
    device: torch.device,
) -> float:
    rho = torch.ones((ny, nx), device=device)
    f = equilibrium(rho, torch.zeros_like(rho), torch.zeros_like(rho))
    for _ in range(n_warmup):
        f = collide_bgk(f, tau)
        f = stream(f)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_measure):
        f = collide_bgk(f, tau)
        f = stream(f)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return (n_measure * ny * nx) / (elapsed * 1e6)


def _mlups_3d(
    nz: int,
    ny: int,
    nx: int,
    n_warmup: int,
    n_measure: int,
    tau: float,
    device: torch.device,
) -> float:
    rho = torch.ones((nz, ny, nx), device=device)
    f = equilibrium3d(
        rho,
        torch.zeros_like(rho),
        torch.zeros_like(rho),
        torch.zeros_like(rho),
        device=device,
    )
    for _ in range(n_warmup):
        f = collide_bgk3d(f, tau)
        f = stream3d(f)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_measure):
        f = collide_bgk3d(f, tau)
        f = stream3d(f)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return (n_measure * nz * ny * nx) / (elapsed * 1e6)


def main() -> None:
    parser = argparse.ArgumentParser(description="TensorLBM MLUPS benchmark")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print()
    print(f"{'Config':<30} {'MLUPS':>10}")
    print("-" * 42)

    for ny, nx in [(256, 256), (512, 512), (1024, 1024)]:
        mlups = _mlups_2d(ny, nx, args.warmup, args.steps, tau=0.6, device=device)
        print(f"D2Q9  {ny}×{nx:<20}  {mlups:>10.2f}")

    for nz, ny, nx in [(32, 32, 32), (64, 64, 64), (128, 64, 64)]:
        mlups = _mlups_3d(nz, ny, nx, args.warmup, args.steps, tau=0.6, device=device)
        print(f"D3Q19 {nz}×{ny}×{nx:<17}  {mlups:>10.2f}")


if __name__ == "__main__":
    main()
