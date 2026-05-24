"""MLUPS (Million Lattice-site Updates Per Second) benchmark.

Measures raw throughput of core LBM kernels on the available device.
Run with::

    PYTHONPATH=src python benchmarks/bench_mlups.py [--device cpu|cuda|mps] [--compile]
    PYTHONPATH=src python benchmarks/bench_mlups.py --collisions all

Results are printed to stdout in a structured table.
"""
from __future__ import annotations

import argparse
import time
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

from tensorlbm import (
    collide_bgk,
    collide_mrt,
    collide_rlbm,
    collide_trt,
    equilibrium,
    stream,
)
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.d3q27 import collide_bgk27, equilibrium27, stream27
from tensorlbm.solver3d import collide_bgk3d, collide_mrt3d, collide_rlbm3d, collide_trt3d, stream3d
from tensorlbm.utils import resolve_device

# Map of collision-name → (2D kernel, 3D kernel, kwargs)
_COLLISIONS_2D: dict[str, Callable] = {
    "bgk": lambda f, tau: collide_bgk(f, tau=tau),
    "mrt": lambda f, tau: collide_mrt(f, tau=tau),
    "trt": lambda f, tau: collide_trt(f, tau_plus=tau),
    "rlbm": lambda f, tau: collide_rlbm(f, tau=tau),
}
_COLLISIONS_3D: dict[str, Callable] = {
    "bgk": lambda f, tau: collide_bgk3d(f, tau=tau),
    "mrt": lambda f, tau: collide_mrt3d(f, tau=tau),
    "trt": lambda f, tau: collide_trt3d(f, tau_plus=tau),
    "rlbm": lambda f, tau: collide_rlbm3d(f, tau=tau),
}


def _maybe_compile(fn: object, use_compile: bool) -> object:
    if not use_compile:
        return fn
    try:
        return torch.compile(fn)  # type: ignore[attr-defined]
    except AttributeError:
        return fn


def _mlups_2d(
    ny: int,
    nx: int,
    n_warmup: int,
    n_measure: int,
    tau: float,
    device: torch.device,
    use_compile: bool = False,
    collision: str = "bgk",
) -> float:
    _collide = _maybe_compile(_COLLISIONS_2D[collision], use_compile)
    _stream = _maybe_compile(stream, use_compile)
    rho = torch.ones((ny, nx), device=device)
    f = equilibrium(rho, torch.zeros_like(rho), torch.zeros_like(rho))
    for _ in range(n_warmup):
        f = _collide(f, tau)
        f = _stream(f)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_measure):
        f = _collide(f, tau)
        f = _stream(f)
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
    use_compile: bool = False,
    collision: str = "bgk",
) -> float:
    _collide = _maybe_compile(_COLLISIONS_3D[collision], use_compile)
    _stream = _maybe_compile(stream3d, use_compile)
    rho = torch.ones((nz, ny, nx), device=device)
    f = equilibrium3d(
        rho,
        torch.zeros_like(rho),
        torch.zeros_like(rho),
        torch.zeros_like(rho),
        device=device,
    )
    for _ in range(n_warmup):
        f = _collide(f, tau)
        f = _stream(f)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_measure):
        f = _collide(f, tau)
        f = _stream(f)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return (n_measure * nz * ny * nx) / (elapsed * 1e6)


def _mlups_d3q27(
    nz: int,
    ny: int,
    nx: int,
    n_warmup: int,
    n_measure: int,
    tau: float,
    device: torch.device,
    use_compile: bool = False,
) -> float:
    _collide = _maybe_compile(collide_bgk27, use_compile)
    _stream = _maybe_compile(stream27, use_compile)
    rho = torch.ones((nz, ny, nx), device=device)
    f = equilibrium27(
        rho,
        torch.zeros_like(rho),
        torch.zeros_like(rho),
        torch.zeros_like(rho),
        device=device,
    )
    for _ in range(n_warmup):
        f = _collide(f, tau)
        f = _stream(f)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_measure):
        f = _collide(f, tau)
        f = _stream(f)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return (n_measure * nz * ny * nx) / (elapsed * 1e6)


def main() -> None:
    parser = argparse.ArgumentParser(description="TensorLBM MLUPS benchmark")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--compile",
        action="store_true",
        dest="use_compile",
        help="Wrap kernels with torch.compile (requires PyTorch >= 2.0)",
    )
    parser.add_argument(
        "--collisions",
        default="bgk",
        help=(
            "Comma-separated list of collision operators to benchmark "
            "(bgk, mrt, trt, rlbm) or 'all'."
        ),
    )
    args = parser.parse_args()

    if args.collisions.lower() == "all":
        collisions = ["bgk", "mrt", "trt", "rlbm"]
    else:
        collisions = [c.strip().lower() for c in args.collisions.split(",") if c.strip()]
    for c in collisions:
        if c not in _COLLISIONS_2D:
            msg = f"Unknown collision '{c}'. Choose from {list(_COLLISIONS_2D)} or 'all'."
            raise SystemExit(msg)

    device = resolve_device(args.device)
    compile_label = " [compiled]" if args.use_compile else ""
    print(f"Device: {device}{compile_label}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Collisions: {', '.join(collisions)}")
    print()
    print(f"{'Config':<32} {'Collision':<8} {'MLUPS':>10}")
    print("-" * 54)

    for collision in collisions:
        for ny, nx in [(256, 256), (512, 512), (1024, 1024)]:
            mlups = _mlups_2d(
                ny, nx, args.warmup, args.steps,
                tau=0.6, device=device, use_compile=args.use_compile,
                collision=collision,
            )
            print(f"D2Q9  {ny}×{nx:<22}  {collision:<8} {mlups:>10.2f}")

        for nz, ny, nx in [(32, 32, 32), (64, 64, 64), (128, 64, 64)]:
            mlups = _mlups_3d(
                nz, ny, nx, args.warmup, args.steps,
                tau=0.6, device=device, use_compile=args.use_compile,
                collision=collision,
            )
            print(f"D3Q19 {nz}×{ny}×{nx:<19}  {collision:<8} {mlups:>10.2f}")

    for nz, ny, nx in [(32, 32, 32), (64, 64, 64)]:
        mlups = _mlups_d3q27(
            nz, ny, nx, args.warmup, args.steps,
            tau=0.6, device=device, use_compile=args.use_compile,
        )
        print(f"D3Q27 {nz}×{ny}×{nx:<19}  {'bgk':<8} {mlups:>10.2f}")


if __name__ == "__main__":
    main()
