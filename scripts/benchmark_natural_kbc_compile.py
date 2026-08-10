#!/usr/bin/env python3
"""Measure eager/compiled natural-KBC equivalence, memory and tau reuse."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.entropic_kbc import (
    _collide_natural_kbc_d3q19_unchecked,
    collide_natural_kbc_d3q19,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--nz", type=int, default=16)
    parser.add_argument(
        "--taus", default="0.50324,0.50162,0.500324,0.500162",
        help="comma-separated relaxation times used to detect float specialization",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--tau-mode",
        choices=("scalar", "tensor"),
        default="scalar",
        help="compile a Python-float or reusable zero-dimensional tensor tau",
    )
    parser.add_argument("--output", type=Path)
    return parser


def _timed(callable_, *args: object) -> tuple[torch.Tensor, float, int]:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = callable_(*args)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return result, elapsed, torch.cuda.max_memory_allocated()


def main() -> None:
    args = _parser().parse_args()
    if min(args.nx, args.ny, args.nz, args.repeats) <= 0:
        raise ValueError("shape and repeats must be positive")
    taus = [float(value) for value in args.taus.split(",")]
    if not taus or min(taus) <= 0.5:
        raise ValueError("every tau must exceed 0.5")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("the compile benchmark requires CUDA")

    shape = (args.nz, args.ny, args.nx)
    rho = torch.ones(shape, device=device)
    x = torch.linspace(0.0, 1.0, args.nx, device=device)
    ux = 0.06 + 1.0e-3 * torch.sin(2.0 * torch.pi * x)[None, None, :]
    ux = ux.expand(shape)
    zero = torch.zeros_like(rho)
    populations = equilibrium3d(rho, ux, zero, zero)
    populations = populations + 1.0e-5 * torch.randn_like(populations)

    _, eager_seconds, eager_peak = _timed(
        collide_natural_kbc_d3q19, populations, taus[0],
    )
    compile_target = (
        collide_natural_kbc_d3q19
        if args.tau_mode == "scalar"
        else _collide_natural_kbc_d3q19_unchecked
    )
    compiled_collision = torch.compile(
        compile_target,
        dynamic=True,
        fullgraph=False,
        mode="reduce-overhead",
    )
    calls = []
    maximum_difference = 0.0
    for tau in taus:
        reference = collide_natural_kbc_d3q19(populations, tau)
        for repeat in range(args.repeats):
            tau_argument = (
                tau
                if args.tau_mode == "scalar"
                else torch.tensor(tau, device=device, dtype=populations.dtype)
            )
            result, seconds, peak = _timed(
                compiled_collision, populations, tau_argument,
            )
            difference = float((result - reference).abs().max().item())
            maximum_difference = max(maximum_difference, difference)
            calls.append({
                "tau": tau,
                "repeat": repeat,
                "seconds": seconds,
                "peak_allocated_bytes": peak,
                "maximum_absolute_difference": difference,
            })

    try:
        from torch._dynamo.utils import counters
        unique_graphs = int(counters["stats"]["unique_graphs"])
    except (ImportError, KeyError, TypeError, ValueError):
        unique_graphs = None
    payload = {
        "schema": "tensorlbm-natural-kbc-compile-benchmark-v1",
        "torch_version": torch.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "shape_qzyx": [19, *shape],
        "cells": args.nx * args.ny * args.nz,
        "taus": taus,
        "tau_mode": args.tau_mode,
        "eager_first_tau": {
            "seconds": eager_seconds,
            "peak_allocated_bytes": eager_peak,
        },
        "compiled_calls": calls,
        "maximum_absolute_difference": maximum_difference,
        "unique_graphs": unique_graphs,
        "admissible_equivalence_tolerance": 2.0e-7,
        "equivalence_pass": maximum_difference <= 2.0e-7,
    }
    text = json.dumps(payload, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
