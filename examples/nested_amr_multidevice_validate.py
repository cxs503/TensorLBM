#!/usr/bin/env python3
"""Validate hierarchy-level device partitioning against a one-device oracle."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.refinement import BoxRegion
from tensorlbm.solver3d import collide_mrt3d, stream3d
from tensorlbm.static_block_amr import (
    AMRAdvanceResult,
    NestedStaticBlockAMR3D,
    StaticBlockAMRConfig,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--devices", default="cuda:0,cuda:1")
    result.add_argument("--steps", type=int, default=12)
    result.add_argument("--seed", type=int, default=20260802)
    result.add_argument("--output", type=Path)
    return result


def _configs() -> tuple[StaticBlockAMRConfig, ...]:
    outer = StaticBlockAMRConfig(
        BoxRegion(x0=3, x1=11, y0=3, y1=9, z0=3, z1=9),
        tau_coarse=0.56,
        ghost_interpolation="trilinear",
        regularize_restriction=True,
        enforce_transfer_positivity=True,
    )
    inner = StaticBlockAMRConfig(
        BoxRegion(x0=4, x1=14, y0=4, y1=10, z0=4, z1=10),
        tau_coarse=outer.tau_fine,
        ghost_interpolation="trilinear",
        regularize_restriction=True,
        enforce_transfer_positivity=True,
    )
    deepest = StaticBlockAMRConfig(
        BoxRegion(x0=3, x1=19, y0=3, y1=11, z0=3, z1=11),
        tau_coarse=inner.tau_fine,
        ghost_interpolation="trilinear",
        regularize_restriction=True,
        enforce_transfer_positivity=True,
    )
    return outer, inner, deepest


def run(args: argparse.Namespace) -> dict[str, object]:
    devices = tuple(
        torch.device(value.strip()) for value in args.devices.split(",") if value.strip()
    )
    if len(devices) != 2 or devices[0] == devices[1]:
        raise ValueError("devices must name two distinct execution devices")
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    for device in devices:
        if device.type != "cuda":
            raise ValueError("the production probe requires two CUDA devices")

    root_device, peer_device = devices
    generator = torch.Generator(device=root_device).manual_seed(args.seed)
    shape = (12, 12, 14)
    rho = torch.full(shape, 1.01, device=root_device)
    ux = torch.full(shape, 0.025, device=root_device)
    uy = torch.full(shape, -0.006, device=root_device)
    uz = torch.full(shape, 0.003, device=root_device)
    initial = equilibrium3d(rho, ux, uy, uz)
    initial += 1.0e-7 * torch.randn(
        initial.shape,
        generator=generator,
        device=root_device,
        dtype=initial.dtype,
    )
    configs = _configs()
    reference = NestedStaticBlockAMR3D(initial.clone(), configs)
    distributed = NestedStaticBlockAMR3D(
        initial.clone(),
        configs,
        fine_devices=(peer_device, peer_device, root_device),
    )

    def advance(
        state: torch.Tensor,
        tau: float,
        level: int,
        substep: int,
    ) -> AMRAdvanceResult:
        del level, substep
        post = collide_mrt3d(state, tau=tau)
        return AMRAdvanceResult(stream3d(post), post)

    maximum_reference_residual = 0.0
    maximum_distributed_residual = 0.0
    for _ in range(args.steps):
        reference_ledgers = reference.step(advance)
        distributed_ledgers = distributed.step(advance)
        maximum_reference_residual = max(
            maximum_reference_residual,
            *(float(ledger.residual.abs().max()) for ledger in reference_ledgers),
        )
        maximum_distributed_residual = max(
            maximum_distributed_residual,
            *(float(ledger.residual.abs().max()) for ledger in distributed_ledgers),
        )

    level_maximum_absolute_difference = [
        float((actual.to(root_device) - expected).abs().max())
        for actual, expected in zip(
            distributed.level_populations,
            reference.level_populations,
            strict=True,
        )
    ]
    initial_mass = float(initial.sum())
    final_mass = float(distributed.coarse_f.sum())
    root_relative_mass_drift = abs(final_mass - initial_mass) / initial_mass
    finite = all(bool(torch.isfinite(level).all()) for level in distributed.level_populations)
    maximum_difference = max(level_maximum_absolute_difference)
    admitted = (
        finite
        and maximum_difference <= 5.0e-6
        and maximum_distributed_residual <= 5.0e-6
        and root_relative_mass_drift <= 5.0e-6
    )
    result: dict[str, object] = {
        "schema": "tensorlbm-nested-amr-multidevice-validation-v1",
        "status": "pass" if admitted else "fail",
        "steps": args.steps,
        "seed": args.seed,
        "reference_level_devices": [str(device) for device in reference.level_devices],
        "distributed_level_devices": [str(device) for device in distributed.level_devices],
        "device_names": {str(device): torch.cuda.get_device_name(device) for device in devices},
        "level_maximum_absolute_difference": (level_maximum_absolute_difference),
        "maximum_absolute_difference": maximum_difference,
        "maximum_reference_reflux_residual": maximum_reference_residual,
        "maximum_distributed_reflux_residual": maximum_distributed_residual,
        "root_relative_mass_drift": root_relative_mass_drift,
        "finite": finite,
        "acceptance": {
            "maximum_absolute_difference": 5.0e-6,
            "maximum_reflux_residual": 5.0e-6,
            "maximum_root_relative_mass_drift": 5.0e-6,
            "admitted": admitted,
        },
    }
    if not all(
        math.isfinite(value)
        for value in (
            maximum_difference,
            maximum_reference_residual,
            maximum_distributed_residual,
            root_relative_mass_drift,
        )
    ):
        raise FloatingPointError("multi-device validation produced non-finite evidence")
    return result


def main() -> None:
    args = parser().parse_args()
    result = run(args)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload, flush=True)
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
