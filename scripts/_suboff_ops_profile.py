#!/usr/bin/env python3
"""Granular per-operator SDAA profile of the suboff_re1000 step.

Runs a warmup, then profiles N steps with torch.profiler (CPU + SDAA) and
prints the top device-time operators with shapes and call counts.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path("/root/TensorLBM_feat2")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks/pending/suboff_re1000"))

import torch  # noqa: E402
import torch_sdaa  # noqa: E402,F401
from torch.profiler import ProfilerActivity, profile  # noqa: E402


def build(L: int, device: str, steps: int):
    import run as runmod  # type: ignore

    from tensorlbm.general_sim import (
        CollisionModel,
        ForceMethod,
        GeneralSimConfig,
        GeneralSimEngine,
        GeometryConfig,
        GeometrySource,
        LatticeModel,
        OutputConfig,
        PhysicsConfig,
        SolverConfig,
        WallTreatment,
    )

    viscosity = runmod.U_PHYS * runmod.SUBOFF_LENGTH_M / 1000.0
    out_dir = Path(f"/root/prof_out/ops_L{L}")
    out_dir.mkdir(parents=True, exist_ok=True)
    config = GeneralSimConfig(
        name=f"prof_suboff_L{L}",
        geometry=GeometryConfig(
            source=GeometrySource.PARAMETRIC_SUBOFF,
            suboff_length=runmod.SUBOFF_LENGTH_M,
            suboff_radius=runmod.SUBOFF_RADIUS_M,
        ),
        physics=PhysicsConfig(
            density=1000.0,
            viscosity=viscosity,
            inlet_velocity=runmod.U_PHYS,
            reference_length=runmod.SUBOFF_LENGTH_M,
        ),
        solver=SolverConfig(
            lattice=LatticeModel.D3Q19,
            collision=CollisionModel.MRT,
            resolution=L,
            domain_padding=(1.0, 4.0, 1.0, 1.0, 1.0, 1.0),
            max_steps=steps,
            warmup_steps=None,
            snapshot_interval=10_000_000,
            force_sample_interval=10,
            device=device,
            wall_treatment=WallTreatment.AUTO,
            force_method=ForceMethod.PRESSURE_FRICTION,
            pressure_extrap="none",
            p0_method="near_wall",
            friction_formula="standard",
            mass_correction=True,
            mass_correction_interval=200,
            smagorinsky_cs=0.05,
        ),
        output=OutputConfig(
            directory=str(out_dir),
            formats=[],
            save_macroscopic=False,
            save_forces=True,
        ),
    )
    eng = GeneralSimEngine(config)
    eng.setup()
    return eng


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=48)
    ap.add_argument("--device", default="sdaa:0")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--prof-steps", type=int, default=5)
    args = ap.parse_args()

    eng = build(args.resolution, args.device, args.warmup + args.prof_steps + 1)
    dev = torch.device(args.device)
    torch.sdaa.set_device(dev.index or 0)

    # warmup (also excludes lazy-init kernels from the trace)
    for _ in range(args.warmup):
        eng.run(steps=1)
    torch.sdaa.synchronize()

    # time a clean window without the profiler for reference
    t0 = time.time()
    eng.run(steps=args.prof_steps)
    torch.sdaa.synchronize()
    print(f"unprofiled wall: {(time.time()-t0)/args.prof_steps*1000:.1f} ms/step", flush=True)

    # ensure a clean start of loop from step 1
    eng.step_count = 0
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.SDAA],
        record_shapes=True,
        profile_memory=False,
    ) as prof:
        eng.run(steps=args.prof_steps)
        torch.sdaa.synchronize()

    print("\n=== TOP SDAA ops (device_time_total) ===", flush=True)
    print(
        prof.key_averages().table(
            sort_by="self_device_time_total", row_limit=40
        ),
        flush=True,
    )
    print("\n=== TOP CPU ops (cpu_time_total) ===", flush=True)
    print(
        prof.key_averages().table(sort_by="cpu_time_total", row_limit=30),
        flush=True,
    )


if __name__ == "__main__":
    main()