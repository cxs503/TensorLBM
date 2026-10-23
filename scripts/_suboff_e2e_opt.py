#!/usr/bin/env python3
"""End-to-end verification of the landed SDAA optimizations (suboff Re=1000).

Runs the *production* GeneralSimEngine.run() loop at resolution L and compares:

  legacy    : TL_STREAM_MODE=gather + TL_FUSE_NODYN_BB=0 + div-check every step
  optimized : TL_STREAM_MODE=cat    + TL_FUSE_NODYN_BB=1 + div-check every --div

Verifies bit-exactness of the field after --verify-steps and reports the
end-to-end ms/step for each.  Optional --mrt-fused to measure the (non
bit-exact, ~1.5e-7) single-gemm MRT variant.

Usage:
  python scripts/_suboff_e2e_opt.py --resolution 48 --steps 20 --div 50
  python scripts/_suboff_e2e_opt.py --resolution 80 --steps 8  --div 50
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path("/root/TensorLBM_feat2")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks/pending/suboff_re1000"))

import torch  # noqa: E402
import torch_sdaa  # noqa: E402,F401

import run as runmod  # type: ignore  # noqa: E402
from tensorlbm.general_sim import (  # noqa: E402
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


def build(L, device):
    viscosity = runmod.U_PHYS * runmod.SUBOFF_LENGTH_M / 1000.0
    out_dir = Path(f"/root/prof_out/e2e_opt_L{L}")
    out_dir.mkdir(parents=True, exist_ok=True)
    config = GeneralSimConfig(
        name=f"e2e_opt_L{L}",
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
            max_steps=100000,
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


def set_env(stream, fuse, div, mrt_fused=0):
    os.environ["TL_STREAM_MODE"] = stream
    os.environ["TL_FUSE_NODYN_BB"] = "1" if fuse else "0"
    os.environ["TL_ISFINITE_INTERVAL"] = str(div)
    os.environ["TL_MRT_FUSED"] = "1" if mrt_fused else "0"


def time_run(eng, steps):
    torch.sdaa.synchronize()
    t = time.time()
    eng.run(steps=steps)
    torch.sdaa.synchronize()
    return (time.time() - t) / steps * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=48)
    ap.add_argument("--device", default="sdaa:0")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--verify-steps", type=int, default=3)
    ap.add_argument("--div", type=int, default=50)
    ap.add_argument("--mrt-fused", action="store_true")
    args = ap.parse_args()

    eng = build(args.resolution, args.device)
    torch.sdaa.synchronize()
    f0 = eng.f.clone()
    sc = eng.step_count
    solid = int(eng.solid.sum()) if eng.solid is not None else 0
    print(
        f"=== L={args.resolution} f.shape={tuple(eng.f.shape)} "
        f"cells={eng.f[0].numel()/1e6:.2f}M solid={solid} "
        f"tau={eng.uc.tau:.4f} div={args.div} mrt_fused={args.mrt_fused} ===",
        flush=True,
    )

    def reset():
        eng.f = f0.clone()
        eng.step_count = sc

    # ---- legacy baseline ----
    set_env("gather", False, 1, 0)
    reset()
    eng.run(steps=args.verify_steps)
    torch.sdaa.synchronize()
    f_legacy = eng.f.clone()
    reset()
    base_ms = time_run(eng, args.steps)
    print(f"{'legacy (gather/no-fuse/div1)':38s} {base_ms:9.1f} ms/step", flush=True)

    # ---- fully optimized ----
    set_env("cat", True, args.div, 1 if args.mrt_fused else 0)
    reset()
    eng.run(steps=args.verify_steps)
    torch.sdaa.synchronize()
    same = torch.equal(eng.f, f_legacy)
    md = (eng.f - f_legacy).abs().max().item()
    reset()
    opt_ms = time_run(eng, args.steps)
    print(
        f"{'optimized (cat/fuse/div%d)':38s} {opt_ms:9.1f} ms/step  "
        f"speedup={base_ms/opt_ms:5.2f}x  bitexact={same} maxdiff={md:.2e}"
        % args.div,
        flush=True,
    )

    # ---- component toggles (isolate each optimization) ----
    for label, stream, fuse, div, mf in [
        ("  +cat stream only", "cat", False, 1, 0),
        ("  +cat +fuse", "cat", True, 1, 0),
        ("  +cat +fuse +div", "cat", True, args.div, 0),
    ]:
        set_env(stream, fuse, div, mf)
        reset()
        ms = time_run(eng, args.steps)
        print(
            f"{label:38s} {ms:9.1f} ms/step  speedup={base_ms/ms:5.2f}x",
            flush=True,
        )


if __name__ == "__main__":
    main()