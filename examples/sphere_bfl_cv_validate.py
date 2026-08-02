#!/usr/bin/env python3
"""Run the canonical Re=100 sphere BFL/control-volume validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensorlbm.sphere_bfl_control_volume import (
    SphereBFLControlVolumeConfig,
    run_sphere_bfl_control_volume,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--nx", type=int, default=192)
    p.add_argument("--ny", type=int, default=128)
    p.add_argument("--nz", type=int, default=128)
    p.add_argument("--radius", type=float, default=8.0)
    p.add_argument("--reynolds", type=float, default=100.0)
    p.add_argument("--lattice-speed", type=float, default=0.06)
    p.add_argument(
        "--collision-model",
        choices=("cumulant_d3q19_cs0", "natural_kbc_d3q19"),
        default="cumulant_d3q19_cs0",
    )
    p.add_argument("--collision-chunk-cells", type=int, default=0)
    p.add_argument("--compile-natural-kbc", action="store_true")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--warmup-steps", type=int, default=1500)
    p.add_argument("--ramp-steps", type=int, default=500)
    p.add_argument("--sponge-width", type=int, default=16)
    p.add_argument("--sponge-strength", type=float, default=0.2)
    p.add_argument("--sponge-inlet", action="store_true")
    p.add_argument("--cv-margin", type=int, default=5)
    p.add_argument("--report-interval", type=int, default=500)
    p.add_argument("--checkpoint-interval", type=int, default=2000)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-v2-checkpoint", action="store_true")
    p.add_argument("--statistics-window-steps", type=int, default=0)
    p.add_argument("--minimum-statistics-convective-times", type=float, default=5.0)
    p.add_argument(
        "--far-field-mode",
        choices=("non_equilibrium_extrapolation", "legacy_hard_equilibrium"),
        default="non_equilibrium_extrapolation",
    )
    p.add_argument("--output", required=True)
    return p


def main() -> None:
    args = parser().parse_args()
    config = SphereBFLControlVolumeConfig(
        nx=args.nx, ny=args.ny, nz=args.nz, radius=args.radius,
        reynolds=args.reynolds, lattice_speed=args.lattice_speed,
        steps=args.steps, warmup_steps=args.warmup_steps,
        ramp_steps=args.ramp_steps, sponge_width=args.sponge_width,
        sponge_strength=args.sponge_strength, sponge_inlet=args.sponge_inlet,
        cv_margin=args.cv_margin,
        report_interval=args.report_interval,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_path=args.checkpoint, resume=args.resume,
        allow_v2_checkpoint=args.allow_v2_checkpoint,
        statistics_window_steps=args.statistics_window_steps,
        minimum_statistics_convective_times=(
            args.minimum_statistics_convective_times
        ),
        far_field_mode=args.far_field_mode, device=args.device,
        collision_model=args.collision_model,
        collision_chunk_cells=args.collision_chunk_cells,
        compile_natural_kbc=args.compile_natural_kbc,
    )
    result = run_sphere_bfl_control_volume(config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
