#!/usr/bin/env python3
"""Run the canonical Re=100 cylinder BFL/control-volume validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensorlbm.cylinder_bfl_control_volume import (
    CylinderBFLControlVolumeConfig,
    run_cylinder_bfl_control_volume,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--nx", type=int, default=320)
    p.add_argument("--ny", type=int, default=200)
    p.add_argument("--nz", type=int, default=3)
    p.add_argument("--radius", type=float, default=12.0)
    p.add_argument("--center-x-fraction", type=float, default=0.30)
    p.add_argument("--reynolds", type=float, default=100.0)
    p.add_argument("--lattice-speed", type=float, default=0.06)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--warmup-steps", type=int, default=4000)
    p.add_argument("--ramp-steps", type=int, default=500)
    p.add_argument("--sponge-width", type=int, default=24)
    p.add_argument("--sponge-strength", type=float, default=0.2)
    p.add_argument("--sponge-inlet", action="store_true")
    p.add_argument("--cv-margin", type=int, default=8)
    p.add_argument("--report-interval", type=int, default=1000)
    p.add_argument("--checkpoint-interval", type=int, default=5000)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--far-field-mode",
        choices=("non_equilibrium_extrapolation", "legacy_hard_equilibrium"),
        default="non_equilibrium_extrapolation",
    )
    p.add_argument("--output", required=True)
    return p


def main() -> None:
    args = parser().parse_args()
    config = CylinderBFLControlVolumeConfig(
        nx=args.nx, ny=args.ny, nz=args.nz, radius=args.radius,
        center_x_fraction=args.center_x_fraction,
        reynolds=args.reynolds, lattice_speed=args.lattice_speed,
        steps=args.steps, warmup_steps=args.warmup_steps,
        ramp_steps=args.ramp_steps, sponge_width=args.sponge_width,
        sponge_strength=args.sponge_strength, sponge_inlet=args.sponge_inlet,
        cv_margin=args.cv_margin,
        report_interval=args.report_interval,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_path=args.checkpoint, resume=args.resume,
        far_field_mode=args.far_field_mode, device=args.device,
    )
    result = run_cylinder_bfl_control_volume(config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
