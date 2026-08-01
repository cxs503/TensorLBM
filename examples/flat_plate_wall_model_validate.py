#!/usr/bin/env python3
"""Run the finite flat-plate BFL wall-model validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensorlbm.flat_plate_wall_model import (
    FlatPlateWallModelConfig,
    run_flat_plate_wall_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nx", type=int, default=512)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--nz", type=int, default=3)
    parser.add_argument("--plate-length", type=int, default=256)
    parser.add_argument("--plate-start-fraction", type=float, default=0.20)
    parser.add_argument("--reynolds", type=float, default=1.0e6)
    parser.add_argument("--resolved-reynolds", type=float, default=1.0e5)
    parser.add_argument("--lattice-speed", type=float, default=0.06)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--warmup-steps", type=int, default=6000)
    parser.add_argument("--ramp-steps", type=int, default=1000)
    parser.add_argument("--sponge-width", type=int, default=24)
    parser.add_argument("--sponge-strength", type=float, default=0.2)
    parser.add_argument("--cv-margin", type=int, default=6)
    parser.add_argument("--wall-law", choices=("log", "reichardt", "musker"), default="log")
    parser.add_argument("--cs-smag", type=float, default=0.05)
    parser.add_argument("--disable-positivity-limiter", action="store_true")
    parser.add_argument("--report-interval", type=int, default=1000)
    parser.add_argument("--checkpoint-interval", type=int, default=2000)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_flat_plate_wall_model(FlatPlateWallModelConfig(
        nx=args.nx, ny=args.ny, nz=args.nz,
        plate_length=args.plate_length,
        plate_start_fraction=args.plate_start_fraction,
        reynolds=args.reynolds, resolved_reynolds=args.resolved_reynolds,
        lattice_speed=args.lattice_speed, steps=args.steps,
        warmup_steps=args.warmup_steps, ramp_steps=args.ramp_steps,
        sponge_width=args.sponge_width, sponge_strength=args.sponge_strength,
        cv_margin=args.cv_margin, wall_law=args.wall_law,
        smagorinsky_cs=args.cs_smag,
        positivity_limiter=not args.disable_positivity_limiter,
        report_interval=args.report_interval,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_path=args.checkpoint, resume=args.resume,
        device=args.device,
    ))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
