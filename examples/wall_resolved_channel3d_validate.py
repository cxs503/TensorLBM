#!/usr/bin/env python3
"""Run the 3-D periodic wall-resolved channel reference."""

from __future__ import annotations

import argparse
from pathlib import Path

from tensorlbm.wall_resolved_channel3d import (
    WallResolvedChannel3DConfig,
    run_wall_resolved_channel3d,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--nz", type=int, default=64)
    parser.add_argument("--re-tau", type=float, default=180.0)
    parser.add_argument("--u-tau", type=float, default=0.003)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--warmup-steps", type=int, default=20000)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--report-interval", type=int, default=500)
    parser.add_argument("--checkpoint-interval", type=int, default=5000)
    parser.add_argument("--collision-model", choices=("natural_kbc", "cumulant"), default="natural_kbc")
    parser.add_argument("--collision-chunk-cells", type=int, default=262144)
    parser.add_argument("--no-compile-natural-kbc", action="store_true")
    parser.add_argument(
        "--initialization-mode",
        choices=("coherent", "spectral_solenoidal"),
        default="spectral_solenoidal",
    )
    parser.add_argument("--perturbation-fraction", type=float, default=1.0)
    parser.add_argument("--random-noise-fraction", type=float, default=0.5)
    parser.add_argument("--spectral-mode-count", type=int, default=32)
    parser.add_argument("--spectral-max-wavenumber", type=int, default=4)
    parser.add_argument(
        "--minimum-statistics-eddy-turnovers", type=float, default=2.0,
    )
    parser.add_argument(
        "--stationarity-window-eddy-turnovers", type=float, default=1.0,
    )
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset-statistics-on-resume", action="store_true")
    args = parser.parse_args()
    config = WallResolvedChannel3DConfig(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        re_tau=args.re_tau,
        u_tau=args.u_tau,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        sample_interval=args.sample_interval,
        report_interval=args.report_interval,
        checkpoint_interval=args.checkpoint_interval,
        collision_model=args.collision_model,
        collision_chunk_cells=args.collision_chunk_cells,
        compile_natural_kbc=not args.no_compile_natural_kbc,
        initialization_mode=args.initialization_mode,
        perturbation_fraction=args.perturbation_fraction,
        random_noise_fraction=args.random_noise_fraction,
        spectral_mode_count=args.spectral_mode_count,
        spectral_max_wavenumber=args.spectral_max_wavenumber,
        minimum_statistics_eddy_turnovers=(
            args.minimum_statistics_eddy_turnovers
        ),
        stationarity_window_eddy_turnovers=(
            args.stationarity_window_eddy_turnovers
        ),
        seed=args.seed,
        device=args.device,
        output=args.output,
        checkpoint=args.checkpoint,
        resume=args.resume,
        reset_statistics_on_resume=args.reset_statistics_on_resume,
    )
    run_wall_resolved_channel3d(config)


if __name__ == "__main__":
    main()
