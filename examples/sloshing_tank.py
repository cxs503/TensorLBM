from __future__ import annotations

import argparse

from tensorlbm import SloshingTankConfig, run_sloshing_tank


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the sloshing-tank benchmark.")
    parser.add_argument("--nx", type=int, default=200, help="Grid width")
    parser.add_argument("--ny", type=int, default=160, help="Grid height")
    parser.add_argument(
        "--water-level",
        dest="water_level",
        type=int,
        default=80,
        help="Initial water depth in cells",
    )
    parser.add_argument("--rho-water", dest="rho_water", type=float, default=0.8)
    parser.add_argument("--rho-air", dest="rho_air", type=float, default=0.4)
    parser.add_argument("--G", type=float, default=0.9, help="CG coupling strength")
    parser.add_argument("--tau", type=float, default=1.0, help="Relaxation time")
    parser.add_argument("--g", type=float, default=2e-5, help="Gravity magnitude")
    parser.add_argument(
        "--forcing-amp",
        dest="forcing_amp",
        type=float,
        default=3e-5,
        help="Horizontal forcing amplitude",
    )
    parser.add_argument(
        "--forcing-omega",
        dest="forcing_omega",
        type=float,
        default=0.0,
        help="Forcing frequency (0 uses the natural frequency)",
    )
    parser.add_argument(
        "--n-steps",
        dest="n_steps",
        type=int,
        default=6000,
        help="Number of time steps",
    )
    parser.add_argument(
        "--output-interval",
        dest="output_interval",
        type=int,
        default=600,
        help="Diagnostic and plot cadence",
    )
    parser.add_argument("--output-root", default="outputs", help="Output root directory")
    parser.add_argument("--run-name", default=None, help="Override run directory name")
    parser.add_argument(
        "--device",
        choices=["cpu", "sdaa", "cuda", "mps"],
        default="cpu",
        help="Execution device",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output directory if it exists",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = SloshingTankConfig(
        nx=args.nx,
        ny=args.ny,
        water_level=args.water_level,
        rho_water=args.rho_water,
        rho_air=args.rho_air,
        G=args.G,
        tau=args.tau,
        g=args.g,
        forcing_amp=args.forcing_amp,
        forcing_omega=args.forcing_omega,
        n_steps=args.n_steps,
        output_interval=args.output_interval,
        output_root=args.output_root,
        run_name=args.run_name,
        device=args.device,
        overwrite=args.overwrite,
    )
    run_sloshing_tank(config)


if __name__ == "__main__":
    main()
