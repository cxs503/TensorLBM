from __future__ import annotations

import argparse

from tensorlbm import TurbulentChannelConfig, run_turbulent_channel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the turbulent-channel benchmark.")
    parser.add_argument("--nx", type=int, default=256, help="Grid width")
    parser.add_argument("--ny", type=int, default=64, help="Grid height")
    parser.add_argument("--re-tau", dest="re_tau", type=float, default=100.0)
    parser.add_argument("--u-tau", dest="u_tau", type=float, default=0.005)
    parser.add_argument(
        "--smagorinsky-cs",
        dest="smagorinsky_cs",
        type=float,
        default=0.1,
        help="Smagorinsky constant",
    )
    parser.add_argument(
        "--n-steps",
        dest="n_steps",
        type=int,
        default=50000,
        help="Number of time steps",
    )
    parser.add_argument(
        "--averaging-start",
        dest="averaging_start",
        type=int,
        default=20000,
        help="Step index at which mean-profile averaging starts",
    )
    parser.add_argument(
        "--output-interval",
        dest="output_interval",
        type=int,
        default=5000,
        help="Diagnostic cadence",
    )
    parser.add_argument("--output-root", default="outputs", help="Output root directory")
    parser.add_argument("--run-name", default=None, help="Override run directory name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
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
    config = TurbulentChannelConfig(
        nx=args.nx,
        ny=args.ny,
        re_tau=args.re_tau,
        u_tau=args.u_tau,
        smagorinsky_cs=args.smagorinsky_cs,
        n_steps=args.n_steps,
        averaging_start=args.averaging_start,
        output_interval=args.output_interval,
        output_root=args.output_root,
        run_name=args.run_name,
        seed=args.seed,
        device=args.device,
        overwrite=args.overwrite,
    )
    run_turbulent_channel(config)


if __name__ == "__main__":
    main()
