from __future__ import annotations

import argparse

from tensorlbm import PipelineFlowConfig, run_pipeline_flow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the pipeline-flow benchmark.")
    parser.add_argument("--nx", type=int, default=400, help="Grid width")
    parser.add_argument("--ny", type=int, default=160, help="Grid height")
    parser.add_argument(
        "--diameter",
        type=float,
        default=20.0,
        help="Cylinder diameter in lattice cells",
    )
    parser.add_argument(
        "--gap-ratio",
        dest="gap_ratio",
        type=float,
        default=0.5,
        help="Gap ratio e/D between the bed and the cylinder",
    )
    parser.add_argument("--u-in", dest="u_in", type=float, default=0.05)
    parser.add_argument("--re", type=float, default=200.0)
    parser.add_argument(
        "--n-steps",
        dest="n_steps",
        type=int,
        default=30000,
        help="Number of time steps",
    )
    parser.add_argument(
        "--output-interval",
        dest="output_interval",
        type=int,
        default=5000,
        help="Diagnostic and plot cadence",
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
    config = PipelineFlowConfig(
        nx=args.nx,
        ny=args.ny,
        diameter=args.diameter,
        gap_ratio=args.gap_ratio,
        u_in=args.u_in,
        re=args.re,
        n_steps=args.n_steps,
        output_interval=args.output_interval,
        output_root=args.output_root,
        run_name=args.run_name,
        seed=args.seed,
        device=args.device,
        overwrite=args.overwrite,
    )
    run_pipeline_flow(config)


if __name__ == "__main__":
    main()
