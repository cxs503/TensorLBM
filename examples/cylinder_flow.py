from __future__ import annotations

import argparse

from tensorlbm import CylinderFlowConfig, run_cylinder_flow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a D2Q9 cylinder-flow LBM demonstration."
    )
    parser.add_argument("--nx", type=int, default=320, help="Grid width")
    parser.add_argument("--ny", type=int, default=100, help="Grid height")
    parser.add_argument(
        "--u-in", dest="u_in", type=float, default=0.08, help="Inlet velocity"
    )
    parser.add_argument("--re", type=float, default=100.0, help="Target Reynolds number")
    parser.add_argument("--radius", type=float, default=12.0, help="Cylinder radius")
    parser.add_argument(
        "--n-steps", dest="n_steps", type=int, default=1200, help="Simulation steps"
    )
    parser.add_argument(
        "--output-interval",
        type=int,
        default=200,
        help="Diagnostic and image cadence",
    )
    parser.add_argument("--output-root", default="outputs", help="Output root directory")
    parser.add_argument(
        "--run-name", default=None, help="Override deterministic run folder name"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu", help="Execution device"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output directory if it already exists",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = CylinderFlowConfig(
        nx=args.nx,
        ny=args.ny,
        u_in=args.u_in,
        re=args.re,
        radius=args.radius,
        n_steps=args.n_steps,
        output_interval=args.output_interval,
        output_root=args.output_root,
        run_name=args.run_name,
        seed=args.seed,
        device=args.device,
        overwrite=args.overwrite,
    )
    run_cylinder_flow(config)


if __name__ == "__main__":
    main()
