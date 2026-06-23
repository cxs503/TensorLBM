from __future__ import annotations

import argparse

from tensorlbm import SphereFlowConfig, run_sphere_flow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a D3Q19 sphere-flow LBM demonstration.")
    parser.add_argument("--nx", type=int, default=120, help="Grid length (x)")
    parser.add_argument("--ny", type=int, default=60, help="Grid height (y)")
    parser.add_argument("--nz", type=int, default=60, help="Grid depth  (z)")
    parser.add_argument(
        "--u-in", dest="u_in", type=float, default=0.06, help="Inlet velocity"
    )
    parser.add_argument("--re", type=float, default=50.0, help="Target Reynolds number")
    parser.add_argument("--radius", type=float, default=8.0, help="Sphere radius")
    parser.add_argument(
        "--n-steps", dest="n_steps", type=int, default=500, help="Simulation steps"
    )
    parser.add_argument(
        "--output-interval",
        type=int,
        default=100,
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
        "--num-threads",
        dest="num_threads",
        type=int,
        default=None,
        help="PyTorch CPU thread count (CPU runs only)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output directory if it already exists",
    )
    parser.add_argument(
        "--resume-checkpoint",
        dest="resume_checkpoint",
        default=None,
        help="Path to checkpoint directory to resume from",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = SphereFlowConfig(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        u_in=args.u_in,
        re=args.re,
        radius=args.radius,
        n_steps=args.n_steps,
        output_interval=args.output_interval,
        output_root=args.output_root,
        run_name=args.run_name,
        seed=args.seed,
        device=args.device,
        num_threads=args.num_threads,
        overwrite=args.overwrite,
        resume_checkpoint=args.resume_checkpoint,
    )
    run_sphere_flow(config)


if __name__ == "__main__":
    main()
