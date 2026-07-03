"""Backward-Facing Step benchmark example.

Run a 2D D2Q9 backward-facing step (BFS) simulation and report the
primary reattachment length x_r* = (x_r − x_step) / h as a function
of the Reynolds number.

Usage::

    PYTHONPATH=src python examples/backward_facing_step.py
    PYTHONPATH=src python examples/backward_facing_step.py --re 200 --n-steps 50000
"""
from __future__ import annotations

import argparse

from tensorlbm import BackwardFacingStepConfig, run_backward_facing_step


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the 2D Backward-Facing Step benchmark (D2Q9 LBM)."
    )
    parser.add_argument("--nx", type=int, default=400,
                        help="Domain width in grid cells")
    parser.add_argument("--ny", type=int, default=80,
                        help="Domain height in grid cells")
    parser.add_argument("--step-h", dest="step_h", type=int, default=40,
                        help="Step height (ny//2 for 2:1 expansion)")
    parser.add_argument("--x-step", dest="x_step", type=int, default=80,
                        help="Pre-step solid length (upstream channel cells)")
    parser.add_argument("--u-in", dest="u_in", type=float, default=0.05,
                        help="Inlet velocity in lattice units")
    parser.add_argument("--re", type=float, default=100.0,
                        help="Reynolds number Re = u_in * step_h / nu")
    parser.add_argument("--n-steps", dest="n_steps", type=int, default=30000,
                        help="Number of simulation steps")
    parser.add_argument("--output-interval", type=int, default=5000,
                        help="Snapshot and diagnostic cadence")
    parser.add_argument("--output-root", default="outputs",
                        help="Output root directory")
    parser.add_argument("--run-name", default=None,
                        help="Override deterministic run folder name")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "sdaa", "cuda"], default="cpu")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace output directory if it already exists")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = BackwardFacingStepConfig(
        nx=args.nx,
        ny=args.ny,
        step_h=args.step_h,
        x_step=args.x_step,
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
    run_backward_facing_step(config)


if __name__ == "__main__":
    main()
