"""Lid-Driven Cavity benchmark example.

Run a 2D D2Q9 lid-driven cavity simulation and compare with the
Ghia et al. (1982) reference data.

Usage::

    PYTHONPATH=src python examples/lid_driven_cavity.py
    PYTHONPATH=src python examples/lid_driven_cavity.py --nx 64 --re 400 --n-steps 20000
"""
from __future__ import annotations

import argparse

from tensorlbm import LidDrivenCavityConfig, run_lid_driven_cavity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the 2D Lid-Driven Cavity benchmark (D2Q9 LBM)."
    )
    parser.add_argument("--nx", type=int, default=128,
                        help="Grid size (square: nx × nx)")
    parser.add_argument("--u-lid", dest="u_lid", type=float, default=0.1,
                        help="Lid velocity in lattice units")
    parser.add_argument("--re", type=float, default=100.0,
                        help="Reynolds number Re = u_lid * nx / nu")
    parser.add_argument("--n-steps", dest="n_steps", type=int, default=10000,
                        help="Number of simulation steps")
    parser.add_argument("--output-interval", type=int, default=2000,
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
    config = LidDrivenCavityConfig(
        nx=args.nx,
        u_lid=args.u_lid,
        re=args.re,
        n_steps=args.n_steps,
        output_interval=args.output_interval,
        output_root=args.output_root,
        run_name=args.run_name,
        seed=args.seed,
        device=args.device,
        overwrite=args.overwrite,
    )
    run_lid_driven_cavity(config)


if __name__ == "__main__":
    main()
