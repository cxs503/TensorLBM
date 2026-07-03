"""Command-line example: 2D dam-break benchmark with multiphase LBM.

Usage
-----
    PYTHONPATH=src python examples/dam_break.py [options]

Models available (--model flag):
    sc   – Shan-Chen two-component (default)
    scmp – Shan-Chen single-component with pseudopotential EOS
    cg   – Color-Gradient (Latva-Kokko & Rothman)
    fe   – Free-Energy (simplified Swift et al.)
"""
from __future__ import annotations

import argparse

from tensorlbm import DamBreakConfig, run_dam_break


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the 2D dam-break multiphase LBM benchmark."
    )
    parser.add_argument("--nx", type=int, default=400)
    parser.add_argument("--ny", type=int, default=200)
    parser.add_argument("--dam-width", type=int, default=100,
                        help="Initial water column width (cells)")
    parser.add_argument("--model", choices=["sc", "scmp", "cg", "fe"], default="sc",
                        help="Multiphase model")
    parser.add_argument("--rho-heavy", type=float, default=2.0)
    parser.add_argument("--rho-light", type=float, default=0.1)
    parser.add_argument("--G", type=float, default=0.9,
                        help="Shan-Chen coupling constant")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--g", type=float, default=5e-5,
                        help="Gravity (lattice units)")
    parser.add_argument("--n-steps", type=int, default=4000)
    parser.add_argument("--output-interval", type=int, default=400)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", choices=["cpu", "sdaa", "cuda"], default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = DamBreakConfig(
        nx=args.nx,
        ny=args.ny,
        dam_width=args.dam_width,
        model=args.model,
        rho_heavy=args.rho_heavy,
        rho_light=args.rho_light,
        G=args.G,
        tau=args.tau,
        g=args.g,
        n_steps=args.n_steps,
        output_interval=args.output_interval,
        output_root=args.output_root,
        run_name=args.run_name,
        device=args.device,
        overwrite=args.overwrite,
    )
    run_dam_break(config)


if __name__ == "__main__":
    main()
