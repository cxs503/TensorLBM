"""Command-line example: multiphase sphere/cylinder water-entry benchmark.

Usage
-----
    PYTHONPATH=src python examples/multiphase_water_entry.py [options]

Modes:
    2d – Cylinder entering water (fast, good for benchmarking)
    3d – Full sphere water entry (more expensive)
"""
from __future__ import annotations

import argparse

from tensorlbm import MultiphaseWaterEntryConfig, run_multiphase_water_entry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the sphere/cylinder water-entry multiphase LBM benchmark."
    )
    parser.add_argument("--mode", choices=["2d", "3d"], default="2d",
                        help="2D cylinder or 3D sphere")
    parser.add_argument("--model", choices=["cg", "sc"], default="cg",
                        help="Multiphase model for 2D (cg=Color-Gradient, sc=Shan-Chen)")
    parser.add_argument("--nx", type=int, default=200)
    parser.add_argument("--ny", type=int, default=160)
    parser.add_argument("--nz", type=int, default=80,
                        help="z-dimension (3D mode only)")
    parser.add_argument("--radius", type=float, default=12.0,
                        help="Sphere/cylinder radius (cells)")
    parser.add_argument("--water-level", type=int, default=80,
                        help="Initial water surface (y index in 2D, z in 3D)")
    parser.add_argument("--clearance", type=int, default=4,
                        help="Gap between sphere bottom and water surface (cells)")
    parser.add_argument("--rho-water", type=float, default=2.0)
    parser.add_argument("--rho-air", type=float, default=0.1)
    parser.add_argument("--G", type=float, default=0.9,
                        help="Shan-Chen coupling constant")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--g", type=float, default=5e-5,
                        help="Gravity (lattice units)")
    parser.add_argument("--n-steps", type=int, default=3000)
    parser.add_argument("--output-interval", type=int, default=300)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = MultiphaseWaterEntryConfig(
        mode=args.mode,
        model=args.model,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        radius=args.radius,
        water_level=args.water_level,
        clearance=args.clearance,
        rho_water=args.rho_water,
        rho_air=args.rho_air,
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
    run_multiphase_water_entry(config)


if __name__ == "__main__":
    main()
