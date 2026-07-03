"""CLI entry-point for the 3-D sphere water-entry LBM simulation.

This script simulates a sphere descending into a pool of still water at a
prescribed entry velocity.  The sphere is held stationary in its own reference
frame while the water flows upward past it, enabling a standard fixed-geometry
D3Q19 LBM calculation.

Usage examples
--------------
Smoke test (small grid, few steps):

.. code-block:: bash

    PYTHONPATH=src python examples/sphere_water_entry.py \\
        --nx 32 --ny 32 --nz 64 \\
        --radius 4 --v-entry 0.05 --re 100 \\
        --n-steps 50 --output-interval 25 \\
        --run-name smoke --overwrite

Standard run:

.. code-block:: bash

    PYTHONPATH=src python examples/sphere_water_entry.py \\
        --nx 48 --ny 48 --nz 96 \\
        --radius 6 --v-entry 0.05 --re 100 --n-ramp 50 \\
        --n-steps 1000 --output-interval 100 --overwrite

High-Reynolds run with Smagorinsky LES:

.. code-block:: bash

    PYTHONPATH=src python examples/sphere_water_entry.py \\
        --nx 64 --ny 64 --nz 128 \\
        --radius 8 --v-entry 0.06 --re 300 --cs 0.1 \\
        --n-steps 2000 --output-interval 200 --overwrite
"""

from __future__ import annotations

import argparse

from tensorlbm.sphere_water_entry import SphereWaterEntryConfig, run_sphere_water_entry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a 3-D D3Q19 sphere water-entry LBM simulation.\n\n"
            "The sphere descends into still water at constant speed v_entry. "
            "In the sphere's reference frame the water flows upward past the "
            "sphere, enabling force diagnostics via the Ladd momentum-exchange "
            "method.  Results (force CSV, PNG snapshots, metadata) are saved "
            "to --output-root/sphere_water_entry/<run-name>/."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Grid
    parser.add_argument("--nx", type=int, default=48,
                        help="Tank width (x direction)")
    parser.add_argument("--ny", type=int, default=48,
                        help="Tank depth (y direction)")
    parser.add_argument("--nz", type=int, default=96,
                        help="Tank height (z direction, flow direction)")

    # Geometry
    parser.add_argument("--radius", type=float, default=6.0,
                        help="Sphere radius in lattice units")
    parser.add_argument("--sphere-z-frac", dest="sphere_z_frac", type=float,
                        default=0.5,
                        help="Fractional z position of sphere centre (0=bottom, 1=top)")

    # Flow
    parser.add_argument("--v-entry", dest="v_entry", type=float, default=0.05,
                        help="Entry velocity in lattice units (≤ 0.1)")
    parser.add_argument("--re", type=float, default=100.0,
                        help="Target Reynolds number Re = v_entry · 2r / ν")
    parser.add_argument("--n-ramp", dest="n_ramp", type=int, default=50,
                        help="Steps to ramp from 0 to v_entry (0 = impulsive)")

    # Turbulence
    parser.add_argument("--cs", dest="smagorinsky_cs", type=float, default=0.0,
                        help="Smagorinsky constant (0 = BGK only, 0.1 typical for LES)")

    # Simulation control
    parser.add_argument("--n-steps", dest="n_steps", type=int, default=1000,
                        help="Total simulation steps")
    parser.add_argument("--output-interval", dest="output_interval", type=int,
                        default=100, help="Steps between diagnostics and PNG output")
    parser.add_argument("--output-root", dest="output_root", default="outputs",
                        help="Root directory for all outputs")
    parser.add_argument("--run-name", dest="run_name", default=None,
                        help="Override auto-generated run folder name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", choices=["cpu", "sdaa", "cuda"], default="cpu",
                        help="Execution device")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace output directory if it already exists")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = SphereWaterEntryConfig(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        radius=args.radius,
        sphere_z_frac=args.sphere_z_frac,
        v_entry=args.v_entry,
        re=args.re,
        n_ramp=args.n_ramp,
        smagorinsky_cs=args.smagorinsky_cs,
        n_steps=args.n_steps,
        output_interval=args.output_interval,
        output_root=args.output_root,
        run_name=args.run_name,
        seed=args.seed,
        device=args.device,
        overwrite=args.overwrite,
    )
    run_sphere_water_entry(config)


if __name__ == "__main__":
    main()
