"""CLI entry-point for the 3-D Wigley ship hull flow simulation.

Usage examples
--------------
Smoke test (small grid, few steps):

.. code-block:: bash

    PYTHONPATH=src python examples/ship_hull_flow.py \\
        --nx 80 --ny 40 --nz 30 \\
        --hull-length 40 --hull-beam 4 --hull-draft 6 \\
        --u-in 0.05 --re 100 \\
        --n-steps 50 --output-interval 25 \\
        --run-name smoke --overwrite

With regular ocean waves:

.. code-block:: bash

    PYTHONPATH=src python examples/ship_hull_flow.py \\
        --nx 160 --ny 60 --nz 40 \\
        --hull-length 80 --hull-beam 8 --hull-draft 12 \\
        --u-in 0.05 --re 300 \\
        --wave-amp 0.005 --wave-period 200 --wave-k 0.04 \\
        --n-steps 2000 --output-interval 200 --overwrite
"""

from __future__ import annotations

import argparse

from tensorlbm.ship_flow import ShipHullFlowConfig, run_ship_hull_flow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a D3Q19 Wigley hull channel-flow LBM simulation."
    )

    # Grid
    parser.add_argument("--nx", type=int, default=160, help="Grid length (x, longitudinal)")
    parser.add_argument("--ny", type=int, default=60, help="Grid width (y, transverse)")
    parser.add_argument("--nz", type=int, default=40, help="Grid height (z, vertical)")

    # Flow
    parser.add_argument("--u-in", dest="u_in", type=float, default=0.05, help="Inlet velocity")
    parser.add_argument("--re", type=float, default=200.0, help="Target Reynolds number")

    # Hull
    parser.add_argument(
        "--hull-length", dest="hull_length", type=float, default=80.0,
        help="Wigley hull length L in lattice units",
    )
    parser.add_argument(
        "--hull-beam", dest="hull_beam", type=float, default=8.0,
        help="Maximum beam B in lattice units",
    )
    parser.add_argument(
        "--hull-draft", dest="hull_draft", type=float, default=12.0,
        help="Hull draft T in lattice units",
    )

    # Turbulence
    parser.add_argument(
        "--cs", dest="smagorinsky_cs", type=float, default=0.1,
        help="Smagorinsky constant (0 to disable LES)",
    )

    # Wave inlet
    parser.add_argument(
        "--wave-amp", dest="wave_amp", type=float, default=0.0,
        help="Airy wave horizontal velocity amplitude at free surface (0 = steady inlet)",
    )
    parser.add_argument(
        "--wave-period", dest="wave_period", type=float, default=200.0,
        help="Wave period in LBM time steps",
    )
    parser.add_argument(
        "--wave-k", dest="wave_k", type=float, default=0.05,
        help="Wave number k = 2π/λ (1/lattice spacing)",
    )
    parser.add_argument(
        "--water-depth", dest="water_depth", type=float, default=0.0,
        help="Water depth in lattice units (0 = use nz)",
    )

    # Simulation control
    parser.add_argument(
        "--n-steps", dest="n_steps", type=int, default=2000, help="Simulation steps"
    )
    parser.add_argument(
        "--output-interval",
        dest="output_interval",
        type=int,
        default=200,
        help="Diagnostic and image cadence",
    )
    parser.add_argument(
        "--output-root", dest="output_root", default="outputs", help="Output root directory"
    )
    parser.add_argument(
        "--run-name",
        dest="run_name",
        default=None,
        help="Override deterministic run folder name",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                        help="Execution device")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace output directory if it already exists")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ShipHullFlowConfig(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        u_in=args.u_in,
        re=args.re,
        hull_length=args.hull_length,
        hull_beam=args.hull_beam,
        hull_draft=args.hull_draft,
        smagorinsky_cs=args.smagorinsky_cs,
        wave_amp=args.wave_amp,
        wave_period=args.wave_period,
        wave_k=args.wave_k,
        water_depth=args.water_depth,
        n_steps=args.n_steps,
        output_interval=args.output_interval,
        output_root=args.output_root,
        run_name=args.run_name,
        seed=args.seed,
        device=args.device,
        overwrite=args.overwrite,
    )
    run_ship_hull_flow(config)


if __name__ == "__main__":
    main()
