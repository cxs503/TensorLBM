#!/usr/bin/env python3
"""CLI entry point for the Wigley hull channel-flow simulation.

Quick smoke run::

    PYTHONPATH=src python examples/ship_hull_flow.py \\
        --nx 60 --ny 24 --nz 24 \\
        --length-lbm 20 --beam-lbm 4 --draft-lbm 4 \\
        --n-steps 10 --output-interval 5 \\
        --run-name smoke --overwrite

Default (full) run::

    PYTHONPATH=src python examples/ship_hull_flow.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensorlbm.ship_flow import ShipHullFlowConfig, run_ship_hull_flow


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TensorLBM – 3D Wigley hull flow (D3Q19 + Smagorinsky LES)"
    )
    p.add_argument("--nx", type=int, default=200, help="Streamwise grid points")
    p.add_argument("--ny", type=int, default=60, help="Transverse grid points")
    p.add_argument("--nz", type=int, default=60, help="Vertical grid points")
    p.add_argument("--u-in", type=float, default=0.05, help="Inlet x-velocity")
    p.add_argument("--re", type=float, default=500.0, help="Reynolds number")
    p.add_argument("--length-lbm", type=int, default=80, help="Ship length [lattice units]")
    p.add_argument("--beam-lbm", type=int, default=12, help="Max beam [lattice units]")
    p.add_argument("--draft-lbm", type=int, default=10, help="Draft [lattice units]")
    p.add_argument("--C-s", type=float, default=0.1, help="Smagorinsky constant")
    p.add_argument("--n-steps", type=int, default=2000, help="Total simulation steps")
    p.add_argument("--output-interval", type=int, default=200, help="Steps per output")
    p.add_argument("--output-root", type=Path, default=Path("outputs"))
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = ShipHullFlowConfig(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        u_in=args.u_in,
        re=args.re,
        length_lbm=args.length_lbm,
        beam_lbm=args.beam_lbm,
        draft_lbm=args.draft_lbm,
        C_s=args.C_s,
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
