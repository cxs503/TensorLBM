"""DG-LBM hybrid SUBOFF submarine-flow case entry script.

Runs a DARPA SUBOFF axisymmetric hull inside a D3Q19 channel using the
DG-LBM hybrid solver: a near-wall DG zone handles high-gradient wall
regions while the LBM exterior uses standard BGK collision.

Usage
-----
Run from the repository root with PYTHONPATH pointing to src::

    PYTHONPATH=src python cases/suboff_hybrid/run_suboff_hybrid.py

Coarse smoke run (fast, ~seconds on CPU)::

    PYTHONPATH=src python cases/suboff_hybrid/run_suboff_hybrid.py \\
        --nx 80 --ny 40 --nz 40 \\
        --hull-length 48 --re 200 \\
        --n-steps 20 --output-interval 10 \\
        --run-name smoke --overwrite

Production-quality run (CPU, minutes)::

    PYTHONPATH=src python cases/suboff_hybrid/run_suboff_hybrid.py \\
        --nx 200 --ny 80 --nz 80 \\
        --hull-length 120 --re 200 \\
        --n-steps 500 --output-interval 100 \\
        --run-name suboff_re200 --overwrite

Outputs are written to ``outputs/dg_lbm_suboff/<run-name>/``.
"""
from __future__ import annotations

import argparse

from tensorlbm import DGLBMSuboffConfig, run_dg_lbm_suboff_flow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DG-LBM hybrid SUBOFF submarine flow simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nx", type=int, default=200, help="Grid length (x / axial)")
    parser.add_argument("--ny", type=int, default=80, help="Grid width  (y)")
    parser.add_argument("--nz", type=int, default=80, help="Grid depth  (z / vertical)")
    parser.add_argument(
        "--u-in", dest="u_in", type=float, default=0.06,
        help="Inlet velocity (lattice units)",
    )
    parser.add_argument(
        "--re", type=float, default=200.0,
        help="Reynolds number (based on hull length)",
    )
    parser.add_argument(
        "--hull-length", dest="hull_length", type=float, default=120.0,
        help="SUBOFF hull length (lattice units)",
    )
    parser.add_argument(
        "--hull-type", dest="hull_type",
        choices=["bare_hull", "with_sail", "full"],
        default="bare_hull",
        help="SUBOFF model variant",
    )
    parser.add_argument(
        "--dg-band", dest="dg_band", type=float, default=4.0,
        help="DG near-wall zone thickness (lattice units)",
    )
    parser.add_argument(
        "--n-steps", dest="n_steps", type=int, default=500,
        help="Total simulation steps",
    )
    parser.add_argument(
        "--output-interval", dest="output_interval", type=int, default=100,
        help="Steps between snapshots and checkpoints",
    )
    parser.add_argument(
        "--output-root", dest="output_root", default="outputs",
        help="Root directory for outputs",
    )
    parser.add_argument(
        "--run-name", dest="run_name", default=None,
        help="Override auto-generated run folder name",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu",
        help="Execution device",
    )
    parser.add_argument(
        "--num-threads", dest="num_threads", type=int, default=None,
        help="PyTorch CPU thread count (ignored for CUDA)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace output directory if it already exists",
    )
    parser.add_argument(
        "--compile", dest="use_compile", action="store_true",
        help="JIT-compile streaming kernel with torch.compile",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = DGLBMSuboffConfig(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        u_in=args.u_in,
        re=args.re,
        hull_length=args.hull_length,
        hull_type=args.hull_type,
        dg_band=args.dg_band,
        n_steps=args.n_steps,
        output_interval=args.output_interval,
        output_root=args.output_root,
        run_name=args.run_name,
        seed=args.seed,
        device=args.device,
        num_threads=args.num_threads,
        overwrite=args.overwrite,
        use_compile=args.use_compile,
    )
    config.validate()
    run_dir = run_dg_lbm_suboff_flow(config)
    print(f"Run complete.  Outputs: {run_dir}")


if __name__ == "__main__":
    main()
