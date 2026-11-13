"""CLI entry point for XVAL-LBM-CARTESIAN-MPI campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .case_spec import PoiseuilleSpec, DEFAULT_SPEC
from .campaign import run_campaign


def main():
    parser = argparse.ArgumentParser(description="XVAL-LBM-CARTESIAN-MPI campaign")
    parser.add_argument("--out-dir", default="runs/xval_cartesian_mpi", help="Output directory")
    parser.add_argument("--nx", type=int, default=None, help="Streamwise resolution")
    parser.add_argument("--ny", type=int, default=None, help="Wall-normal resolution")
    parser.add_argument("--re", type=float, default=None, help="Reynolds number")
    parser.add_argument("--u-max", type=float, default=None, help="Centreline velocity")
    parser.add_argument("--steps", type=int, default=None, help="Total steps")
    parser.add_argument("--npx", type=int, default=None, help="Subdomains in x")
    parser.add_argument("--npy", type=int, default=None, help="Subdomains in y")
    parser.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    parser.add_argument("--threads", type=int, default=8, help="CPU threads")
    parser.add_argument("--quick", action="store_true", help="Quick run (reduced steps)")

    args = parser.parse_args()

    # Build spec
    spec_kwargs = {}
    if args.nx is not None:
        spec_kwargs["nx"] = args.nx
    if args.ny is not None:
        spec_kwargs["ny"] = args.ny
    if args.re is not None:
        spec_kwargs["re"] = args.re
    if args.u_max is not None:
        spec_kwargs["u_max"] = args.u_max
    if args.steps is not None:
        spec_kwargs["steps"] = args.steps
    if args.npx is not None:
        spec_kwargs["npx"] = args.npx
    if args.npy is not None:
        spec_kwargs["npy"] = args.npy

    if args.quick:
        spec_kwargs["steps"] = 2000
        spec_kwargs["sample_every"] = 100
        spec_kwargs["warmup_steps"] = 500
        spec_kwargs["nx"] = 128
        spec_kwargs["ny"] = 32

    spec = PoiseuilleSpec(**spec_kwargs) if spec_kwargs else DEFAULT_SPEC

    print(f"[xval_cartesian_mpi] Running campaign with spec:")
    print(f"  nx={spec.nx}, ny={spec.ny}, Re={spec.re}, U_max={spec.u_max}")
    print(f"  tau={spec.tau_computed:.4f}, Ma={spec.mach:.4f}")
    print(f"  decomposition: {spec.npx}x{spec.npy} = {spec.npx * spec.npy} subdomains")
    print(f"  steps={spec.steps}, sample_every={spec.sample_every}")

    result = run_campaign(spec, args.out_dir, dtype=args.dtype, threads=args.threads)

    print(f"\n[xval_cartesian_mpi] Result: {result['status']}")
    print(f"  Artefacts in: {args.out_dir}")

    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
