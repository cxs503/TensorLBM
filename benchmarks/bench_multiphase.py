"""Multiphase LBM model benchmark suite runner.

Runs the full TensorLBM multiphase benchmark suite across three canonical
tests — Laplace pressure, spinodal decomposition, and two-phase Poiseuille —
and prints a quantitative comparison report.

Usage::

    PYTHONPATH=src python benchmarks/bench_multiphase.py
    PYTHONPATH=src python benchmarks/bench_multiphase.py --device cuda --fast
    PYTHONPATH=src python benchmarks/bench_multiphase.py --output outputs/bench_run

Options:
    --device   PyTorch device (default: cpu)
    --fast     Use smaller domains and fewer steps for a quick sanity-check run
    --output   Output root directory (default: outputs/multiphase_benchmark)
    --overwrite   Overwrite existing output directories
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tensorlbm.multiphase_benchmarks import (
    MultiphaseBenchmarkSuiteConfig,
    SpinodaleConfig,
    StaticDropletConfig,
    TwoPhaseChannelCompareConfig,
    run_multiphase_benchmark_suite,
)


def _fast_config(output: Path, device: str, overwrite: bool) -> MultiphaseBenchmarkSuiteConfig:
    """Return a reduced config for a quick sanity-check run (< 60 s on CPU)."""
    return MultiphaseBenchmarkSuiteConfig(
        droplet=StaticDropletConfig(
            nx=60,
            ny=60,
            radii=(10.0, 15.0),
            n_steps=500,
            output_interval=500,
        ),
        spinodal=SpinodaleConfig(
            nx=32,
            ny=32,
            G=-4.0,
            tau=1.0,
            rho0=0.7,
            noise_amp=0.05,
            n_steps=500,
            output_interval=500,
        ),
        poiseuille=TwoPhaseChannelCompareConfig(
            nx=4,
            ny=30,
            G_x=5e-5,
            n_steps=1000,
            output_interval=1000,
        ),
        output_root=output,
        device=device,
        overwrite=overwrite,
    )


def _full_config(output: Path, device: str, overwrite: bool) -> MultiphaseBenchmarkSuiteConfig:
    """Return the full production benchmark config."""
    return MultiphaseBenchmarkSuiteConfig(
        droplet=StaticDropletConfig(
            nx=100,
            ny=100,
            radii=(10.0, 15.0, 20.0),
            n_steps=4000,
            output_interval=1000,
        ),
        spinodal=SpinodaleConfig(
            nx=64,
            ny=64,
            G=-4.0,
            tau=1.0,
            rho0=0.7,
            noise_amp=0.05,
            n_steps=3000,
            output_interval=500,
        ),
        poiseuille=TwoPhaseChannelCompareConfig(
            nx=4,
            ny=40,
            G_x=5e-5,
            n_steps=8000,
            output_interval=2000,
        ),
        output_root=output,
        device=device,
        overwrite=overwrite,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TensorLBM multiphase model benchmark suite"
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda", "mps"],
        help="PyTorch device (default: cpu)",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Run a quick sanity-check with reduced domain and step count",
    )
    parser.add_argument(
        "--output", default="outputs/multiphase_benchmark",
        help="Output root directory (default: outputs/multiphase_benchmark)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output directories",
    )
    args = parser.parse_args()

    output = Path(args.output)

    if args.fast:
        cfg = _fast_config(output, args.device, args.overwrite)
    else:
        cfg = _full_config(output, args.device, args.overwrite)

    run_multiphase_benchmark_suite(cfg)


if __name__ == "__main__":
    main()
