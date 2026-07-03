"""Multiphase LBM model benchmark suite runner.

Runs the full TensorLBM multiphase benchmark suite across four canonical
tests — Laplace pressure, spinodal decomposition, free-energy droplet
relaxation, and two-phase Poiseuille — and prints a quantitative comparison
report.

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
    FreeEnergyDropletConfig,
    MultiphaseBenchmarkSuiteConfig,
    Spinodal3DConfig,
    SpinodaleConfig,
    StaticDroplet3DConfig,
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
        free_energy=FreeEnergyDropletConfig(
            nx=40,
            ny=40,
            radius=8.0,
            interface_width=2.0,
            n_steps=200,
            output_interval=100,
        ),
        droplet_3d=StaticDroplet3DConfig(
            nx=20,
            ny=20,
            nz=20,
            radii=(4.0,),
            n_steps=100,
            output_interval=100,
        ),
        spinodal_3d=Spinodal3DConfig(
            nx=20,
            ny=20,
            nz=20,
            G=-4.0,
            tau=1.0,
            rho0=0.7,
            noise_amp=0.05,
            n_steps=120,
            output_interval=120,
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
        free_energy=FreeEnergyDropletConfig(
            nx=80,
            ny=80,
            radius=16.0,
            interface_width=2.5,
            n_steps=1200,
            output_interval=300,
        ),
        droplet_3d=StaticDroplet3DConfig(
            nx=40,
            ny=40,
            nz=40,
            radii=(8.0, 12.0),
            n_steps=1200,
            output_interval=400,
        ),
        spinodal_3d=Spinodal3DConfig(
            nx=40,
            ny=40,
            nz=40,
            G=-4.0,
            tau=1.0,
            rho0=0.7,
            noise_amp=0.05,
            n_steps=1200,
            output_interval=300,
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
        "--device", default="cpu", choices=["cpu", "sdaa", "cuda", "mps"],
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
