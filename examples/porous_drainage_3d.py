"""3D porous drainage example using TensorLBM.

Simulates gas (non-wetting phase) invading a water-saturated 3D porous
medium composed of randomly placed spheres.  Gas saturation is tracked
over time.

Usage
-----
Default run (small domain, random spheres)::

    PYTHONPATH=src python examples/porous_drainage_3d.py

Tube array medium::

    PYTHONPATH=src python examples/porous_drainage_3d.py --medium tube_array

Custom domain::

    PYTHONPATH=src python examples/porous_drainage_3d.py \\
        --nz 60 --ny 32 --nx 32 --n-spheres 20 --n-steps 5000

Options
-------
--nz / --ny / --nx     Domain size (default 40 × 24 × 24).
--medium               ``random_spheres`` or ``tube_array``.
--n-spheres            Number of spheres (random_spheres only, default 8).
--r-min / --r-max      Sphere radius range (default 2.0 – 4.0).
--G                    SC coupling constant (default 0.9).
--n-steps              Number of time steps (default 2000).
--output-interval      Diagnostic output interval (default 500).
--run-name             Run folder name.
--overwrite            Overwrite existing output folder.
--device               ``cpu`` (default), ``cuda``, or ``mps``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tensorlbm.porous_media3d import PorousDrainageConfig3D, run_porous_drainage_3d


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3D porous drainage LBM example")
    p.add_argument("--nz", type=int, default=40)
    p.add_argument("--ny", type=int, default=24)
    p.add_argument("--nx", type=int, default=24)
    p.add_argument(
        "--medium",
        choices=["random_spheres", "tube_array"],
        default="random_spheres",
    )
    p.add_argument("--n-spheres", type=int, default=8, dest="n_spheres")
    p.add_argument("--r-min", type=float, default=2.0, dest="r_min")
    p.add_argument("--r-max", type=float, default=4.0, dest="r_max")
    p.add_argument("--n-tubes-y", type=int, default=2, dest="n_tubes_y")
    p.add_argument("--n-tubes-x", type=int, default=2, dest="n_tubes_x")
    p.add_argument("--tube-width", type=int, default=4, dest="tube_width")
    p.add_argument("--G", type=float, default=0.9)
    p.add_argument("--G-ads", type=float, default=0.3, dest="G_ads")
    p.add_argument("--u-inlet", type=float, default=0.005, dest="u_inlet")
    p.add_argument("--n-steps", type=int, default=2000, dest="n_steps")
    p.add_argument("--output-interval", type=int, default=500, dest="output_interval")
    p.add_argument("--run-name", default=None, dest="run_name")
    p.add_argument("--output-root", default="outputs", dest="output_root")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    config = PorousDrainageConfig3D(
        nz=args.nz,
        ny=args.ny,
        nx=args.nx,
        medium=args.medium,
        n_spheres=args.n_spheres,
        r_min=args.r_min,
        r_max=args.r_max,
        n_tubes_y=args.n_tubes_y,
        n_tubes_x=args.n_tubes_x,
        tube_width=args.tube_width,
        G_12=args.G,
        G_ads=args.G_ads,
        u_inlet=args.u_inlet,
        n_steps=args.n_steps,
        output_interval=args.output_interval,
        run_name=args.run_name,
        output_root=Path(args.output_root),
        overwrite=args.overwrite,
        device=args.device,
        seed=args.seed,
    )

    result = run_porous_drainage_3d(config)

    print("\n=== Summary ===")
    print(f"Porosity:          {result['porosity']:.4f}")
    series = result["saturation_series"]
    if series:
        last = series[-1]
        print(f"Final saturation:  {last['gas_saturation']:.4f}")


if __name__ == "__main__":
    main()
