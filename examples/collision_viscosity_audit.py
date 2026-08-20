#!/usr/bin/env python3
"""Audit recovered D3Q19 viscosity with periodic shear-wave decay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensorlbm.collision_viscosity_audit import (
    CollisionViscosityAuditConfig,
    run_collision_viscosity_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collision-model",
        choices=(
            "bgk",
            "cumulant",
            "cumulant_wale",
            "cumulant_vreman",
            "planar_cumulant_d2q9",
            "entropic_kbc",
            "natural_kbc",
        ),
        required=True,
    )
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--wavelength-cells", type=int, default=32)
    parser.add_argument("--transverse-cells", type=int, default=4)
    parser.add_argument("--amplitude", type=float, default=1.0e-3)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--fit-start-step", type=int, default=20)
    parser.add_argument("--maximum-relative-error-pct", type=float, default=2.0)
    parser.add_argument("--kbc-max-iterations", type=int, default=12)
    parser.add_argument("--wale-cw", type=float, default=0.5)
    parser.add_argument("--vreman-cv", type=float, default=0.025)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--natural-kbc-compute-dtype",
        choices=("storage", "float64"),
        default="storage",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_collision_viscosity_audit(
        CollisionViscosityAuditConfig(
            collision_model=args.collision_model,
            tau=args.tau,
            wavelength_cells=args.wavelength_cells,
            transverse_cells=args.transverse_cells,
            amplitude=args.amplitude,
            steps=args.steps,
            fit_start_step=args.fit_start_step,
            maximum_relative_error_pct=args.maximum_relative_error_pct,
            kbc_max_iterations=args.kbc_max_iterations,
            wale_cw=args.wale_cw,
            vreman_cv=args.vreman_cv,
            device=args.device,
            dtype=args.dtype,
            natural_kbc_compute_dtype=args.natural_kbc_compute_dtype,
        )
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
