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
        choices=("bgk", "cumulant", "entropic_kbc"),
        required=True,
    )
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--wavelength-cells", type=int, default=32)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--fit-start-step", type=int, default=20)
    parser.add_argument("--kbc-max-iterations", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_collision_viscosity_audit(CollisionViscosityAuditConfig(
        collision_model=args.collision_model,
        tau=args.tau,
        wavelength_cells=args.wavelength_cells,
        steps=args.steps,
        fit_start_step=args.fit_start_step,
        kbc_max_iterations=args.kbc_max_iterations,
        device=args.device,
        dtype=args.dtype,
    ))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
