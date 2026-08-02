#!/usr/bin/env python3
"""Audit recovered viscosity for every relaxation time in a level schedule."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

from tensorlbm.collision_viscosity_audit import (
    CollisionViscosityAuditConfig,
    run_collision_viscosity_audit,
)


def assess(
    collision_model: str,
    taus: list[float],
    *,
    wavelength_cells: int,
    transverse_cells: int,
    amplitude: float,
    steps: int,
    fit_start_step: int,
    maximum_relative_error_pct: float,
    device: str,
    dtype: str,
    natural_kbc_compute_dtype: str = "storage",
) -> dict:
    if not taus or any(not math.isfinite(tau) for tau in taus):
        raise ValueError("taus must contain finite values")
    if len(set(taus)) != len(taus):
        raise ValueError("taus must be unique")
    audits = []
    for level, tau in enumerate(taus):
        config = CollisionViscosityAuditConfig(
            collision_model=collision_model,
            tau=tau,
            wavelength_cells=wavelength_cells,
            transverse_cells=transverse_cells,
            amplitude=amplitude,
            steps=steps,
            fit_start_step=fit_start_step,
            maximum_relative_error_pct=maximum_relative_error_pct,
            device=device,
            dtype=dtype,
            natural_kbc_compute_dtype=natural_kbc_compute_dtype,
        )
        result = run_collision_viscosity_audit(config)
        audits.append({
            "level": level,
            "configuration": asdict(config),
            "result": result["result"],
            "admitted": bool(result["acceptance"]["admitted"]),
        })
    all_admitted = all(item["admitted"] for item in audits)
    return {
        "schema": "tensorlbm-collision-viscosity-schedule-v1",
        "status": "admitted" if all_admitted else "rejected",
        "physical_validation": False,
        "collision_model": collision_model,
        "dtype": dtype,
        "natural_kbc_compute_dtype": natural_kbc_compute_dtype,
        "taus": taus,
        "audits": audits,
        "acceptance": {
            "maximum_relative_error_pct": maximum_relative_error_pct,
            "all_levels_recover_configured_viscosity": all_admitted,
            "configured_reynolds_sequence_admitted": all_admitted,
            "flow_solution_validated": False,
        },
        "prohibition": (
            "A configured Reynolds number must not be treated as a recovered "
            "collision Reynolds number when any scheduled tau fails this audit."
        ),
    }


def _parse_taus(value: str) -> list[float]:
    try:
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("taus must be comma-separated floats") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collision-model", required=True)
    parser.add_argument("--taus", type=_parse_taus, required=True)
    parser.add_argument("--wavelength-cells", type=int, default=16)
    parser.add_argument("--transverse-cells", type=int, default=3)
    parser.add_argument("--amplitude", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--fit-start-step", type=int, default=1000)
    parser.add_argument("--maximum-relative-error-pct", type=float, default=5.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument(
        "--natural-kbc-compute-dtype",
        choices=("storage", "float64"),
        default="storage",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assess(
        args.collision_model,
        args.taus,
        wavelength_cells=args.wavelength_cells,
        transverse_cells=args.transverse_cells,
        amplitude=args.amplitude,
        steps=args.steps,
        fit_start_step=args.fit_start_step,
        maximum_relative_error_pct=args.maximum_relative_error_pct,
        device=args.device,
        dtype=args.dtype,
        natural_kbc_compute_dtype=args.natural_kbc_compute_dtype,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
