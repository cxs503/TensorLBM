#!/usr/bin/env python3
"""Assess a declared resolution/value sequence and write JSON evidence."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from tensorlbm.spatial_convergence import assess_spatial_convergence


def _csv_floats(value: str) -> list[float]:
    try:
        parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from error
    if not parsed:
        raise argparse.ArgumentTypeError("sequence cannot be empty")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", type=_csv_floats, required=True)
    parser.add_argument("--values", type=_csv_floats, required=True)
    parser.add_argument("--quantity", required=True)
    parser.add_argument("--maximum-finest-error-pct", type=float, default=2.0)
    parser.add_argument("--maximum-fit-rms-pct", type=float, default=1.0)
    parser.add_argument("--minimum-order", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assessment = assess_spatial_convergence(args.resolutions, args.values)
    result = {
        "schema": "tensorlbm-spatial-convergence-evidence-v1",
        "quantity": args.quantity,
        "assessment": asdict(assessment),
        "acceptance": {
            "maximum_finest_error_pct": args.maximum_finest_error_pct,
            "maximum_fit_rms_pct": args.maximum_fit_rms_pct,
            "minimum_order": args.minimum_order,
            "admitted": assessment.meets(
                maximum_finest_error_pct=args.maximum_finest_error_pct,
                maximum_fit_rms_pct=args.maximum_fit_rms_pct,
                minimum_order=args.minimum_order,
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
