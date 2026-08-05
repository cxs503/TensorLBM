#!/usr/bin/env python3
"""Validate a static 2:1 interface against a uniform-fine reference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensorlbm.amr_interface_validation import (
    AMRInterfaceValidationConfig,
    run_amr_interface_validation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument(
        "--reflux-correction-stencil",
        choices=("exterior_cells", "crossing_links"),
        default="exterior_cells",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_amr_interface_validation(
        AMRInterfaceValidationConfig(
            device=args.device,
            steps=args.steps,
            reflux_correction_stencil=args.reflux_correction_stencil,
        ),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
