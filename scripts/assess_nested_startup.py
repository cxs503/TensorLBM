#!/usr/bin/env python3
"""Assess a nested LBM result as startup evidence, never force validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensorlbm.nested_startup_stability import assess_nested_startup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--maximum-speed", type=float, default=0.3)
    parser.add_argument("--maximum-limited-fraction", type=float, default=1.0e-3)
    parser.add_argument("--maximum-reflux-residual", type=float, default=1.0e-6)
    parser.add_argument("--allow-before-target-reynolds", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    assessment = assess_nested_startup(
        payload,
        maximum_speed=args.maximum_speed,
        maximum_limited_fraction=args.maximum_limited_fraction,
        maximum_reflux_residual=args.maximum_reflux_residual,
        require_target_reynolds=not args.allow_before_target_reynolds,
    ).to_dict()
    rendered = json.dumps(assessment, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
