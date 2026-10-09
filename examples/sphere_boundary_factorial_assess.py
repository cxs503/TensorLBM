#!/usr/bin/env python3
"""Assess the complete sphere width-by-inlet-sponge 2x2 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tensorlbm.sphere_boundary_factorial import (
    assess_sphere_domain_inlet_factorial,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs=4, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sources = []
    records = []
    for path in args.inputs:
        payload = path.read_bytes()
        sources.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        records.append(json.loads(payload))
    result = assess_sphere_domain_inlet_factorial(records)
    result["sources"] = sources
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
