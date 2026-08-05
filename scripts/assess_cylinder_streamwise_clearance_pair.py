#!/usr/bin/env python3
"""Assess a cylinder streamwise-clearance intervention with source hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tensorlbm.cylinder_domain_convergence import (
    assess_cylinder_streamwise_clearance_pair,
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assess_cylinder_streamwise_clearance_pair(
        _load(args.baseline), _load(args.candidate),
    )
    result["sources"] = [
        {"path": str(path), "sha256": _sha256(path)}
        for path in (args.baseline, args.candidate)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    if not result["acceptance"]["causal_pair_admitted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
