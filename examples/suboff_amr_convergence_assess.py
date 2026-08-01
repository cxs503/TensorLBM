#!/usr/bin/env python3
"""Assess equivalent production SUBOFF static-AMR results as one grid sequence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensorlbm.suboff_amr_convergence import assess_suboff_amr_convergence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in args.inputs
    ]
    assessment = assess_suboff_amr_convergence(records)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(assessment, indent=2), encoding="utf-8")
    print(json.dumps(assessment, indent=2), flush=True)


if __name__ == "__main__":
    main()
