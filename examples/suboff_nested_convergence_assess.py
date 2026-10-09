#!/usr/bin/env python3
"""Assess an exact three-grid nested SUBOFF sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensorlbm.suboff_nested_convergence import assess_suboff_nested_convergence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = [json.loads(path.read_text(encoding="utf-8")) for path in args.records]
    result = assess_suboff_nested_convergence(records)
    rendered = json.dumps(result, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
