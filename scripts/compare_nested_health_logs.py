#!/usr/bin/env python3
"""Compare two nested-LBM health logs at identical root steps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensorlbm.nested_health_comparison import (
    compare_nested_health,
    read_nested_health_log,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare_nested_health(
        read_nested_health_log(args.baseline),
        read_nested_health_log(args.candidate),
    )
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
