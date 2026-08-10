#!/usr/bin/env python3
"""Assess equivalent production SUBOFF static-AMR results as one grid sequence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tensorlbm.suboff_amr_convergence import assess_suboff_amr_convergence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = []
    sources = []
    for path_value in args.inputs:
        path = Path(path_value)
        payload = path.read_bytes()
        records.append(json.loads(payload))
        sources.append({
            "path": str(path),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
    assessment = assess_suboff_amr_convergence(records)
    assessment["sources"] = sources
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(assessment, indent=2), encoding="utf-8")
    print(json.dumps(assessment, indent=2), flush=True)


if __name__ == "__main__":
    main()
