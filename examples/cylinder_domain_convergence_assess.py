#!/usr/bin/env python3
"""Assess cylinder lateral-domain convergence with hashed source records."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tensorlbm.cylinder_domain_convergence import assess_cylinder_domain_convergence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = []
    sources = []
    for value in args.inputs:
        path = Path(value)
        payload = path.read_bytes()
        records.append(json.loads(payload))
        sources.append({
            "path": str(path),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
    result = assess_cylinder_domain_convergence(records)
    result["sources"] = sources
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
