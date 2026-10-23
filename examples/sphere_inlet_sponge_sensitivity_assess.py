#!/usr/bin/env python3
"""Assess a matched current-code sphere inlet-sponge A/B pair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tensorlbm.sphere_boundary_sensitivity import assess_sphere_inlet_sponge_pair


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("disabled", type=Path)
    parser.add_argument("enabled", type=Path)
    parser.add_argument("--maximum-drag-change-pct", type=float, default=1.0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sources = []
    records = []
    for path in (args.disabled, args.enabled):
        payload = path.read_bytes()
        sources.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        records.append(json.loads(payload))
    result = assess_sphere_inlet_sponge_pair(
        records,
        maximum_drag_change_pct=args.maximum_drag_change_pct,
    )
    result["sources"] = sources
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
