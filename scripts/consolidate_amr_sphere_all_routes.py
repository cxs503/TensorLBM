#!/usr/bin/env python3
"""Consolidate all AMR sphere drag validation runs (3 routes + combo + controls)
into one evidence record with source hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records = []
    for value in args.inputs:
        path = Path(value)
        if not path.exists():
            print(f"skip missing: {path}")
            continue
        payload = path.read_bytes()
        record = json.loads(payload)
        sources = record.setdefault("sources", [])
        sources.append({"path": str(path), "sha256": _sha256_file(path)})
        result = record.get("result", {})
        config = record.get("configuration", {})
        records.append({
            "file": path.name,
            "schema": record.get("schema"),
            "status": record.get("status"),
            "case": record.get("case"),
            "cd_control_volume": result.get("cd_control_volume"),
            "reference_cd": result.get("reference_cd"),
            "reference_error_pct": result.get("reference_error_pct"),
            "cell_saving_fraction": config.get("cell_saving_fraction"),
            "max_reflux_residual": result.get("max_reflux_residual"),
            "wall_time_s": result.get("wall_time_s"),
            "mean_patch_count": result.get("mean_patch_count"),
            "sphere_solid_covered_fraction": result.get(
                "sphere_solid_covered_fraction"
            ),
            "configuration": config,
            "sources": sources,
        })

    evidence = {
        "schema": "tensorlbm-amr-sphere-all-routes-validation-v1",
        "status": "measured_candidate",
        "physical_validation": False,
        "case": "AMR sphere drag Re=100: route1 shell / route2 nested-L2 / route3 cellwise / combo shell+L2 vs uniform controls",
        "reference": {
            "schiller_naumann_cd": 1.0917,
            "uniform_R16_cd": 1.1403,
            "uniform_R16_error_pct": 4.45,
        },
        "runs": records,
        "conclusion": (
            "Route2 nested-L2 is the best AMR accuracy (Cd=1.1573, 6.00%, "
            "45% memory saving), approaching uniform-R16 (4.45%). Route1 "
            "body-fitted shell gives 86.3% memory saving at 7.97% (≈ single-"
            "block 7.80%). Route3 cellwise adaptive failed on GPU "
            "(sphere cover 90.8%, Cd error >600%) — coarse-grid patch BFL "
            "weighting unreliable. Combo shell+L2 saves 97.9% memory but "
            "error degrades to 18.8%: the shell removes the thick fine region "
            "around the whole sphere that carries the boundary-layer/pressure "
            "development — blunt-body flows need that volume, shell-only "
            "refinement suits slender bodies (SUBOFF) instead."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2), flush=True)


if __name__ == "__main__":
    main()
