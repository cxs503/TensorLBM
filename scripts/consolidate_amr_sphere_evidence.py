#!/usr/bin/env python3
"""Consolidate AMR sphere drag validation runs into one evidence record.

Reads all outputs/amr-sphere-*.json plus the uniform-grid control, attaches
source hashes, and writes a single evidence JSON with the key metrics.
"""
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
        payload = path.read_bytes()
        record = json.loads(payload)
        sources = record.setdefault("sources", [])
        sources.append({"path": str(path), "sha256": _sha256_file(path)})
        result = record.get("result", {})
        records.append({
            "file": path.name,
            "status": record.get("status"),
            "case": record.get("case"),
            "cd_control_volume": result.get("cd_control_volume"),
            "reference_cd": result.get("reference_cd"),
            "reference_error_pct": result.get("reference_error_pct"),
            "max_reflux_residual": result.get("max_reflux_residual"),
            "wall_time_s": result.get("wall_time_s"),
            "configuration": record.get("configuration"),
            "sources": sources,
        })

    evidence = {
        "schema": "tensorlbm-amr-sphere-drag-validation-v1",
        "status": "measured_candidate",
        "physical_validation": False,
        "case": "AMR sphere drag Re=100: coarse grid + 2:1 fine block vs uniform-grid control vs Schiller-Naumann",
        "reference": {
            "schiller_naumann_cd": 1.0917,
            "uniform_grid_extrapolated_cd": 1.1263,
            "uniform_grid_extrapolated_error_pct": 3.17,
            "note": "uniform R9/R12/R15 convergence admitted 2026-08-02 (sphere-v9-corrected-bfl)",
        },
        "runs": records,
        "conclusion": (
            "AMR mechanics are healthy (reflux residual ~1e-9, no NaN). "
            "3000-step runs are dominated by transient statistics (drift 19-21%); "
            "8000-step runs reach perfect stationarity. Best AMR (m28+trilinear, "
            "8.8M cells): Cd=1.1769, 7.80% vs Schiller-Naumann 1.0917. "
            "Uniform-grid R16 (16.1M cells, same 8000 steps): Cd=1.1403, 4.45% "
            "with drift 0.12% — the R16 fine block does NOT fully deliver R16 "
            "accuracy: AMR sits at uniform-R8 accuracy (7.71%) despite R16 "
            "resolution in the block. Interface error (ghost interpolation + "
            "block margin) consumes ~3.4pp of the expected 2x-resolution gain. "
            "Speed/memory: AMR m16 = 2.8x faster + 76% memory saving at 9.76% "
            "error; AMR m28 = 1.3x + 45% saving at 7.80%. The classic AMR "
            "value proposition (same accuracy, faster) is NOT yet demonstrated "
            "— reaching uniform-R16 accuracy (4.45%) requires two-level nested "
            "refinement (L2 hugging the sphere surface) or a larger fine block."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2), flush=True)


if __name__ == "__main__":
    main()
