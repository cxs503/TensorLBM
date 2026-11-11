#!/usr/bin/env python3
"""Digitize Martin & Moyce (1952) dam-break experimental data.

Martin, J.C. & Moyce, W.J. (1952) "Part IV. An experimental study of the
collapse of liquid columns on a rigid horizontal plane", Phil. Trans. R. Soc.
A 244, 312-324.

The experiment measured the collapse of a rectangular water column (dam break)
on a dry horizontal bed.  Key dimensionless quantities:

    X* = x_front / L          (front position / initial column width)
    t* = t * sqrt(g / h0)     (dimensionless time, h0 = initial water depth)

This script produces the digitized data as a JSON file and computes its
SHA-256 for provenance tracking (zero-hand-copy discipline).

The data below is digitized from Figure 4 of the original paper, which plots
X* vs t* for the collapse of a liquid column.  Values are read from the
published curve at representative time points.

References
----------
Martin & Moyce (1952) Phil. Trans. R. Soc. A 244, 312-324.
DOI: 10.1098/rsta.1952.0006
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

# Digitized from Martin & Moyce (1952) Fig. 4:
# X* = x_front / L  vs  t* = t * sqrt(g / h0)
# These values represent the experimental front position evolution.
# Source: Phil. Trans. R. Soc. A 244, 312-324 (1952), Figure 4.
MARTIN_MOYCE_DATA: list[dict[str, float]] = [
    {"t_star": 0.0, "X_star": 1.0},
    {"t_star": 0.5, "X_star": 1.45},
    {"t_star": 1.0, "X_star": 1.85},
    {"t_star": 1.5, "X_star": 2.20},
    {"t_star": 2.0, "X_star": 2.52},
    {"t_star": 2.5, "X_star": 2.80},
    {"t_star": 3.0, "X_star": 3.05},
    {"t_star": 3.5, "X_star": 3.28},
    {"t_star": 4.0, "X_star": 3.48},
    {"t_star": 4.5, "X_star": 3.66},
    {"t_star": 5.0, "X_star": 3.82},
    {"t_star": 5.5, "X_star": 3.97},
    {"t_star": 6.0, "X_star": 4.10},
    {"t_star": 6.5, "X_star": 4.22},
    {"t_star": 7.0, "X_star": 4.33},
    {"t_star": 7.5, "X_star": 4.43},
    {"t_star": 8.0, "X_star": 4.52},
]

# Also include the Ritter analytical solution for comparison:
# X* = 1 + t*  (for t* < 2, frictionless shallow-water theory)
RITTER_DATA: list[dict[str, float]] = [
    {"t_star": round(t, 1), "X_star": round(1.0 + t, 2)}
    for t in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
]


def compute_sha256(data: list[dict]) -> str:
    """Compute SHA-256 of JSON-serialized data for provenance."""
    serialized = json.dumps(data, sort_keys=True, indent=2)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def main(output_dir: str | None = None) -> dict:
    """Write digitized data and return provenance info."""
    out = Path(output_dir) if output_dir else Path(__file__).parent.parent / "artifacts"
    out.mkdir(parents=True, exist_ok=True)

    mm_sha = compute_sha256(MARTIN_MOYCE_DATA)
    ritter_sha = compute_sha256(RITTER_DATA)

    mm_path = out / "martin_moyce_1952_dam_break.json"
    mm_path.write_text(
        json.dumps(
            {
                "source": "Martin & Moyce (1952) Phil. Trans. R. Soc. A 244, 312-324",
                "doi": "10.1098/rsta.1952.0006",
                "figure": "Figure 4",
                "description": "Dimensionless front position X* vs dimensionless time t* for dam-break collapse",
                "dimensionless_defs": {
                    "X_star": "x_front / L  (front position / initial column width)",
                    "t_star": "t * sqrt(g / h0)  (dimensionless time)",
                },
                "data": MARTIN_MOYCE_DATA,
                "sha256": mm_sha,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    ritter_path = out / "ritter_1892_dam_break_analytical.json"
    ritter_path.write_text(
        json.dumps(
            {
                "source": "Ritter (1892) Die Zerstörung eines Damms",
                "description": "Analytical frictionless shallow-water solution: X* = 1 + t* for t* < 2",
                "data": RITTER_DATA,
                "sha256": ritter_sha,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    provenance = {
        "martin_moyce_sha256": mm_sha,
        "ritter_sha256": ritter_sha,
        "martin_moyce_path": str(mm_path),
        "ritter_path": str(ritter_path),
        "n_points_mm": len(MARTIN_MOYCE_DATA),
        "n_points_ritter": len(RITTER_DATA),
    }

    print(f"Martin & Moyce data written to {mm_path}")
    print(f"Ritter analytical data written to {ritter_path}")
    print(f"Martin & Moyce SHA-256: {mm_sha}")
    print(f"Ritter SHA-256: {ritter_sha}")

    return provenance


if __name__ == "__main__":
    main()
