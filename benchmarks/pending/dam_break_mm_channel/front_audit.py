"""Inspect interface connectivity and liquid-inventory changes over one transition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from benchmarks.pending.dam_break_mm_channel.run import measure


def inspect_state(path: Path, n: int, former_front_cell: int) -> dict:
    with np.load(path, allow_pickle=False) as snapshot:
        fill = snapshot["fill"][1:-1, 1:-1, 1:-1]
    if not n <= former_front_cell < fill.shape[2]:
        raise ValueError("front cell must leave at least one column width upstream")
    connectivity = {}
    for threshold in (0.05, 0.25, 0.5):
        wet = fill >= threshold
        seed = np.zeros_like(wet)
        seed[:, 0, 0] = wet[:, 0, 0]
        connected = ndimage.binary_propagation(seed, mask=wet)
        indices = np.flatnonzero(connected.any(axis=(0, 1)))
        connectivity[str(threshold)] = int(indices[-1]) if indices.size else None
    ahead = slice(former_front_cell, None)
    rightmost_wet = np.flatnonzero(wet.any(axis=(0, 1)))
    bed = fill[:, 0, :]
    return {
        "reported_Z": measure(fill, n)[0],
        "diagnostic_Z_with_quarter_fill_connectivity": measure(
            fill, n, connectivity_threshold=0.25
        )[0],
        "rightmost_half_fill_cell": int(rightmost_wet[-1]) if rightmost_wet.size else None,
        "half_fill_cells_ahead": int(wet[:, :, ahead].sum()),
        "connected_half_fill_cells_ahead": int(connected[:, :, ahead].sum()),
        "rightmost_connected_cell_by_threshold": connectivity,
        "liquid_volume_ahead_cell_units": float(fill[:, :, ahead].sum()),
        "minimum_bed_column_max_in_front_region": float(
            bed[:, former_front_cell - n : former_front_cell + 1].max(axis=0).min()
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--step", type=int, required=True, help="Step before the reported jump")
    parser.add_argument(
        "--front-cell",
        type=int,
        required=True,
        help="Zero-based x cell for the downstream inventory cut",
    )
    parser.add_argument("--cells", type=int, required=True, help="Initial half-width in cells")
    args = parser.parse_args()
    result = {
        "before_step": args.step,
        "front_cell_cut": args.front_cell,
        **{
            label: inspect_state(
                args.directory / f"state_{step:06d}.npz", args.cells, args.front_cell
            )
            for label, step in (("before", args.step), ("after", args.step + 1))
        },
        "inventory": json.loads((args.directory / f"transition_{args.step:06d}.json").read_text()),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
