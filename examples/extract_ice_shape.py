#!/usr/bin/env python
"""Post-process aircraft-icing results (ice shape metrics from saved run).

Reads ``result.npz`` written by ``benchmark_aircraft_icing.py`` (or by
``tensorlbm.aircraft_icing.save_icing_artifacts``) and prints the ice
shape metrics, the mass audit is echoed from ``icing_metrics.json``.

Usage
-----
    python examples/extract_ice_shape.py outputs/aircraft_icing/result.npz
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tensorlbm.aircraft_icing import ice_shape_metrics  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        npz = Path("outputs/aircraft_icing/result.npz")
    else:
        npz = Path(sys.argv[1])
    d = np.load(npz)
    airfoil = d["airfoil"]
    solid = d["solid"]
    stag = tuple(int(v) for v in d["stag"])
    out_dir = npz.parent

    # chord length in lattice units from the airfoil extent
    xs = np.nonzero(airfoil.any(axis=0))[0]
    chord_lu = float(xs.max() - xs.min() + 1)

    # physical scaling re-read from the JSON sidecar (kept alongside npz)
    dx_phys = chord_phys = None
    js = out_dir / "icing_metrics.json"
    if js.exists():
        with open(js) as fh:
            blob = json.load(fh)
        mp = blob["mapping"]
        dx_phys, chord_phys = mp["dx_phys"], mp["chord_phys"]
        print(f"  (mapping from icing_metrics.json: dx={dx_phys:.4e} m, chord={chord_phys} m)")
    if dx_phys is None:
        dx_phys, chord_phys = 1.0, chord_lu

    m = ice_shape_metrics(airfoil, solid, dx_phys, chord_phys, chord_lu, stag)

    print("=" * 60)
    print(f"  ICE SHAPE METRICS — {npz}")
    print("=" * 60)
    print(f"  chord            : {chord_lu:.0f} cells")
    print(f"  ice cells        : {m['n_ice_cells']} ({m['ice_area_pct_chord2']:.2f} % chord^2)")
    if "upper_horn_pct_chord" in m:
        print(
            f"  upper horn       : {m['upper_horn_cells']} cells "
            f"({m['upper_horn_pct_chord']:.2f} % chord)"
        )
    if "lower_horn_pct_chord" in m:
        print(
            f"  lower horn       : {m['lower_horn_cells']} cells "
            f"({m['lower_horn_pct_chord']:.2f} % chord)"
        )
    if "horn_symmetry_pct" in m:
        print(f"  horn symmetry    : {m['horn_symmetry_pct']:+.1f} %")
    print(
        f"  ice x extent     : LE + {m.get('ice_x_offset_min', 0):.3f} .. "
        f"{m.get('ice_x_offset_max', 0):.3f} chord"
    )
    print(f"  max layer depth  : {m.get('ice_max_layer', 0)} cells")
    print("=" * 60)
    if js.exists():
        with open(js) as fh:
            blob = json.load(fh)
        a = blob["audit"]
        print(f"  mass audit closure error: {a['closure_error'] * 100:.4f} %")


if __name__ == "__main__":
    main()
