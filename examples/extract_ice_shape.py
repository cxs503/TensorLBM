#!/usr/bin/env python
"""Extract ice shape metrics from aircraft icing simulation.
Computes: ice area, horn height, horn spacing (vs NASA glaze).
"""
import sys, os
import numpy as np
import torch

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark_aircraft_icing import run_aircraft_icing


def extract_ice_shape(r):
    """Extract ice shape metrics from result."""
    ice = r["ice_mask"]      # (nz, ny, nx)
    solid = r["original_solid"]  # airfoil only (no ice)
    nx, ny, nz = r["nx"], r["ny"], r["nz"]
    chord = r["chord"]
    cx0 = int(nx * 0.3)  # airfoil leading edge

    # 3D ice projected to 2D (any z has ice/solid)
    ice2d = ice.any(dim=0).cpu().numpy()
    solid2d = solid.any(dim=0).cpu().numpy()
    ice_only = ice2d & ~solid2d

    ice_area = int(ice_only.sum())

    # Ice front (min x of ice per y row)
    x_arr = np.arange(nx)
    ice_x = np.where(ice_only, x_arr, nx)
    ice_front = ice_x.min(axis=1)  # (ny,)

    # Horn height (upstream extension above airfoil leading edge)
    ice_height = np.maximum(cx0 - ice_front, 0)  # (ny,)

    # Find horns (peaks of ice_height)
    horn_y = np.where(ice_height > 0)[0]
    mid_y = ny // 2

    metrics = {"ice_area": ice_area, "ice_area_pct": ice_area / chord**2 * 100,
               "chord": chord, "cx0": cx0}

    uy = ly = None
    if len(horn_y) > 0:
        upper = horn_y[horn_y < mid_y]
        lower = horn_y[horn_y > mid_y]
        if len(upper) > 0:
            uh = int(ice_height[upper].max())
            uy = int(upper[ice_height[upper].argmax()])
            metrics["upper_horn_h"] = uh
            metrics["upper_horn_pct"] = uh / chord * 100
            metrics["upper_horn_y"] = uy
        if len(lower) > 0:
            lh = int(ice_height[lower].max())
            ly = int(lower[ice_height[lower].argmax()])
            metrics["lower_horn_h"] = lh
            metrics["lower_horn_pct"] = lh / chord * 100
            metrics["lower_horn_y"] = ly
        if uy is not None and ly is not None:
            sp = ly - uy
            metrics["horn_spacing"] = sp
            metrics["horn_spacing_pct"] = sp / chord * 100

    return metrics


def main():
    r = run_aircraft_icing(nx=200, ny=100, nz=32, tau=0.52, steps=5000,
                           device="sdaa:0", log_every=9999)
    m = extract_ice_shape(r)
    chord = m["chord"]
    print(f"\n{'='*60}")
    print(f"  ICE SHAPE METRICS (NACA 0012, 5000 steps)")
    print(f"{'='*60}")
    print(f"  Chord: {chord} cells")
    print(f"  Ice area: {m['ice_area']} cells ({m['ice_area_pct']:.1f}% chord²)")
    if "upper_horn_h" in m:
        print(f"  Upper horn: h={m['upper_horn_h']} ({m['upper_horn_pct']:.1f}% chord), y={m['upper_horn_y']}")
    if "lower_horn_h" in m:
        print(f"  Lower horn: h={m['lower_horn_h']} ({m['lower_horn_pct']:.1f}% chord), y={m['lower_horn_y']}")
    if "horn_spacing" in m:
        print(f"  Horn spacing: {m['horn_spacing']} ({m['horn_spacing_pct']:.1f}% chord)")
    print(f"\n  NASA glaze reference (NACA 0012, c=0.5334m):")
    print(f"    Horn height: ~5-10% chord")
    print(f"    Horn spacing: ~10-20% chord")
    print(f"    Ice area: limited (front region)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
