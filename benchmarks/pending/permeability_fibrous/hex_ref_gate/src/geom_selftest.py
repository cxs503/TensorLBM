#!/usr/bin/env python3
"""Geometry-only self-test for the hex lattices (NO flow solve, no K/f values).

1. dist2_to_cylinders vs brute-force images m,n in [-3,3] at random points.
2. Cell-centre solid-fraction (mask only) -> vf trend for hexA rungs.
3. Nearest-neighbour distance and gap/r of the designed lattices.
Run locally: python geom_selftest.py
"""

import math
import sys

import numpy as np

sys.path.insert(0, "/root/hexref_work")
from hexref_fd import LATTICES, dist2_to_cylinders


def brute_d2(X, Y, cylinders, a1, a2, rng=3):
    best = np.full(X.shape, np.inf)
    for cx, cy in cylinders:
        for m in range(-rng, rng + 1):
            for n in range(-rng, rng + 1):
                dx = X - (cx + m * a1[0] + n * a2[0])
                dy = Y - (cy + m * a1[1] + n * a2[1])
                best = np.minimum(best, dx * dx + dy * dy)
    return best


def main():
    rng = np.random.default_rng(42)
    ok = True

    # --- test 1: distance function vs brute force
    for name, (aspect, cyl, a1, a2) in LATTICES.items():
        X = rng.uniform(0, 1, 400)
        Y = rng.uniform(0, aspect, 400)
        fast = dist2_to_cylinders(X, Y, cyl, a1, a2)
        slow = brute_d2(X, Y, cyl, a1, a2, rng=3)
        err = np.max(np.abs(fast - slow) / np.maximum(slow, 1e-300))
        status = "OK" if err < 1e-12 else "FAIL"
        ok &= err < 1e-12
        print(f"[{status}] dist2 {name}: max rel diff vs brute 7x7 = {err:.2e}")

    # --- test 2: mask-only solid fraction trend for hexA rungs
    print("\nhexA cell-centre solid fraction (mask only, no solve):")
    for vf in (0.70, 0.75):
        aspect, cyl, a1, a2 = LATTICES["hexA"]
        n_cyl = len(cyl)
        R2 = vf * aspect / (n_cyl * math.pi)
        for m in (2, 4, 8, 14):
            nx, ny = 56 * m, 97 * m
            h = 1.0 / nx
            ii = np.arange(nx)
            jj = np.arange(ny)
            Xc, Yc = np.meshgrid((ii + 0.5) * h, (jj + 0.5) * h, indexing="ij")
            d2c = dist2_to_cylinders(Xc, Yc, cyl, a1, a2)
            phi = float(np.mean(d2c <= R2))
            print(
                f"  vf={vf:.2f} m={m:2d} ({nx}x{ny}): phi_actual={phi:.6f} "
                f"dev={100 * (phi / vf - 1):+.3f}%"
            )
    # hexB slope ladder m=20
    aspect, cyl, a1, a2 = LATTICES["hexB"]
    for vf in (0.70, 0.75):
        R2 = vf * aspect / (2 * math.pi)
        for m in (20, 28):
            nx, ny = 15 * m, 26 * m
            h = 1.0 / nx
            Xc, Yc = np.meshgrid(
                (np.arange(nx) + 0.5) * h, (np.arange(ny) + 0.5) * h, indexing="ij"
            )
            d2c = dist2_to_cylinders(Xc, Yc, cyl, a1, a2)
            phi = float(np.mean(d2c <= R2))
            print(
                f"  hexB vf={vf:.2f} m={m} ({nx}x{ny}): phi={phi:.6f} "
                f"dev={100 * (phi / vf - 1):+.3f}%"
            )

    # --- test 3: nearest-neighbour distance & gap/r
    print("\nlattice metric check:")
    for name in ("hexA", "hexB"):
        aspect, cyl, a1, a2 = LATTICES[name]
        d_nn = math.hypot(a2[0], a2[1])  # cyl2-cyl1 vector = a2
        for vf in (0.70, 0.75):
            R = math.sqrt(vf * aspect / (2 * math.pi))
            print(f"  {name} vf={vf:.2f}: d_nn={d_nn:.6f} R={R:.6f} gap/r={(d_nn - 2 * R) / R:.4f}")

    print("\nALL OK" if ok else "\nFAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
