"""Diagnostic: does the moving-wall momentum exchange produce lift on a single
inclined plate in the same channel/BC setup used for the E779A face?

A flat plate of chord ``c`` and span equal to the channel width is placed at
incidence ``alpha`` in the uniform inflow.  If the load model is sound, the
measured lift coefficient must approach the thin-aerofoil value ``2 pi alpha``
(±25 % at these lattice Reynolds numbers).  This isolates "load model" from
"propeller resolution" in the P129W3-N04 error attribution.
"""

from __future__ import annotations

import json
import math
import sys
import time

import torch

torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 2)

import tensorlbm.insean_e779a as m
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.solver3d import collide_mrt3d_fused, stream3d


def plate_mask(nz, ny, nx, cx, cy, cz, chord, alpha_deg, thickness=1.0):
    """One flat plate: mid-chord at (cx, cy, cz), inclined by alpha about z."""
    mask = torch.zeros((nz, ny, nx), dtype=torch.bool)
    a = math.radians(alpha_deg)
    ca, sa = math.cos(a), math.sin(a)
    z0, z1 = 1, nz - 1                       # span = channel width (no slip walls)
    rng = int(0.5 * chord + 2)
    for i in range(int(cx) - rng, int(cx) + rng + 1):
        for j in range(int(cy) - rng, int(cy) + rng + 1):
            dx, dy = i - cx, j - cy
            xi = dx * ca + dy * sa           # chordwise
            eta = -dx * sa + dy * ca         # normal
            if abs(xi) <= 0.5 * chord and abs(eta) <= 0.5 * thickness:
                mask[z0:z1, j, i] = True
    return mask


def main() -> int:
    nx, ny, nz = 96, 40, 16
    cx, cy, cz = 0.4 * nx, 0.5 * (ny - 1), 0.5 * (nz - 1)
    chord = 20.0
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
    u_in = 0.05
    tau = 0.5008
    steps = 1400
    rows = []
    mask = plate_mask(nz, ny, nx, cx, cy, cz, chord, alpha)
    wall = m._wall_mask(nz, ny, nx, torch.device("cpu"))
    rho0 = torch.ones((nz, ny, nx))
    f = equilibrium3d(rho0, torch.full_like(rho0, u_in), torch.zeros_like(rho0), torch.zeros_like(rho0))
    zero = torch.zeros((nz, ny, nx))
    fx = fy = 0.0
    n_acc = 0
    t0 = time.perf_counter()
    for step in range(steps):
        f = collide_mrt3d_fused(f, tau=tau)
        f = stream3d(f)
        m._open_water_bc(f, u_in, wall)
        imp, idx, _ = m._masked_moving_wall(f, mask, zero, zero, zero)
        if step >= steps - 600:
            fx += -float(imp[:, 0].sum())
            fy += -float(imp[:, 1].sum())
            n_acc += 1
    fx /= n_acc
    fy /= n_acc
    area = chord * (nz - 2)
    q = 0.5 * u_in * u_in * area
    cl = fy / q
    cd = fx / q
    nu = (tau - 0.5) / 3.0
    re_c = u_in * chord / nu
    out = {
        "chord": chord,
        "span": nz - 2,
        "alpha_deg": alpha,
        "u_in": u_in,
        "re_c": re_c,
        "cl": cl,
        "cd": cd,
        "thin_aerofoil_cl": 2 * math.pi * math.radians(alpha),
        "cl_error_pct": (
            (cl - 2 * math.pi * math.radians(alpha)) / (2 * math.pi * math.radians(alpha)) * 100
            if abs(alpha) > 1e-9 else float("nan")
        ),
        "wall_time_s": time.perf_counter() - t0,
        "solid_cells": int(mask.sum()),
    }
    for r in rows:
        print(r)
    print(json.dumps(out, indent=1))
    tag = f"{alpha:+.1f}".replace(".", "p")
    with open(f"artifacts/p129w3_n04_e779a/diagnostic_plate_{tag}.json", "w") as fh:
        json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())