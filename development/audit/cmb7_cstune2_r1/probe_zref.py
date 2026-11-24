"""CMB7-CSTUNE2-R1 mechanism probe (fast): does TL_FS_HYDRO_Z_REF engage?

Replicates the CMB7 arm setup and runs N steps for a given z_ref (env
TL_FS_HYDRO_Z_REF).  Prints a compact JSON line: final f-field sha1, final
depth, and the analytic correction magnitude at the water surface and at the
would-be cavity bottom.  Compare fhash across z_ref arms: identical => the
correction is not perturbing the trajectory.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from benchmarks.water_entry.single_drop.run import (
    equivalent_radius,
    load_case,
    open_depth,
    scales,
)
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.free_surface_lbm import (
    GAS,
    INTERFACE,
    LIQUID,
    SOLID,
    free_surface_step,
    init_flags_from_fill,
    init_mass_from_fill,
)
from tensorlbm.free_surface_topology_transaction import TopologyTransactionError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    n = 8
    case = load_case(False)
    sc = scales(case, n, 0.06)
    nz, side = round((3.0 + 2) * n) + 2, round(4.0 * n)
    level = round(3.0 * n) + 0.5
    aspect = case["aspect_ratio"]
    gap = 0.18 * n
    mid = (side - 1) / 2
    z, y, x = (
        v.to(torch.float64)
        for v in torch.meshgrid(torch.arange(nz), torch.arange(side), torch.arange(side), indexing="ij")
    )
    center = level + gap + 0.5 * n * aspect ** (2 / 3)
    drop = torch.zeros_like(z)
    for dz in (-0.375, -0.125, 0.125, 0.375):
        for dy in (-0.375, -0.125, 0.125, 0.375):
            for dx in (-0.375, -0.125, 0.125, 0.375):
                drop += (equivalent_radius(z + dz - center, y + dy - mid, x + dx - mid, aspect) < 0.5 * n) / 64
    fill = torch.maximum(drop, (level - z + 0.5).clamp(0, 1))
    solid = torch.zeros_like(fill, dtype=torch.bool)
    solid[[0, -1]] = True
    fill[solid] = 0
    flags = init_flags_from_fill(fill, solid)
    active = (flags != GAS) & (flags != SOLID)
    moving = z > level + 0.5 * gap
    release_lattice = 1.53 * sc["dt_s"] / sc["dx_m"]
    velocity = -release_lattice * (moving & active)
    rho = 1 + 3 * (sc["g"] * (level - z).clamp_min(0) + (4 * sc["sigma"] / n) * moving)
    f = equilibrium3d(rho * active, torch.zeros_like(z), torch.zeros_like(z), velocity)
    mass = init_mass_from_fill(fill, flags) * rho

    zref_env = os.environ.get("TL_FS_HYDRO_Z_REF")
    zref = float(zref_env) if zref_env not in (None, "") else None
    gz = -sc["g"]
    cs2 = 1.0 / 3.0

    reason = "step_limit"
    for step in range(a.steps):
        try:
            f, fill, flags, mass, _ = free_surface_step(
                f, fill, flags, solid, mass=mass,
                tau=0.5003877, gz=gz, surface_tension=sc["sigma"], rho_gas=1.0,
                collision="rr", capacity_redistribution=True, simple_mass_exchange=True,
                u_limit=0.15, C_s=0.075,
            )
        except TopologyTransactionError as e:
            reason = f"topo_{step+1}:{e}"
            break

    dens, ux, uy, uz = macroscopic3d(f)
    active = (flags != GAS) & (flags != SOLID)
    mach = float((3 * (ux**2 + uy**2 + uz**2))[active].max().sqrt())
    field = fill[1:-1].numpy()
    depth = open_depth(field, level - 1, 1.5 * n) / n
    fhash = hashlib.sha1(f.detach().contiguous().view(-1).cpu().numpy().tobytes()).hexdigest()[:16]

    # analytic correction at key z levels: corr(z) = rho_l*gz*(z-zref)/cs2
    def corr(zz):
        return 1.0 * gz * (zz - zref) / cs2 if zref is not None else 0.0
    out = {
        "tag": a.tag, "steps": a.steps, "stop": reason,
        "z_ref": zref, "level": level, "gz": gz,
        "corr_at_surface": corr(level),
        "corr_at_z12": corr(12.0),
        "corr_at_z14": corr(14.0),
        "mach_max": mach, "depth_over_D": depth, "fhash": fhash,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
