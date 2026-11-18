"""Release a rectangular column occupying the full channel width with D3Q19."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.free_surface_lbm import (
    GAS,
    INTERFACE,
    LIQUID,
    SOLID,
    free_surface_step,
    init_flags_from_fill,
)
from tensorlbm.free_surface_topology_transaction import TopologyTransactionError

WIDTH_M = 2.25 * 0.0254
GRAVITY = 9.81
NU = 1.0e-6
DENSITY = 998.0
SURFACE_TENSION = 0.072
END_T = 11.5
MACH_LIMIT = 0.30


def outer_crossing(profile):
    """Outermost half-fill crossing measured from the halfway wall, in cells."""
    wet = np.flatnonzero(profile >= 0.5)
    if not len(wet):
        return 0.0
    i = wet[-1]
    if i + 1 >= len(profile):
        raise ValueError("wet surface reached outer wall")
    return i + 0.5 + (profile[i] - 0.5) / (profile[i] - profile[i + 1])


def measure(fill, n):
    # Axis order (span, vertical, streamwise), with boundary cells removed.
    # Ignore isolated spray. Project the reservoir-connected body as in the
    # overhead and side views in the original experiment.
    wet = fill >= 0.5
    seed = np.zeros_like(wet)
    seed[:, 0, 0] = wet[:, 0, 0]
    if not seed.any():
        raise ValueError("reservoir no longer wets upstream base")
    body = ndimage.binary_propagation(seed, mask=wet)
    adjacent = ndimage.binary_dilation(body)
    interface = np.where(body | (adjacent & ~wet), fill, 0.0)
    front = outer_crossing(interface.max(axis=(0, 1))) / n
    crest = outer_crossing(interface[:, :, 0].max(axis=0)) / (2 * n)
    return front, crest


def simulate(n, c0, span, output, collision="bgk"):
    dx = WIDTH_M / n
    dt = c0 * dx / math.sqrt(2 * GRAVITY * WIDTH_M)
    gravity = GRAVITY * dt**2 / dx
    tau = 0.5 + 3 * NU * dt / dx**2
    sigma = SURFACE_TENSION * dt**2 / (DENSITY * dx**3)
    channel = span == "channel"
    shape = (n + 2 if channel else 3, math.ceil(2.6 * n) + 2, 18 * n + 2)
    fill = torch.zeros(shape, dtype=torch.float64)
    solid = torch.zeros_like(fill, dtype=torch.bool)
    solid[:, (0, -1), :] = True
    solid[:, :, (0, -1)] = True
    if channel:
        solid[(0, -1), :, :] = True
    fill[:, 1 : 2 * n + 1, 1 : n + 1] = 1
    fill[solid] = 0
    flags = init_flags_from_fill(fill, solid)
    y = torch.arange(shape[1], dtype=fill.dtype)[None, :, None] - 0.5
    rho = (1 + 3 * gravity * (2 * n - y).clamp_min(0)) * ((flags != GAS) & (flags != SOLID))
    zero = torch.zeros_like(fill)
    f = equilibrium3d(rho, zero, zero, zero)
    mass = fill * rho
    initial_mass, initial_volume = float(mass.sum()), float(fill.sum())
    stop = "window_complete"
    output.mkdir(parents=True, exist_ok=True)
    t_scale = c0 / n
    with (output / "history.csv").open("w", newline="") as fp:
        log = csv.writer(fp, lineterminator="\n")
        log.writerow(["step", "T", "Z", "H", "mass_drift", "volume_drift", "mach", "rho_min"])
        for step in range(math.ceil(END_T / t_scale) + 1):
            dens, ux, uy, uz = macroscopic3d(f)
            active = (flags == LIQUID) | (flags == INTERFACE)
            mach = float(torch.sqrt(3 * (ux**2 + uy**2 + uz**2))[active].max())
            rho_min = float(dens[active].min())
            represented = float(dens[flags == LIQUID].sum() + mass[flags == INTERFACE].sum())
            drift = represented / initial_mass - 1
            geom = float(fill.sum()) / initial_volume - 1
            valid = (
                all(map(math.isfinite, (mach, rho_min, drift, geom)))
                and mach <= MACH_LIMIT
                and rho_min > 0
            )
            front, crest = None, None
            if valid:
                slab = fill[1:-1] if channel else fill
                try:
                    front, crest = measure(slab[:, 1:-1, 1:-1].numpy(), n)
                except ValueError as err:
                    stop = str(err)
                    valid = False
            else:
                stop = "mach_density_or_finiteness_limit"
            log.writerow([step, step * t_scale, front, crest, drift, geom, mach, rho_min])
            if step % 200 == 0:
                fp.flush()
            if not valid or step * t_scale >= END_T:
                break
            try:
                f, fill, flags, mass, _ = free_surface_step(
                    f,
                    fill,
                    flags,
                    solid,
                    mass=mass,
                    tau=tau,
                    gy=-gravity,
                    surface_tension=sigma,
                    rho_gas=1.0,
                    collision=collision,
                    capacity_redistribution=True,
                    simple_mass_exchange=True,
                )
            except TopologyTransactionError as err:
                stop = str(err)
                break
    meta = dict(
        n=n,
        c0=c0,
        span=span,
        collision=collision,
        shape=shape,
        dt_s=dt,
        dx_m=dx,
        relaxation_time=tau,
        lattice_gravity=gravity,
        lattice_surface_tension=sigma,
        stop=stop,
        last_step=step,
        solver="tensorlbm.free_surface_lbm.free_surface_step",
        torch_version=str(torch.__version__),
    )
    (output / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(meta, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--celerity", type=float, default=0.06)
    parser.add_argument("--span", choices=("channel", "periodic"), default="channel")
    parser.add_argument("--collision", choices=("bgk", "mrt"), default="bgk")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.cells < 4 or not 0 < args.celerity < 0.15:
        parser.error("invalid resolution or lattice celerity")
    torch.set_num_threads(1)
    simulate(args.cells, args.celerity, args.span, args.out, args.collision)
