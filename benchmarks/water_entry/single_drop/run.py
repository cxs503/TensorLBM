"""Run the D3Q19 water/air impact with fixed SI parameters and report failed steps."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from tensorlbm.cac_lbm import (
    CACSim,
    _macro_fp64,
    f_equilibrium,
    g_equilibrium,
    rho_of_phi,
)

CASE = Path(__file__).with_name("protocol.json")


def load_case(pilot=False):
    case = json.loads(CASE.read_text())
    if pilot:
        case.update(case["pilot_materials"], aspect_ratio=1.0)
    return case


def equivalent_radius(dz, dy, dx, aspect):
    """Volume-preserving spheroid in coordinates of its equivalent sphere."""
    return ((dz / aspect ** (2 / 3)) ** 2 + (dy**2 + dx**2) * aspect ** (2 / 3)).sqrt()


def scales(case, cells, speed):
    dx = case["diameter_m"] / cells
    dt = speed * dx / case["velocity_m_s"]
    rho = case["water_density_kg_m3"]
    return {
        "dx_m": dx,
        "dt_s": dt,
        "rho_g": case["air_density_kg_m3"] / rho,
        "mu_l": case["water_dynamic_viscosity_pa_s"] * dt / (rho * dx**2),
        "mu_g": case["air_dynamic_viscosity_pa_s"] * dt / (rho * dx**2),
        "sigma": case["surface_tension_n_m"] * dt**2 / (rho * dx**3),
        "g": case["gravity_m_s2"] * dt**2 / dx,
    }


def open_depth(phi, level, radius):
    """Depth below initial surface of deepest half-fill crossing open to headspace."""
    gas = phi < 0.5
    seeds = np.zeros_like(gas)
    seeds[-1] = gas[-1]
    atmosphere = ndimage.binary_propagation(seeds, mask=gas)
    y, x = np.indices(phi.shape[1:])
    roi = (y - (phi.shape[1] - 1) / 2) ** 2 + (x - (phi.shape[2] - 1) / 2) ** 2 <= radius**2
    k, j, i = np.where(atmosphere[1:] & ~gas[:-1] & roi[None])
    if not len(k):
        raise ValueError("no gas-to-liquid floor within measurement area")
    low, high = phi[k, j, i], phi[k + 1, j, i]
    floor = k + (low - 0.5) / (low - high)
    return float(max(0.0, level - floor.min()))


class DropTracker:
    """Follow the detached bulk by overlap until it actually joins the pool."""

    def __init__(self):
        self.previous = None
        self.contact_step = None

    def update(self, phi, step, level):
        labels, _ = ndimage.label(phi >= 0.5)
        pool = labels[0, phi.shape[1] // 2, phi.shape[2] // 2]
        if not pool:
            raise ValueError("pool lost at base")
        if self.contact_step is not None:
            return None
        if self.previous is None:
            counts = np.bincount(labels.ravel())
            counts[[0, pool]] = 0
            if not counts.max():
                raise ValueError("no separated drop at initialization")
        else:
            counts = np.bincount(labels[self.previous], minlength=pool + 1)
            counts[0] = 0
            if counts.max() <= self.previous.sum() / 2:
                raise ValueError("drop identity lost between consecutive steps")
            if counts.argmax() == pool:
                self.contact_step = step
                return None
        self.previous = labels == counts.argmax()
        if np.nonzero(self.previous)[0].mean() <= level:
            raise ValueError("detached drop below initial level")
        return self.previous


def run(
    n,
    speed,
    steps,
    width,
    gap,
    gas_flow,
    odd_rate=None,
    at_rest=False,
    uniform_drop=False,
    stencil_wall=False,
    pilot=False,
    span=4.0,
    pool_depth=3.0,
):
    case = load_case(pilot)
    sc = scales(case, n, speed)
    nz, side = round((pool_depth + 2) * n) + 2, round(span * n)
    shape = (nz, side, side)
    z0 = round(pool_depth * n) + 0.5
    aspect = case["aspect_ratio"]
    cz = z0 + (gap + 0.5 * aspect ** (2 / 3)) * n
    cy = (side - 1) / 2
    dtype = torch.float64
    z = torch.arange(nz, dtype=dtype)[:, None, None]
    y = torch.arange(side, dtype=dtype)[None, :, None]
    x = torch.arange(side, dtype=dtype)[None, None, :]
    distance = equivalent_radius(z - cz, y - cy, x - cy, aspect)
    drop = 0.5 + 0.5 * torch.tanh(2 * (n / 2 - distance) / width)
    pool = 0.5 + 0.5 * torch.tanh(2 * (z0 - z) / width)
    phi = torch.maximum(drop, pool)
    wall = torch.zeros(shape, dtype=torch.bool)
    wall[[0, -1]] = True
    sim = CACSim(
        shape,
        rho_l=1.0,
        rho_g=sc["rho_g"],
        mu_l=sc["mu_l"],
        mu_g=sc["mu_g"],
        sigma=sc["sigma"],
        width=width,
        gravity=(0.0, 0.0, -sc["g"]),
        gravity_mode="rho",
        hydro_odd_rate=odd_rate,
        p0=0.0,
        axes="wpp",
        wall_mask=wall,
        wall_phi_policy="stencil_neumann" if stencil_wall else "neumann",
        dtype=dtype,
    )
    sim.initialize(phi)
    # Ballistic correction for the specified gap, measured velocity must still be checked.
    u0 = math.sqrt(speed**2 - 2 * sc["g"] * gap * n)
    if at_rest:
        u0 = 0.0
    zero = torch.zeros_like(phi)
    velocity_weight = drop
    if uniform_drop:
        # Prescribe the impact velocity throughout the liquid side of the
        # half-fill surface. Smooth only its extension into the exterior air.
        t = (2 * drop).clamp(max=1.0)
        velocity_weight = t * t * (3 - 2 * t)
    u = [zero, zero, -u0 * velocity_weight]
    if gas_flow:
        # Potential flow of the exterior of a translating sphere, decaying as r^-3.
        r = ((x - cy) ** 2 + (y - cy) ** 2 + (z - cz) ** 2).sqrt().clamp_min(n / 2)
        scale = -0.5 * u0 * (n / (2 * r)) ** 3 * (1 - velocity_weight) * (1 - pool)
        u = [
            scale * 3 * (x - cy) * (z - cz) / r**2,
            scale * 3 * (y - cy) * (z - cz) / r**2,
            u[2] + scale * (3 * ((z - cz) / r) ** 2 - 1),
        ]
    u = [v.masked_fill(wall, 0.0) for v in u]
    # Hydrostatic flat pool plus spherical Young-Laplace pressure in the drop.
    rho = rho_of_phi(phi, 1.0, sc["rho_g"])
    planar_density = rho_of_phi(pool, 1.0, sc["rho_g"])
    increments = (planar_density[1:] + planar_density[:-1]) * 0.5 * sc["g"]
    base = torch.cat((increments.flip((0,)).cumsum(0).flip((0,)), torch.zeros_like(pool[-1:])))
    sim.f = f_equilibrium(phi, u, sim.wv, sim.cf)
    sim.g = g_equilibrium(base + 4 * sc["sigma"] / n * drop, rho, u, sim.wv, sim.cf, sim.w0)
    sim.phi_u_prev = tuple(phi * v for v in u)
    stop_ms = 24
    limit = (
        steps
        if steps is not None
        else math.ceil((stop_ms * 1e-3) / sc["dt_s"] + 2 * gap * n / speed)
    )
    rows = []
    tracker = DropTracker()
    impact_speed = None
    last_speed = None
    drift_limit = case["gates"]["maximum_liquid_volume_relative_drift"]
    first_mass = None
    reason = "step_limit"
    for step in range(limit + 1):
        field, density, _, _, _, velocity, _, _ = _macro_fp64(sim)
        liquid = field[1:-1]
        mass = float(liquid.sum())
        if first_mass is None:
            first_mass = mass
        mach_field = (3 * sum(v[1:-1] ** 2 for v in velocity)).sqrt()
        worst = np.unravel_index(int(mach_field.argmax()), mach_field.shape)
        mach = float(mach_field[worst])
        drift = abs(mass / first_mass - 1)
        row = {
            "step": step,
            "time_ms": 1000 * step * sc["dt_s"],
            "mach_max": mach,
            "rho_min": float(density[1:-1].min()),
            "min_phi": float(liquid.min()),
            "volume_drift": drift,
            "phi_at_mach_max": float(liquid[worst]),
            "mach_liquid_max": float(mach_field[liquid >= 0.5].max()),
        }
        rows.append(row)
        if (
            not all(math.isfinite(v) for v in row.values())
            or not torch.isfinite(sim.f).all()
            or not torch.isfinite(sim.g).all()
        ):
            reason = "nonfinite"
            break
        if (
            row["rho_min"] <= 0
            or mach > case["gates"]["maximum_lattice_mach"]
            or drift > drift_limit
        ):
            reason = (
                "negative_density"
                if row["rho_min"] <= 0
                else "mach_limit"
                if mach > case["gates"]["maximum_lattice_mach"]
                else "volume_drift"
            )
            break
        phase = liquid.numpy()
        try:
            detached = tracker.update(phase, step, z0 - 1)
            row["depth_over_D"] = open_depth(phase, z0 - 1, 1.5 * n) / n
        except ValueError as error:
            reason = str(error)
            break
        if detached is not None:
            last_speed = float(-velocity[2][1:-1][detached].mean()) * sc["dx_m"] / sc["dt_s"]
            row["detached_downward_speed_m_s"] = last_speed
        elif tracker.contact_step == step:
            impact_speed = last_speed
        if (
            tracker.contact_step is not None
            and (step - tracker.contact_step) * sc["dt_s"] >= stop_ms * 1e-3
        ):
            reason = "window_complete"
            break
        if step and step % 100 == 0:
            print(
                f"step {step}, time {row['time_ms']:.3f} ms, contact={tracker.contact_step}",
                flush=True,
            )
        if step < limit:
            sim.step()
    return {
        "solver": "CACSim/D3Q19",
        "case_id": "pilot" if pilot else "santini_case3",
        "span_over_D": span,
        "pool_depth_over_D": pool_depth,
        "cells_per_D": n,
        "lattice_speed": speed,
        "shape_zyx": shape,
        "interface_width_cells": width,
        "gap_over_D": gap,
        "exterior_potential_flow": gas_flow,
        "hydro_odd_rate": odd_rate,
        "scales": sc,
        "stop_reason": reason,
        "contact_step": tracker.contact_step,
        "impact_speed_m_s": impact_speed,
        "at_rest_control": at_rest,
        "uniform_drop": uniform_drop,
        "stencil_wall": stencil_wall,
        "history": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--span", type=float, default=4.0)
    parser.add_argument("--pool-depth", type=float, default=3.0)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--speed", type=float, default=0.04)
    parser.add_argument("--width", type=float, default=4.0)
    parser.add_argument("--gap", type=float, default=0.4)
    parser.add_argument("--gas-flow", action="store_true")
    parser.add_argument("--stencil-wall", action="store_true")
    parser.add_argument("--uniform-drop", action="store_true")
    parser.add_argument(
        "--at-rest",
        action="store_true",
        help="start with the drop at rest for a diagnostic control",
    )
    parser.add_argument("--odd-rate", type=float)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.span < 3
        or args.pool_depth < 2
        or args.cells < 8
        or args.width <= 0
        or not 0.1 <= args.gap <= 0.6
        or not 0 < args.speed < 0.15
        or (args.steps is not None and args.steps < 0)
    ):
        parser.error("invalid resolution, width, speed, gap or step limit")
    torch.set_num_threads(1)
    result = run(
        args.cells,
        args.speed,
        args.steps,
        args.width,
        args.gap,
        args.gas_flow,
        args.odd_rate,
        args.at_rest,
        args.uniform_drop,
        args.stencil_wall,
        args.pilot,
        args.span,
        args.pool_depth,
    )
    for row in result["history"]:
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                row[key] = None
    args.out.parent.mkdir(exist_ok=True, parents=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(result["stop_reason"], result["history"][-1])
    return 0 if result["stop_reason"] == "window_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
