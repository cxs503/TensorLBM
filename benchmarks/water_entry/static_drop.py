"""Measure 2-D Laplace pressure with the same closed-tank FE/CH solver as water entry.

Run with PYTHONPATH=src python benchmarks/water_entry/static_drop.py --output results.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from tensorlbm.boundary.tank import TankStreamingPlan
from tensorlbm.d2q9 import C, equilibrium, macroscopic
from tensorlbm.multiphase import free_energy_step, init_free_energy_g
from tensorlbm.phasefield.operators import central_gradient_2d, laplacian_2d


@torch.no_grad()
def measure(radius, steps, size=88):
    # Same material parameters as the entry benchmark, with gravity switched off.
    A, kappa, xi, gamma = 0.004, 0.018, 3.0, 0.5
    axis = torch.arange(size, dtype=torch.float64) - (size - 1) / 2
    r = torch.hypot(axis[:, None], axis[None, :])
    phi0 = torch.tanh((radius - r) / xi)
    wall = torch.zeros_like(r, dtype=torch.bool)
    wall[-1] = True
    wall[:, [0, -1]] = True
    bc = dict(boundary="no_flux", solid_mask=wall)

    def chemical(phi):
        return A * (phi**3 - phi) - kappa * laplacian_2d(phi, **bc)

    f = equilibrium(torch.ones_like(r), torch.zeros_like(r), torch.zeros_like(r))
    g = init_free_energy_g(phi0, geq_convention="lee_lin", mu=chemical(phi0), Gamma=gamma)
    tank = TankStreamingPlan(wall)
    inner = r < radius - 2.5 * xi
    outer = (r > radius + 2.5 * xi) & (r < size / 2 - xi)
    sigma = (2 / 3) * math.sqrt(2 * kappa * A)
    c2 = C.square().sum(1)[:, None, None]
    rows = []
    healthy = True
    for step in range(steps + 1):
        if step % 250 == 0 or step == steps:
            phi = g.sum(0)
            rho, ux, uy = macroscopic(f)
            healthy = bool(torch.isfinite(f).all() and torch.isfinite(g).all() and (rho > 0).all())
            if not healthy:
                break
            mu = chemical(phi)
            # Half trace of the kinetic pressure plus the 2D Korteweg stress.
            # The gradient dyad and isotropic gradient energy cancel in this trace.
            bulk = A * (-phi.square() / 2 + phi**4 / 4)
            pressure = ((f * c2).sum(0) - rho * (ux.square() + uy.square())) / 2 + phi * mu - bulk
            jump = float(pressure[inner].mean() - pressure[outer].mean())
            # Area-equivalent radius from the evolved zero-level liquid region.
            R = math.sqrt(int(((phi > 0) & ~wall).sum()) / math.pi)
            if R == 0:
                healthy = False
                break
            mx, my = central_gradient_2d(mu, **bc)
            u = torch.hypot(ux - 0.5 * phi * mx / rho, uy - 0.5 * phi * my / rho)
            rows.append(
                dict(
                    step=step,
                    radius=R,
                    pressure_jump=jump,
                    analytic_jump=sigma / R,
                    relative_error=abs(jump * R / sigma - 1),
                    max_speed=float(u[~wall].max()),
                )
            )
        if step < steps:
            f, g = free_energy_step(
                f,
                g,
                tau_f=0.8,
                tau_g=1.0,
                A=A,
                B=A,
                kappa=kappa,
                Gamma=gamma,
                geq_convention="lee_lin",
                **bc,
            )
            f, g = tank.stream(f, g)
    # Require agreement and stationarity in the last 25% of the run.
    tail = [v for v in rows if v["step"] >= 0.75 * steps]
    settled = bool(
        len(tail) >= 2
        and max(v["pressure_jump"] for v in tail) - min(v["pressure_jump"] for v in tail)
        < 0.05 * tail[-1]["analytic_jump"]
    )
    passed = (
        healthy
        and settled
        and all(v["relative_error"] < 0.15 and v["max_speed"] < 1e-3 for v in tail)
    )
    return dict(
        initial_radius=radius,
        size=size,
        steps_requested=steps,
        healthy=healthy,
        settled=settled,
        passed=passed,
        sigma_input=sigma,
        history=rows,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.steps < 1000:
        ap.error("--steps must be >=1000 for a stationarity window")
    torch.set_num_threads(1)
    cases = [measure(r, args.steps) for r in (10, 14, 18)]
    passed = all(c["passed"] for c in cases)
    slope = None
    intercept = None
    if all(c["healthy"] and c["history"] for c in cases):
        last = [c["history"][-1] for c in cases]
        x = [1.0 / r["radius"] for r in last]
        slope, intercept = map(float, np.polyfit(x, [r["pressure_jump"] for r in last], 1))
        passed = passed and abs(slope / cases[0]["sigma_input"] - 1) < 0.15
    report = dict(
        scope="2D closed tank, same FE/CH closure as droplet entry, gravity off",
        passed=passed,
        slope=slope,
        intercept=intercept,
        cases=cases,
        acceptance=dict(
            tail_relative_jump_error=0.15,
            tail_pressure_range_over_reference=0.05,
            tail_max_speed=1e-3,
            relative_slope_error=0.15,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Passed: {passed}, fitted slope: {slope}, output: {args.output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
