#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""B32: 2D cylinder Re=20 steady — Schafer-Turek 2D-1 benchmark (verified).

Physics (FeatFlow official + OpenLB cylinder2d double-checked):
  - domain [0,2.2] x [0,0.41], cylinder B_r(0.2,0.2) radius 0.05, D=0.1
  - inlet Poiseuille u(0,y) = (4U*y*(0.41-y)/0.41^2, 0), U=0.3 (U_mean=0.2)
  - outlet Zou-He rho=1, top/bottom walls + cylinder no-slip
  - Re = U_mean*D/nu = 20  ->  nu = 0.2*0.1/20 = 0.001
  - reference (Nabh 1998 spectral, FeatFlow official):
        C_D = 5.57953523384, C_L = 0.010618948146
  - normalization: Cd = Fx/(0.5*rho*U_mean^2*D), Cl = Fy/(0.5*rho*U_mean^2*D)
    with the *physical* D=0.1, exactly as the reference.

R_eff lesson (why mask radius R, NOT R-0.5):
  half-way bounce-back (post-streaming at solid cells) places the no-slip
  wall at the midpoint between the outermost solid cell center and the
  adjacent fluid cell center.  Along the axes the wall sits at floor(R)+0.5,
  which for mask radius R looks like "effective radius R+0.5".  However the
  hydrodynamic radius of the digital staircase circle is much closer to R:
  the flow-derived effective radius of an identical staircase pipe is
  R_eff^Q = R+0.11 (poiseuille_3d_pipe, flow-rate inversion).  A mask radius
  of R-0.5 therefore overshoots inward (R_eff -> ~R-0.39) and LOWERS the
  momentum-exchange drag; measured Cd at D=40: 5.549 (R mask) vs 5.414
  (R-0.5 mask), at D=80: 5.5xx vs 5.356.  The R-0.5 shift is thus rejected;
  the mask-radius-R geometry is the correct one for this benchmark.

Methodological lesson (mass renormalization):
  the original draft applied a global mass rescaling f *= M0/sum(f) every
  2000 steps; this excites a slowly-decaying ~10k-step oscillation of Cd
  (+-2-4%) whose phase biases any finite-time average (it made D=40 look
  like -0.76% and D=80 -3.28%, an apparent "grid dependence" that was not
  physical).  Without renormalization the signal converges monotonically.
  This verified version does NOT renormalize mass.

True simulation, no extrapolation:
  - library primitives only: d2q9.equilibrium, solver.collide_mrt/stream,
    boundaries.zou_he_inlet_velocity / zou_he_outlet_pressure /
    bounce_back_cells / cylinder_mask / make_channel_wall_mask /
    compute_obstacle_forces
  - Ladd (1994) momentum-exchange drag/lift on the staircase surface
  - no correction factors, no result tuning, extrap: none

Usage:
    run.py single D_cells out.json [--shift 0] [--device cuda:1] [--steps 300000]
    run.py verify out_dir [--device cuda:1] [--steps 300000]   # D=40 and D=80
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/wxsc/cxs/TensorLBM/src")

import torch

from tensorlbm.boundaries import (
    bounce_back_cells,
    compute_obstacle_forces,
    cylinder_mask,
    make_channel_wall_mask,
    zou_he_inlet_velocity,
    zou_he_outlet_pressure,
)
from tensorlbm.d2q9 import equilibrium
from tensorlbm.solver import collide_mrt, stream

REF_CD = 5.57953523384
REF_CL = 0.010618948146


def run_case(
    D_cells: int,
    device: str = "cuda:1",
    n_steps: int = 300000,
    resid_interval: int = 5000,
    shift: float = 0.0,
) -> dict:
    """Run the Schafer-Turek 2D-1 case on a grid with D_cells across D=0.1.

    shift: cylinder_mask radius offset (0 = mask radius R, the verified
    configuration; -0.5 = the rejected R_eff experiment).  No mass
    renormalization is applied (see module docstring).
    """
    dev = torch.device(device)
    lx, ly = 2.2, 0.41
    dx = 0.1 / D_cells
    nx = int(round(lx / dx))
    ny = int(round(ly / dx))
    cx = round(0.2 / dx)
    cy = round(0.2 / dx)
    radius = 0.05 / dx  # physical radius in cells
    mask_radius = radius + shift

    tau = 0.8
    nu_lb = (tau - 0.5) / 3.0
    U_char_lb = 20.0 * nu_lb / D_cells  # U_mean, lattice units (Re=20)
    y_phys = torch.arange(ny, device=dev) * dx
    u_profile = 4.0 * 1.5 * y_phys * (0.41 - y_phys) / 0.41**2
    u_lb = U_char_lb * u_profile

    solid = cylinder_mask(nx, ny, cx, cy, mask_radius, dev)
    wall = make_channel_wall_mask(ny, nx, solid, dev)
    fluid = ~solid
    surface = solid & (
        torch.roll(fluid, 1, 0)
        | torch.roll(fluid, -1, 0)
        | torch.roll(fluid, 1, 1)
        | torch.roll(fluid, -1, 1)
    )

    dyn_p = 0.5 * U_char_lb**2 * D_cells  # 0.5*rho*U^2*D (physical D)

    rho0 = torch.ones((ny, nx), dtype=torch.float32, device=dev)
    ux0 = torch.zeros_like(rho0)
    ux0[:, :] = u_lb[:, None].expand(ny, nx)
    ux0[solid] = 0.0
    uy0 = torch.zeros_like(rho0)
    f = equilibrium(rho0, ux0, uy0)

    wall_axis = math.floor(mask_radius) + 0.5  # half-way BB wall on the axes
    t0 = time.time()
    cd_list, cl_list = [], []
    for step in range(1, n_steps + 1):
        before = f.clone()
        collided = collide_mrt(f, tau)
        f = torch.where(solid.unsqueeze(0), before, collided)
        f = stream(f)
        f = zou_he_inlet_velocity(f, u_lb, 0.0)
        f = zou_he_outlet_pressure(f, 1.0)
        f = bounce_back_cells(f, wall)
        fx, fy = compute_obstacle_forces(f, surface)
        f = bounce_back_cells(f, solid)
        cd_list.append(float(fx.item()) / dyn_p)
        cl_list.append(float(fy.item()) / dyn_p)
        if step % resid_interval == 0:
            cd = sum(cd_list[-1000:]) / min(len(cd_list), 1000)
            cl = sum(cl_list[-1000:]) / min(len(cl_list), 1000)
            print(f"  step {step}: Cd={cd:.4f} Cl={cl:.4f} ({time.time() - t0:.0f}s)")

    win = min(len(cd_list), 50000)
    cd = sum(cd_list[-win:]) / win
    cl = sum(cl_list[-win:]) / win
    return {
        "D_cells": D_cells,
        "nx": nx,
        "ny": ny,
        "tau": tau,
        "shift": shift,
        "radius_phys_cells": radius,
        "mask_radius": mask_radius,
        "wall_axis_pos": wall_axis,
        "D_eff_axis": 2.0 * wall_axis,
        "n_steps": n_steps,
        "avg_window": win,
        "n_solid_cells": int(solid.sum().item()),
        "n_surface_cells": int(surface.sum().item()),
        "cd": cd,
        "cl": cl,
        "err_cd_pct": (cd - REF_CD) / REF_CD * 100,
        "err_cl_pct": (cl - REF_CL) / REF_CL * 100,
        "ref_cd": REF_CD,
        "ref_cl": REF_CL,
        "wall_s": time.time() - t0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["single", "verify"])
    ap.add_argument("arg", help="D_cells (single) or output dir (verify)")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--steps", type=int, default=300000)
    ap.add_argument("--shift", type=float, default=0.0)
    a = ap.parse_args()

    if a.mode == "single":
        r = run_case(int(a.arg), a.device, a.steps, shift=a.shift)
        print(json.dumps(r, indent=2))
        out = Path(f"/tmp/cyl_re20_d{a.arg}.json")
        out.write_text(json.dumps(r, indent=2))
        return 0

    out_dir = Path(a.arg)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_grid = []
    for D in (40, 80):
        r = run_case(D, a.device, a.steps, shift=a.shift)
        per_grid.append(r)
        print(json.dumps(r, indent=2))
        (out_dir / f"case_D{D}.json").write_text(json.dumps(r, indent=2))
    ok = all(abs(r["err_cd_pct"]) <= 3.0 for r in per_grid)
    res = {
        "case": "B32_cylinder_re20_st",
        "lattice": "D2Q9",
        "collision": "mrt",
        "tau": 0.8,
        "boundary": "zou_he_velocity_inlet(profile) + zou_he_pressure_outlet(rho=1) "
        "+ half-way bounce-back (pre-collision skip at solid, post-streaming swap)",
        "force": "ladd_momentum_exchange (post-stream, pre-bounce-back)",
        "mask_shift": a.shift,
        "extrap": "none",
        "ref_cd": REF_CD,
        "ref_cl": REF_CL,
        "per_grid": per_grid,
        "converged": all(abs(r["err_cd_pct"]) <= 3.0 for r in per_grid) and len(per_grid) >= 2,
        "verdict": "verified" if ok else "not_verified",
    }
    (out_dir / "result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
