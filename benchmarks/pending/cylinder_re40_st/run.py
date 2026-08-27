#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""B33: 2D cylinder Re=40 steady — Schafer-Turek 2D-1 channel benchmark (no renorm).

Physics (same setup as the verified Re=20 run, only Re changed):
  - domain [0,2.2] x [0,0.41], cylinder B_r(0.2,0.2) radius 0.05, D=0.1
  - inlet Poiseuille u(0,y) = (4U*y*(0.41-y)/0.41^2, 0), U=0.3 (U_mean=0.2)
  - outlet Zou-He rho=1, top/bottom walls + cylinder no-slip
  - Re = U_mean*D/nu = 40  ->  nu = 0.2*0.1/40 = 0.0005
  - normalization: Cd = Fx/(0.5*rho*U_mean^2*D), Cl = Fy/(0.5*rho*U_mean^2*D)
    with the *physical* D=0.1, exactly as the reference.

Reference values (VERIFY carefully):
  - Schafer-Turek/FeatFlow official values exist ONLY for Re=20 (2D-1,
    Cd=5.57953523384, Cl=0.010618948146, Nabh 1998 spectral) and Re=100
    (2D-2 ECCENTRIC cylinder, Cd=3.22-3.24, Cl=0.99-1.01).
  - There is NO official Re=40 value. We therefore use a log-space
    interpolation between Re=20 and Re=100: Cd_ref(40) ~ 4.41
    (ln-linear: 5.5795^(1-t) * 3.23^t, t=ln(40/20)/ln(100/20)=0.4307
     -> exp(1.4838)=4.41).  This is an INTERPOLATION ESTIMATE, not an
     official reference; the 3%-window verdict is judged against it and
     cross-checked for grid convergence (same-code, same-grid family as
     the verified Re=20 run: D40 ~ -0.56%, D80 ~ -2.87%).
  - Cl at Re=40 has no reference either; recorded as-is (expected to
    mismatch, same as Re=20: lift is a 2nd-order effect ~0.01 swamped by
    staircase momentum-exchange discretization).

Methodology (identical to verified cylinder_re20_st):
  - mask radius = R (shift=0; R-0.5 rejected by experiment, see re20 README)
  - NO mass renormalization (the every-2000-step global rescale was proven
    to excite ~10k-step Cd oscillations and was removed)
  - 300k steps, average of last 50k steps
  - library primitives only, Ladd (1994) momentum exchange, no correction
    factors, no extrapolation, no result tuning.
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

# Official references (Schafer-Turek / FeatFlow / Nabh 1998)
REF_CD_20 = 5.57953523384  # official Re=20 (2D-1)
REF_CL_20 = 0.010618948146  # official Re=20 (2D-1)
REF_CD_100 = 3.23  # official Re=100 (2D-2 eccentric) mid-range 3.22-3.24
# Re=40: log-space interpolation estimate (NOT official)
_t = math.log(40.0 / 20.0) / math.log(100.0 / 20.0)
REF_CD = math.exp((1 - _t) * math.log(REF_CD_20) + _t * math.log(REF_CD_100))
REF_CL = None  # no Re=40 lift reference; recorded as-is


def run_case(
    D_cells: int,
    device: str = "cuda:0",
    n_steps: int = 300000,
    resid_interval: int = 5000,
    shift: float = 0.0,
    tau: float = 0.8,
) -> dict:
    """Run the Schafer-Turek 2D-1 case at Re=40, D_cells across D=0.1.

    shift: cylinder_mask radius offset (0 = mask radius R, the verified
    configuration).  No mass renormalization is applied.

    tau: relaxation parameter.  U_mean_lb = Re*nu_lb/D_cells with
    nu_lb=(tau-0.5)/3.  tau=0.8 -> U_lb=0.1 (Ma~0.17, default "natural"
    choice); tau=0.65 -> U_lb=0.05 (Ma~0.087, SAME Ma as the verified
    Re=20 run -> same-code same-Ma grid-convergence comparison).
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

    nu_lb = (tau - 0.5) / 3.0
    U_char_lb = 40.0 * nu_lb / D_cells  # U_mean, lattice units (Re=40)
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
            print(f"  step {step}: Cd={cd:.4f} Cl={cl:.4f} ({time.time() - t0:.0f}s)", flush=True)

    win = min(len(cd_list), 50000)
    cd = sum(cd_list[-win:]) / win
    cl = sum(cl_list[-win:]) / win
    return {
        "case": "B33_cylinder_re40_st",
        "D_cells": D_cells,
        "nx": nx,
        "ny": ny,
        "tau": tau,
        "nu_lb": nu_lb,
        "u_mean_lb": U_char_lb,
        "ma": U_char_lb / (1.0 / math.sqrt(3.0)),
        "re": 40.0,
        "nu_phys": 0.0005,
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
        "ref_cd": REF_CD,
        "ref_cd_note": "log-interpolation Re20(5.5795)->Re100(3.23), NOT official",
        "ref_cd_20_official": REF_CD_20,
        "ref_cl_20_official": REF_CL_20,
        "ref_cl": None,
        "wall_s": time.time() - t0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["single", "verify"])
    ap.add_argument("arg", help="D_cells (single) or output dir (verify)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=300000)
    ap.add_argument("--shift", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=0.8)
    a = ap.parse_args()

    if a.mode == "single":
        r = run_case(int(a.arg), a.device, a.steps, shift=a.shift, tau=a.tau)
        print(json.dumps(r, indent=2))
        out = Path(f"/tmp/cyl_re40_d{a.arg}.json")
        out.write_text(json.dumps(r, indent=2))
        return 0

    out_dir = Path(a.arg)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_grid = []
    for D in (40, 80):
        r = run_case(D, a.device, a.steps, shift=a.shift, tau=a.tau)
        per_grid.append(r)
        print(json.dumps(r, indent=2))
        (out_dir / f"case_D{D}.json").write_text(json.dumps(r, indent=2))
    ok = all(abs(r["err_cd_pct"]) <= 3.0 for r in per_grid)
    res = {
        "case": "B33_cylinder_re40_st",
        "lattice": "D2Q9",
        "collision": "mrt",
        "tau": 0.8,
        "boundary": "zou_he_velocity_inlet(profile) + zou_he_pressure_outlet(rho=1) "
        "+ half-way bounce-back (pre-collision skip at solid, post-streaming swap)",
        "force": "ladd_momentum_exchange (post-stream, pre-bounce-back)",
        "mask_shift": a.shift,
        "renormalization": "none",
        "extrap": "none",
        "ref_cd": REF_CD,
        "ref_cd_note": "log-interpolation Re20(5.5795)->Re100(3.23), NOT official",
        "ref_cd_20_official": REF_CD_20,
        "ref_cl_20_official": REF_CL_20,
        "per_grid": per_grid,
        "converged": all(abs(r["err_cd_pct"]) <= 3.0 for r in per_grid) and len(per_grid) >= 2,
        "verdict": "verified" if ok else "not_verified",
    }
    (out_dir / "result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
