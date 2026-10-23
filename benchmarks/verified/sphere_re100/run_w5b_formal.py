"""W5-B formal runner — sphere Re=100, direct-force judgment routes.

Routes
------
bb  : GeneralSimEngine (patched package) with force_method=MOMENTUM_EXCHANGE,
      mem_variant='wet_node' — the repaired estimator (2·Σ c_q f_q(x_s) over
      the complete fluid->solid link set) is the engine's primary reported
      force.  Historical-estimator comparison columns live in diag/.
bfl : engine kernels + Bouzidi interpolated wall on the analytic sphere
      (compute_q_sphere + bouzidi_bounce_back_d3q19, laboratory-frame link
      ledger) — the directly measured per-step momentum transfer.

Only library kernels are used (grep rule: no hand-written collide/stream/
equilibrium/bounce/zou_he/far_field definitions in this file).

Usage:
  PYTHONPATH=src_patched python run.py D STEPS ROUTE [--lat M] [--sample K]
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import sys

import torch

STAGE = "/nfs/wangxi/runs/bm_widen_w5_20260921/sphere_force"
SRC_PATCHED = os.path.join(STAGE, "src_patched")
sys.path.insert(0, SRC_PATCHED)

from tensorlbm.bfl_d3q19 import bouzidi_bounce_back_d3q19  # noqa: E402
from tensorlbm.boundaries3d import (  # noqa: E402
    bounce_back_cells_3d,
    far_field_bc_3d,
)
from tensorlbm.external_open_boundary import (  # noqa: E402
    non_equilibrium_far_field_bc_3d,
)
from tensorlbm.general_sim import (  # noqa: E402
    CollisionModel,
    ForceMethod,
    GeneralSimConfig,
    GeneralSimEngine,
    GeometryConfig,
    GeometrySource,
    LatticeModel,
    OutputConfig,
    OutputFormat,
    PhysicsConfig,
    SolverConfig,
    WallTreatment,
)
from tensorlbm.interpolated_bc import compute_q_sphere  # noqa: E402
from tensorlbm.solver3d import (  # noqa: E402
    collide_mrt3d_low_memory,
    correct_mass3d,
    stream3d,
)
from tensorlbm.sponge_layer import (  # noqa: E402
    apply_equilibrium_difference_sponge,
    build_sponge_sigma_3d,
)

sys.path.insert(0, os.path.join(STAGE, "diag"))
from bfl_worker import cv_box_force_exact  # noqa: E402

DEV = "cuda:6"
U_LB = 0.05
RE = 100.0


def build_config(D, lat, up, down):
    pad = (up, down, lat, lat, lat, lat)
    return GeneralSimConfig(
        name=f"w5b_formal_sphere_re100_D{D}",
        geometry=GeometryConfig(
            source=GeometrySource.PARAMETRIC_SPHERE,
            sphere_radius=0.5,
            sphere_center=(0.0, 0.0, 0.0),
        ),
        physics=PhysicsConfig(
            density=1000.0,
            viscosity=1.0e-6,
            inlet_velocity=1.0e-4,
            reference_length=1.0,
        ),
        solver=SolverConfig(
            lattice=LatticeModel.D3Q19,
            collision=CollisionModel.MRT,
            resolution=D,
            domain_padding=pad,
            max_steps=10,
            snapshot_interval=10**9,
            force_sample_interval=50,
            device=DEV,
            wall_treatment=WallTreatment.BOUNCE_BACK,
            mass_correction=True,
            mass_correction_interval=200,
            force_method=ForceMethod.MOMENTUM_EXCHANGE,
            mem_variant="wet_node",
        ),
        output=OutputConfig(
            directory="/nfs/wangxi/tmp/w5b_nodump",
            formats=[OutputFormat.NPY],
            save_macroscopic=False,
            save_forces=True,
        ),
    )


def route_bb(D, steps, lat, up, down, sample):
    cfg = build_config(D, lat, up, down)
    engine = GeneralSimEngine(cfg)
    info = engine.setup()
    nu_lb = U_LB * D / RE
    tau = 0.5 + 3.0 * nu_lb
    R_lb = cfg.geometry.sphere_radius / (cfg.physics.reference_length / cfg.solver.resolution)
    dpS = 0.5 * U_LB**2 * math.pi * R_lb**2
    run = engine.run(steps=steps)
    nz, ny, nx = engine.solid.shape
    meta = {
        "route": "bb",
        "D": D,
        "steps": run["steps"],
        "diverged": run["diverged"],
        "domain_lu": [nz, ny, nx],
        "tau": tau,
        "u_lb": U_LB,
        "nu_lb": nu_lb,
        "dpS": dpS,
        "R_lb": R_lb,
        "blockage": math.pi * R_lb**2 / (ny * nz),
        "engine_setup_Re": info["Re"],
        "engine_tau": info["tau"],
    }
    return meta, engine.forces_log


def route_bfl(D, steps, lat, up, down, sample, treat="hard"):
    cfg = build_config(D, lat, up, down)
    engine = GeneralSimEngine(cfg)
    engine.setup()
    nu_lb = U_LB * D / RE
    tau = 0.5 + 3.0 * nu_lb
    R_lb = cfg.geometry.sphere_radius / (cfg.physics.reference_length / cfg.solver.resolution)
    dpS = 0.5 * U_LB**2 * math.pi * R_lb**2
    solid = engine.solid
    nz, ny, nx = solid.shape
    dx = cfg.physics.reference_length / cfg.solver.resolution
    cx_lb = (cfg.geometry.sphere_center[0] - engine.domain_phys[0]) / dx
    cy_lb = (cfg.geometry.sphere_center[1] - engine.domain_phys[2]) / dx
    cz_lb = (cfg.geometry.sphere_center[2] - engine.domain_phys[4]) / dx
    dev = torch.device(DEV)
    bfl_mask, bfl_q = compute_q_sphere(nx, ny, nz, cx_lb, cy_lb, cz_lb, R_lb, dev)

    bc_config = {"far_field_faces": ["y-", "y+", "z-", "z+"], "periodic_faces": []}
    far_field_fn = functools.partial(far_field_bc_3d, bc_config=bc_config)
    if treat == "prod":
        sponge_sigma = build_sponge_sigma_3d(
            (nz, ny, nx),
            width=18,
            max_strength=0.2,
            device=dev,
            faces=("x+", "y-", "y+", "z-", "z+"),
        )
        sponge_fn = functools.partial(
            apply_equilibrium_difference_sponge,
            sigma=sponge_sigma,
            rho_target=1.0,
            velocity_target=(U_LB, 0.0, 0.0),
        )
    cv0, box0 = cv_box_force_exact(engine.f.clone(), solid)  # t=0: ~0 by symmetry

    f = engine.f.clone()
    target_mass = float(f.sum().item())
    initial_mass = target_mass
    hist = []
    finite = True
    diverged_at = None
    sample_set = set(range(sample, steps + 1, sample))
    sm = solid.unsqueeze(0).expand_as(f)
    for step in range(1, steps + 1):
        f_pre = f.clone()
        f = collide_mrt3d_low_memory(f, tau=tau)
        for q in range(f.shape[0]):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        f = bounce_back_cells_3d(f, solid, f_pre=f_pre)
        f_pre_stream = f.clone()
        f = stream3d(f)
        if treat == "prod":
            f = non_equilibrium_far_field_bc_3d(
                f,
                u_in=U_LB,
                faces=("x-", "x+", "y-", "y+", "z-", "z+"),
            )
            f = sponge_fn(f)
        else:
            f = far_field_fn(f, U_LB)
        f, force3 = bouzidi_bounce_back_d3q19(f, f_pre_stream, bfl_mask, bfl_q, return_force=True)
        if step in sample_set:
            cv, _ = cv_box_force_exact(f_pre_stream, solid)
            hist.append(
                {
                    "step": step,
                    "cd_bfl": float(force3[0].item()) / dpS,
                    "cl_bfl": float(force3[1].item()) / dpS,
                    "cs_bfl": float(force3[2].item()) / dpS,
                    "cv_mom_cd": [v / dpS for v in cv],
                    "mass": float(f.sum().item()),
                }
            )
        if step % 200 == 0:
            f = correct_mass3d(f, target_mass)
        if step % 500 == 0 and not bool(torch.isfinite(f).all()):
            finite = False
            diverged_at = step
            break
        if step % 2000 == 0:
            print(f"step {step}/{steps}", flush=True)

    n_links = int(bfl_mask[1:].sum().item())
    q_active = bfl_q[1:][bfl_mask[1:]]
    meta = {
        "route": "bfl",
        "treatment": treat,
        "D": D,
        "steps": steps if finite else diverged_at,
        "diverged": not finite,
        "domain_lu": [nz, ny, nx],
        "tau": tau,
        "u_lb": U_LB,
        "nu_lb": nu_lb,
        "dpS": dpS,
        "R_lb": R_lb,
        "center_lu": [cz_lb, cy_lb, cx_lb],
        "blockage": math.pi * R_lb**2 / (ny * nz),
        "bfl_links": n_links,
        "q_min": float(q_active.min().item()),
        "q_max": float(q_active.max().item()),
        "cv0": cv0,
        "cv_box_bounds": list(box0),
        "mass_drift_ppm": (
            (float(f.sum().item()) - initial_mass) / initial_mass * 1e6 if finite else None
        ),
    }
    return meta, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("D", type=int)
    ap.add_argument("steps", type=int)
    ap.add_argument("route", choices=["bb", "bfl"])
    ap.add_argument("--lat", type=float, default=1.25)
    ap.add_argument("--up", type=float, default=1.25)
    ap.add_argument("--down", type=float, default=2.25)
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--treat", choices=["hard", "prod"], default="hard")
    args = ap.parse_args()

    if args.route == "bb":
        meta, hist = route_bb(args.D, args.steps, args.lat, args.up, args.down, args.sample)
    else:
        meta, hist = route_bfl(
            args.D, args.steps, args.lat, args.up, args.down, args.sample, args.treat
        )
    meta["lat"] = args.lat
    meta["up"] = args.up
    meta["down"] = args.down
    meta["sample_interval"] = args.sample

    out = os.path.join(STAGE, f"formal_{args.route}_D{args.D}.json")
    with open(out, "w") as fh:
        json.dump({"meta": meta, "history": hist}, fh)
    print("written", out)


if __name__ == "__main__":
    main()
