"""W7-B runner — sphere Re=100 solution-side floor attack.

Reuses the W5-B validated manual loop (engine kernels + library public
entrances only; grep rule: no collide/stream/equilibrium/bounce/zou_he/
far_field definitions in this file — only calls).

Routes
------
bfl : Bouzidi interpolated wall on the analytic sphere, laboratory-frame
      per-link momentum ledger (primary judgment observable, W5-B same).
bb  : voxel staircase + wet-node MEM (secondary observable; manual loop
      because GeneralSimEngine has no TRT dispatch).

Collisions (all library kernels)
    mrt        : collide_mrt3d_low_memory defaults (W5-B baseline)
    mrt_magic  : same with s_q = 1/(0.5 + (3/16)/(tau-0.5))  [Ginzburg magic]
    trt        : collide_trt3d, lambda_trt = 3/16 (default)
    bgk        : collide_bgk3d

Usage:
  python run_diag.py OUT.json --route bfl --collision mrt --D 40 \
      --steps 12000 --lat 2.0 --up 1.25 --down 2.25 --tau 0.56 --ulb 0.05
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import sys
from pathlib import Path

import torch

SRC_CANDIDATES = [
    os.environ.get("W7B_SRC_PATCHED", ""),
    str(Path(__file__).resolve().parents[4] / "src"),  # <repo>/src
]
for _p in SRC_CANDIDATES:
    if _p and os.path.isdir(_p):
        sys.path.insert(0, _p)
        break

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
from tensorlbm.momentum_exchange import momentum_exchange_wet_node  # noqa: E402
from tensorlbm.solver3d import (  # noqa: E402
    collide_bgk3d,
    collide_mrt3d_low_memory,
    collide_trt3d,
    correct_mass3d,
    stream3d,
    stream3d_roll,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cv_instrument import cv_box_force_exact  # noqa: E402

DEV = os.environ.get("W7B_DEV", "cuda:6")
RE_NOMINAL = 100.0


def build_engine_config(D, lat, up, down):
    pad = (up, down, lat, lat, lat, lat)
    return GeneralSimConfig(
        name=f"w7b_sphere_D{D}",
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
            directory=os.path.join(os.environ.get("TMPDIR", "/tmp"), "w7b_nodump"),
            formats=[OutputFormat.NPY],
            save_macroscopic=False,
            save_forces=True,
        ),
    )


def magic_s_q(tau: float, lam: float = 3.0 / 16.0) -> float:
    """Ginzburg/Pan magic relation (tau-0.5)*(1/s_q-0.5)=lam -> s_q."""
    return 1.0 / (0.5 + lam / (tau - 0.5))


def make_collider(kind: str, tau: float):
    if kind == "mrt":
        return lambda f: collide_mrt3d_low_memory(f, tau=tau)
    if kind == "mrt_magic":
        return lambda f: collide_mrt3d_low_memory(f, tau=tau, s_q=magic_s_q(tau))
    if kind == "trt":
        return lambda f: collide_trt3d(f, tau_plus=tau, lambda_trt=3.0 / 16.0)
    if kind == "bgk":
        return lambda f: collide_bgk3d(f, tau=tau)
    raise ValueError(kind)


def cd_ref_family(re: float) -> float:
    return 24.0 / re * (1.0 + 0.15 * re**0.687)


def run_case(
    out_path, route, collision, D, steps, lat, up, down, tau, ulb, treat, sample, cv_tail=0
):
    cfg = build_engine_config(D, lat, up, down)
    engine = GeneralSimEngine(cfg)
    engine.setup()
    nu_lb = (tau - 0.5) / 3.0
    re_eff = ulb * D / nu_lb
    R_lb = cfg.geometry.sphere_radius / (cfg.physics.reference_length / cfg.solver.resolution)
    dpS = 0.5 * ulb**2 * math.pi * R_lb**2
    solid = engine.solid
    nz, ny, nx = solid.shape
    dx = cfg.physics.reference_length / cfg.solver.resolution
    cx_lb = (cfg.geometry.sphere_center[0] - engine.domain_phys[0]) / dx
    cy_lb = (cfg.geometry.sphere_center[1] - engine.domain_phys[2]) / dx
    cz_lb = (cfg.geometry.sphere_center[2] - engine.domain_phys[4]) / dx
    dev = torch.device(DEV)

    if route == "bfl":
        bfl_mask, bfl_q = compute_q_sphere(nx, ny, nz, cx_lb, cy_lb, cz_lb, R_lb, dev)
        n_links = int(bfl_mask[1:].sum().item())
        q_active = bfl_q[1:][bfl_mask[1:]]
    else:
        bfl_mask = bfl_q = None
        n_links = 0

    bc_config = {"far_field_faces": ["y-", "y+", "z-", "z+"], "periodic_faces": []}
    far_field_fn = functools.partial(far_field_bc_3d, bc_config=bc_config)
    collide = make_collider(collision, tau)
    # stream3d caches 4x[19,N] int64 index tensors (~0.6 KB/cell, permanent);
    # stream3d_roll is the library's memory-efficient equivalent (same values,
    # pure permutation). Flag for A/B; default gather (W5-B bit-path).
    stream_fn = stream3d_roll if os.environ.get("W7B_STREAM") == "roll" else stream3d

    f = engine.f.clone()
    engine.f = None  # memory trim: release engine's distribution reference
    target_mass = float(f.sum().item())
    initial_mass = target_mass
    hist = []
    finite = True
    diverged_at = None
    sample_set = set(range(sample, steps + 1, sample))
    sm = solid.unsqueeze(0).expand_as(f)
    for step in range(1, steps + 1):
        f_pre = f.clone()
        f = collide(f)
        for q in range(f.shape[0]):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        f = bounce_back_cells_3d(f, solid, f_pre=f_pre)
        del f_pre  # memory trim: not needed past the solid restore/bounce
        # bfl needs the pre-stream state every step (BFL interpolation);
        # bb only at CV tail samples.
        if route == "bfl":
            f_pre_stream = f.clone()
            f = stream_fn(f)
        else:
            f_pre_stream = None
            if cv_tail > 0 and (step + 1) in sample_set and step + 1 > steps - cv_tail:
                f_pre_stream = f.clone()
            f = stream_fn(f)
        if treat == "noneq":
            f = non_equilibrium_far_field_bc_3d(
                f,
                u_in=ulb,
                faces=("x-", "x+", "y-", "y+", "z-", "z+"),
            )
        else:
            f = far_field_fn(f, ulb)
        if route == "bfl":
            f, force3 = bouzidi_bounce_back_d3q19(
                f, f_pre_stream, bfl_mask, bfl_q, return_force=True
            )
        if step in sample_set:
            if route == "bfl":
                cd = float(force3[0].item()) / dpS
                cl = float(force3[1].item()) / dpS
                cs = float(force3[2].item()) / dpS
            else:
                fx, fy, fz = momentum_exchange_wet_node(f, solid)
                cd = fx / dpS
                cl = fy / dpS
                cs = fz / dpS
            entry = {
                "step": step,
                "cd": cd,
                "cl": cl,
                "cs": cs,
                "mass": float(f.sum().item()),
            }
            if cv_tail > 0 and step > steps - cv_tail:
                cv, _ = cv_box_force_exact(f_pre_stream, solid)
                entry["cv_mom_cd"] = cv[0] / dpS
                entry["cv_mom_cl"] = cv[1] / dpS
            hist.append(entry)
        if f_pre_stream is not None and route == "bfl":
            del f_pre_stream  # memory trim: consumed by bouzidi/sample above
        if os.environ.get("W7B_EMPTY_CACHE") == "1":
            torch.cuda.empty_cache()  # memory trim: return stage caches
        if step % 200 == 0:
            f = correct_mass3d(f, target_mass)
        if step % 500 == 0 and not bool(torch.isfinite(f).all()):
            finite = False
            diverged_at = step
            break
        if step % 2000 == 0:
            print(f"step {step}/{steps}", flush=True)

    meta = {
        "route": route,
        "collision": collision,
        "treat": treat,
        "D": D,
        "lat": lat,
        "up": up,
        "down": down,
        "tau": tau,
        "u_lb": ulb,
        "nu_lb": nu_lb,
        "re_eff": re_eff,
        "steps": steps if finite else diverged_at,
        "diverged": not finite,
        "domain_lu": [nz, ny, nx],
        "dpS": dpS,
        "R_lb": R_lb,
        "center_lu": [cz_lb, cy_lb, cx_lb],
        "blockage": math.pi * R_lb**2 / (ny * nz),
        "s_q_magic": magic_s_q(tau) if collision == "mrt_magic" else None,
        "cd_ref_at_re_eff": cd_ref_family(re_eff),
        "mass_drift_ppm": (
            (float(f.sum().item()) - initial_mass) / initial_mass * 1e6 if finite else None
        ),
    }
    if route == "bfl":
        meta["bfl_links"] = n_links
        meta["q_min"] = float(q_active.min().item())
        meta["q_max"] = float(q_active.max().item())

    with open(out_path, "w") as fh:
        json.dump({"meta": meta, "history": hist}, fh)
    print("written", out_path)
    if hist:
        tail = [h["cd"] for h in hist[-8:]]
        print("last-8 cd mean:", sum(tail) / len(tail))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--route", choices=["bfl", "bb"], default="bfl")
    ap.add_argument("--collision", choices=["mrt", "mrt_magic", "trt", "bgk"], default="mrt")
    ap.add_argument("--D", type=int, default=40)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--lat", type=float, default=2.0)
    ap.add_argument("--up", type=float, default=1.25)
    ap.add_argument("--down", type=float, default=2.25)
    ap.add_argument("--tau", type=float, default=0.56)
    ap.add_argument("--ulb", type=float, default=0.05)
    ap.add_argument("--treat", choices=["hard", "noneq"], default="hard")
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument(
        "--cv_tail", type=int, default=0, help="sample CV budget at steps within this tail window"
    )
    args = ap.parse_args()
    run_case(
        args.out,
        args.route,
        args.collision,
        args.D,
        args.steps,
        args.lat,
        args.up,
        args.down,
        args.tau,
        args.ulb,
        args.treat,
        args.sample,
        args.cv_tail,
    )


if __name__ == "__main__":
    main()
