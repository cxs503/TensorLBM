#!/usr/bin/env python3
"""Sphere Re=100 drag PROBE.

Tests which lever fixes the sphere Cd:
  H1 domain size (wake truncation / blockage)  -> --pad up,down,lat,lat,lat,lat  (units of D)
  H2 force method: pressure+friction (GeneralSimEngine common modules) vs
     discrete kinetic control-volume momentum balance (control_volume_force)
  H3 friction formula: standard / faces / mix50 / bfl_smooth

Compile-routed whole-step chain (benchmarks/compile_route), force sampling and
control-volume probe stay eager in the driver loop.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "benchmarks"))

import torch  # noqa: E402
import torch_sdaa  # noqa: F401,E402

try:  # registers the SDAA device for torch._inductor (teco_inductor backend)
    import torch_sdaa._inductor  # noqa: F401,E402
except Exception as _exc:  # pragma: no cover - CPU-only hosts
    print(f"[warn] torch_sdaa._inductor not imported: {_exc}", flush=True)

from compile_route import add_compile_mode_arg, compile_mode_from_args, route_step  # noqa: E402

from tensorlbm.boundaries3d import far_field_bc_3d  # noqa: E402
from tensorlbm.control_volume_force import (  # noqa: E402
    box_control_volume,
    fluid_momentum_change,
    streaming_momentum_import,
)
from tensorlbm.drag_pressure import (  # noqa: E402
    drag_friction_integration,
    drag_pressure_integration,
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
    PhysicsConfig,
    SolverConfig,
    WallTreatment,
)
from tensorlbm.lbm_step_correct import lbm_step_correct  # noqa: E402
from tensorlbm.solver3d import stream3d  # noqa: E402

REF_SN = 24.0 / 100.0 * (1.0 + 0.15 * 100.0**0.687)          # 1.09173
REF_CGW = 24.0 / 100.0 * (1.0 + 0.1935 * 100.0**0.6305)      # Clift-Grace-Weber


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diam", type=int, default=20)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--pad", type=str, default="1.75,1.75,1,1,1,1",
                    help="upstream,downstream,ylo,yhi,zlo,zhi in units of D")
    ap.add_argument("--device", default="sdaa:0")
    ap.add_argument("--collision", default="mrt")
    ap.add_argument("--cov-margin", type=int, default=2)
    add_compile_mode_arg(ap)
    args = ap.parse_args()

    compile_mode = compile_mode_from_args(args)
    D = args.diam
    pad = tuple(float(v) for v in args.pad.split(","))
    assert len(pad) == 6

    config = GeneralSimConfig(
        name=f"probe_sphere_re100_d{D}",
        geometry=GeometryConfig(source=GeometrySource.PARAMETRIC_SPHERE,
                                sphere_radius=0.5, sphere_center=(0.0, 0.0, 0.0)),
        physics=PhysicsConfig(density=1000.0, viscosity=1.0e-6,
                              inlet_velocity=1.0e-4, reference_length=1.0),
        solver=SolverConfig(
            lattice=LatticeModel.D3Q19,
            collision=CollisionModel.MRT if args.collision == "mrt" else CollisionModel.AUTO,
            resolution=D, domain_padding=pad, max_steps=args.steps,
            snapshot_interval=10_000_000, force_sample_interval=20,
            device=args.device, wall_treatment=WallTreatment.AUTO,
            force_method=ForceMethod.PRESSURE_FRICTION,
            pressure_extrap="none", p0_method="near_wall",
            friction_formula="standard", mass_correction=True,
            mass_correction_interval=200, smagorinsky_cs=0.05),
        output=OutputConfig(directory="/tmp/probe_sphere", formats=[],
                            save_macroscopic=False, save_forces=True),
    )
    engine = GeneralSimEngine(config)
    info = engine.setup()
    print(f"[setup] D={D} domain={info['domain_lu']} cells={info['total_cells']} "
          f"solid={info['obstacle_cells']} near={info['near_wall_cells']} "
          f"tau={info['tau']:.4f} u_lb={info['u_lb']:.4f}", flush=True)

    tau = engine.uc.tau
    nu_lb = engine.uc.nu_lb
    u_in = engine.uc.u_lb
    collide_fn, collide_kwargs = engine._get_collide_fn()
    far_field_fn = functools.partial(far_field_bc_3d, bc_config=engine._build_bc_config())
    solid = engine.solid
    from tensorlbm.d3q19 import OPPOSITE as _OPP
    f_pre_opp_idx = _OPP.to(solid.device)
    target_mass = engine._initial_mass

    def _step(f):
        return lbm_step_correct(f, collide_fn, tau, solid, u_in, far_field_fn, **collide_kwargs)

    step_fn = route_step(_step, compile_mode, name=f"probe_sphere_re100[D{D}]")

    # control volume: box around the sphere, strictly interior
    dpS = engine._compute_dpS()
    dx = config.physics.reference_length / D
    R_lb = 0.5 / dx
    cx = (0.0 - engine.domain_phys[0]) / dx
    cy = (0.0 - engine.domain_phys[2]) / dx
    cz = (0.0 - engine.domain_phys[4]) / dx
    m = args.cov_margin
    cv = box_control_volume(solid.shape, x0=int(cx - R_lb - m), x1=int(cx + R_lb + m) + 1,
                            y0=int(cy - R_lb - m), y1=int(cy + R_lb + m) + 1,
                            z0=int(cz - R_lb - m), z1=int(cz + R_lb + m) + 1,
                            device=solid.device)
    print(f"[cv] cx={cx:.1f} R_lb={R_lb:.1f} cells={int(cv.sum())}", flush=True)

    t0 = time.time()
    for s in range(1, args.steps + 1):
        engine.f = step_fn(engine.f)
        engine.step_count += 1
        if s % 200 == 0:
            from tensorlbm.solver3d import correct_mass3d
            engine.f = correct_mass3d(engine.f, target_mass)
    t_step = time.time() - t0
    print(f"[run] {args.steps} steps in {t_step:.0f}s "
          f"({t_step / args.steps * 1000:.1f} ms/step)", flush=True)

    # analytic smooth-sphere wall distance for the BFL friction formula
    nz_g, ny_g, nx_g = solid.shape
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz_g, device=solid.device, dtype=torch.float32),
        torch.arange(ny_g, device=solid.device, dtype=torch.float32),
        torch.arange(nx_g, device=solid.device, dtype=torch.float32),
        indexing="ij")
    q_sph = (torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) - R_lb
             ).clamp(0.05, 1.0) * engine.near.float()

    # ---- final-field force sweep (all common-module formulas), no extras ----
    f = engine.f
    mesh = engine.mesh
    rows = {}
    for p0 in ("near_wall", "far_field", "domain_avg", "inlet"):
        cd_p = drag_pressure_integration(f, mesh, dpS, extrap="none",
                                         p0_method=p0, solid=solid)[0]
        r = {"cd_p": cd_p}
        for formula in ("standard", "faces", "mix50", "lagrange"):
            cd_f = drag_friction_integration(f, mesh, dpS, nu_lb,
                                             formula=formula, solid=solid)[0]
            r[formula] = cd_f
        r["bfl"] = drag_friction_integration(f, mesh, dpS, nu_lb,
                                             formula="bfl", q_wall=q_sph, solid=solid)[0]
        rows[p0] = r

    # ---- discrete control-volume momentum balance over 20 steps ----
    cov_sum = torch.zeros(3, dtype=torch.float64, device=solid.device)
    n_cov = 20
    for _ in range(n_cov):
        f_old = engine.f
        fp = collide_fn(f_old, tau=tau, **collide_kwargs)
        fp = torch.where(solid.unsqueeze(0),
                         f_old[f_pre_opp_idx.to(f_old.device)], fp)
        imported = streaming_momentum_import(fp, cv)
        f_new = far_field_fn(stream3d(fp), u_in)
        change = fluid_momentum_change(f_old, f_new, cv, solid=solid)
        cov_sum += imported - change
        engine.f = f_new
    cov_f = (cov_sum / n_cov).tolist()
    cd_cov = cov_f[0] / dpS
    print(f"[cov] force_x={cov_f[0]:.6e}  Cd_cov={cd_cov:.4f}", flush=True)

    print("\n=== final-field pressure/friction sweep (extrap=none) ===")
    for p0, r in rows.items():
        print(f"  p0={p0:11s} cd_p={r['cd_p']:.4f} "
              f"Std={r['standard']:.4f} Faces={r['faces']:.4f} "
              f"mix50={r['mix50']:.4f} bfl(=.5std)={r['bfl']:.4f} lag={r['lagrange']:.4f}")
        for base, lbl in (("standard", "standard"), ("faces", "faces"), ("mix50", "mix50")):
            tot = r["cd_p"] + r[base]
            print(f"      tot({lbl}) = {tot:.4f}  err_SN={100*(tot-REF_SN)/REF_SN:+.2f}%  "
                  f"err_CGW={100*(tot-REF_CGW)/REF_CGW:+.2f}%")

    out = {
        "D": D, "domain": info["domain_lu"], "pad": pad,
        "u_lb": u_in, "nu_lb": nu_lb, "tau": tau, "steps": args.steps,
        "Cd_cov": cd_cov, "cov_force": cov_f,
        "cd_pressure_by_p0": {k: v["cd_p"] for k, v in rows.items()},
        "cd_friction": {k: {ff: v[ff] for ff in ("standard", "faces", "mix50", "bfl", "lagrange")}
                        for k, v in rows.items()},
        "ref_SN": REF_SN, "ref_CGW": REF_CGW,
        "ms_per_step": t_step / args.steps * 1000,
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()