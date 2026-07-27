#!/usr/bin/env python3
"""Optimize benchmark accuracy — push below 10% (SDAA 0-3).

ALL benchmarks use ONLY the common interface modules:

  - Geometry:  get_near_wall_3d(solid)
  - Normals:   SurfaceMesh.from_cylinder / from_suboff / from_gradient
  - Force:     drag_pressure_integration(f, mesh, dpS, ...)
               drag_friction_integration(f, mesh, dpS, nu, ...)
  - BC:        bounce_back_cells_3d + far_field_bc_3d  (via lbm_step_correct)
  - Main loop: lbm_step_correct(f, collide_fn, tau, solid, u_in, ...)

Benchmarks (one per SDAA card 0-3):
  1. cyl_re200_opt:  Cylinder Re=200, D=48, nx=1000, ny=400, nz=4,
                     MRT+Smag(0.05), 5000 steps, from_cylinder  (SDAA:0)
                     Cd_ref=1.30  Target: <10%  (was 16.8% with 15% blockage)
  2. cyl_re40_opt:   Cylinder Re=40, D=48, nx=1000, ny=400, nz=4,
                     MRT (no Smag), 10000 steps, from_cylinder  (SDAA:1)
                     Cd_ref=1.50  Target: <10%  (was 12.0% with 15% blockage)
  3. suboff_re1000_opt: SUBOFF Re=1000, L=80, nx=200, ny=80, nz=80,
                        MRT+Smag(0.05), 10000 steps, warmup=5000, from_suboff  (SDAA:2)
                        Cd_ref=0.042  Target: <5%  (was 5.6%)
  4. kvlcc2_re1000_grad: KVLCC2 from_gradient (NOT STL), Re=1000 (NOT 1e5),
                         MRT+Smag(0.05), 5000 steps  (SDAA:3)
                         Target: Cd_p positive, <50%  (was 93% with STL)

Usage:
  PYTHONPATH=src python optimize_bench_worker.py <benchmark> <device_id> [output_path]
  benchmark: cyl_re200_opt | cyl_re40_opt | suboff_re1000_opt | kvlcc2_re1000_grad
"""
from __future__ import annotations

import functools
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_mrt3d, correct_mass3d
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.lbm_step_correct import lbm_step_correct
from tensorlbm.drag_pressure import (
    get_near_wall_3d,
    SurfaceMesh,
    drag_pressure_integration,
    drag_friction_integration,
)
from tensorlbm.postprocess import detect_strouhal
from tensorlbm.suboff_cad import build_suboff_mask, SuboffConfig
from tensorlbm.ship_cad import build_hull_mask, ShipHullType


# ------------------------------------------------------------------ #
#  Geometry builders                                                 #
# ------------------------------------------------------------------ #

def build_cylinder_solid(nx, ny, nz, cx, cy, radius, device):
    """Boolean solid mask for a cylinder extruded along the z-axis."""
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    return circle.unsqueeze(0).expand(nz, ny, nx).clone()


# ------------------------------------------------------------------ #
#  Benchmark configurations                                          #
# ------------------------------------------------------------------ #

STL_DIR = Path(
    "/root/ship-performance-platform-incoming/ship-performance-platform/"
    "backend/data/geometry/ships"
)

BENCHMARKS = {
    "cyl_re200_opt": {
        "shape": "cylinder",
        "Re": 200,
        "D": 48,
        "nx": 1000, "ny": 400, "nz": 4,
        "u_in": 0.08,
        "collision": "mrt_smag",
        "Cs": 0.05,
        "n_steps": 5000,
        "warmup": 1000,
        "avg_window": 1000,
        "Cd_ref": 1.30,
        "ref_name": "Cd=1.30 (Re=200, Henderson 1997)",
        "note": "Optimized: D=48, nx=1000, ny=400 → blockage D/ny=12%. "
                "Previous nx=800 ny=320 (15% blockage) gave 16.8%.",
        "bc_config": {"far_field_faces": ["y-", "y+"], "periodic_faces": ["z-", "z+"]},
    },
    "cyl_re40_opt": {
        "shape": "cylinder",
        "Re": 40,
        "D": 48,
        "nx": 1000, "ny": 400, "nz": 4,
        "u_in": 0.08,
        "collision": "mrt",
        "Cs": None,
        "n_steps": 10000,
        "warmup": 2000,
        "avg_window": 2000,
        "Cd_ref": 1.50,
        "ref_name": "Cd=1.50 (Re=40 steady)",
        "note": "Optimized: D=48, nx=1000, ny=400 → blockage D/ny=12%. "
                "Previous nx=800 ny=320 (15% blockage) gave 12.0%.",
        "bc_config": {"far_field_faces": ["y-", "y+"], "periodic_faces": ["z-", "z+"]},
    },
    "suboff_re1000_opt": {
        "shape": "suboff",
        "Re": 1000,
        "L": 80,
        "nx": 200, "ny": 80, "nz": 80,
        "u_in": 0.06,
        "collision": "mrt_smag",
        "Cs": 0.05,
        "n_steps": 10000,
        "warmup": 5000,
        "avg_window": 2000,
        "Cd_ref": 0.042,
        "ref_name": "Cf=0.042 (Blasius 1.328/sqrt(Re))",
        "note": "L=80, 10000 steps, warmup=5000. Previous 5000 steps gave 5.6%.",
        "bc_config": {"far_field_faces": ["y-", "y+", "z-", "z+"], "periodic_faces": []},
    },
    "kvlcc2_re1000_grad": {
        "shape": "kvlcc2",
        "Re": 1000,
        "nx": 200, "ny": 80, "nz": 80,
        "u_in": 0.06,
        "collision": "mrt_smag",
        "Cs": 0.05,
        "n_steps": 5000,
        "warmup": 1000,
        "avg_window": 1000,
        "Cd_ref": None,  # ITTC formula
        "ref_name": "Cf_ITTC = 0.075/(log10(Re)-2)^2",
        "note": "KVLCC2 from_gradient (NOT STL), Re=1000 (NOT 1e5). "
                "Previous STL Re=1e5 gave 93% error, Cd_p negative.",
        "bc_config": {"far_field_faces": ["y-", "y+", "z-", "z+"], "periodic_faces": []},
    },
}


# ------------------------------------------------------------------ #
#  KVLCC2 analytical hull (from_gradient normals, NOT STL)           #
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  Main benchmark runner                                             #
# ------------------------------------------------------------------ #

def run_benchmark(bench_name: str, device_id: int, output_path: str | None = None):
    """Run a single benchmark on the specified SDAA card."""
    cfg = BENCHMARKS[bench_name]
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    tag = f"[{bench_name} SDAA:{device_id}]"

    # --- Physical parameters ---
    shape = cfg["shape"]
    Re = cfg["Re"]
    u_in = cfg["u_in"]
    n_steps = cfg["n_steps"]
    warmup = cfg["warmup"]
    avg_window = cfg["avg_window"]

    # Collision operator
    if cfg["collision"] == "mrt":
        collide_fn = collide_mrt3d
        collide_kwargs: dict = {}
    else:
        collide_fn = collide_smagorinsky_mrt3d
        collide_kwargs = {"C_s": cfg["Cs"]}

    # Far-field BC wrapper
    far_field_fn = functools.partial(far_field_bc_3d, bc_config=cfg["bc_config"])

    t0 = time.time()

    # === Step 1: Build geometry ===
    if shape == "cylinder":
        D = cfg["D"]
        R = D / 2.0
        nx, ny, nz = cfg["nx"], cfg["ny"], cfg["nz"]
        nu = u_in * D / Re
        tau = 3.0 * nu + 0.5
        cx = nx * 0.25
        cy = ny * 0.5
        A_frontal = D * nz
        dpS = 0.5 * u_in ** 2 * A_frontal
        length_ref = D
        solid = build_cylinder_solid(nx, ny, nz, cx, cy, R, device)
    elif shape == "suboff":
        L = cfg["L"]
        nx, ny, nz = cfg["nx"], cfg["ny"], cfg["nz"]
        config = SuboffConfig()
        radius = config.r_over_l * L
        D = 2.0 * radius
        nu = u_in * L / Re
        tau = 3.0 * nu + 0.5
        cx = nx * 0.30
        cy = ny * 0.5
        cz = nz * 0.5
        A_frontal = math.pi * D * L
        dpS = 0.5 * u_in ** 2 * A_frontal
        length_ref = D
        solid, stats = build_suboff_mask(
            hull_type="bare_hull", nx=nx, ny=ny, nz=nz,
            cx=cx, cy=cy, cz=cz, length=L, radius=radius,
            config=config, device=device,
        )
    elif shape == "kvlcc2":
        nx, ny, nz = cfg["nx"], cfg["ny"], cfg["nz"]
        L = 80.0
        solid, hull_info = build_hull_mask(
            ShipHullType.KVLCC2, nx, ny, nz, length=L, device=str(device)
        )
        L_lattice = L
        nu = u_in * L_lattice / Re
        tau = 3.0 * nu + 0.5
        length_ref = L_lattice
    else:
        raise ValueError(f"Unknown shape: {shape}")

    n_solid = int(solid.sum().item())

    # === Step 2: Near-wall mask ===
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())

    # === Step 3: Surface mesh with normals ===
    if shape == "cylinder":
        mesh = SurfaceMesh.from_cylinder(solid, near, cx, cy, R, axis="z")
    elif shape == "suboff":
        mesh = SurfaceMesh.from_suboff(solid, near, cx, cy, cz, L, radius, config)
    elif shape == "kvlcc2":
        # KEY: from_gradient (NOT STL) — Bug 47+48 fix
        mesh = SurfaceMesh.from_gradient(solid, near)
        # Wetted area from voxel surface
        S_wetted = float(n_near)
        dpS = 0.5 * u_in ** 2 * S_wetted

    # ITTC reference
    if cfg["Cd_ref"] is not None:
        cd_ref = cfg["Cd_ref"]
    else:
        cd_ref = 0.075 / (math.log10(Re) - 2.0) ** 2

    blockage = D / ny * 100 if shape == "cylinder" else None

    print(
        f"{tag} {shape} Re={Re} grid={nx}x{ny}x{nz} u_in={u_in} "
        f"nu={nu:.6e} tau={tau:.6f} dpS={dpS:.6e}",
        flush=True,
    )
    print(
        f"{tag} collision={cfg['collision']} Cs={cfg.get('Cs')} "
        f"n_steps={n_steps} warmup={warmup} avg_window={avg_window}",
        flush=True,
    )
    if blockage is not None:
        print(f"{tag} blockage D/ny={blockage:.1f}%", flush=True)
    print(f"{tag} Cd_ref={cd_ref:.6f} ({cfg['ref_name']})", flush=True)
    print(f"{tag} solid cells: {n_solid}", flush=True)
    print(f"{tag} near-wall cells: {n_near}", flush=True)
    print(f"{tag} SurfaceMesh normals: from_{shape if shape != 'kvlcc2' else 'gradient'}",
          flush=True)

    # --- Initialise flow field ---
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(
        rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device
    )
    initial_mass = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s) mass={initial_mass}", flush=True)

    # --- History accumulators ---
    cd_p_hist: list[float] = []
    cd_f_hist: list[float] = []
    cd_tot_hist: list[float] = []
    cl_hist: list[float] = []

    # === Step 4: Main loop via lbm_step_correct ===
    for step in range(1, n_steps + 1):
        f = lbm_step_correct(
            f,
            collide_fn,
            tau,
            solid,
            u_in,
            far_field_fn,
            correct_mass_fn=correct_mass3d,
            target_mass=initial_mass,
            step=step,
            mass_interval=200,
            **collide_kwargs,
        )

        # === Step 5: Force via common interface ===
        if step > warmup:
            fx_p, fy_p, _ = drag_pressure_integration(
                f, mesh, dpS, extrap="none",
            )
            fx_f, fy_f, _ = drag_friction_integration(
                f, mesh, dpS, nu, q_wall=None, formula="standard"
            )
            cd_p = float(fx_p)
            cd_f = float(fx_f)
            cd_tot = cd_p + cd_f
            cl = float(fy_p + fy_f)

            cd_p_hist.append(cd_p)
            cd_f_hist.append(cd_f)
            cd_tot_hist.append(cd_tot)
            cl_hist.append(cl)

        # Divergence guard
        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        # Progress
        if step % 500 == 0 or step == n_steps:
            n_avg = min(500, len(cd_tot_hist))
            if n_avg > 0:
                cd_p_avg = sum(cd_p_hist[-n_avg:]) / n_avg
                cd_f_avg = sum(cd_f_hist[-n_avg:]) / n_avg
                cd_tot_avg = sum(cd_tot_hist[-n_avg:]) / n_avg
            else:
                cd_p_avg = cd_f_avg = cd_tot_avg = 0.0
            elapsed = time.time() - t0
            print(
                f"{tag} step={step}/{n_steps} "
                f"Cd_p={cd_p_avg:.6f} Cd_f={cd_f_avg:.6f} "
                f"Cd_tot={cd_tot_avg:.6f} ({elapsed:.0f}s)",
                flush=True,
            )

    elapsed = time.time() - t0

    # --- Final time-averaged drag ---
    n_final = min(avg_window, len(cd_tot_hist))
    if n_final > 0:
        cd_p_final = sum(cd_p_hist[-n_final:]) / n_final
        cd_f_final = sum(cd_f_hist[-n_final:]) / n_final
        cd_tot_final = sum(cd_tot_hist[-n_final:]) / n_final
        cl_final = sum(cl_hist[-n_final:]) / n_final
    else:
        cd_p_final = cd_f_final = cd_tot_final = cl_final = 0.0

    # === Step 6: Strouhal number ===
    st = detect_strouhal(
        cl_hist, sample_rate=1.0, u_ref=u_in, length_ref=length_ref, min_cycles=3
    )

    err_pct = abs(cd_tot_final - cd_ref) / cd_ref * 100 if cd_ref > 0 else float("nan")

    result = {
        "case": bench_name,
        "device": f"sdaa:{device_id}",
        "shape": shape,
        "Re": Re,
        "grid": f"{nx}x{ny}x{nz}",
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "collision": cfg["collision"],
        "Cs": cfg.get("Cs"),
        "n_steps": n_steps,
        "warmup": warmup,
        "avg_window": avg_window,
        "n_solid": n_solid,
        "n_near": n_near,
        "dpS": dpS,
        "Cd_pressure": float(cd_p_final),
        "Cd_friction": float(cd_f_final),
        "Cd_total": float(cd_tot_final),
        "Cl": float(cl_final),
        "St": st,
        "Cd_ref": cd_ref,
        "ref_name": cfg["ref_name"],
        "error_pct": float(err_pct),
        "n_samples": len(cd_tot_hist),
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
        "note": cfg.get("note", ""),
        "modules_used": [
            "drag_pressure.get_near_wall_3d",
            "drag_pressure.SurfaceMesh.from_" + ("gradient" if shape == "kvlcc2" else shape),
            "drag_pressure.drag_pressure_integration",
            "drag_pressure.drag_friction_integration",
            "boundaries3d.bounce_back_cells_3d",
            "boundaries3d.far_field_bc_3d",
            "lbm_step_correct.lbm_step_correct",
            "postprocess.detect_strouhal",
        ],
    }

    if blockage is not None:
        result["blockage_pct"] = blockage

    print(
        f"\n{'=' * 60}\n{tag} FINAL RESULTS\n{'=' * 60}",
        flush=True,
    )
    print(f"{tag} Cd_p  = {cd_p_final:.6f}", flush=True)
    print(f"{tag} Cd_f  = {cd_f_final:.6f}", flush=True)
    print(f"{tag} Cd_tot= {cd_tot_final:.6f}", flush=True)
    print(f"{tag} Cd_ref= {cd_ref:.6f}  err={err_pct:.1f}%", flush=True)
    print(f"{tag} St    = {st}", flush=True)
    print(f"{tag} time  = {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)
    print(f"{tag} finite= {result['finite']}", flush=True)

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} Results → {output_path}", flush=True)

    return result


def main():
    if len(sys.argv) < 3:
        print("Usage: python optimize_bench_worker.py <benchmark> <device_id> [output_path]")
        print(f"Benchmarks: {', '.join(BENCHMARKS.keys())}")
        sys.exit(1)

    bench_name = sys.argv[1]
    device_id = int(sys.argv[2])
    output_path = sys.argv[3] if len(sys.argv) > 3 else None

    if bench_name not in BENCHMARKS:
        print(f"Unknown benchmark: {bench_name}")
        print(f"Available: {', '.join(BENCHMARKS.keys())}")
        sys.exit(1)

    run_benchmark(bench_name, device_id, output_path)


if __name__ == "__main__":
    main()
