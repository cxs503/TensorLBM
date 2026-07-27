#!/usr/bin/env python3
"""Universal common-interface benchmark worker.

ALL benchmarks use ONLY the common interface modules — NO custom force
computation, NO custom bounce-back:

  - Geometry:  get_near_wall_3d(solid)            from drag_pressure.py
  - Normals:   SurfaceMesh.from_cylinder/from_sphere
  - Force:     drag_pressure_integration(f, mesh, dpS, extrap, p0_method)
               drag_friction_integration(f, mesh, dpS, nu, q_wall, formula)
  - BC:        bounce_back_cells_3d + far_field_bc_3d  (via lbm_step_correct)
  - Main loop: lbm_step_correct(f, collide_fn, tau, solid, u_in, far_field_fn, ...)
  - St:        detect_strouhal(cl_hist, sample_rate, u_ref, length_ref)

Benchmarks (one per SDAA card 0-3):
  1. cyl_re40:    Cylinder Re=40,   D=48, MRT,           Cd_ref=1.50  (SDAA:0)
  2. cyl_re3900:  Cylinder Re=3900, D=48, MRT+Smag,      Cd_ref=0.98  (SDAA:1)
  3. sph_re300:   Sphere  Re=300,  D=40, MRT,            Cd_ref=0.44  (SDAA:2)
  4. sph_re10000: Sphere  Re=10000, D=40, MRT+Smag(0.1),  Cd_ref=0.40  (SDAA:3)

Usage:
  python _common_interface_worker.py <benchmark> <device_id> [output_path]
"""
from __future__ import annotations

import functools
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch

# ---- Common interface imports (ONLY these modules) ----
from tensorlbm.d3q19 import equilibrium3d
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


# ------------------------------------------------------------------ #
#  Geometry builders (mask construction — not force/BC computation)   #
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


def build_sphere_solid(nx, ny, nz, cx, cy, cz, R, device):
    """Boolean solid mask for a sphere."""
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return ((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) < R ** 2


# ------------------------------------------------------------------ #
#  Benchmark configurations                                          #
# ------------------------------------------------------------------ #

BENCHMARKS = {
    "cyl_re40": {
        "shape": "cylinder",
        "Re": 40,
        "D": 96,
        "nx": 960, "ny": 384, "nz": 4,
        "u_in": 0.08,
        "collision": "mrt_smag",
        "Cs": 0.05,
        "n_steps": 5000,
        "avg_window": 1500,
        "Cd_ref": 1.50,
        "note": "D=48 gives Cd_tot=2.17 (45% high). D=96 with MRT(no smag) gives 2.22. Trying MRT+Smag(Cs=0.05) matching verified cylinder_large_sdaa16.py.",
        "bc_config": {"far_field_faces": ["y-", "y+"], "periodic_faces": ["z-", "z+"]},
    },
    "cyl_re3900": {
        "shape": "cylinder",
        "Re": 3900,
        "D": 192,
        "nx": 768, "ny": 576, "nz": 4,
        "u_in": 0.1,
        "collision": "mrt_smag",
        "Cs": 0.15,
        "n_steps": 5000,
        "avg_window": 1500,
        "Cd_ref": 0.98,
        "note": "D=48 tau=0.503 diverges. D=96 tau=0.506 diverges. D=192 u_in=0.08 tau=0.512 diverges at step 3329. D=192 u_in=0.1 tau=0.515 Cs=0.15 for stability.",
        "bc_config": {"far_field_faces": ["y-", "y+"], "periodic_faces": ["z-", "z+"]},
    },
    "sph_re300": {
        "shape": "sphere",
        "Re": 300,
        "D": 40,
        "nx": 120, "ny": 120, "nz": 120,
        "u_in": 0.08,
        "collision": "mrt_smag",
        "Cs": 0.05,
        "n_steps": 2000,
        "avg_window": 800,
        "Cd_ref": 0.44,
        "note": "MRT(no smag) gives Cd~1.13 (2.6x high). Trying MRT+Smag(Cs=0.05) matching verified sphere_3d_drag_worker.py.",
        "bc_config": {"far_field_faces": ["y-", "y+", "z-", "z+"], "periodic_faces": []},
    },
    "sph_re10000": {
        "shape": "sphere",
        "Re": 10000,
        "D": 40,
        "nx": 120, "ny": 120, "nz": 120,
        "u_in": 0.08,
        "collision": "mrt_smag",
        "Cs": 0.1,
        "n_steps": 2500,
        "avg_window": 1000,
        "Cd_ref": 0.40,
        "bc_config": {"far_field_faces": ["y-", "y+", "z-", "z+"], "periodic_faces": []},
    },
}


def run_benchmark(bench_name: str, device_id: int, output_path: str | None = None):
    """Run a single benchmark on the specified SDAA card."""
    cfg = BENCHMARKS[bench_name]
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    tag = f"[{bench_name} SDAA:{device_id}]"

    # --- Physical parameters ---
    D = cfg["D"]
    R = D / 2.0
    nx, ny, nz = cfg["nx"], cfg["ny"], cfg["nz"]
    u_in = cfg["u_in"]
    Re = cfg["Re"]
    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5
    n_steps = cfg["n_steps"]
    avg_window = cfg["avg_window"]

    # Frontal area → dynamic-pressure × area (dpS)
    if cfg["shape"] == "cylinder":
        A_frontal = D * nz          # 2-D extruded: diameter × span
    else:
        A_frontal = math.pi * R ** 2  # sphere: πR²
    dpS = 0.5 * u_in ** 2 * A_frontal

    # --- Collision operator ---
    if cfg["collision"] == "mrt":
        collide_fn = collide_mrt3d
        collide_kwargs: dict = {}
    else:
        collide_fn = collide_smagorinsky_mrt3d
        collide_kwargs = {"C_s": cfg["Cs"]}

    # --- Far-field BC wrapper for lbm_step_correct ---
    # lbm_step_correct calls far_field_bc_fn(f, u_in); we fix bc_config via partial.
    far_field_fn = functools.partial(far_field_bc_3d, bc_config=cfg["bc_config"])

    print(
        f"{tag} Re={Re} D={D} grid={nx}x{ny}x{nz} u_in={u_in} "
        f"nu={nu:.6e} tau={tau:.6f} dpS={dpS:.6e}",
        flush=True,
    )
    print(
        f"{tag} collision={cfg['collision']} Cs={cfg.get('Cs')} "
        f"n_steps={n_steps} avg_window={avg_window}",
        flush=True,
    )

    t0 = time.time()

    # === Step 1: Build geometry (solid mask) ===
    if cfg["shape"] == "cylinder":
        cx = nx * 0.25
        cy = ny * 0.5
        cz = nz * 0.5  # not used for cylinder
        solid = build_cylinder_solid(nx, ny, nz, cx, cy, R, device)
    else:
        cx = nx * 0.25
        cy = ny * 0.5
        cz = nz * 0.5
        solid = build_sphere_solid(nx, ny, nz, cx, cy, cz, R, device)

    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells: {n_solid}", flush=True)

    # === Step 2: Near-wall mask (common interface) ===
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells: {n_near}", flush=True)

    # === Step 3: Surface mesh with normals (common interface) ===
    if cfg["shape"] == "cylinder":
        mesh = SurfaceMesh.from_cylinder(solid, near, cx, cy, R, axis="z")
    else:
        mesh = SurfaceMesh.from_sphere(solid, near, cx, cy, cz, R)
    print(f"{tag} SurfaceMesh.from_{cfg['shape']} built", flush=True)

    # --- Initialise flow field: uniform flow, zero inside solid ---
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

    # === Step 4: Main loop via lbm_step_correct (common interface) ===
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
        # Pressure drag: drag_pressure_integration(f, mesh, dpS, extrap, p0_method)
        # Using default p0_method='near_wall' (same as verified sphere/cylinder workers)
        fx_p, fy_p, _ = drag_pressure_integration(f, mesh, dpS, extrap="none")
        # Friction drag: drag_friction_integration(f, mesh, dpS, nu, q_wall, formula)
        fx_f, fy_f, _ = drag_friction_integration(
            f, mesh, dpS, nu, q_wall=None, formula="standard"
        )

        cd_p = fx_p
        cd_f = fx_f
        cd_tot = cd_p + cd_f
        cl = fy_p + fy_f  # lift (transverse force)

        cd_p_hist.append(cd_p)
        cd_f_hist.append(cd_f)
        cd_tot_hist.append(cd_tot)
        cl_hist.append(cl)

        # Divergence guard
        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        # Progress
        if step % 500 == 0:
            n_avg = min(500, len(cd_tot_hist))
            cd_p_avg = sum(cd_p_hist[-n_avg:]) / n_avg
            cd_f_avg = sum(cd_f_hist[-n_avg:]) / n_avg
            cd_tot_avg = sum(cd_tot_hist[-n_avg:]) / n_avg
            elapsed = time.time() - t0
            print(
                f"{tag} step={step}/{n_steps} "
                f"Cd_p={cd_p_avg:.4f} Cd_f={cd_f_avg:.4f} "
                f"Cd_tot={cd_tot_avg:.4f} ({elapsed:.0f}s)",
                flush=True,
            )

    elapsed = time.time() - t0

    # --- Final time-averaged drag ---
    n_final = min(avg_window, len(cd_tot_hist))
    cd_p_final = sum(cd_p_hist[-n_final:]) / n_final
    cd_f_final = sum(cd_f_hist[-n_final:]) / n_final
    cd_tot_final = sum(cd_tot_hist[-n_final:]) / n_final
    cl_final = sum(cl_hist[-n_final:]) / n_final

    # === Step 6: Strouhal number via common interface ===
    # detect_strouhal(cl_signal, sample_rate, u_ref, length_ref)
    st = detect_strouhal(
        cl_hist, sample_rate=1.0, u_ref=u_in, length_ref=D, min_cycles=3
    )

    cd_ref = cfg["Cd_ref"]
    err_pct = abs(cd_tot_final - cd_ref) / cd_ref * 100 if cd_ref > 0 else float("nan")

    result = {
        "case": bench_name,
        "device": f"sdaa:{device_id}",
        "shape": cfg["shape"],
        "Re": Re,
        "D": D,
        "grid": f"{nx}x{ny}x{nz}",
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "collision": cfg["collision"],
        "Cs": cfg.get("Cs"),
        "n_steps": n_steps,
        "n_solid": n_solid,
        "n_near": n_near,
        "Cd_pressure": cd_p_final,
        "Cd_friction": cd_f_final,
        "Cd_total": cd_tot_final,
        "Cl": cl_final,
        "St": st,
        "Cd_ref": cd_ref,
        "error_pct": err_pct,
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
        "note": cfg.get("note", ""),
        "modules_used": [
            "drag_pressure.get_near_wall_3d",
            "drag_pressure.SurfaceMesh.from_" + cfg["shape"],
            "drag_pressure.drag_pressure_integration",
            "drag_pressure.drag_friction_integration",
            "boundaries3d.bounce_back_cells_3d",
            "boundaries3d.far_field_bc_3d",
            "lbm_step_correct.lbm_step_correct",
            "postprocess.detect_strouhal",
        ],
    }

    print(
        f"{tag} DONE  Cd_p={cd_p_final:.4f}  Cd_f={cd_f_final:.4f}  "
        f"Cd_tot={cd_tot_final:.4f}  St={st}  "
        f"(ref={cd_ref:.2f})  err={err_pct:.1f}%  time={elapsed:.0f}s",
        flush=True,
    )

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} Results → {output_path}", flush=True)

    return result


def main():
    if len(sys.argv) < 3:
        print("Usage: python _common_interface_worker.py <benchmark> <device_id> [output_path]")
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
