#!/usr/bin/env python3
"""IBM stability test — verifies the fixed ibm_common.py.

Tests:
1. Stationary cylinder (Re=100, D=24) — should be stable and produce
   reasonable Cd (within 20% of BB reference).
2. Oscillating cylinder (A=0.05*u_in, St=0.1) — small amplitude, should
   be stable (no NaN) for at least 500 steps.

Runs on CPU for quick verification.  Uses small grids for speed.
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

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_mrt3d, correct_mass3d, stream3d
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.ibm_common import (
    ibm_step_correct,
    generate_cylinder_markers,
    compute_ibm_drag_from_markers,
)
from tensorlbm.drag_pressure import (
    get_near_wall_3d,
    SurfaceMesh,
    drag_pressure_integration,
    drag_friction_integration,
)


def build_cylinder_solid(nx, ny, nz, cx, cy, radius, device):
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    return circle.unsqueeze(0).expand(nz, ny, nx).clone()


def make_stationary_target(n_markers, device):
    zero = torch.zeros(n_markers, dtype=torch.float32, device=device)
    def fn(step):
        return zero.clone(), zero.clone(), zero.clone()
    return fn


def make_oscillating_target(n_markers, device, A_vel, f_reduced, D, u_in):
    f_lattice = f_reduced * u_in / D
    def fn(step):
        t = float(step)
        u_y = A_vel * math.sin(2.0 * math.pi * f_lattice * t)
        ux = torch.zeros(n_markers, dtype=torch.float32, device=device)
        uy = torch.full((n_markers,), u_y, dtype=torch.float32, device=device)
        uz = torch.zeros(n_markers, dtype=torch.float32, device=device)
        return ux, uy, uz
    return fn


def run_test(test_name, device, **kwargs):
    """Run a single IBM stability test."""
    tag = f"[{test_name}]"
    D = kwargs["D"]
    R = D / 2.0
    nx, ny, nz = kwargs["nx"], kwargs["ny"], kwargs["nz"]
    u_in = kwargs["u_in"]
    Re = kwargs["Re"]
    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5
    n_steps = kwargs["n_steps"]

    A_frontal = D * nz
    dpS = 0.5 * u_in ** 2 * A_frontal

    collide_fn = collide_mrt3d
    collide_kwargs = {}
    far_field_fn = functools.partial(
        far_field_bc_3d,
        bc_config={"far_field_faces": ["y-", "y+"], "periodic_faces": ["z-", "z+"]},
    )

    print(f"{tag} Re={Re} D={D} grid={nx}x{ny}x{nz} u_in={u_in} "
          f"nu={nu:.6e} tau={tau:.6f}", flush=True)

    t0 = time.time()

    # Build geometry
    cx = nx * 0.25
    cy = ny * 0.5
    cz = nz * 0.5
    solid = build_cylinder_solid(nx, ny, nz, cx, cy, R, device)
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells: {n_solid}", flush=True)

    # Near-wall + mesh for drag
    near = get_near_wall_3d(solid)
    mesh = SurfaceMesh.from_cylinder(solid, near, cx, cy, R, axis="z")

    # IBM markers
    marker_x0, marker_y0, marker_z0 = generate_cylinder_markers(
        cx, cy, cz, R, nz, axis="z", n_theta=kwargs.get("n_theta", 32),
        device=device,
    )
    n_markers = marker_x0.shape[0]
    print(f"{tag} IBM markers: {n_markers}", flush=True)

    # Target velocity
    u_target_fn = make_stationary_target(n_markers, device)
    if test_name == "oscillating":
        A_vel = kwargs["A_vel"]
        f_reduced = kwargs["f_reduced"]
        u_target_fn = make_oscillating_target(
            n_markers, device, A_vel, f_reduced, D, u_in
        )
        f_lattice = f_reduced * u_in / D
        print(f"{tag} oscillating: A_vel={A_vel} St={f_reduced} "
              f"f_lattice={f_lattice:.6e}", flush=True)
    markers = (marker_x0, marker_y0, marker_z0)

    # Initialize flow
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.zeros((nz, ny, nx), device=device)  # Start from rest (ramp handles startup)
    f = equilibrium3d(
        rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device
    )
    initial_mass = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s) mass={initial_mass}", flush=True)

    # History
    cd_ibm_hist = []
    cd_p_hist = []
    cd_f_hist = []
    cd_tot_hist = []
    cl_hist = []
    max_u_hist = []

    # Main loop
    diverged = False
    for step in range(1, n_steps + 1):
        f, (mfx, mfy, mfz) = ibm_step_correct(
            f,
            collide_fn,
            tau,
            solid,
            u_in,
            far_field_fn,
            markers,
            u_target_fn,
            lattice="D3Q19",
            kernel="4pt",
            correct_mass_fn=correct_mass3d,
            target_mass=initial_mass,
            step=step,
            mass_interval=200,
            ramp_steps=kwargs.get("ramp_steps", 500),
            n_force_iter=kwargs.get("n_force_iter", 4),
            force_clip=kwargs.get("force_clip", 0.05),
            **collide_kwargs,
        )

        # Force computation
        cd_ibm, cl_ibm, _ = compute_ibm_drag_from_markers(mfx, mfy, mfz, dpS)
        fx_p, fy_p, _ = drag_pressure_integration(f, mesh, dpS, extrap="none")
        fx_f, fy_f, _ = drag_friction_integration(
            f, mesh, dpS, nu, q_wall=None, formula="standard"
        )
        cd_p = fx_p
        cd_f = fx_f
        cd_tot = cd_p + cd_f
        cl = fy_p + fy_f

        cd_ibm_hist.append(cd_ibm)
        cd_p_hist.append(cd_p)
        cd_f_hist.append(cd_f)
        cd_tot_hist.append(cd_tot)
        cl_hist.append(cl)

        # Check max velocity for stability monitoring
        _, ux, uy, uz = macroscopic3d(f)
        max_u = float(torch.max(ux.abs()).item())
        max_u_hist.append(max_u)

        # Divergence check
        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step} (max_u={max_u:.4f})", flush=True)
            diverged = True
            break

        if max_u > 1.0:
            print(f"{tag} VELOCITY BLOWUP at step {step} (max_u={max_u:.4f})", flush=True)
            diverged = True
            break

        # Progress
        if step % 100 == 0:
            n_avg = min(100, len(cd_tot_hist))
            cd_tot_avg = sum(cd_tot_hist[-n_avg:]) / n_avg
            cd_ibm_avg = sum(cd_ibm_hist[-n_avg:]) / n_avg
            elapsed = time.time() - t0
            print(f"{tag} step={step}/{n_steps} max_u={max_u:.4f} "
                  f"Cd_tot={cd_tot_avg:.4f} Cd_ibm={cd_ibm_avg:.4f} "
                  f"({elapsed:.1f}s)", flush=True)

    elapsed = time.time() - t0

    # Final averages
    avg_window = min(kwargs.get("avg_window", 200), len(cd_tot_hist))
    cd_tot_final = sum(cd_tot_hist[-avg_window:]) / avg_window
    cd_ibm_final = sum(cd_ibm_hist[-avg_window:]) / avg_window
    cd_p_final = sum(cd_p_hist[-avg_window:]) / avg_window
    cd_f_final = sum(cd_f_hist[-avg_window:]) / avg_window
    cl_final = sum(cl_hist[-avg_window:]) / avg_window
    max_u_final = max(max_u_hist[-avg_window:]) if len(max_u_hist) >= avg_window else max(max_u_hist)

    result = {
        "test": test_name,
        "device": str(device),
        "Re": Re,
        "D": D,
        "grid": f"{nx}x{ny}x{nz}",
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "n_steps": n_steps,
        "n_markers": n_markers,
        "n_solid": n_solid,
        "Cd_ibm": cd_ibm_final,
        "Cd_pressure": cd_p_final,
        "Cd_friction": cd_f_final,
        "Cd_total": cd_tot_final,
        "Cl": cl_final,
        "max_u": max_u_final,
        "finite": not diverged,
        "elapsed_s": elapsed,
        "ramp_steps": kwargs.get("ramp_steps", 500),
        "n_force_iter": kwargs.get("n_force_iter", 4),
    }

    status = "PASS" if not diverged else "FAIL"
    print(f"\n{tag} {status} — Cd_tot={cd_tot_final:.4f} Cd_ibm={cd_ibm_final:.4f} "
          f"max_u={max_u_final:.4f} finite={not diverged} ({elapsed:.1f}s)\n", flush=True)

    return result


def main():
    device = torch.device("cpu")
    print(f"Device: {device}\n")

    results = {}

    # Test 1: Stationary cylinder (Re=100, small grid)
    print("=" * 60)
    print("TEST 1: Stationary cylinder Re=100")
    print("=" * 60)
    results["stationary"] = run_test(
        "stationary",
        device,
        D=24,
        nx=160, ny=80, nz=4,
        u_in=0.1,
        Re=100,
        n_steps=1000,
        avg_window=200,
        n_theta=32,
        ramp_steps=500,
        n_force_iter=4,
        force_clip=0.05,
    )

    # Test 2: Oscillating cylinder (small amplitude)
    print("=" * 60)
    print("TEST 2: Oscillating cylinder A=0.05*u_in, St=0.1")
    print("=" * 60)
    results["oscillating"] = run_test(
        "oscillating",
        device,
        D=24,
        nx=160, ny=80, nz=4,
        u_in=0.1,
        Re=100,
        n_steps=1000,
        avg_window=200,
        n_theta=32,
        A_vel=0.005,  # 0.05 * u_in = 0.005
        f_reduced=0.1,  # St = 0.1
        ramp_steps=500,
        n_force_iter=4,
        force_clip=0.05,
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, r in results.items():
        status = "PASS" if r["finite"] else "FAIL"
        print(f"  {name}: {status}  Cd_tot={r['Cd_total']:.4f}  "
              f"Cd_ibm={r['Cd_ibm']:.4f}  max_u={r['max_u']:.4f}")

    # Save results
    with open("ibm_stability_test_results.json", "w") as fout:
        json.dump(results, fout, indent=2)
    print("\nResults saved to ibm_stability_test_results.json")


if __name__ == "__main__":
    main()
