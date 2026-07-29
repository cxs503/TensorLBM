#!/usr/bin/env python3
"""Quick IBM stability test — SDAA, 500 steps, small grid.

Tests:
1. Stationary cylinder — should be stable, Cd should develop
2. Oscillating cylinder (A=0.05*u_in, St=0.1) — should be stable (no NaN)
"""
from __future__ import annotations
import functools, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
import torch
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_mrt3d, correct_mass3d
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.ibm_common import (
    ibm_step_correct, generate_cylinder_markers, compute_ibm_drag_from_markers,
)
from tensorlbm.drag_pressure import (
    get_near_wall_3d, SurfaceMesh,
    drag_pressure_integration, drag_friction_integration,
)


def build_cylinder_solid(nx, ny, nz, cx, cy, radius, device):
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    return circle.unsqueeze(0).expand(nz, ny, nx).clone()


def run_test(test_name, device, **kw):
    tag = f"[{test_name}]"
    D = kw["D"]; R = D / 2.0
    nx, ny, nz = kw["nx"], kw["ny"], kw["nz"]
    u_in = kw["u_in"]; Re = kw["Re"]
    nu = u_in * D / Re; tau = 3.0 * nu + 0.5
    n_steps = kw["n_steps"]
    dpS = 0.5 * u_in ** 2 * D * nz
    far_field_fn = functools.partial(far_field_bc_3d,
        bc_config={"far_field_faces": ["y-", "y+"], "periodic_faces": ["z-", "z+"]})

    print(f"{tag} Re={Re} D={D} grid={nx}x{ny}x{nz} u_in={u_in} "
          f"nu={nu:.6e} tau={tau:.6f} n_steps={n_steps}", flush=True)
    t0 = time.time()

    cx = nx * 0.25; cy = ny * 0.5; cz = nz * 0.5
    solid = build_cylinder_solid(nx, ny, nz, cx, cy, R, device)
    near = get_near_wall_3d(solid)
    mesh = SurfaceMesh.from_cylinder(solid, near, cx, cy, R, axis="z")
    mx0, my0, mz0 = generate_cylinder_markers(cx, cy, cz, R, nz, axis="z",
                                               n_theta=kw.get("n_theta", 32), device=device)
    n = mx0.shape[0]
    print(f"{tag} solid={int(solid.sum().item())} markers={n}", flush=True)

    # Target velocity
    zero = torch.zeros(n, dtype=torch.float32, device=device)
    if test_name == "oscillating":
        A_vel = kw["A_vel"]; f_red = kw["f_reduced"]
        f_lat = f_red * u_in / D
        def u_target_fn(step):
            u_y = A_vel * math.sin(2.0 * math.pi * f_lat * float(step))
            return zero.clone(), torch.full((n,), u_y, dtype=torch.float32, device=device), zero.clone()
        print(f"{tag} oscillating: A_vel={A_vel} St={f_red} f_lat={f_lat:.6e}", flush=True)
    else:
        def u_target_fn(step):
            return zero.clone(), zero.clone(), zero.clone()

    # Init from rest
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.zeros((nz, ny, nx), device=device)
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    mass0 = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s)", flush=True)

    cd_tot_hist = []; cd_ibm_hist = []; cl_hist = []; max_u_hist = []
    diverged = False

    for step in range(1, n_steps + 1):
        f, (mfx, mfy, mfz) = ibm_step_correct(
            f, collide_mrt3d, tau, solid, u_in, far_field_fn,
            (mx0, my0, mz0), u_target_fn,
            lattice="D3Q19", kernel="4pt",
            correct_mass_fn=correct_mass3d, target_mass=mass0,
            step=step, mass_interval=200,
            ramp_steps=kw.get("ramp_steps", 300),
            n_force_iter=kw.get("n_force_iter", 4),
            force_clip=kw.get("force_clip", 0.05),
        )

        cd_ibm, cl_ibm, _ = compute_ibm_drag_from_markers(mfx, mfy, mfz, dpS)
        fx_p, fy_p, _ = drag_pressure_integration(f, mesh, dpS, extrap="none")
        fx_f, fy_f, _ = drag_friction_integration(f, mesh, dpS, nu, q_wall=None, formula="standard")
        cd_tot = fx_p + fx_f; cl = fy_p + fy_f
        cd_tot_hist.append(cd_tot); cd_ibm_hist.append(cd_ibm); cl_hist.append(cl)

        _, ux, _, _ = macroscopic3d(f)
        max_u = float(torch.max(ux.abs()).item())
        max_u_hist.append(max_u)

        if not torch.isfinite(f).all() or max_u > 1.0:
            print(f"{tag} DIVERGED at step {step} max_u={max_u:.4f}", flush=True)
            diverged = True; break

        if step % 100 == 0:
            n_avg = min(100, len(cd_tot_hist))
            cd_avg = sum(cd_tot_hist[-n_avg:]) / n_avg
            cd_ibm_avg = sum(cd_ibm_hist[-n_avg:]) / n_avg
            print(f"{tag} step={step}/{n_steps} max_u={max_u:.4f} "
                  f"Cd_tot={cd_avg:.4f} Cd_ibm={cd_ibm_avg:.4f} ({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0
    avg_w = min(kw.get("avg_window", 100), len(cd_tot_hist))
    cd_tot_f = sum(cd_tot_hist[-avg_w:]) / avg_w
    cd_ibm_f = sum(cd_ibm_hist[-avg_w:]) / avg_w
    cl_f = sum(cl_hist[-avg_w:]) / avg_w
    max_u_f = max(max_u_hist[-avg_w:]) if len(max_u_hist) >= avg_w else max(max_u_hist)

    status = "PASS" if not diverged else "FAIL"
    print(f"\n{tag} {status} — Cd_tot={cd_tot_f:.4f} Cd_ibm={cd_ibm_f:.4f} "
          f"Cl={cl_f:.4f} max_u={max_u_f:.4f} ({elapsed:.0f}s)\n", flush=True)

    return {
        "test": test_name, "Re": Re, "D": D, "grid": f"{nx}x{ny}x{nz}",
        "u_in": u_in, "tau": tau, "n_steps": n_steps, "n_markers": n,
        "Cd_ibm": cd_ibm_f, "Cd_total": cd_tot_f, "Cl": cl_f,
        "max_u": max_u_f, "finite": not diverged, "elapsed_s": elapsed,
    }


def main():
    device = torch.device("sdaa:0")
    torch.sdaa.set_device(device)
    print(f"Device: {device}\n")

    results = {}

    print("=" * 60)
    print("TEST 1: Stationary cylinder Re=100")
    print("=" * 60)
    results["stationary"] = run_test("stationary", device,
        D=24, nx=200, ny=100, nz=4, u_in=0.1, Re=100,
        n_steps=500, avg_window=100, n_theta=32,
        ramp_steps=300, n_force_iter=4, force_clip=0.05)

    print("=" * 60)
    print("TEST 2: Oscillating cylinder A=0.05*u_in, St=0.1")
    print("=" * 60)
    results["oscillating"] = run_test("oscillating", device,
        D=24, nx=200, ny=100, nz=4, u_in=0.1, Re=100,
        n_steps=500, avg_window=100, n_theta=32,
        A_vel=0.005, f_reduced=0.1,
        ramp_steps=300, n_force_iter=4, force_clip=0.05)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, r in results.items():
        s = "PASS" if r["finite"] else "FAIL"
        print(f"  {name}: {s}  Cd_tot={r['Cd_total']:.4f}  "
              f"Cd_ibm={r['Cd_ibm']:.4f}  max_u={r['max_u']:.4f}")

    with open("ibm_stability_test_results.json", "w") as fout:
        json.dump(results, fout, indent=2)
    print("\nResults saved to ibm_stability_test_results.json")


if __name__ == "__main__":
    main()
