#!/usr/bin/env python3
"""Ahmed body + ellipsoid benchmark — unified pressure integration.

Both benchmarks use:
  - SurfaceMesh.from_gradient  (generic normal from solid mask gradient)
  - Verified main loop: NoDynamics + half-way BB + far_field_bc_3d
  - MRT + Smagorinsky LES (Cs=0.05)
  - drag_pressure_integration + drag_friction_integration (unified)

BENCHMARK 1: Ahmed body (simplified, Re=1000)
  - Geometry: rectangular box L=60, W=20, H=15 with 25° slanted rear
  - Grid: nx=300, ny=100, nz=80
  - u_in=0.05, tau=0.509, 10000 steps
  - Reference: Cd ≈ 0.25 (experimental, Re=1000 simplified)

BENCHMARK 2: Ellipsoid (prolate spheroid 2:1, Re=100)
  - Geometry: a=20, b=10, c=10 (prolate, aspect ratio 2:1)
  - Grid: nx=120, ny=80, nz=80
  - u_in=0.05, tau=0.515, 5000 steps
  - Reference: Cd ≈ 0.47 (sphere at Re=100, ellipsoid similar)

Usage:
  python ahmed_ellipsoid_worker.py <benchmark> <device_id> <output_path>
  benchmark: ahmed | ellipsoid
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from tensorlbm.boundaries3d import far_field_bc_3d, bounce_back_cells_3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.drag_pressure import (
    get_near_wall_3d,
    SurfaceMesh,
    drag_pressure_integration,
    drag_friction_integration,
)
from tensorlbm.ellipsoid_benchmark import build_ellipsoid_mask


# ---------------------------------------------------------------------------
# Ahmed body geometry: rectangular box with slanted rear
# ---------------------------------------------------------------------------
def build_ahmed_solid(nx, ny, nz, L, W, H, slant_deg, x_start, cy, cz, device):
    """Build Ahmed body solid mask (nz, ny, nx).

    The body is a rectangular box of length L, width W, height H with a
    slanted rear top at *slant_deg* from horizontal.  The slant occupies
    the rear third of the body (L_slant = L/3); the top drops from H to
    H - L_slant * tan(slant) over that distance.

    Coordinate convention: x=streamwise, y=vertical, z=spanwise.
    """
    slant_rad = math.radians(slant_deg)
    L_flat = L * 2.0 / 3.0          # front portion with flat top
    L_slant = L - L_flat            # rear slant portion
    y_base = cy - H / 2.0          # bottom of body
    y_top = cy + H / 2.0           # top of flat portion
    z_min = cz - W / 2.0
    z_max = cz + W / 2.0
    x_slant_start = x_start + L_flat
    x_end = x_start + L

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )

    # x within body extent
    in_x = (xx >= x_start) & (xx <= x_end)
    # z within width
    in_z = (zz >= z_min) & (zz <= z_max)

    # y top depends on x position
    # Flat top region: y_top
    # Slant region: y_top - (x - x_slant_start) * tan(slant)
    y_top_local = torch.where(
        xx <= x_slant_start,
        torch.full_like(xx, y_top),
        y_top - (xx - x_slant_start) * math.tan(slant_rad),
    )
    in_y = (yy >= y_base) & (yy <= y_top_local)

    solid = in_x & in_y & in_z
    return solid


# ---------------------------------------------------------------------------
# Generic run function (used by both benchmarks)
# ---------------------------------------------------------------------------
def run_benchmark(
    device_id,
    tag,
    nx, ny, nz,
    solid,
    u_in,
    tau,
    n_steps,
    dpS,
    cd_ref,
    ref_name,
    output_path=None,
):
    """Run LBM simulation with unified pressure integration and return results."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    nu = (tau - 0.5) / 3.0
    cs_smag = 0.05

    solid = solid.to(device)
    nz, ny, nx = solid.shape

    print(
        f"{tag} nx={nx} ny={ny} nz={nz} "
        f"u_in={u_in} nu={nu:.6e} tau={tau:.6f} Cs={cs_smag} "
        f"dpS={dpS:.6e} Cd_ref={cd_ref}",
        flush=True,
    )

    t0 = time.time()

    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells: {n_solid}", flush=True)

    # Precompute near-wall mask and surface mesh (from_gradient)
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells: {n_near}", flush=True)

    mesh = SurfaceMesh.from_gradient(solid, near)

    # Normal statistics
    nx_n_vals = mesh.nx_n[near]
    ny_n_vals = mesh.ny_n[near]
    nz_n_vals = mesh.nz_n[near]
    print(
        f"{tag} normal stats: "
        f"nx_n=[{float(nx_n_vals.min()):.3f}, {float(nx_n_vals.max()):.3f}] "
        f"ny_n=[{float(ny_n_vals.min()):.3f}, {float(ny_n_vals.max()):.3f}] "
        f"nz_n=[{float(nz_n_vals.min()):.3f}, {float(nz_n_vals.max()):.3f}]",
        flush=True,
    )
    norm_check = torch.sqrt(mesh.nx_n ** 2 + mesh.ny_n ** 2 + mesh.nz_n ** 2)
    norm_near = norm_check[near]
    print(
        f"{tag} |n| stats: min={float(norm_near.min()):.6f} "
        f"max={float(norm_near.max()):.6f} mean={float(norm_near.mean()):.6f}",
        flush=True,
    )
    n_nx_nonzero = int((nx_n_vals.abs() > 1e-6).sum().item())
    n_ny_nonzero = int((ny_n_vals.abs() > 1e-6).sum().item())
    n_nz_nonzero = int((nz_n_vals.abs() > 1e-6).sum().item())
    print(
        f"{tag} 3D check: nx_n nonzero={n_nx_nonzero}/{n_near} "
        f"ny_n nonzero={n_ny_nonzero}/{n_near} nz_n nonzero={n_nz_nonzero}/{n_near}",
        flush=True,
    )

    # Solid mask for NoDynamics (19, nz, ny, nx)
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    # Initialize flow field: uniform flow, zero inside solid
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(
        rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device
    )
    im = float(rho0.sum().item())

    print(f"{tag} init done ({time.time() - t0:.1f}s), initial_mass={im}",
          flush=True)

    # History
    cd_p_hist = []
    cd_f_hist = []
    cd_tot_hist = []
    cl_hist = []
    fz_hist = []

    for step in range(1, n_steps + 1):
        # 1. Save pre-collision state
        f_pre = f.clone()

        # 2. Collision (MRT + Smagorinsky LES)
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)

        # 3. NoDynamics: restore solid cells to pre-collision values
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        # 4. Half-way bounce-back (BEFORE streaming)
        f = bounce_back_cells_3d(f, solid)

        # 5. Streaming
        f = stream3d(f)

        # 6. Far-field BC (without obstacle_mask → don't touch solid)
        f = far_field_bc_3d(f, u_in)

        # 7. Mass correction every 200 steps
        if step % 200 == 0:
            f = correct_mass3d(f, im)

        # 8. Drag computation (unified pressure integration)
        fx_p, fy_p, fz_p = drag_pressure_integration(f, mesh, dpS)
        fx_f, fy_f, fz_f = drag_friction_integration(f, mesh, dpS, nu)

        cd_p = fx_p
        cd_f = fx_f
        cd_tot = cd_p + cd_f
        cl = fy_p + fy_f
        fz_tot = fz_p + fz_f

        cd_p_hist.append(cd_p)
        cd_f_hist.append(cd_f)
        cd_tot_hist.append(cd_tot)
        cl_hist.append(cl)
        fz_hist.append(fz_tot)

        # Check for divergence
        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step % 500 == 0:
            n_avg = min(500, len(cd_tot_hist))
            cd_p_avg = sum(cd_p_hist[-n_avg:]) / n_avg
            cd_f_avg = sum(cd_f_hist[-n_avg:]) / n_avg
            cd_tot_avg = sum(cd_tot_hist[-n_avg:]) / n_avg
            cl_avg = sum(cl_hist[-n_avg:]) / n_avg
            elapsed = time.time() - t0
            print(
                f"{tag} step={step} Cd_p={cd_p_avg:.4f} Cd_f={cd_f_avg:.4f} "
                f"Cd_tot={cd_tot_avg:.4f} Cl={cl_avg:.6f} ({elapsed:.0f}s)",
                flush=True,
            )

    elapsed = time.time() - t0

    # Final averages (last 1000 steps or all if fewer)
    n_final = min(1000, len(cd_tot_hist))
    cd_p_final = sum(cd_p_hist[-n_final:]) / n_final
    cd_f_final = sum(cd_f_hist[-n_final:]) / n_final
    cd_tot_final = sum(cd_tot_hist[-n_final:]) / n_final
    cl_final = sum(cl_hist[-n_final:]) / n_final
    fz_final = sum(fz_hist[-n_final:]) / n_final

    err_pct = abs(cd_tot_final - cd_ref) / cd_ref * 100 if cd_ref > 0 else float("nan")

    result = {
        "case": tag,
        "device": f"sdaa:{device_id}",
        "grid": f"{nx}x{ny}x{nz}",
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "n_steps": n_steps,
        "n_solid": n_solid,
        "n_near": n_near,
        "Cs": cs_smag,
        "Cd_pressure": cd_p_final,
        "Cd_friction": cd_f_final,
        "Cd_total": cd_tot_final,
        "Cl": cl_final,
        "fz": fz_final,
        "Cd_ref": cd_ref,
        "ref_name": ref_name,
        "error_pct": err_pct,
        "normal_stats": {
            "nx_n_min": float(nx_n_vals.min()),
            "nx_n_max": float(nx_n_vals.max()),
            "ny_n_min": float(ny_n_vals.min()),
            "ny_n_max": float(ny_n_vals.max()),
            "nz_n_min": float(nz_n_vals.min()),
            "nz_n_max": float(nz_n_vals.max()),
            "n_norm_min": float(norm_near.min()),
            "n_norm_max": float(norm_near.max()),
            "n_norm_mean": float(norm_near.mean()),
            "nx_n_nonzero": n_nx_nonzero,
            "ny_n_nonzero": n_ny_nonzero,
            "nz_n_nonzero": n_nz_nonzero,
        },
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
    }

    print(
        f"{tag} DONE Cd_p={cd_p_final:.4f} Cd_f={cd_f_final:.4f} "
        f"Cd_tot={cd_tot_final:.4f} Cl={cl_final:.6f} "
        f"(ref={cd_ref:.4f}) err={err_pct:.1f}% time={elapsed:.0f}s",
        flush=True,
    )

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} Results written to {output_path}", flush=True)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 4:
        print("Usage: python ahmed_ellipsoid_worker.py <benchmark> <device_id> <output_path>")
        print("  benchmark: ahmed | ellipsoid")
        sys.exit(1)

    benchmark = sys.argv[1]
    device_id = int(sys.argv[2])
    output_path = sys.argv[3]

    if benchmark == "ahmed":
        # Ahmed body (simplified, Re=1000)
        nx, ny, nz = 300, 100, 80
        L, W, H = 60, 20, 15
        slant_deg = 25.0
        u_in = 0.05
        tau = 0.509
        n_steps = 10000
        cd_ref = 0.25
        ref_name = "Ahmed body experimental (Re=1000, simplified)"

        x_start = nx * 0.25   # 75
        cy = ny * 0.5          # 50
        cz = nz * 0.5          # 40

        # Frontal area = W * H
        dpS = 0.5 * u_in ** 2 * W * H

        device = torch.device(f"sdaa:{device_id}")
        torch.sdaa.set_device(device)
        solid = build_ahmed_solid(
            nx, ny, nz, L, W, H, slant_deg, x_start, cy, cz, device
        )
        tag = f"ahmed_re1000_sdaa{device_id}"

        run_benchmark(
            device_id, tag, nx, ny, nz, solid,
            u_in, tau, n_steps, dpS, cd_ref, ref_name, output_path,
        )

    elif benchmark == "ellipsoid":
        # Ellipsoid (prolate spheroid 2:1, Re=100)
        nx, ny, nz = 120, 80, 80
        a, b = 20, 10   # a/b = 2:1
        u_in = 0.05
        tau = 0.515
        n_steps = 5000
        cd_ref = 0.47
        ref_name = "Sphere Re=100 (Henderson, ellipsoid similar)"

        cx = nx / 3.0     # 40
        cy = ny / 2.0     # 40
        cz = nz / 2.0     # 40

        # Frontal area = pi * b^2 (minor axis)
        dpS = 0.5 * u_in ** 2 * math.pi * b ** 2

        device = torch.device(f"sdaa:{device_id}")
        torch.sdaa.set_device(device)
        solid = build_ellipsoid_mask(
            nx, ny, nz, a, b, alpha_deg=0.0,
            cx=cx, cy=cy, cz=cz, device=device,
        )
        tag = f"ellipsoid_re100_sdaa{device_id}"

        run_benchmark(
            device_id, tag, nx, ny, nz, solid,
            u_in, tau, n_steps, dpS, cd_ref, ref_name, output_path,
        )

    else:
        print(f"Unknown benchmark: {benchmark}")
        sys.exit(1)


if __name__ == "__main__":
    main()
