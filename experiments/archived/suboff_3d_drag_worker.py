#!/usr/bin/env python3
"""SUBOFF submarine 3D drag verification — analytical surface normals.

Validates SurfaceMesh.from_suboff normal computation for an axisymmetric
body of revolution.  Uses the verified main loop:
  NoDynamics + half-way BB + far-field BC + MRT/Smagorinsky.

Benchmarks:
  1. Re=1000  (tau=0.5144, laminar, stable)
  2. Re=1e4   (tau=0.50144, transitional)
  3. Re=1e5   (tau=0.500144, turbulent — if stable)

Reference: ITTC-1957 Cf = 0.075 / (log10(Re) - 2)^2
  Re=1000:  Cf=0.075
  Re=1e4:   Cf=0.01875
  Re=1e5:   Cf=0.00833
  Re=2e6:   Cf=0.00405

Usage:
  python suboff_3d_drag_worker.py <benchmark> <device_id> <output_path>
  benchmark: re1000 | re1e4 | re1e5 | re2e6
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
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
from tensorlbm.suboff_cad import build_suboff_mask, SuboffConfig


def run_suboff(
    device_id,
    Re,
    L,
    nx,
    ny,
    nz,
    u_in,
    tau,
    n_steps,
    tag,
    output_path=None,
):
    """Run SUBOFF bare-hull drag simulation and return results dict."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    # SUBOFF bare hull geometry
    config = SuboffConfig()
    radius = config.r_over_l * L  # R_max in lattice units
    cx = nx * 0.30   # bow at 30% of domain (room for wake)
    cy = ny * 0.5
    cz = nz * 0.5

    nu = u_in * L / Re
    cs_smag = 0.05

    # dpS = 0.5 * u_in^2 * pi * R_max^2 (dynamic pressure × frontal area)
    dpS = 0.5 * u_in ** 2 * math.pi * radius ** 2

    # ITTC-1957 reference
    Cf_ittc = 0.075 / (np.log10(Re) - 2.0) ** 2

    print(
        f"{tag} nx={nx} ny={ny} nz={nz} L={L} R_max={radius:.3f} "
        f"u_in={u_in} nu={nu:.6e} tau={tau:.6f} Cs={cs_smag} "
        f"dpS={dpS:.6e} Cf_ITTC={Cf_ittc:.6f}",
        flush=True,
    )

    t0 = time.time()

    # Build SUBOFF solid mask
    solid, stats = build_suboff_mask(
        hull_type="bare_hull",
        nx=nx, ny=ny, nz=nz,
        cx=cx, cy=cy, cz=cz,
        length=L, radius=radius,
        config=config,
        device=device,
    )
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells: {n_solid}  L/D={stats['L_D_ratio']}", flush=True)

    # Precompute near-wall mask and surface mesh
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells: {n_near}", flush=True)

    mesh = SurfaceMesh.from_suboff(
        solid, near, cx, cy, cz, L, radius, config)

    # --- Normal statistics (validate 3D normal) ---
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
    # Check normal is unit vector
    norm_check = torch.sqrt(mesh.nx_n ** 2 + mesh.ny_n ** 2 + mesh.nz_n ** 2)
    norm_near = norm_check[near]
    print(
        f"{tag} |n| stats: min={float(norm_near.min()):.6f} "
        f"max={float(norm_near.max()):.6f} mean={float(norm_near.mean()):.6f}",
        flush=True,
    )
    # Check all 3 components are non-zero (TRUE 3D normal)
    n_nx_nonzero = int((nx_n_vals.abs() > 1e-6).sum().item())
    n_ny_nonzero = int((ny_n_vals.abs() > 1e-6).sum().item())
    n_nz_nonzero = int((nz_n_vals.abs() > 1e-6).sum().item())
    print(
        f"{tag} 3D check: nx_n nonzero={n_nx_nonzero}/{n_near} "
        f"ny_n nonzero={n_ny_nonzero}/{n_near} nz_n nonzero={n_nz_nonzero}/{n_near}",
        flush=True,
    )

    # Sign check: bow cells should have nx_n < 0, stern cells nx_n > 0
    # Bow region: xi < 0.233, Stern region: xi > 0.748
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing='ij')
    x_bow = cx - L / 2.0
    xi_field = (xx - x_bow) / L
    bow_mask = near & (xi_field < 0.233)
    stern_mask = near & (xi_field > 0.748)
    mid_mask = near & (xi_field >= 0.233) & (xi_field <= 0.748)
    if bow_mask.any():
        bow_nx_mean = float(mesh.nx_n[bow_mask].mean())
        print(f"{tag} bow nx_n mean={bow_nx_mean:.4f} (expect < 0)", flush=True)
    if stern_mask.any():
        stern_nx_mean = float(mesh.nx_n[stern_mask].mean())
        print(f"{tag} stern nx_n mean={stern_nx_mean:.4f} (expect > 0)", flush=True)
    if mid_mask.any():
        mid_nx_mean = float(mesh.nx_n[mid_mask].mean())
        print(f"{tag} midbody nx_n mean={mid_nx_mean:.6f} (expect ~ 0)", flush=True)

    # dA statistics (default dA=1.0)
    dA_vals = mesh.dA[near]
    print(
        f"{tag} dA stats: min={float(dA_vals.min()):.4f} "
        f"max={float(dA_vals.max()):.4f} mean={float(dA_vals.mean()):.4f}",
        flush=True,
    )

    # Solid mask for NoDynamics (19, nz, ny, nx)
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    # Initialize flow field: uniform flow, zero inside hull
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

        # 6. Far-field BC
        f = far_field_bc_3d(f, u_in)

        # 7. Mass correction every 200 steps
        if step % 200 == 0:
            f = correct_mass3d(f, im)

        # 8. Drag computation
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

        if step % 200 == 0:
            n_avg = min(200, len(cd_tot_hist))
            cd_p_avg = sum(cd_p_hist[-n_avg:]) / n_avg
            cd_f_avg = sum(cd_f_hist[-n_avg:]) / n_avg
            cd_tot_avg = sum(cd_tot_hist[-n_avg:]) / n_avg
            cl_avg = sum(cl_hist[-n_avg:]) / n_avg
            elapsed = time.time() - t0
            print(
                f"{tag} step={step} Cd_p={cd_p_avg:.6f} Cd_f={cd_f_avg:.6f} "
                f"Cd_tot={cd_tot_avg:.6f} Cl={cl_avg:.6f} ({elapsed:.0f}s)",
                flush=True,
            )

    elapsed = time.time() - t0

    # Final averages (last 500 steps or all if fewer)
    n_final = min(500, len(cd_tot_hist))
    cd_p_final = sum(cd_p_hist[-n_final:]) / n_final
    cd_f_final = sum(cd_f_hist[-n_final:]) / n_final
    cd_tot_final = sum(cd_tot_hist[-n_final:]) / n_final
    cl_final = sum(cl_hist[-n_final:]) / n_final
    fz_final = sum(fz_hist[-n_final:]) / n_final

    # Reference: ITTC-1957 friction coefficient
    cd_ref = float(Cf_ittc)
    ref_name = "ITTC-1957 Cf=0.075/(log10(Re)-2)^2"

    err_pct = abs(cd_tot_final - cd_ref) / cd_ref * 100 if cd_ref > 0 else float("nan")

    result = {
        "case": tag,
        "device": f"sdaa:{device_id}",
        "Re": Re,
        "L": L,
        "R_max": radius,
        "L_D_ratio": stats["L_D_ratio"],
        "grid": f"{nx}x{ny}x{nz}",
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "n_steps": n_steps,
        "n_solid": n_solid,
        "n_near": n_near,
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
            "bow_nx_mean": float(mesh.nx_n[bow_mask].mean()) if bow_mask.any() else None,
            "stern_nx_mean": float(mesh.nx_n[stern_mask].mean()) if stern_mask.any() else None,
            "mid_nx_mean": float(mesh.nx_n[mid_mask].mean()) if mid_mask.any() else None,
        },
        "dA_stats": {
            "min": float(dA_vals.min()),
            "max": float(dA_vals.max()),
            "mean": float(dA_vals.mean()),
        },
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
    }

    print(
        f"{tag} DONE Cd_p={cd_p_final:.6f} Cd_f={cd_f_final:.6f} "
        f"Cd_tot={cd_tot_final:.6f} Cl={cl_final:.6f} "
        f"(ref Cf={cd_ref:.6f}) err={err_pct:.1f}% time={elapsed:.0f}s",
        flush=True,
    )

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} Results written to {output_path}", flush=True)

    return result


def main():
    if len(sys.argv) < 4:
        print("Usage: python suboff_3d_drag_worker.py <benchmark> <device_id> <output_path>")
        print("  benchmark: re1000 | re1e4 | re1e5 | re2e6")
        sys.exit(1)

    benchmark = sys.argv[1]
    device_id = int(sys.argv[2])
    output_path = sys.argv[3]

    if benchmark == "re1000":
        # Re=1000: tau=0.5144, laminar, stable
        # L=80, u_in=0.06, nu=0.06*80/1000=0.0048, tau=3*0.0048+0.5=0.5144
        run_suboff(
            device_id=device_id, Re=1000, L=80,
            nx=300, ny=120, nz=120,
            u_in=0.06, tau=0.5144,
            n_steps=5000,
            tag=f"[SDAA:{device_id} SUBOFF Re=1000]",
            output_path=output_path,
        )
    elif benchmark == "re1e4":
        # Re=1e4: tau=0.50144, transitional
        # nu=0.06*80/1e4=0.00048, tau=3*0.00048+0.5=0.50144
        run_suboff(
            device_id=device_id, Re=10000, L=80,
            nx=300, ny=120, nz=120,
            u_in=0.06, tau=0.50144,
            n_steps=5000,
            tag=f"[SDAA:{device_id} SUBOFF Re=1e4]",
            output_path=output_path,
        )
    elif benchmark == "re1e5":
        # Re=1e5: tau=0.500144, turbulent (may be unstable)
        # nu=0.06*80/1e5=4.8e-5, tau=3*4.8e-5+0.5=0.500144
        run_suboff(
            device_id=device_id, Re=100000, L=80,
            nx=300, ny=120, nz=120,
            u_in=0.06, tau=0.500144,
            n_steps=5000,
            tag=f"[SDAA:{device_id} SUBOFF Re=1e5]",
            output_path=output_path,
        )
    elif benchmark == "re2e6":
        # Re=2e6: tau≈0.5, turbulent (likely unstable without wall function)
        # nu=0.06*80/2e6=2.4e-6, tau=3*2.4e-6+0.5=0.5000072
        run_suboff(
            device_id=device_id, Re=2000000, L=80,
            nx=300, ny=120, nz=120,
            u_in=0.06, tau=0.5000072,
            n_steps=5000,
            tag=f"[SDAA:{device_id} SUBOFF Re=2e6]",
            output_path=output_path,
        )
    else:
        print(f"Unknown benchmark: {benchmark}")
        sys.exit(1)


if __name__ == "__main__":
    main()
