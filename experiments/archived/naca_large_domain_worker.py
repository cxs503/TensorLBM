#!/usr/bin/env python3
"""NACA 0012 at Re=1000 with large domain (12 chord) — single card.

Reference: Cd≈0.05 (laminar, Re=1000)

Domain: chord=100, nx=1200, ny=400, nz=4 (12 chord domain)
u_in=0.05, Re=1000, tau=3*0.05*100/1000+0.5=0.515
10000 steps, MRT+Smagorinsky (Cs=0.05)
Uses from_gradient normal for the airfoil surface.

Previous result: chord100 6L domain (nx=600, ny=200) gave Cd_tot=0.0600, err=20%.
This 12L domain should reduce blockage and improve accuracy.

Usage:
  PYTHONPATH=src python naca_large_domain_worker.py <device_id> <output_path>
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.boundaries3d import far_field_bc_3d, bounce_back_cells_3d
from tensorlbm.drag_pressure import (
    get_near_wall_2d,
    SurfaceMesh,
    drag_pressure_integration,
    drag_friction_integration,
)


def build_naca(chord, nx, ny, x_le, y_c, device):
    """Build NACA 0012 solid mask (2D extruded in z, nz=4).

    Standard NACA 4-digit thickness formula:
      yt = 0.6 * (0.2969*sqrt(xc) - 0.1260*xc - 0.3516*xc^2 + 0.2843*xc^3 - 0.1015*xc^4)
    where xc = (i - x_le) / chord, 0 <= xc <= 1.
    """
    nz = 4
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    for k in range(nz):
        for i in range(nx):
            xc = (i - x_le) / chord
            if 0 <= xc <= 1:
                yt = 0.6 * (
                    0.2969 * math.sqrt(xc)
                    - 0.1260 * xc
                    - 0.3516 * xc ** 2
                    + 0.2843 * xc ** 3
                    - 0.1015 * xc ** 4
                )
                j_lo = max(0, int(y_c - yt * chord))
                j_hi = min(ny - 1, int(y_c + yt * chord))
                solid[k, j_lo : j_hi + 1, i] = True
    return solid


def run_naca_large_domain(device_id, output_path):
    """Run NACA 0012 at Re=1000 with 12-chord domain."""
    chord = 100
    nx = 1200  # 12 chord
    ny = 400   # 4 chord
    nz = 4
    u_in = 0.05
    Re = 1000
    nu = u_in * chord / Re  # 0.005
    tau = 3.0 * nu + 0.5    # 0.515
    cs_smag = 0.05
    n_steps = 10000
    ref_cd = 0.05

    x_le = int(nx * 0.25)  # 3 chords from inlet
    y_c = ny // 2           # centered

    # dpS = 0.5 * u_in^2 * chord * nz (dynamic pressure × frontal area)
    dpS = 0.5 * u_in ** 2 * chord * nz

    tag = f"[SDAA:{device_id} NACA0012 Re=1000 12L {nx}x{ny}x{nz}]"
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    print(
        f"{tag} chord={chord} nx={nx} ny={ny} nz={nz} "
        f"u_in={u_in} nu={nu:.6f} tau={tau:.6f} Cs={cs_smag} "
        f"x_le={x_le} y_c={y_c} dpS={dpS:.6f}",
        flush=True,
    )

    t0 = time.time()

    # Build NACA solid mask
    solid = build_naca(chord, nx, ny, x_le, y_c, device)
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells: {n_solid}", flush=True)

    # Near-wall mask and surface mesh (from_gradient normal)
    near = get_near_wall_2d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells: {n_near}", flush=True)

    mesh = SurfaceMesh.from_gradient(solid, near)

    # Normal statistics
    nx_n_vals = mesh.nx_n[near]
    ny_n_vals = mesh.ny_n[near]
    print(
        f"{tag} normal stats: "
        f"nx_n=[{float(nx_n_vals.min()):.3f}, {float(nx_n_vals.max()):.3f}] "
        f"ny_n=[{float(ny_n_vals.min()):.3f}, {float(ny_n_vals.max()):.3f}]",
        flush=True,
    )

    # Solid mask for NoDynamics (19, nz, ny, nx)
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    # Initialize flow field
    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(
        rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device
    )
    im = float(rho0.sum().item())

    print(f"{tag} init done ({time.time() - t0:.1f}s), initial_mass={im}", flush=True)

    # History
    cd_p_hist = []
    cd_f_hist = []
    cd_tot_hist = []
    cl_hist = []

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

        # 8. Drag computation
        fx_p, fy_p, _ = drag_pressure_integration(f, mesh, dpS)
        fx_f, fy_f, _ = drag_friction_integration(f, mesh, dpS, nu)

        cd_p = fx_p
        cd_f = fx_f
        cd_tot = cd_p + cd_f
        cl = fy_p + fy_f  # lift (should be ~0 for symmetric airfoil at 0° AoA)

        cd_p_hist.append(cd_p)
        cd_f_hist.append(cd_f)
        cd_tot_hist.append(cd_tot)
        cl_hist.append(cl)

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

    # Final averages (last 1000 steps)
    n_final = min(1000, len(cd_tot_hist))
    cd_p_final = sum(cd_p_hist[-n_final:]) / n_final
    cd_f_final = sum(cd_f_hist[-n_final:]) / n_final
    cd_tot_final = sum(cd_tot_hist[-n_final:]) / n_final
    cl_final = sum(cl_hist[-n_final:]) / n_final

    err_pct = abs(cd_tot_final - ref_cd) / ref_cd * 100

    result = {
        "case": tag,
        "benchmark": "naca0012_large_domain",
        "device": f"sdaa:{device_id}",
        "Re": Re,
        "chord": chord,
        "grid": f"{nx}x{ny}x{nz}",
        "domain_ratio": f"{nx/chord:.0f}L",
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "Cs": cs_smag,
        "n_steps": n_steps,
        "n_solid": n_solid,
        "n_near": n_near,
        "dpS": dpS,
        "x_le": x_le,
        "y_c": y_c,
        "Cd_pressure": cd_p_final,
        "Cd_friction": cd_f_final,
        "Cd_total": cd_tot_final,
        "Cl": cl_final,
        "Cd_ref": ref_cd,
        "ref_name": "laminar Re=1000",
        "error_pct": err_pct,
        "previous_6L_result": {
            "Cd_p": 0.0301,
            "Cd_f": 0.0299,
            "Cd_tot": 0.0600,
            "err_pct": 20.1,
        },
        "normal_method": "from_gradient",
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
    }

    print(
        f"{tag} DONE Cd_p={cd_p_final:.4f} Cd_f={cd_f_final:.4f} "
        f"Cd_tot={cd_tot_final:.4f} Cl={cl_final:.6f} "
        f"(ref={ref_cd:.4f}) err={err_pct:.1f}% time={elapsed:.0f}s",
        flush=True,
    )
    print(
        f"{tag} Previous 6L: Cd_tot=0.0600 err=20.1% → "
        f"12L: Cd_tot={cd_tot_final:.4f} err={err_pct:.1f}%",
        flush=True,
    )

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} Results written to {output_path}", flush=True)

    return result


def main():
    if len(sys.argv) < 3:
        print("Usage: python naca_large_domain_worker.py <device_id> <output_path>")
        sys.exit(1)

    device_id = int(sys.argv[1])
    output_path = sys.argv[2]

    run_naca_large_domain(device_id, output_path)


if __name__ == "__main__":
    main()
