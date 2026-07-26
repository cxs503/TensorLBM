#!/usr/bin/env python3
"""NACA 4412 cambered airfoil at Re=1000, alpha=5 deg — single card.

Reference: Cl~0.4, Cd~0.05 (experimental, cambered + angle of attack)

Domain: chord=100, nx=600, ny=300, nz=4 (6L domain)
u_in=0.05, Re=1000, tau=3*0.05*100/1000+0.5=0.515
10000 steps, MRT+Smagorinsky (Cs=0.05)
Uses from_naca analytical normal (thickness-based).

NACA 4412: m=0.04 (4% camber), p=0.40 (camber position), t=0.12 (12% thick)
Mean camber line:
  0 <= x <= p:  yc = (m/p^2)*(2*p*x - x^2)
  p <= x <= 1:  yc = (m/(1-p)^2)*((1-2*p) + 2*p*x - x^2)
Surface:
  upper: y = yc + yt,  lower: y = yc - yt
  where yt = 5*0.12*(0.2969*sqrt(x) - 0.1260*x - 0.3516*x^2 + 0.2843*x^3 - 0.1015*x^4)

Angle of attack alpha=5 deg applied by rotating the airfoil coordinates.

Usage:
  PYTHONPATH=src python naca4412_cambered_worker.py <device_id> <output_path>
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
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


def naca4412_camber_line(xc):
    """NACA 4412 mean camber line y_c(x) for x in [0,1].

    m=0.04, p=0.40.
    """
    m = 0.04
    p = 0.40
    yc = np.where(
        xc < p,
        (m / p ** 2) * (2.0 * p * xc - xc ** 2),
        (m / (1.0 - p) ** 2) * ((1.0 - 2.0 * p) + 2.0 * p * xc - xc ** 2),
    )
    return yc


def naca_thickness(xc, t=0.12):
    """NACA 4-digit thickness distribution y_t(x) for x in [0,1]."""
    yt = 5.0 * t * (
        0.2969 * np.sqrt(xc)
        - 0.1260 * xc
        - 0.3516 * xc ** 2
        + 0.2843 * xc ** 3
        - 0.1015 * xc ** 4
    )
    return yt


def build_naca4412(chord, nx, ny, x_le, y_c_chord, alpha_deg, device):
    """Build NACA 4412 solid mask (2D extruded in z, nz=4) at angle of attack.

    The airfoil is built in body coordinates (chord along x, camber along y),
    then rotated by alpha_deg about the quarter-chord point.
    """
    nz = 4
    alpha_rad = math.radians(alpha_deg)
    cos_a = math.cos(alpha_rad)
    sin_a = math.sin(alpha_rad)

    # Quarter-chord point (rotation center) in lattice coords
    x_qc = x_le + 0.25 * chord
    y_qc = y_c_chord

    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)

    # Sample the airfoil surface at high resolution and rasterize
    n_samples = 2000
    xc = np.linspace(0.0, 1.0, n_samples)
    yc = naca4412_camber_line(xc)
    yt = naca_thickness(xc, t=0.12)

    # Upper and lower surface in body coordinates (lattice units)
    x_upper = x_le + xc * chord
    y_upper = y_c_chord + (yc + yt) * chord
    x_lower = x_le + xc * chord
    y_lower = y_c_chord + (yc - yt) * chord

    # Rotate by alpha about quarter-chord
    def rotate(xp, yp):
        dx = xp - x_qc
        dy = yp - y_qc
        xr = x_qc + cos_a * dx - sin_a * dy
        yr = y_qc + sin_a * dx + cos_a * dy
        return xr, yr

    x_upper_r, y_upper_r = rotate(x_upper, y_upper)
    x_lower_r, y_lower_r = rotate(x_lower, y_lower)

    # Build a polygon: upper surface TE->LE, lower surface LE->TE
    # Upper: from LE (xc=0) to TE (xc=1)
    # Lower: from TE (xc=1) to LE (xc=0)
    poly_x = np.concatenate([x_upper_r, x_lower_r[::-1]])
    poly_y = np.concatenate([y_upper_r, y_lower_r[::-1]])

    # Rasterize: for each x column, find y range inside polygon
    # Use the min/max y of the polygon at each x
    for k in range(nz):
        for i in range(nx):
            # Find polygon y-values at this x
            # Use linear interpolation along upper and lower surfaces
            xi = float(i)
            # Upper surface y at xi
            if xi < x_upper_r[0] or xi > x_upper_r[-1]:
                continue
            y_u = np.interp(xi, x_upper_r, y_upper_r)
            y_l = np.interp(xi, x_lower_r, y_lower_r)
            j_lo = max(0, int(math.floor(min(y_u, y_l))))
            j_hi = min(ny - 1, int(math.ceil(max(y_u, y_l))))
            if j_hi >= j_lo:
                solid[k, j_lo:j_hi + 1, i] = True

    return solid


def run_naca4412(device_id, output_path):
    """Run NACA 4412 cambered airfoil at Re=1000, alpha=5 deg."""
    chord = 100
    nx = 600   # 6 chord
    ny = 300   # 3 chord
    nz = 4
    u_in = 0.05
    Re = 1000
    nu = u_in * chord / Re  # 0.005
    tau = 3.0 * nu + 0.5    # 0.515
    cs_smag = 0.05
    n_steps = 10000
    alpha_deg = 5.0
    ref_cl = 0.4
    ref_cd = 0.05

    x_le = int(nx * 0.25)  # 1.5 chords from inlet
    y_c_chord = ny // 2     # chord centerline

    # dpS = 0.5 * u_in^2 * chord * nz (dynamic pressure x frontal area)
    dpS = 0.5 * u_in ** 2 * chord * nz

    tag = f"[SDAA:{device_id} NACA4412 Re=1000 a={alpha_deg}deg {nx}x{ny}x{nz}]"
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    print(
        f"{tag} chord={chord} nx={nx} ny={ny} nz={nz} "
        f"u_in={u_in} nu={nu:.6f} tau={tau:.6f} Cs={cs_smag} "
        f"alpha={alpha_deg}deg x_le={x_le} y_c={y_c_chord} dpS={dpS:.6f}",
        flush=True,
    )

    t0 = time.time()

    # Build NACA 4412 solid mask with camber + angle of attack
    solid = build_naca4412(chord, nx, ny, x_le, y_c_chord, alpha_deg, device)
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells: {n_solid}", flush=True)

    # Near-wall mask and surface mesh (from_naca analytical normal)
    near = get_near_wall_2d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells: {n_near}", flush=True)

    # Use from_naca with y_c = chord centerline
    # (analytical normal based on thickness derivative)
    mesh = SurfaceMesh.from_naca(solid, near, x_le, y_c_chord, chord)

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

        # 6. Far-field BC
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
        cl = fy_p + fy_f  # lift (non-zero due to camber + angle of attack)

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

    err_cd = abs(cd_tot_final - ref_cd) / ref_cd * 100
    err_cl = abs(cl_final - ref_cl) / ref_cl * 100 if ref_cl > 0 else float("nan")

    result = {
        "case": tag,
        "benchmark": "naca4412_cambered",
        "device": f"sdaa:{device_id}",
        "Re": Re,
        "chord": chord,
        "alpha_deg": alpha_deg,
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
        "y_c": y_c_chord,
        "Cd_pressure": cd_p_final,
        "Cd_friction": cd_f_final,
        "Cd_total": cd_tot_final,
        "Cl": cl_final,
        "Cd_ref": ref_cd,
        "Cl_ref": ref_cl,
        "ref_name": "experimental cambered+alpha",
        "error_cd_pct": err_cd,
        "error_cl_pct": err_cl,
        "normal_method": "from_naca",
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
    }

    print(
        f"{tag} DONE Cd_p={cd_p_final:.4f} Cd_f={cd_f_final:.4f} "
        f"Cd_tot={cd_tot_final:.4f} Cl={cl_final:.6f} "
        f"(ref Cd={ref_cd:.4f} Cl={ref_cl:.4f}) "
        f"err_Cd={err_cd:.1f}% err_Cl={err_cl:.1f}% time={elapsed:.0f}s",
        flush=True,
    )

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} Results written to {output_path}", flush=True)

    return result


def main():
    if len(sys.argv) < 3:
        print("Usage: python naca4412_cambered_worker.py <device_id> <output_path>")
        sys.exit(1)

    device_id = int(sys.argv[1])
    output_path = sys.argv[2]

    run_naca4412(device_id, output_path)


if __name__ == "__main__":
    main()
