#!/usr/bin/env python3
"""DTMB 5415 destroyer (simplified) drag benchmark — single card.

Uses the Series 60 hull (DTMB family, Cb=0.60) from ship_cad as a
simplified DTMB 5415 destroyer hull.  The DTMB 5415 is a standard
naval-architecture benchmark destroyer model; the Series 60 Cb=0.60
hull is the closest parametric form available in the ship_cad module.

Reference: Ct~0.042 (Blasius, same as SUBOFF at Re=1000)

Domain: L=80, nx=200, ny=80, nz=80
u_in=0.06, Re=1000, tau=3*0.06*80/1000+0.5=0.5144
5000 steps, MRT+Smagorinsky (Cs=0.05)
from_gradient normal
dpS = 0.5*u^2*pi*D*L (wetted area, D=draft)

Usage:
  PYTHONPATH=src python dtmb5415_simplified_worker.py <device_id> <output_path>
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
from tensorlbm.ship_cad import build_hull_mask, ShipHullType


def run_dtmb5415(device_id, output_path=None):
    """Run DTMB 5415 (simplified Series 60) ship hull drag on sdaa:<device_id>."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)
    tag = f"[SDAA:{device_id} DTMB5415-simplified]"

    # ---- Problem parameters (from task spec) ----
    L = 80.0
    nx, ny, nz = 200, 80, 80
    u_in = 0.06
    Re = 1000
    nu = u_in * L / Re          # = 0.0048
    tau = 3.0 * nu + 0.5       # = 0.5144
    n_steps = 5000
    warmup = 500
    cs_smag = 0.05

    # Reference: Blasius (same as SUBOFF at Re=1000)
    ct_ref = 0.042
    # ITTC-1957 for comparison
    cf_ittc = 0.075 / (math.log10(Re) - 2.0) ** 2

    print(
        f"{tag} nx={nx} ny={ny} nz={nz} L={L} u_in={u_in} "
        f"nu={nu:.6e} tau={tau:.6f} Cs={cs_smag} Re={Re:.0e}",
        flush=True,
    )
    print(f"{tag} Ct_ref(Blasius)={ct_ref:.6f}  Cf_ITTC={cf_ittc:.6f}",
          flush=True)

    t0 = time.time()

    # ---- Build Series 60 (DTMB) hull mask ----
    # Hull placement: bow at 30% of domain (room for wake)
    cx = nx * 0.30
    cy = ny * 0.5
    cz_keel = nz * 0.5 - 8   # keel slightly below center

    # Hull dimensions (lattice units)
    # DTMB 5415: L/B ~ 7.0, B/T ~ 3.0
    beam = ny * 0.20       # beam = 16
    draft = nz * 0.15      # draft = 12

    solid, stats = build_hull_mask(
        hull_type=ShipHullType.SERIES60,
        nx=nx, ny=ny, nz=nz,
        cx=cx, cy=cy,
        cz_keel=cz_keel,
        length=L,
        beam=beam,
        draft=draft,
        device=str(device),
    )
    n_solid = int(solid.sum().item())
    cb_num = stats.get("Cb_numerical", 0)
    print(f"{tag} solid cells: {n_solid}  Cb_numerical={cb_num:.4f} "
          f"(theoretical ~0.60)", flush=True)
    print(f"{tag} hull: L={L} B={beam} T={draft} cx={cx} cy={cy} "
          f"cz_keel={cz_keel}", flush=True)

    # ---- Wetted area normalization: dpS = 0.5 * u^2 * pi * D * L ----
    # D = draft (characteristic transverse dimension for wetted surface)
    D = draft
    dpS = 0.5 * 1.0 * u_in ** 2 * math.pi * D * L
    print(f"{tag} dpS = 0.5*u^2*pi*D*L = {dpS:.6e} (D=draft={D})", flush=True)

    # Near-wall mask and surface mesh (from_gradient normal)
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells: {n_near}", flush=True)

    mesh = SurfaceMesh.from_gradient(solid, near)

    # Normal stats
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

    # Solid mask for NoDynamics (19, nz, ny, nx)
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    # ---- Initialize flow field ----
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(
        rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device
    )
    initial_mass = float(rho0.sum().item())
    print(f"{tag} init done ({time.time() - t0:.1f}s), initial_mass={initial_mass}",
          flush=True)

    # ---- BC config: far-field on lateral faces (3D) ----
    bc_config = {
        'far_field_faces': ['y-', 'y+', 'z-', 'z+'],
        'periodic_faces': [],
    }

    # ---- Time history ----
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

        # 6. Far-field BC (all lateral faces)
        f = far_field_bc_3d(f, u_in=u_in, bc_config=bc_config)

        # 7. Mass correction every 200 steps
        if step % 200 == 0:
            f = correct_mass3d(f, initial_mass)

        # 8. Drag computation (after warmup)
        if step > warmup:
            fx_p, fy_p, fz_p = drag_pressure_integration(f, mesh, dpS)
            fx_f, fy_f, fz_f = drag_friction_integration(f, mesh, dpS, nu)
            cd_p_hist.append(fx_p)
            cd_f_hist.append(fx_f)
            cd_tot_hist.append(fx_p + fx_f)
            cl_hist.append(fy_p + fy_f)

        # Check for divergence
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
                cl_avg = sum(cl_hist[-n_avg:]) / n_avg
            else:
                cd_p_avg = cd_f_avg = cd_tot_avg = cl_avg = 0.0
            elapsed = time.time() - t0
            print(
                f"{tag} step={step} Cd_p={cd_p_avg:.6f} Cd_f={cd_f_avg:.6f} "
                f"Cd_tot={cd_tot_avg:.6f} (Blasius={ct_ref:.6f}, "
                f"ITTC={cf_ittc:.6f}) Cl={cl_avg:.6f} [{elapsed:.0f}s]",
                flush=True,
            )

    elapsed = time.time() - t0

    # ---- Final statistics: average over last 1000 samples ----
    win = min(1000, len(cd_tot_hist))
    cd_p_final = sum(cd_p_hist[-win:]) / max(win, 1)
    cd_f_final = sum(cd_f_hist[-win:]) / max(win, 1)
    cd_tot_final = sum(cd_tot_hist[-win:]) / max(win, 1)
    cl_final = sum(cl_hist[-win:]) / max(win, 1)

    err_blasius = abs(cd_tot_final - ct_ref) / ct_ref * 100 if ct_ref > 0 else float('nan')
    err_ittc = abs(cd_tot_final - cf_ittc) / cf_ittc * 100 if cf_ittc > 0 else float('nan')

    result = {
        "benchmark": "dtmb5415_simplified",
        "device": f"sdaa:{device_id}",
        "lattice": "D3Q19",
        "collision": "MRT+Smagorinsky",
        "C_s": cs_smag,
        "normal_method": "from_gradient",
        "hull_type": "Series60 (DTMB family, Cb=0.60)",
        "L": L,
        "nx": nx, "ny": ny, "nz": nz,
        "beam": beam,
        "draft": draft,
        "u_in": u_in,
        "Re": Re,
        "nu": nu,
        "tau": tau,
        "n_steps": n_steps,
        "warmup": warmup,
        "n_samples": len(cd_tot_hist),
        "D_wetted": D,
        "dpS": dpS,
        "Cb_numerical": cb_num,
        "Ct_ref_Blasius": ct_ref,
        "Cf_ITTC": cf_ittc,
        "Cd_pressure": cd_p_final,
        "Cd_friction": cd_f_final,
        "Cd_total": cd_tot_final,
        "Cl_total": cl_final,
        "error_vs_Blasius_pct": err_blasius,
        "error_vs_ITTC_pct": err_ittc,
        "finite": bool(torch.isfinite(f).all().item()),
        "wall_time_s": elapsed,
    }

    print(f"\n{'='*60}")
    print(f"{tag} FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  Cd_pressure  = {cd_p_final:.6f}")
    print(f"  Cd_friction  = {cd_f_final:.6f}")
    print(f"  Cd_total     = {cd_tot_final:.6f}")
    print(f"  Ct_ref(Blasius) = {ct_ref:.6f}  (err={err_blasius:.1f}%)")
    print(f"  Cf_ITTC      = {cf_ittc:.6f}  (err={err_ittc:.1f}%)")
    print(f"  Cl_total     = {cl_final:.6f}")
    print(f"  Cb_numerical = {cb_num:.4f}  (theoretical ~0.60)")
    print(f"  Wall time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Finite: {result['finite']}")

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"  Results saved to {output_path}")

    return result


if __name__ == "__main__":
    did = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    out = sys.argv[2] if len(sys.argv) > 2 else None
    run_dtmb5415(did, out)
