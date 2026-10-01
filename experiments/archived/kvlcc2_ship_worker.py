#!/usr/bin/env python3
"""KVLCC2 ship hull drag benchmark worker.

Validates the LBM simulation against ITTC-1957 friction reference:
  Cf = 0.075 / (log10(Re) - 2)^2
  Ct_ref ≈ 0.0035 (task reference for Re=1e5)

Uses the verified main loop (NoDynamics + half-way BB + far-field BC)
with MRT+Smagorinsky (Cs=0.05) and unified pressure-integration drag
(SurfaceMesh.from_gradient normal).

Geometry: KVLCC2 VLCC tanker hull (Cb≈0.81) from ship_cad module.
  L=80, nx=300, ny=120, nz=120
  u_in=0.06, Re=1e5, tau=0.500144
  Wetted area normalization: dpS = 0.5 * u^2 * pi * D * L

Usage:
  python kvlcc2_ship_worker.py <device_id> <output_path>
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


def run_kvlcc2(device_id, output_path=None):
    """Run KVLCC2 ship hull drag benchmark on sdaa:<device_id>."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)
    tag = f"[SDAA:{device_id} KVLCC2]"

    # ---- Problem parameters (from task spec) ----
    L = 80.0
    nx, ny, nz = 300, 120, 120
    u_in = 0.06
    Re = 1e5
    nu = u_in * L / Re          # = 4.8e-5
    tau = 3.0 * nu + 0.5       # = 0.500144
    n_steps = 5000
    warmup = 1000
    cs_smag = 0.05

    # ITTC-1957 reference
    cf_ittc = 0.075 / (math.log10(Re) - 2.0) ** 2   # = 0.00833 at Re=1e5
    ct_ref_task = 0.0035   # task-specified reference

    print(
        f"{tag} nx={nx} ny={ny} nz={nz} L={L} u_in={u_in} "
        f"nu={nu:.6e} tau={tau:.6f} Cs={cs_smag} Re={Re:.0e}",
        flush=True,
    )
    print(f"{tag} Cf_ITTC={cf_ittc:.6f}  Ct_ref(task)={ct_ref_task:.6f}",
          flush=True)

    t0 = time.time()

    # ---- Build KVLCC2 hull mask ----
    # Use build_hull_mask with KVLCC2 hull type
    # Hull placement: bow at 25% of domain (room for wake)
    cx = nx * 0.30
    cy = ny * 0.5
    cz_keel = nz * 0.5 - 10   # keel below center to leave room above waterline

    # Hull dimensions (lattice units)
    beam = ny * 0.20       # beam = 24
    draft = nz * 0.15      # draft = 18

    solid, stats = build_hull_mask(
        hull_type=ShipHullType.KVLCC2,
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
          f"(theoretical ~0.81)", flush=True)
    print(f"{tag} hull: L={L} B={beam} T={draft} cx={cx} cy={cy} "
          f"cz_keel={cz_keel}", flush=True)

    # ---- Wetted area normalization: dpS = 0.5 * u^2 * pi * D * L ----
    # D = draft (characteristic transverse dimension for wetted surface)
    D = draft
    dpS = 0.5 * 1.0 * u_in ** 2 * math.pi * D * L
    print(f"{tag} dpS = 0.5*u^2*pi*D*L = {dpS:.6e} (D=draft={D})", flush=True)

    # Also compute voxel wetted surface area for reference
    # (count fluid cells adjacent to solid)
    fluid = ~solid
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells: {n_near}", flush=True)

    # ---- Precompute surface mesh (from_gradient) ----
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

    # ---- BC config: far-field on all lateral faces (3D) ----
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
                f"{tag} step={step} Ct_p={cd_p_avg:.6f} Ct_f={cd_f_avg:.6f} "
                f"Ct_tot={cd_tot_avg:.6f} (ITTC={cf_ittc:.6f}, "
                f"task_ref={ct_ref_task:.6f}) Cl={cl_avg:.6f} [{elapsed:.0f}s]",
                flush=True,
            )

    elapsed = time.time() - t0

    # ---- Final statistics: average over last 1000 samples ----
    win = min(1000, len(cd_tot_hist))
    ct_p_final = sum(cd_p_hist[-win:]) / max(win, 1)
    ct_f_final = sum(cd_f_hist[-win:]) / max(win, 1)
    ct_tot_final = sum(cd_tot_hist[-win:]) / max(win, 1)
    cl_final = sum(cl_hist[-win:]) / max(win, 1)

    err_ittc = abs(ct_tot_final - cf_ittc) / cf_ittc * 100 if cf_ittc > 0 else float('nan')
    err_task = abs(ct_tot_final - ct_ref_task) / ct_ref_task * 100 if ct_ref_task > 0 else float('nan')

    result = {
        "benchmark": "kvlcc2_ship_hull",
        "device": f"sdaa:{device_id}",
        "lattice": "D3Q19",
        "collision": "MRT+Smagorinsky",
        "C_s": cs_smag,
        "normal_method": "from_gradient",
        "hull_type": "KVLCC2",
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
        "Cf_ITTC": cf_ittc,
        "Ct_ref_task": ct_ref_task,
        "Ct_pressure": ct_p_final,
        "Ct_friction": ct_f_final,
        "Ct_total": ct_tot_final,
        "Cl_total": cl_final,
        "error_vs_ITTC_pct": err_ittc,
        "error_vs_task_ref_pct": err_task,
        "finite": bool(torch.isfinite(f).all().item()),
        "wall_time_s": elapsed,
    }

    print(f"\n{'='*60}")
    print(f"{tag} FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  Ct_pressure  = {ct_p_final:.6f}")
    print(f"  Ct_friction  = {ct_f_final:.6f}")
    print(f"  Ct_total     = {ct_tot_final:.6f}")
    print(f"  Cf_ITTC      = {cf_ittc:.6f}  (err={err_ittc:.1f}%)")
    print(f"  Ct_ref(task) = {ct_ref_task:.6f}  (err={err_task:.1f}%)")
    print(f"  Cl_total     = {cl_final:.6f}")
    print(f"  Cb_numerical = {cb_num:.4f}  (theoretical ~0.81)")
    print(f"  Wall time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Finite: {result['finite']}")

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"  Results saved to {output_path}")

    return result


if __name__ == "__main__":
    did = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    out = sys.argv[2] if len(sys.argv) > 2 else None
    run_kvlcc2(did, out)
