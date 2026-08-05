#!/usr/bin/env python3
"""BFL cylinder Re=200 2nd-order — SDAA:15.

BFL with 2nd-order Lagrange friction formula:
  τ = ν·(3·u₁ - u₂/3) / (2·q)

D=48, nx=600, ny=240, nz=4 (20% blockage).
MRT+Smag(0.05), 5000 steps, from_cylinder.
BB baseline: Cd=1.220 (6.1%).
Previous BFL: 16.8% vs BB (1st-order).
Target: <10% vs BB with 2nd-order.
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
import torch_sdaa  # noqa: F401

from tensorlbm.boundaries3d import far_field_bc_3d, bounce_back_cells_3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.drag_pressure import (
    get_near_wall_2d,
    SurfaceMesh,
    drag_pressure_integration,
    drag_friction_integration,
)
from tensorlbm.bfl_d3q19_vec import bouzidi_bounce_back_d3q19_vec
from tensorlbm.bfl_common import compute_q_cylinder_common, compute_q_wall_cylinder


def build_cylinder_mask(nx, ny, nz, cx, cy, radius, device):
    """Boolean solid mask for a cylinder extruded along z-axis."""
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    solid = circle.unsqueeze(0).expand(nz, ny, nx).clone()
    return solid


def run(device_id, mode, output_path=None):
    """Cylinder Re=200, D=48. mode: 'bfl_2nd' or 'bb'."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    D = 48.0
    nx, ny, nz = 600, 240, 4
    Re = 200.0
    u_in = 0.08
    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5
    n_steps = 5000
    warmup = 1250  # 25% warmup
    Cd_ref = 1.30  # experimental reference
    Cd_bb_ref = 1.220  # BB baseline from previous run
    cs_smag = 0.05
    radius = D / 2.0
    cx_c = nx * 0.25
    cy_c = ny * 0.5
    A_frontal = D * nz
    dpS = 0.5 * u_in ** 2 * A_frontal

    use_bfl = mode == "bfl_2nd"
    tag = f"[Cyl-Re200-{mode} SDAA:{device_id}]"
    print(
        f"{tag} mode={mode} use_bfl={use_bfl} "
        f"nx={nx} ny={ny} nz={nz} D={D} R={radius:.4f} "
        f"u_in={u_in} nu={nu:.6e} tau={tau:.6f} Cs={cs_smag} "
        f"dpS={dpS:.6e} Cd_ref={Cd_ref} Cd_bb={Cd_bb_ref} "
        f"blockage={D/ny*100:.1f}%",
        flush=True,
    )

    t0 = time.time()
    solid = build_cylinder_mask(nx, ny, nz, cx_c, cy_c, radius, device)
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells: {n_solid}", flush=True)

    near = get_near_wall_2d(solid, axis="z")
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells: {n_near}", flush=True)

    mesh = SurfaceMesh.from_cylinder(solid, near, cx_c, cy_c, radius, axis="z")

    # BFL q-values (analytical cylinder)
    bfl_mask = None
    bfl_q = None
    q_wall = None
    bfl_stats = {}
    if use_bfl:
        print(f"{tag} computing BFL q-values (analytical cylinder)...", flush=True)
        t_q = time.time()
        bfl_mask, bfl_q = compute_q_cylinder_common(
            nx, ny, nz, cx_c, cy_c, radius, device, axis="z", lattice="D3Q19"
        )
        n_links = int(bfl_mask.sum().item())
        q_at_boundary = bfl_q[bfl_mask]
        bfl_stats = {
            "n_links": n_links,
            "q_min": float(q_at_boundary.min()) if n_links > 0 else None,
            "q_max": float(q_at_boundary.max()) if n_links > 0 else None,
            "q_mean": float(q_at_boundary.mean()) if n_links > 0 else None,
        }
        print(
            f"{tag} BFL q-field: {n_links} links ({time.time()-t_q:.1f}s) "
            f"q=[{bfl_stats['q_min']:.4f}, {bfl_stats['q_max']:.4f}] "
            f"mean={bfl_stats['q_mean']:.4f}",
            flush=True,
        )

        # Analytical normal wall distance for friction formula
        print(f"{tag} computing analytical q_wall (normal distance)...", flush=True)
        q_wall = compute_q_wall_cylinder(
            near, cx_c, cy_c, radius, device, axis="z"
        )
        q_at_near = q_wall[near]
        print(
            f"  q_wall: n_near={n_near} "
            f"q_min={float(q_at_near.min()):.4f} q_max={float(q_at_near.max()):.4f} "
            f"q_mean={float(q_at_near.mean()):.4f}",
            flush=True,
        )

    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(
        rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device
    )
    im = float(rho0.sum().item())
    print(f"{tag} init done ({time.time() - t0:.1f}s), initial_mass={im}", flush=True)

    cd_p_hist, cd_f_hist, cd_tot_hist, cl_hist = [], [], [], []
    bc_config = {"far_field_faces": ["y-", "y+"], "periodic_faces": ["z-", "z+"]}
    far_field_fn = functools.partial(far_field_bc_3d, bc_config=bc_config)

    friction_formula = "bfl_lagrange" if use_bfl else "standard"

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)
        # NoDynamics: restore solid cells to pre-collision
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        if use_bfl:
            f_pre_stream = f.clone()
            f = stream3d(f)
            f = far_field_bc_3d(f, u_in, bc_config=bc_config)
            f = bouzidi_bounce_back_d3q19_vec(f, f_pre_stream, bfl_mask, bfl_q)
        else:
            f = bounce_back_cells_3d(f, solid)
            f = stream3d(f)
            f = far_field_bc_3d(f, u_in, bc_config=bc_config)

        if step % 200 == 0:
            f = correct_mass3d(f, im)

        # Drag measurement
        fx_p, fy_p, _ = drag_pressure_integration(f, mesh, dpS)
        fx_f, fy_f, _ = drag_friction_integration(
            f, mesh, dpS, nu, q_wall=q_wall, formula=friction_formula
        )
        cd_tot = fx_p + fx_f
        cl = fy_p + fy_f

        if step > warmup:
            if math.isfinite(cd_tot):
                cd_p_hist.append(fx_p)
                cd_f_hist.append(fx_f)
                cd_tot_hist.append(cd_tot)
                cl_hist.append(cl)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step % 500 == 0:
            n_avg = min(500, len(cd_tot_hist))
            if n_avg > 0:
                print(
                    f"{tag} step={step} Cd_p={sum(cd_p_hist[-n_avg:])/n_avg:.6f} "
                    f"Cd_f={sum(cd_f_hist[-n_avg:])/n_avg:.6f} "
                    f"Cd_tot={sum(cd_tot_hist[-n_avg:])/n_avg:.6f} "
                    f"Cl={sum(cl_hist[-n_avg:])/n_avg:.6f} "
                    f"({time.time()-t0:.0f}s)",
                    flush=True,
                )
            else:
                print(f"{tag} step={step} (warmup, {time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0
    n_final = max(1, len(cd_tot_hist))
    cd_p_final = sum(cd_p_hist) / n_final if cd_p_hist else float("nan")
    cd_f_final = sum(cd_f_hist) / n_final if cd_f_hist else float("nan")
    cd_tot_final = sum(cd_tot_hist) / n_final if cd_tot_hist else float("nan")
    cl_final = sum(cl_hist) / n_final if cl_hist else float("nan")

    err_vs_exp = abs(cd_tot_final - Cd_ref) / Cd_ref * 100 if Cd_ref > 0 and math.isfinite(cd_tot_final) else float("nan")
    err_vs_bb = abs(cd_tot_final - Cd_bb_ref) / Cd_bb_ref * 100 if Cd_bb_ref > 0 and math.isfinite(cd_tot_final) else float("nan")

    result = {
        "case": tag,
        "mode": mode,
        "device": f"sdaa:{device_id}",
        "Re": Re,
        "D": D,
        "grid": f"{nx}x{ny}x{nz}",
        "blockage_pct": D / ny * 100,
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "Cs": cs_smag,
        "n_steps": n_steps,
        "warmup": warmup,
        "n_solid": n_solid,
        "n_near": n_near,
        "dpS": dpS,
        "Cd_pressure": cd_p_final,
        "Cd_friction": cd_f_final,
        "Cd_total": cd_tot_final,
        "Cl": cl_final,
        "Cd_ref_exp": Cd_ref,
        "Cd_ref_bb": Cd_bb_ref,
        "error_pct_vs_exp": err_vs_exp,
        "error_pct_vs_bb": err_vs_bb,
        "bfl_stats": bfl_stats,
        "friction_formula": friction_formula,
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
    }

    print(
        f"{tag} DONE mode={mode} Cd_p={cd_p_final:.6f} Cd_f={cd_f_final:.6f} "
        f"Cd_tot={cd_tot_final:.6f} Cl={cl_final:.6f} "
        f"(exp Cd={Cd_ref}) err_exp={err_vs_exp:.1f}% "
        f"(bb Cd={Cd_bb_ref}) err_bb={err_vs_bb:.1f}% "
        f"time={elapsed:.0f}s",
        flush=True,
    )

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} Results written to {output_path}", flush=True)

    return result


def main():
    if len(sys.argv) < 3:
        print("Usage: python _bfl_cyl_2nd_order_worker.py <mode> <device_id> [output_path]")
        print("  mode: bfl_2nd | bb")
        sys.exit(1)

    mode = sys.argv[1]
    device_id = int(sys.argv[2])
    output_path = sys.argv[3] if len(sys.argv) > 3 else None

    run(device_id, mode, output_path)


if __name__ == "__main__":
    main()
