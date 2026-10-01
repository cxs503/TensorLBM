"""Grid convergence test: wall-surface BFL+ME vs BB+pressure on cylinder Re=200.

Method 1 (SDAA even): Standard BB + pressure integration (after stream)
Method 2 (SDAA odd):  Wall-surface BFL + momentum exchange (after stream)

Main loop:
  collide -> NoDynamics -> stream -> far_field_bc -> BB/BFL(after stream) -> compute drag -> correct_mass

Usage:
  PYTHONPATH=src python grid_conv_bfl_worker.py <device_id> <method> <D> <output_json>
  method: 1 = BB+pressure, 2 = BFL+ME
  D: 24, 48, 96, 200
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from tensorlbm.d3q19 import C, OPPOSITE, W, equilibrium3d, macroscopic3d
from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d
from tensorlbm.bfl_d3q19 import compute_q_cylinder_d3q19
from tensorlbm.wall_surface_bfl import (
    bouzidi_bounce_back_wallsurface,
    drag_momentum_exchange_bfl,
)
from tensorlbm.solver3d import stream3d, correct_mass3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.drag_pressure import (
    SurfaceMesh, drag_total, get_near_wall_2d,
)


GRID_PARAMS = {
    24:  {"nx": 200,  "ny": 80,  "nz": 4, "n_steps": 3000, "warmup": 300},
    48:  {"nx": 400,  "ny": 160, "nz": 4, "n_steps": 3000, "warmup": 300},
    96:  {"nx": 800,  "ny": 320, "nz": 4, "n_steps": 3000, "warmup": 300},
    200: {"nx": 1660, "ny": 660, "nz": 4, "n_steps": 2000, "warmup": 200},
}


def build_cylinder_mask(nx, ny, nz, cx, cy, radius, device):
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    solid = circle.unsqueeze(0).expand(nz, ny, nx).clone()
    return solid


def run(device_id, method, D, output_path):
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    params = GRID_PARAMS[D]
    nx, ny, nz = params["nx"], params["ny"], params["nz"]
    n_steps = params["n_steps"]
    warmup = params["warmup"]

    diameter = float(D)
    radius = diameter / 2.0
    u_in = 0.08
    Re = 200.0
    nu = u_in * diameter / Re
    tau = 3.0 * nu + 0.5
    cs_smag = 0.05

    cx_c = nx * 0.25
    cy_c = ny * 0.5

    A_frontal = diameter * nz
    dpS = 0.5 * 1.0 * u_in ** 2 * A_frontal

    method_name = "BB+pressure" if method == 1 else "BFL+ME"
    tag = f"[D={D} {method_name} SDAA:{device_id}]"
    print(f"{tag} nx={nx} ny={ny} nz={nz} D={diameter} u_in={u_in} "
          f"nu={nu:.6e} tau={tau:.6f} Cs={cs_smag} dpS={dpS:.4f}", flush=True)

    t0 = time.time()

    solid = build_cylinder_mask(nx, ny, nz, cx_c, cy_c, radius, device)
    n_solid = int(solid.sum().item())
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    near = get_near_wall_2d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} solid cells={n_solid} near-wall cells={n_near}", flush=True)

    fbm = None
    qf = None
    if method == 2:
        fbm, qf = compute_q_cylinder_d3q19(nx, ny, nz, cx_c, cy_c, radius, device)
        n_links = int(fbm.sum().item())
        q_min = float(qf[fbm].min().item())
        q_max = float(qf[fbm].max().item())
        q_mean = float(qf[fbm].mean().item())
        print(f"{tag} BFL: {n_links} links, q=[{q_min:.4f},{q_max:.4f}], mean={q_mean:.4f}", flush=True)

    mesh = None
    if method == 1:
        mesh = SurfaceMesh.from_cylinder(solid, near, cx_c, cy_c, radius)

    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    initial_mass = float(rho0.sum().item())

    print(f"{tag} init done ({time.time()-t0:.1f}s), starting loop...", flush=True)

    cd_hist = []

    for step in range(1, n_steps + 1):
        # 1. Save pre-collision
        f_pre = f.clone()

        # 2. Collision (MRT + Smagorinsky)
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)

        # 3. NoDynamics: restore solid cells
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        # 4. Save post-collision (pre-stream) for BFL
        f_post_coll = f.clone()

        # 5. Stream
        f = stream3d(f)

        # 6. Far-field BC
        f = far_field_bc_3d(f, u_in=u_in)

        # 7. BB/BFL (after stream)
        if method == 1:
            f = bounce_back_cells_3d(f, solid)
            cd_tot, cd_p, cd_f = drag_total(f, mesh, dpS, nu)
            cd_val = cd_tot
        else:
            f = bouzidi_bounce_back_wallsurface(f, f_post_coll, fbm, qf)
            cd_val = drag_momentum_exchange_bfl(f, f_post_coll, fbm, qf, dpS, use_q_scaling=False)

        # 8. Mass correction
        if step % 200 == 0:
            f = correct_mass3d(f, initial_mass)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step > warmup:
            if math.isfinite(cd_val):
                cd_hist.append(cd_val)

        if step % 200 == 0:
            _, ux, _, _ = macroscopic3d(f)
            ms = float(torch.sqrt(ux * ux).max().item())
            elapsed = time.time() - t0
            cd_avg = sum(cd_hist) / max(len(cd_hist), 1)
            print(f"{tag} step={step} Cd={cd_avg:.4f} "
                  f"max|ux|={ms:.4f} ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    ref_cd = 1.30

    cd_mean = sum(cd_hist) / max(len(cd_hist), 1) if cd_hist else float("nan")

    def _std(vals):
        if len(vals) < 2:
            return 0.0
        m = sum(vals) / len(vals)
        return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))

    cd_std = _std(cd_hist)
    err = abs(cd_mean - ref_cd) / ref_cd * 100 if math.isfinite(cd_mean) else float("nan")

    print(f"\n{tag} === FINAL RESULTS ===", flush=True)
    print(f"{tag} Cd = {cd_mean:.4f} +/- {cd_std:.4f}  (err={err:.2f}%)", flush=True)
    print(f"{tag} ref Cd = {ref_cd}", flush=True)
    print(f"{tag} time = {elapsed:.0f}s", flush=True)

    result = {
        "method": method_name,
        "method_id": method,
        "device": f"sdaa:{device_id}",
        "grid": f"{nx}x{ny}x{nz}",
        "D": D,
        "diameter": diameter,
        "Re": Re,
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "Cs": cs_smag,
        "n_steps": n_steps,
        "warmup": warmup,
        "Cd_mean": cd_mean,
        "Cd_std": cd_std,
        "Cd_err_pct": err,
        "Cd_ref": ref_cd,
        "n_samples": len(cd_hist),
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
    }
    Path(output_path).write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    device_id = int(sys.argv[1])
    method = int(sys.argv[2])
    D = int(sys.argv[3])
    output_path = sys.argv[4]
    run(device_id, method, D, output_path)
