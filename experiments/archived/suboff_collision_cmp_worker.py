#!/usr/bin/env python3
"""SUBOFF bare-hull collision-model comparison — Re=1000.

Runs a single collision model on SUBOFF bare hull and writes results JSON.
Launched 4× in parallel (one per SDAA card) to compare:
  BGK, MRT, MRT+Smagorinsky, Cumulant D3Q19.

Parameters (fixed across all models):
  L=80, nx=200, ny=80, nz=80, u_in=0.06, Re=1000, tau=0.5144
  dpS = 0.5*u_in^2 * pi * D * L   (WETTED area, D=2*R_max=9.335)
  Reference: Cf = 1.328/sqrt(Re) = 0.042  (Blasius laminar flat plate)
  5000 steps, averaging window = 500

Usage:
  python suboff_collision_cmp_worker.py <model> <device_id> <output_path>
  model: bgk | mrt | mrt_smag | cumulant
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch

from tensorlbm import build_suboff_mask
from tensorlbm.boundaries3d import far_field_bc_3d, bounce_back_cells_3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import (
    stream3d,
    correct_mass3d,
    collide_bgk3d,
    collide_mrt3d,
)
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.cumulant import collide_cumulant_d3q19
from tensorlbm.drag_pressure import (
    get_near_wall_3d,
    SurfaceMesh,
    drag_pressure_integration,
    drag_friction_integration,
)

MODELS = {
    "bgk":      ("BGK",              lambda f, tau: collide_bgk3d(f, tau=tau)),
    "mrt":      ("MRT",              lambda f, tau: collide_mrt3d(f, tau=tau)),
    "mrt_smag": ("MRT+Smag(Cs=0.05)", lambda f, tau: collide_smagorinsky_mrt3d(f, tau=tau, C_s=0.05)),
    "cumulant": ("Cumulant D3Q19",   lambda f, tau: collide_cumulant_d3q19(f, tau=tau)),
}


def run(model_key, device_id, output_path=None):
    model_name, collide_fn = MODELS[model_key]
    tag = f"[SDAA:{device_id} {model_name}]"

    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    # ---- Fixed simulation parameters ----
    L = 80.0
    nx, ny, nz = 200, 80, 80
    u_in = 0.06
    Re = 1000
    tau = 0.5144
    n_steps = 5000
    win = 500  # averaging window for final stats

    nu = u_in * L / Re  # 0.0048

    # Geometry: SUBOFF bare hull, default placement (centre)
    solid, meta = build_suboff_mask(
        hull_type="bare_hull",
        nx=nx, ny=ny, nz=nz,
        length=L,
        device=device,
    )
    radius = meta["radius"]
    D = 2.0 * radius
    n_solid = int(solid.sum().item())

    # Wetted-area dynamic pressure scale
    dpS = 0.5 * u_in ** 2 * math.pi * D * L

    # Reference: Blasius laminar flat-plate skin-friction coefficient
    Cf_ref = 1.328 / math.sqrt(Re)

    print(
        f"{tag} nx={nx} ny={ny} nz={nz} L={L} R={radius:.3f} D={D:.3f} "
        f"u_in={u_in} nu={nu:.6e} tau={tau:.6f} "
        f"dpS={dpS:.6e} Cf_ref={Cf_ref:.6f} n_solid={n_solid}",
        flush=True,
    )

    t0 = time.time()

    # Near-wall mask + surface mesh (gradient-based normals, dA=1.0)
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    mesh = SurfaceMesh.from_gradient(solid, near)
    print(f"{tag} near-wall cells: {n_near}", flush=True)

    # NoDynamics solid mask (19, nz, ny, nx)
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    # Initialise: uniform inflow, zero inside hull
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    im = float(rho0.sum().item())

    print(f"{tag} init done ({time.time() - t0:.1f}s), initial_mass={im}", flush=True)

    # History buffers
    cd_p_hist, cd_f_hist, cd_tot_hist = [], [], []

    step_done = 0
    for step in range(1, n_steps + 1):
        # 1. Save pre-collision state
        f_pre = f.clone()

        # 2. Collision (model-specific)
        f = collide_fn(f, tau)

        # 3. NoDynamics: restore solid cells to pre-collision values
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        # 4. Half-way bounce-back (before streaming)
        f = bounce_back_cells_3d(f, solid)

        # 5. Streaming
        f = stream3d(f)

        # 6. Far-field BC
        f = far_field_bc_3d(f, u_in)

        # 7. Mass correction every 200 steps
        if step % 200 == 0:
            f = correct_mass3d(f, im)

        # 8. Drag computation
        fx_p, _, _ = drag_pressure_integration(f, mesh, dpS)
        fx_f, _, _ = drag_friction_integration(f, mesh, dpS, nu)
        cd_p_hist.append(fx_p)
        cd_f_hist.append(fx_f)
        cd_tot_hist.append(fx_p + fx_f)
        step_done = step

        # Divergence guard
        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step % 500 == 0:
            n_avg = min(500, len(cd_tot_hist))
            print(
                f"{tag} step={step} Cd_p={sum(cd_p_hist[-n_avg:]) / n_avg:.6f} "
                f"Cd_f={sum(cd_f_hist[-n_avg:]) / n_avg:.6f} "
                f"Cd_tot={sum(cd_tot_hist[-n_avg:]) / n_avg:.6f} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )

    elapsed = time.time() - t0
    time_per_step = elapsed / max(step_done, 1)

    # Final averages (last `win` steps)
    n_final = min(win, len(cd_tot_hist))
    cd_p_final = sum(cd_p_hist[-n_final:]) / n_final
    cd_f_final = sum(cd_f_hist[-n_final:]) / n_final
    cd_tot_final = sum(cd_tot_hist[-n_final:]) / n_final

    # Cf = Cd_total (since dpS is wetted-area based, Cd_tot IS the skin-friction coeff)
    Cf_num = cd_tot_final
    err_pct = abs(Cf_num - Cf_ref) / Cf_ref * 100 if Cf_ref > 0 else float("nan")

    result = {
        "model_key": model_key,
        "model_name": model_name,
        "device": f"sdaa:{device_id}",
        "Re": Re,
        "L": L,
        "D": D,
        "radius": radius,
        "grid": f"{nx}x{ny}x{nz}",
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "n_steps": n_steps,
        "steps_completed": step_done,
        "n_solid": n_solid,
        "n_near": n_near,
        "dpS": dpS,
        "dpS_formula": "0.5*u_in^2*pi*D*L (wetted area)",
        "Cd_pressure": cd_p_final,
        "Cd_friction": cd_f_final,
        "Cd_total": cd_tot_final,
        "Cf_numerical": Cf_num,
        "Cf_ref": Cf_ref,
        "ref_name": "Blasius laminar flat plate Cf=1.328/sqrt(Re)",
        "error_pct": err_pct,
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
        "time_per_step_ms": time_per_step * 1000.0,
        "avg_window": win,
    }

    print(
        f"{tag} DONE Cd_p={cd_p_final:.6f} Cd_f={cd_f_final:.6f} "
        f"Cd_tot={cd_tot_final:.6f} Cf={Cf_num:.6f} "
        f"(ref Cf={Cf_ref:.6f}) err={err_pct:.1f}% "
        f"time={elapsed:.0f}s ({time_per_step * 1000:.1f}ms/step)",
        flush=True,
    )

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} Results written to {output_path}", flush=True)

    return result


def main():
    if len(sys.argv) < 3:
        print("Usage: python suboff_collision_cmp_worker.py <model> <device_id> [output_path]")
        print("  model: bgk | mrt | mrt_smag | cumulant")
        sys.exit(1)

    model_key = sys.argv[1]
    device_id = int(sys.argv[2])
    output_path = sys.argv[3] if len(sys.argv) > 3 else None

    if model_key not in MODELS:
        print(f"Unknown model: {model_key}. Choose from {list(MODELS.keys())}")
        sys.exit(1)

    run(model_key, device_id, output_path)


if __name__ == "__main__":
    main()
