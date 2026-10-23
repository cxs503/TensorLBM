"""Cylinder wall-function benchmark worker — tests Cd at Re=100/200/500.

D3Q19 MRT + Smagorinsky Cs=0.05 + wall_function_3d + farfield BC.
2D cylinder extruded to 3D thin span: D=24 cells, domain 200×80×4.
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.wall_model import wall_function_3d


def build_cylinder_mask(nx, ny, nz, cx, cy, cz, radius, device):
    """Boolean solid mask for a cylinder extruded along z-axis.
    
    Cylinder cross-section: (x-cx)^2 + (y-cy)^2 <= radius^2, for all z.
    """
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2  # (ny, nx)
    solid = circle.unsqueeze(0).expand(nz, ny, nx).clone()    # (nz, ny, nx)
    return solid


def compute_drag_forces(f, solid, u_in, diameter, nz):
    """Compute Cd from integrated wall shear (friction) and pressure (form).

    Uses the same method as wall_function_3d's returned drag values,
    but also computes total Cd directly from the momentum exchange for comparison.
    """
    from tensorlbm.d3q19 import C as C3D

    device = f.device
    c_t = C3D.to(device).float()
    cx = c_t[:, 0].view(19, 1, 1, 1)
    # Momentum exchange: Fx = sum over solid-adjacent fluid cells
    # Using direct force: sum over all cells of 2 * f_i * cx_i * solid
    fx_me = 2.0 * (cx * f * solid.unsqueeze(0).float()).sum().item()

    A_frontal = diameter * nz  # Projected frontal area
    dyn_p = 0.5 * 1.0 * u_in ** 2 * A_frontal

    cd_me = -fx_me / dyn_p if dyn_p > 0 else float("nan")
    return cd_me


def main():
    device_id = int(sys.argv[1])
    re = float(sys.argv[2])
    n_steps = int(sys.argv[3])
    warmup = int(sys.argv[4])
    output_path = sys.argv[5]

    # Fixed parameters
    nx, ny, nz = 200, 80, 4
    diameter = 24.0
    radius = diameter / 2.0
    u_in = 0.08
    cs_smag = 0.05

    nu = u_in * diameter / re
    tau = 3.0 * nu + 0.5

    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    tag = f"[SDAA:{device_id} Re={int(re)}]"
    print(f"{tag} nx={nx} ny={ny} nz={nz} D={diameter} u_in={u_in} nu={nu:.6e} tau={tau:.6f} Cs={cs_smag}",
          flush=True)

    t0 = time.time()

    # Build cylinder mask
    cx_cyl = nx * 0.25  # quarter from inlet
    cy_cyl = ny * 0.5   # centered vertically
    cz_cyl = nz * 0.5   # centered in span
    solid = build_cylinder_mask(nx, ny, nz, cx_cyl, cy_cyl, cz_cyl, radius, device)

    A_frontal = diameter * nz
    dyn_p = 0.5 * 1.0 * u_in ** 2 * A_frontal

    # Initialize flow field
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    initial_mass = float(torch.ones_like(rho0).sum().item())

    print(f"{tag} init done ({time.time() - t0:.1f}s)", flush=True)

    # Accumulators for running average after warmup
    cd_hist = []  # per-step Cd from wall function

    for step in range(1, n_steps + 1):
        # 1. Collision: MRT + Smagorinsky LES
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)

        # 2. Stream
        f = stream3d(f)

        # 3. Wall function (body force + drag computation)
        f, drag_fric, drag_pres = wall_function_3d(f, solid, nu, y_val=0.5)

        # 4. Far-field BC
        f = far_field_bc_3d(f, u_in=u_in)

        # 5. Mass correction every 100 steps
        if step % 100 == 0:
            f = correct_mass3d(f, initial_mass)

        # Compute Cd
        cd_fric = drag_fric / dyn_p if dyn_p > 0 else 0.0
        cd_pres = drag_pres / dyn_p if dyn_p > 0 else 0.0
        cd_total = cd_fric + cd_pres

        if step > warmup and math.isfinite(cd_total):
            cd_hist.append(cd_total)

        # Check for divergence
        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step % 200 == 0:
            cd_avg = sum(cd_hist) / max(len(cd_hist), 1) if cd_hist else float("nan")
            elapsed = time.time() - t0
            print(f"{tag} step={step} Cd={cd_total:.4f} Cd_avg={cd_avg:.4f} ({elapsed:.0f}s)",
                  flush=True)

    elapsed = time.time() - t0

    # Final results
    cd_mean = sum(cd_hist) / max(len(cd_hist), 1) if cd_hist else float("nan")
    cd_std = (sum((c - cd_mean) ** 2 for c in cd_hist) / max(len(cd_hist) - 1, 1)) ** 0.5 if len(cd_hist) > 1 else 0.0

    # Reference Cd for 2D cylinder
    ref_cd = {100: 1.35, 200: 1.30, 500: 1.20}
    ref = ref_cd.get(int(re), float("nan"))
    err_pct = abs(cd_mean - ref) / ref * 100 if ref > 0 and math.isfinite(cd_mean) else float("nan")

    result = {
        "case": f"cylinder_Re{int(re)}",
        "device": f"sdaa:{device_id}",
        "lattice": "D3Q19",
        "collision": f"MRT+Smag(Cs={cs_smag})",
        "boundary": "wall_function_3d+farfield",
        "grid": f"{nx}x{ny}x{nz}",
        "diameter": diameter,
        "radius": radius,
        "u_in": u_in,
        "Re": re,
        "nu": nu,
        "tau": tau,
        "n_steps": n_steps,
        "warmup": warmup,
        "Cd_mean": cd_mean,
        "Cd_std": cd_std,
        "Cd_ref": ref,
        "error_pct": err_pct,
        "cd_samples": len(cd_hist),
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
    }

    print(f"{tag} DONE Cd={cd_mean:.4f} (ref={ref}) err={err_pct:.1f}% time={elapsed:.0f}s",
          flush=True)
    Path(output_path).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
