#!/usr/bin/env python3
"""Thermal heated cylinder — SDAA:13.

Custom worker (avoids lbm_step_correct which crashes on SDAA).
Cylinder Re=200, D=48, T_cyl=1, Pr=0.71.
5000 steps.
Reference: Nu≈6.5 (Kruger 2017).
Target: Nu > 4 (<40%).
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch_sdaa  # noqa: F401

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_bgk3d, stream3d, correct_mass3d
from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d
from tensorlbm.thermal_common import (
    thermal_equilibrium_3d,
    thermal_collide_bgk_3d,
    thermal_stream_3d,
    thermal_macroscopic_3d,
    thermal_fixed_temp_mask_3d,
    thermal_dirichlet_wall_3d,
    nusselt_cylinder_3d,
)


def main():
    device_id = int(sys.argv[1]) if len(sys.argv) > 1 else 13
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    tag = f"[HeatedCylinder SDAA:{device_id}]"

    torch.sdaa.set_device(device_id)
    device = torch.device(f"sdaa:{device_id}")

    # Parameters
    D = 48.0
    R = D / 2.0
    nx, ny, nz = 200, 100, 4
    Re = 200.0
    Pr = 0.71
    u_in = 0.08
    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5
    tau_T = 0.8
    alpha = (tau_T - 0.5) / 3.0
    T_cyl, T_inf = 1.0, 0.0
    n_steps = 5000

    # Cylinder position (ensure it fits in domain)
    cx = max(nx * 0.25, R + 5)
    cy = ny * 0.5

    print(f"{tag} Starting: heated cylinder Re=200, D={D}, Pr={Pr}, "
          f"{nx}x{ny}x{nz}, {n_steps} steps", flush=True)
    print(f"{tag} cx={cx}, cy={cy}, R={R}, u_in={u_in}, nu={nu:.6e}, "
          f"tau={tau:.6f}, tau_T={tau_T}", flush=True)

    t0 = time.time()

    # Build cylinder mask
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= R ** 2
    solid = circle.unsqueeze(0).expand(nz, ny, nx).clone()
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells: {n_solid}", flush=True)

    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    # Initialize momentum field
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(
        rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device
    )
    im = float(rho0.sum().item())

    # Initialize thermal field
    T0 = torch.full((nz, ny, nx), T_inf, device=device)
    T0[solid] = T_cyl
    g = thermal_equilibrium_3d(
        T0, torch.zeros_like(T0), torch.zeros_like(T0), torch.zeros_like(T0)
    )

    bc_config = {"far_field_faces": ["y-", "y+"], "periodic_faces": ["z-", "z+"]}
    nu_hist = []

    print(f"{tag} init done ({time.time() - t0:.1f}s), initial_mass={im}", flush=True)

    for step in range(n_steps):
        # --- Momentum step (manual, avoids lbm_step_correct) ---
        f_pre = f.clone()
        f = collide_bgk3d(f, tau)
        if step == 0: print(f"{tag} step0: collide OK", flush=True)
        # NoDynamics: restore solid cells
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        if step == 0: print(f"{tag} step0: NoDynamics OK", flush=True)
        # Bounce-back BEFORE streaming (half-way BB)
        f = bounce_back_cells_3d(f, solid)
        if step == 0: print(f"{tag} step0: bounce_back OK", flush=True)
        # Streaming
        f = stream3d(f)
        if step == 0: print(f"{tag} step0: stream OK", flush=True)
        # Far-field BC
        f = far_field_bc_3d(f, u_in, bc_config=bc_config)
        if step == 0: print(f"{tag} step0: far_field OK", flush=True)
        # Mass correction
        if step % 200 == 0:
            f = correct_mass3d(f, im)
        if step == 0: print(f"{tag} step0: mass_correct OK", flush=True)

        # --- Thermal step ---
        rho, ux, uy, uz = macroscopic3d(f)
        if step == 0: print(f"{tag} step0: macro OK", flush=True)
        ux = ux.masked_fill(solid, 0.0)
        uy = uy.masked_fill(solid, 0.0)
        uz = uz.masked_fill(solid, 0.0)
        T = thermal_macroscopic_3d(g)
        if step == 0: print(f"{tag} step0: thermal_macro OK", flush=True)
        g = thermal_collide_bgk_3d(g, T, ux, uy, uz, tau_T=tau_T)
        if step == 0: print(f"{tag} step0: thermal_collide OK", flush=True)
        g = thermal_stream_3d(g)
        if step == 0: print(f"{tag} step0: thermal_stream OK", flush=True)
        # Dirichlet T on cylinder
        g = thermal_fixed_temp_mask_3d(g, T_cyl, solid)
        if step == 0: print(f"{tag} step0: thermal_fixed OK", flush=True)
        # Far-field T = T_inf on y walls
        g = thermal_dirichlet_wall_3d(g, T_inf, "y-")
        g = thermal_dirichlet_wall_3d(g, T_inf, "y+")
        if step == 0: print(f"{tag} step0: thermal_bc OK", flush=True)

        if step % 200 == 0 or step == n_steps - 1:
            T = thermal_macroscopic_3d(g)
            nu_val = nusselt_cylinder_3d(T, solid, T_cyl, T_inf, alpha, D)
            nu_hist.append(nu_val)
            elapsed = time.time() - t0
            print(f"{tag} step={step:5d} Nu={nu_val:.4f} ({elapsed:.1f}s)", flush=True)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

    elapsed = time.time() - t0
    T_final = thermal_macroscopic_3d(g)
    nu_final = nusselt_cylinder_3d(T_final, solid, T_cyl, T_inf, alpha, D)
    nu_ref = 6.5
    rel_err = abs(nu_final - nu_ref) / abs(nu_ref) if abs(nu_ref) > 1e-12 else abs(nu_final - nu_ref)
    passed = rel_err < 0.40

    print(f"{tag} DONE in {elapsed:.1f}s — Nu={nu_final:.4f}, ref={nu_ref:.4f}, "
          f"rel_err={rel_err:.2%} (tol=40%)", flush=True)
    print(f"{tag} PASS={passed}", flush=True)

    output = {
        "benchmark": "heated_cylinder",
        "device": f"sdaa:{device_id}",
        "elapsed_s": round(elapsed, 1),
        "passed": passed,
        "metric": f"Nu={nu_final:.4f}, ref={nu_ref:.4f}, rel_err={rel_err:.2%}",
        "result": {
            "nusselt": nu_final,
            "nusselt_ref": nu_ref,
            "nusselt_history": nu_hist,
            "Re": Re,
            "Pr": Pr,
            "D": D,
            "nx": nx, "ny": ny, "nz": nz,
            "n_steps": n_steps,
        },
    }

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as fh:
            json.dump(output, fh, indent=2, default=str)
        print(f"{tag} Results saved to {out_p}", flush=True)

    print(f"{tag} Summary: {json.dumps(output, default=str)}", flush=True)


if __name__ == "__main__":
    main()
