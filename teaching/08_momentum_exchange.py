#!/usr/bin/env python3
"""Teaching Example 08: Momentum Exchange vs Pressure+Friction Integration.

PHYSICS
=======
There are two independent methods to compute drag in LBM:

METHOD 1: Momentum Exchange (MEM — Ladd 1994)
  F = Σ (f_i + f_opp_i) * c_i   over all fluid→solid links
  Uses momentum_exchange_standard() from tensorlbm.momentum_exchange

METHOD 2: Pressure + Friction Integration
  F_pressure = -Σ (p_wall - p_0) * n * dA
  F_friction = Σ τ_wall * dA
  Uses drag_pressure_integration() and drag_friction_integration()

Both methods should give the SAME result for the same flow.

Uses the COMMON INTERFACE ONLY:
  - lbm_step_correct() for the main loop
  - SurfaceMesh.from_cylinder() for surface normals
  - bounce_back_cells_3d(f_pre) for half-way BB (inside lbm_step_correct)
  - far_field_bc_3d for far-field boundary
  - momentum_exchange_standard for MEM
  - drag_pressure_integration / drag_friction_integration for PF

Usage:
    PYTHONPATH=src python teaching/08_momentum_exchange.py [device_id]
"""
from __future__ import annotations
import sys, json, math, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d
from tensorlbm.lbm_step_correct import lbm_step_correct
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.drag_pressure import (
    SurfaceMesh, drag_pressure_integration,
    drag_friction_integration, get_near_wall_3d,
)
from tensorlbm.momentum_exchange import momentum_exchange_standard


def build_cylinder_mask(nx, ny, nz, cx, cy, radius, device):
    """Boolean solid mask for a cylinder extruded along z-axis."""
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2


def main():
    device_id = int(sys.argv[1]) if len(sys.argv) > 1 else 19
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    # ---- Parameters ----
    D = 24.0
    radius = D / 2.0
    nx, ny, nz = 200, 80, 4
    u_in = 0.08
    Re = 200.0
    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5
    cs_smag = 0.05
    n_steps = 4000
    warmup = 1000
    Cd_ref = 1.30

    cx_c = nx * 0.25
    cy_c = ny * 0.5
    A_frontal = D * nz
    dpS = 0.5 * 1.0 * u_in ** 2 * A_frontal

    tag = f"[MEM-vs-PF SDAA:{device_id}]"
    print(f"{tag} === Momentum Exchange vs Pressure+Friction ===")
    print(f"{tag} Grid: {nx}x{ny}x{nz}  D={D}  Re={Re}  u_in={u_in}")
    print(f"{tag} nu={nu:.6e}  tau={tau:.6f}  Cs={cs_smag}")

    t0 = time.time()
    solid = build_cylinder_mask(nx, ny, nz, cx_c, cy_c, radius, device)
    near = get_near_wall_3d(solid)
    mesh = SurfaceMesh.from_cylinder(solid, near, cx_c, cy_c, radius, axis='z')

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)

    cd_pf_hist = []
    cd_me_hist = []

    for step in range(1, n_steps + 1):
        # Common interface: lbm_step_correct handles the full step
        f = lbm_step_correct(f, collide_smagorinsky_mrt3d, tau, solid, u_in,
                             far_field_bc_3d, C_s=cs_smag)

        # MEM: computed on post-stream, post-BC distribution
        fx_me, _, _ = momentum_exchange_standard(f, solid, near)
        cd_me = fx_me / dpS

        # PF: computed on post-stream, post-BC distribution
        cd_p, _, _ = drag_pressure_integration(f, mesh, dpS, solid=solid)
        cd_f, _, _ = drag_friction_integration(f, mesh, dpS, nu)
        cd_pf = cd_p + cd_f

        cd_pf_hist.append(cd_pf)
        cd_me_hist.append(cd_me)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}")
            break

        if step % 500 == 0:
            n_avg = min(500, len(cd_pf_hist))
            cd_pf_avg = sum(cd_pf_hist[-n_avg:]) / n_avg
            cd_me_avg = sum(cd_me_hist[-n_avg:]) / n_avg
            print(f"{tag} step={step}  Cd_PF={cd_pf_avg:.4f}  "
                  f"Cd_ME={cd_me_avg:.4f}  "
                  f"diff={abs(cd_pf_avg-cd_me_avg):.4f}  "
                  f"({time.time()-t0:.0f}s)")

    elapsed = time.time() - t0

    # Final averages
    n_final = min(warmup, len(cd_pf_hist))
    cd_pf = sum(cd_pf_hist[-n_final:]) / n_final if n_final > 0 else float("nan")
    cd_me = sum(cd_me_hist[-n_final:]) / n_final if n_final > 0 else float("nan")
    diff = abs(cd_pf - cd_me) / max(abs(cd_pf), 1e-10) * 100

    cd_p_final, _, _ = drag_pressure_integration(f, mesh, dpS, solid=solid)
    cd_f_final, _, _ = drag_friction_integration(f, mesh, dpS, nu)

    print(f"\n{'='*60}")
    print(f"  MOMENTUM EXCHANGE vs PRESSURE+FRICTION")
    print(f"{'='*60}")
    print(f"  Method                    Cd        err vs ref")
    print(f"  {'-'*48}")
    print(f"  Pressure + Friction     {cd_pf:.4f}   {abs(cd_pf-Cd_ref)/Cd_ref*100:.1f}%")
    print(f"  Momentum Exchange (MEM)  {cd_me:.4f}   {abs(cd_me-Cd_ref)/Cd_ref*100:.1f}%")
    print(f"  Reference (Re=200)       {Cd_ref:.4f}")
    print(f"  {'-'*48}")
    print(f"  Difference (PF vs MEM):  {diff:.2f}%")
    print(f"  Cd_p (pressure):         {cd_p_final:.4f}")
    print(f"  Cd_f (friction):         {cd_f_final:.4f}")
    print(f"  Time: {elapsed:.0f}s")

    passed = diff < 15.0
    print(f"\n  PASS (MEM vs PF <15%): {passed}")

    result = {
        "Cd_pressure_friction": cd_pf, "Cd_momentum_exchange": cd_me,
        "Cd_ref": Cd_ref, "difference_pct": diff,
        "Cd_p": cd_p_final, "Cd_f": cd_f_final,
        "n_steps": n_steps, "elapsed_s": elapsed, "passed": passed,
    }
    out = Path(__file__).parent / "results_08_momentum_exchange.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\n  Results saved to {out}")


if __name__ == "__main__":
    main()
