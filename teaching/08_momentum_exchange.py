#!/usr/bin/env python3
"""Example 08: Momentum Exchange vs Pressure+Friction Integration.

PHYSICS
=======
There are two independent methods to compute drag in LBM:

METHOD 1: Momentum Exchange (MEM — Ladd 1994)
  F = Σ (f_i + f_opp_i) * c_i   over all fluid→solid links

  The force on the wall is the rate of momentum change when populations
  bounce back at solid boundaries.  This method does NOT need surface
  normals — the lattice velocity c_i provides the direction automatically.

  Key: count ALL 18 directions (not just opp_i > i) — equilibrium
  cancellation requires both directions in each opposite pair.

METHOD 2: Pressure + Friction Integration
  F_pressure = -Σ (p_wall - p_0) * n * dA
  F_friction = Σ τ_wall * dA

  This method needs surface normals (analytical or numerical) and
  integrates pressure and shear stress over the surface.

Both methods should give the SAME result for the same flow.  Comparing
them is a powerful cross-validation:

  - If they agree → both implementations are correct
  - If they disagree → one has a bug (normal, timing, or formula)

TIMING IS CRITICAL
==================
Momentum exchange must be computed on the POST-BOUNCE-BACK,
PRE-STREAMING distribution.  At this point:
  - f_i at fluid cells is post-collision (will stream toward wall)
  - f_opp_i at solid cells is post-bounce-back (bounced-back population)

RUN
===
    cd /root/TensorLBM_dev
    SDAA_VISIBLE_DEVICES=0,1,...,31 PYTHONPATH=src python teaching/08_momentum_exchange.py
"""
from __future__ import annotations
import sys, json, math, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d
from tensorlbm.solver3d import stream3d, correct_mass3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.drag_pressure import (
    SurfaceMesh, drag_total, drag_pressure_integration,
    drag_friction_integration, get_near_wall_2d,
)
from tensorlbm.drag_momentum import drag_momentum_exchange, drag_momentum_exchange_vec


def build_cylinder_mask(nx, ny, nz, cx, cy, radius, device):
    """Boolean solid mask for a cylinder extruded along z-axis."""
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    return circle.unsqueeze(0).expand(nz, ny, nx).clone()


def main():
    device_id = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    # ---- Parameters ----
    D = 48.0
    radius = D / 2.0
    nx, ny, nz = 400, 160, 4
    u_in = 0.08
    Re = 200.0
    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5
    cs_smag = 0.05
    n_steps = 5000
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
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)
    near = get_near_wall_2d(solid)
    mesh = SurfaceMesh.from_cylinder(solid, near, cx_c, cy_c, radius)

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    im = float(rho0.sum().item())

    # Accumulators for both methods
    cd_pf_hist = []    # pressure + friction
    cd_me_hist = []    # momentum exchange

    for step in range(1, n_steps + 1):
        f_pre = f.clone()

        # 1. Collision (MRT + Smagorinsky)
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)

        # 2. NoDynamics: restore solid cells
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        # 3. Half-way bounce-back (BEFORE streaming)
        f = bounce_back_cells_3d(f, solid)

        # === MOMENTUM EXCHANGE: compute HERE (post-BB, pre-stream) ===
        cd_me = drag_momentum_exchange(f, near, solid, dpS)

        # 4. Stream
        f = stream3d(f)

        # 5. Far-field BC
        f = far_field_bc_3d(f, u_in=u_in)

        # 6. Mass correction
        if step % 200 == 0:
            f = correct_mass3d(f, im)

        # === PRESSURE + FRICTION: compute HERE (post-stream, post-BC) ===
        cd_tot, cd_p, cd_f = drag_total(f, mesh, dpS, nu)
        cd_pf_hist.append(cd_tot)
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

    # Also get pressure/friction breakdown
    cd_p_final = cd_f_final = float("nan")
    if n_final > 0:
        # Recompute at final state
        cd_p_final, _, _ = drag_pressure_integration(f, mesh, dpS)
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

    print(f"\n  TEACHING POINTS:")
    print(f"  1. MEM (Ladd 1994): F = Σ (f_i + f_opp_i) * c_i")
    print(f"     No surface normal needed — c_i gives direction")
    print(f"  2. PF integration: F = -Σ(p-p0)*n*dA + Σ τ*dA")
    print(f"     Needs surface normals (analytical or numerical)")
    print(f"  3. TIMING: MEM on post-BB pre-stream f")
    print(f"     PF on post-stream post-BC f (physical state)")
    print(f"  4. Both methods should agree (cross-validation)")
    print(f"  5. Count ALL 18 directions in MEM (equilibrium cancellation)")

    result = {
        "Cd_pressure_friction": cd_pf, "Cd_momentum_exchange": cd_me,
        "Cd_ref": Cd_ref, "difference_pct": diff,
        "Cd_p": cd_p_final, "Cd_f": cd_f_final,
        "n_steps": n_steps, "elapsed_s": elapsed,
    }
    out = Path(__file__).parent / "results_08_momentum_exchange.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\n  Results saved to {out}")


if __name__ == "__main__":
    main()
