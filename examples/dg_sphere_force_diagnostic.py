"""Force-method diagnostic: is the 2x Cd over-prediction a force bug or real?

Runs the BFL sphere and computes drag THREE independent ways:
  1. BFL momentum-exchange (wall bounce-back injection) — the codebase default.
  2. Control-volume momentum + pressure flux integral (independent of wall BC).
  3. Wake momentum-deficit survey (classic external-aero method).

If (1) ~2x but (2)/(3) ~1x the reference, the BFL force formula is the bug and
the flow field is fine.  If all three agree at ~2x, the flow itself
over-predicts (physics/resolution).

    PYTHONPATH=src python examples/dg_sphere_force_diagnostic.py
"""
from __future__ import annotations

import math

import torch

from tensorlbm.d3q19 import C as C3D, equilibrium3d, macroscopic3d
from tensorlbm.boundaries3d import make_channel_wall_mask_3d, sphere_mask, apply_zou_he_channel_boundaries_3d
from tensorlbm.interpolated_bc import bouzidi_bounce_back_3d, compute_q_sphere
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d


def cd_ref(re):
    return 24.0 / re * (1 + 0.15 * re ** 0.687) + 0.42 / (1 + 4.25e4 / re ** 1.16)


def cv_momentum_drag(rho, ux, uy, uz, cx, cy, cz, r):
    """Drag on the body = -∮_CV [ρ u_x (u·n̂) + p n̂_x] dA over a box CV.

    CV = [cx-r-2, cx+r+2] × [cy-r-2, cy+r+2] × [cz-r-2, cz+r+2].  Gauge
    pressure p = (ρ-1)/3 (c_s²=1/3, ρ₀=1).
    """
    p = (rho - 1.0) / 3.0
    x0 = int(cx - r - 2); x1 = int(cx + r + 2)
    y0 = int(cy - r - 2); y1 = int(cy + r + 2)
    z0 = int(cz - r - 2); z1 = int(cz + r + 2)
    nz, ny, nx = rho.shape
    x0 = max(x0, 0); x1 = min(x1, nx - 1)
    y0 = max(y0, 0); y1 = min(y1, ny - 1)
    z0 = max(z0, 0); z1 = min(z1, nz - 1)
    M = 0.0  # F_body->fluid,x = ∮[ρ ux(u·n) + p n_x] dA ; drag = -M
    # x-faces (n_x = ∓1)
    sl = (slice(z0, z1 + 1), slice(y0, y1 + 1))
    M += (-rho[z0:z1+1, y0:y1+1, x0] * ux[z0:z1+1, y0:y1+1, x0]**2 - p[z0:z1+1, y0:y1+1, x0]).sum().item()
    M += ( rho[z0:z1+1, y0:y1+1, x1] * ux[z0:z1+1, y0:y1+1, x1]**2 + p[z0:z1+1, y0:y1+1, x1]).sum().item()
    # y-faces (n_y=∓1, n_x=0): contribution ρ ux (u·n) = ρ ux (±uy)
    M += (-rho[z0:z1+1, y0, x0:x1+1] * ux[z0:z1+1, y0, x0:x1+1] * uy[z0:z1+1, y0, x0:x1+1]).sum().item()
    M += ( rho[z0:z1+1, y1, x0:x1+1] * ux[z0:z1+1, y1, x0:x1+1] * uy[z0:z1+1, y1, x0:x1+1]).sum().item()
    # z-faces
    M += (-rho[z0, y0:y1+1, x0:x1+1] * ux[z0, y0:y1+1, x0:x1+1] * uz[z0, y0:y1+1, x0:x1+1]).sum().item()
    M += ( rho[z1, y0:y1+1, x0:x1+1] * ux[z1, y0:y1+1, x0:x1+1] * uz[z1, y0:y1+1, x0:x1+1]).sum().item()
    return -M  # drag on body (positive)


def wake_deficit_drag(rho, ux, cx, cy, cz, r, u_in, ny, nz):
    """D = ∫_wake ρ u_x (u_in - u_x) dA at x = cx + 2r."""
    xw = min(int(cx + 2 * r), ux.shape[2] - 1)
    uw = ux[:, :, xw]
    rw = rho[:, :, xw]
    return (rw * uw * (u_in - uw)).sum().item()


def run(radius=12.0, u_in=0.06, re=100.0, n_steps=2000, warmup=1000, cs=0.1, device="cuda", far_field=False, nx=120, ny=64, nz=64):
    tau = 3.0 * (u_in * 2 * radius / re) + 0.5
    cx, cy, cz = nx / 2.0, ny / 2.0, nz / 2.0
    solid = sphere_mask(nx, ny, nz, cx, cy, cz, radius, device=device)
    fluid_bc, q_field = compute_q_sphere(nx, ny, nz, cx, cy, cz, radius, device=device)
    wall_mask = make_channel_wall_mask_3d(nz, ny, nx, solid, device=device)
    c_dev = C3D.to(device).float()

    rho0 = torch.ones(nz, ny, nx, device=device)
    f_eq_solid = equilibrium3d(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0), torch.zeros_like(rho0), device=device)
    f = equilibrium3d(rho0, torch.full_like(rho0, u_in), torch.zeros_like(rho0), torch.zeros_like(rho0), device=device)
    initial_mass = float(f.sum().item())
    dyn_p = 0.5 * u_in ** 2 * math.pi * radius ** 2

    f_force = []; cv_force = []; wake_force = []
    window_f = []
    print(f"Re={re:.0f} r={radius:.0f} ref Cd={cd_ref(re):.4f}  convergence (BFL Cd per 500-step window):")
    for step in range(1, n_steps + 1):
        f[:, solid] = f_eq_solid[:, solid]
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs)
        f = stream3d(f)
        if far_field:
            # Far-field (free-stream Dirichlet) on inlet + lateral faces; outlet zero-gradient.
            feq_fs = equilibrium3d(torch.ones_like(rho0), torch.full_like(rho0, u_in),
                                   torch.zeros_like(rho0), torch.zeros_like(rho0), device=device)
            f[:, :, :, 0] = feq_fs[:, :, :, 0]                      # inlet
            f[:, :, :, -1] = f[:, :, :, -2]                          # outlet zero-grad
            f[:, 0, :, :] = feq_fs[:, 0, :, :]                       # y- lateral
            f[:, -1, :, :] = feq_fs[:, -1, :, :]
            f[:, :, 0, :] = feq_fs[:, :, 0, :]                       # z- lateral
            f[:, :, -1, :] = feq_fs[:, :, -1, :]
        else:
            f = apply_zou_he_channel_boundaries_3d(f, u_in=u_in, wall_mask=wall_mask, obstacle_mask=torch.zeros_like(solid))
        f_pre = f.clone()
        for d in range(1, 19):
            if bool(fluid_bc[d].any()):
                f = bouzidi_bounce_back_3d(f, f_pre, fluid_bc[d], q_field[d], d)
        if step % 200 == 0:
            f = correct_mass3d(f, initial_mass)
        if step > warmup:
            fx_b = 0.0
            for d in range(1, 19):
                if bool(fluid_bc[d].any()):
                    delta = f[d][fluid_bc[d]] - f_pre[d][fluid_bc[d]]
                    fx_b -= float((delta * c_dev[d, 0]).sum().item())
            f_force.append(fx_b); window_f.append(fx_b)
            rho, ux, uy, uz = macroscopic3d(f)
            cv_force.append(cv_momentum_drag(rho, ux, uy, uz, cx, cy, cz, radius))
            wake_force.append(wake_deficit_drag(rho, ux, cx, cy, cz, radius, u_in, ny, nz))
            if step % 500 == 0 and len(window_f) > 0:
                print(f"  step {step:5d}: window Cd = {sum(window_f)/len(window_f)/dyn_p:.4f}")
                window_f = []

    def avg(L): return sum(L) / max(len(L), 1)
    ref = cd_ref(re)
    print(f"\nRe={re:.0f} r={radius:.0f} ref Cd={ref:.4f}  (dyn_p={dyn_p:.3f})")
    for label, L in (("BFL momentum-exchange", f_force), ("CV momentum+pressure", cv_force), ("wake deficit", wake_force)):
        cd = avg(L) / dyn_p
        print(f"  {label:>24}: Cd={cd:.4f}  err={abs(cd-ref)/ref*100:6.1f}%")


if __name__ == "__main__":
    run()
