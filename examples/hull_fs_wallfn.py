"""CG multiphase wall function for free-surface ship resistance.

Integrates the SUBOFF log-law wall function into the Color-Gradient two-phase
framework.  The body force is split between phases by local density fraction.
This replaces bounce-back on the hull with a τ-decoupled wall shear, the same
trick that brought SUBOFF from 320× to <1%.

    PYTHONPATH=src python examples/hull_fs_wallfn.py
"""
from __future__ import annotations

import math

import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d, C as C3D, W as W3D
from tensorlbm.multiphase3d import color_gradient_step_3d
from tensorlbm.obstacles import compute_obstacle_forces_3d, wigley_hull_mask
from tensorlbm.boundaries3d import bounce_back_cells_3d
from tensorlbm.solver3d import stream3d
from tensorlbm.ibm import ibm_apply_body_force_3d

_KAPPA = 0.41
_B_LOG = 5.0


def wall_function_cg_3d(f_r, f_b, solid, nu, y_val=0.5, wall_law="reichardt"):
    """CG multiphase wall function: body force split by phase density.

    Computes τ_w from the combined (ρ_r+ρ_b) velocity via the log-law, applies
    the Guo body force to each phase proportional to its local density share.
    Returns (f_r, f_b, drag_friction_x, drag_pressure_x).

    This replaces bounce-back on the hull, decoupling wall shear from τ.
    """
    device = f_r.device
    fluid = ~solid
    near = torch.zeros_like(solid)
    for ax, sgn in [(2, 1), (2, -1), (1, 1), (1, -1), (0, 1), (0, -1)]:
        near |= torch.roll(solid, sgn, dims=ax) & fluid

    # Combined macroscopic
    f_combined = f_r + f_b
    rho, ux, uy, uz = macroscopic3d(f_combined)
    u_mag = torch.sqrt(ux * ux + uy * uy + uz * uz).clamp(min=1e-12)

    # Phase densities for force splitting
    rho_r = f_r.sum(dim=0)
    rho_b = f_b.sum(dim=0)
    rho_total = (rho_r + rho_b).clamp(min=1e-12)
    alpha_r = (rho_r / rho_total)          # water fraction
    alpha_b = (rho_b / rho_total)          # air fraction

    # Log-law / Reichardt solve for u_tau
    ut = torch.sqrt(nu * u_mag / y_val).clamp(min=1e-12)
    if wall_law == "reichardt":
        for _ in range(10):
            yp = (y_val * ut / nu).clamp(min=1e-6)
            up = (1.0 / _KAPPA) * torch.log1p(_KAPPA * yp) + 7.8 * (
                1.0 - torch.exp(-yp / 11.0) - (yp / 11.0) * torch.exp(-yp / 3.0)
            )
            ut = (u_mag / up.clamp(min=1e-6)).clamp(min=1e-12)
    else:
        yp = (y_val * ut / nu)
        turb = (yp > 11.6) & near
        if bool(turb.any()):
            ut_t = ut[turb].clone(); um_t = u_mag[turb]
            for _ in range(8):
                lyp = torch.log(y_val * ut_t / nu)
                fv = ut_t * (lyp / _KAPPA + _B_LOG) - um_t
                fp = (lyp / _KAPPA + _B_LOG) + 1.0 / _KAPPA
                ut_t = (ut_t - fv / fp.clamp(min=1e-10)).clamp(min=1e-12)
            ut[turb] = ut_t
    u_tau = torch.where(near, ut, torch.zeros_like(ut))
    tau_w = u_tau * u_tau

    # Body force on near-wall cells (WATER-SIDE ONLY): F = -(τ_w/dy)·û
    inv_umag = 1.0 / u_mag
    # Only apply wall function to near-wall cells in the water phase
    near_water = near & (rho_r > 0.5)   # water fraction > 0.5
    coef = -(tau_w / y_val) * near_water.to(f_r.dtype)
    fx = coef * (ux * inv_umag)
    fy = coef * (uy * inv_umag)
    fz = coef * (uz * inv_umag)

    # Split force by phase density fraction (water gets almost all)
    fx_r = fx * alpha_r; fy_r = fy * alpha_r; fz_r = fz * alpha_r
    fx_b = fx * alpha_b; fy_b = fy * alpha_b; fz_b = fz * alpha_b

    f_r = ibm_apply_body_force_3d(f_r, fx_r, fy_r, fz_r)
    f_b = ibm_apply_body_force_3d(f_b, fx_b, fy_b, fz_b)

    # Friction drag: Σ τ_w·(u_x/|u|) over WATER-SIDE near-wall cells only
    drag_fric = float((tau_w * (ux * inv_umag) * near_water.to(f_r.dtype)).sum().item())
    # Pressure drag: use water-phase density only (avoid air ρ=0.1 contamination)
    p_water = (rho_r - 1.0) / 3.0  # gauge pressure from water phase
    sp = torch.roll(solid, 1, dims=2); sm = torch.roll(solid, -1, dims=2)
    water_fluid = fluid & (rho_r > 0.5)
    drag_pres = float((p_water * (sp.to(f_r.dtype) - sm.to(f_r.dtype)) * water_fluid.to(f_r.dtype)).sum().item())
    return f_r, f_b, drag_fric, drag_pres


def run(hull_type="wigley", nx=200, ny=80, nz=80, re=1000, u_in=0.05,
        fill_fraction=0.55, n_steps=4000, warmup=1500, output_interval=500, device="cuda"):
    fill_height = max(int(fill_fraction * nz), 1)

    hull = wigley_hull_mask(
        nx=nx, ny=ny, nz=nz,
        cx=int(0.4 * nx), cy=0.5 * (ny - 1),
        cz_keel=1.0, length=max(6.0, 0.35 * nx),
        beam=max(3.0, 0.25 * ny), draft=fill_height + 4,
        device=device,
    )
    solid_mask = hull.clone()

    zz = torch.arange(nz, device=device).view(nz, 1, 1)
    water_mask = (zz < fill_height).expand(nz, ny, nx)

    rho_r0 = torch.where(water_mask, torch.ones((nz, ny, nx), device=device), 0.1)
    rho_b0 = torch.where(water_mask, 0.1, torch.ones((nz, ny, nx), device=device))
    ux0 = torch.where(water_mask, torch.full((nz, ny, nx), u_in, device=device), 0.0)
    f_r = equilibrium3d(rho_r0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0))
    f_b = equilibrium3d(rho_b0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0))

    zero3 = torch.zeros((nz, ny, nx), device=device)
    f_r_seq = equilibrium3d(rho_r0, zero3, zero3, zero3)
    f_b_seq = equilibrium3d(rho_b0, zero3, zero3, zero3)

    nu = u_in * max(1.0, 0.35 * nx) / max(re, 1e-6)
    tau = 3.0 * nu + 0.5
    S_wet = float((hull & water_mask).sum().item())  # underwater solid cells
    dyn_p_S = 0.5 * u_in ** 2 * max(S_wet, 1.0)      # wetted-area normalization

    # Far-field equilibrium
    rho_r_fs = torch.where(water_mask, torch.ones_like(zero3), torch.full_like(zero3, 0.1))
    rho_b_fs = torch.where(water_mask, torch.full_like(zero3, 0.1), torch.ones_like(zero3))
    ux_fs = torch.where(water_mask, torch.full_like(zero3, u_in), zero3)
    f_r_fs = equilibrium3d(rho_r_fs, ux_fs, zero3, zero3)
    f_b_fs = equilibrium3d(rho_b_fs, ux_fs, zero3, zero3)

    water_slice = water_mask[:, :, 0]
    fric, pres = [], []

    print(f"CG wall-function hull ({hull_type}): grid={nx}x{ny}x{nz} Re={re} tau={tau:.4f}")
    print(f"  fill_z={fill_height}  wetted_cells={S_wet:.0f}  nu={nu:.4f}\n")

    for step in range(1, n_steps + 1):
        # 1. CG collision
        f_r, f_b = color_gradient_step_3d(f_r, f_b, tau=tau, A=0.005, beta=0.7, solid_mask=solid_mask)
        # 2. Stream
        f_r = stream3d(f_r); f_b = stream3d(f_b)
        # 3. CG wall function (replaces bounce-back on hull)
        f_r, f_b, df, dp = wall_function_cg_3d(f_r, f_b, solid_mask, nu, y_val=0.5, wall_law="reichardt")
        # 4. Reset solid cells
        f_r = torch.where(solid_mask.unsqueeze(0), f_r_seq, f_r)
        f_b = torch.where(solid_mask.unsqueeze(0), f_b_seq, f_b)
        # 5. Far-field BC (inlet/outlet/lateral, per phase)
        rho_ir = torch.where(water_slice, torch.ones_like(water_slice, dtype=torch.float32), 0.1)
        rho_ib = torch.where(water_slice, 0.1, torch.ones_like(water_slice, dtype=torch.float32))
        ux_in = torch.where(water_slice, torch.full_like(rho_ir, u_in), torch.zeros_like(rho_ir))
        f_r[:, :, :, 0] = equilibrium3d(rho_ir.unsqueeze(-1), ux_in.unsqueeze(-1),
                                        torch.zeros_like(ux_in).unsqueeze(-1), torch.zeros_like(ux_in).unsqueeze(-1))[:, :, :, 0]
        f_b[:, :, :, 0] = equilibrium3d(rho_ib.unsqueeze(-1), ux_in.unsqueeze(-1),
                                        torch.zeros_like(ux_in).unsqueeze(-1), torch.zeros_like(ux_in).unsqueeze(-1))[:, :, :, 0]
        f_r[:, :, :, -1] = f_r[:, :, :, -2]; f_b[:, :, :, -1] = f_b[:, :, :, -2]
        f_r[:, 0, :] = f_r_fs[:, 0, :]; f_r[:, -1, :] = f_r_fs[:, -1, :]
        f_b[:, 0, :] = f_b_fs[:, 0, :]; f_b[:, -1, :] = f_b_fs[:, -1, :]
        f_r[0, :, :] = f_r_fs[0, :, :]; f_r[-1, :, :] = f_r_fs[-1, :, :]
        f_b[0, :, :] = f_b_fs[0, :, :]; f_b[-1, :, :] = f_b_fs[-1, :, :]

        if step > warmup and math.isfinite(df):
            fric.append(df); pres.append(dp)
        if step % output_interval == 0 or step == n_steps:
            cf = (sum(fric)/max(len(fric),1))/dyn_p_S if fric else 0.0
            cp = (sum(pres)/max(len(pres),1))/dyn_p_S if pres else 0.0
            _, ux, uy, uz = macroscopic3d(f_r + f_b)
            ms = float(torch.sqrt(ux*ux+uy*uy+uz*uz).max().item())
            print(f"  step={step:5d}  Cf={cf:.4f} Cp={cp:.4f} Ct={cf+cp:.4f}  max|u|={ms:.4f}  "
                  f"{'UNSTABLE' if (not math.isfinite(ms) or ms > 0.5) else ''}")

    cf = (sum(fric)/max(len(fric),1))/dyn_p_S if fric else 0.0
    cp = (sum(pres)/max(len(pres),1))/dyn_p_S if pres else 0.0
    print(f"\nFinal: Cf={cf:.4f}  Cp={cp:.4f}  Ct={cf+cp:.4f}")
    return {"Cf": cf, "Cp": cp, "Ct": cf + cp}


if __name__ == "__main__":
    run()
