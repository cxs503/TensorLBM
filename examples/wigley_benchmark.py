"""Wigley hull free-surface resistance benchmark.

Runs the Wigley standard hull at model-scale Re with free surface (CG
multiphase + wall function + CV integral), comparing total resistance Ct
against published experimental / ITTC data.

Standard Wigley hull benchmark references:
  - Wave resistance Cw vs Fn: Professor Kawai's data, Tokyo Univ.
  - Total resistance Ct vs Fn: ITTC benchmark (L=2.5m, B=0.5m, T=0.25m)
  - Typical values at Fn=0.20: Ct ~ 0.004-0.006
                           Fn=0.25: Ct ~ 0.005-0.008
                           Fn=0.30: Ct ~ 0.008-0.012

The Froude number Fn = U / sqrt(g·L) sets the gravity (density-ratio body
force in the CG model).

    PYTHONPATH=src python examples/wigley_benchmark.py
"""

from __future__ import annotations

import math

import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.ibm import ibm_apply_body_force_3d
from tensorlbm.multiphase3d import color_gradient_step_3d
from tensorlbm.obstacles import wigley_hull_mask
from tensorlbm.solver3d import stream3d

_KAPPA = 0.41


def _cv_drag_water(rho_r, ux, uy, uz, nx, ny, fill_height, x0_frac=0.2, x1_frac=0.65):
    """CV momentum integral in the water region only."""
    x0 = int(x0_frac * nx)
    x1 = int(x1_frac * nx)
    y0 = 2
    y1 = ny - 3
    z0 = 1
    z1 = fill_height - 1
    M_in = (
        (rho_r[z0 : z1 + 1, y0 : y1 + 1, x0] * ux[z0 : z1 + 1, y0 : y1 + 1, x0] ** 2).sum().item()
    )
    M_out = (
        (rho_r[z0 : z1 + 1, y0 : y1 + 1, x1] * ux[z0 : z1 + 1, y0 : y1 + 1, x1] ** 2).sum().item()
    )
    M_y0 = (
        (
            rho_r[z0 : z1 + 1, y0, x0 : x1 + 1]
            * ux[z0 : z1 + 1, y0, x0 : x1 + 1]
            * uy[z0 : z1 + 1, y0, x0 : x1 + 1]
        )
        .sum()
        .item()
    )
    M_y1 = (
        (
            rho_r[z0 : z1 + 1, y1, x0 : x1 + 1]
            * ux[z0 : z1 + 1, y1, x0 : x1 + 1]
            * uy[z0 : z1 + 1, y1, x0 : x1 + 1]
        )
        .sum()
        .item()
    )
    M_z0 = (
        (
            rho_r[z0, y0 : y1 + 1, x0 : x1 + 1]
            * ux[z0, y0 : y1 + 1, x0 : x1 + 1]
            * uz[z0, y0 : y1 + 1, x0 : x1 + 1]
        )
        .sum()
        .item()
    )
    M_z1 = (
        (
            rho_r[z1, y0 : y1 + 1, x0 : x1 + 1]
            * ux[z1, y0 : y1 + 1, x0 : x1 + 1]
            * uz[z1, y0 : y1 + 1, x0 : x1 + 1]
        )
        .sum()
        .item()
    )
    return M_in - M_out + M_y0 - M_y1 + M_z0 - M_z1


def _wall_fn_cg(f_r, f_b, solid, nu, near_water, y_val=0.5):
    """Apply Reichardt wall function to CG multiphase (water-side only)."""
    f_comb = f_r + f_b
    rho, ux, uy, uz = macroscopic3d(f_comb)
    u_mag = torch.sqrt(ux * ux + uy * uy + uz * uz).clamp(min=1e-12)
    rho_r = f_r.sum(dim=0)
    rho_total = (rho_r + f_b.sum(dim=0)).clamp(min=1e-12)
    alpha_r = rho_r / rho_total
    alpha_b = 1.0 - alpha_r

    ut = torch.sqrt(nu * u_mag / y_val).clamp(min=1e-12)
    for _ in range(10):
        yp = (y_val * ut / nu).clamp(min=1e-6)
        up = (1.0 / _KAPPA) * torch.log1p(_KAPPA * yp) + 7.8 * (
            1.0 - torch.exp(-yp / 11.0) - (yp / 11.0) * torch.exp(-yp / 3.0)
        )
        ut = (u_mag / up.clamp(min=1e-6)).clamp(min=1e-12)
    tau_w = ut * ut
    inv_umag = 1.0 / u_mag
    coef = -(tau_w / y_val) * near_water.to(f_r.dtype)
    fx = coef * (ux * inv_umag)
    fy = coef * (uy * inv_umag)
    fz = coef * (uz * inv_umag)
    f_r = ibm_apply_body_force_3d(f_r, fx * alpha_r, fy * alpha_r, fz * alpha_r)
    f_b = ibm_apply_body_force_3d(f_b, fx * alpha_b, fy * alpha_b, fz * alpha_b)
    # friction drag
    drag_fric = float((tau_w * (ux * inv_umag) * near_water.to(f_r.dtype)).sum().item())
    return f_r, f_b, drag_fric


def run(
    fn_target=0.25,
    nx=240,
    ny=96,
    nz=96,
    u_in=0.05,
    fill_fraction=0.55,
    n_steps=5000,
    warmup=1500,
    device="cuda",
):
    fill_height = max(int(fill_fraction * nz), 1)
    hull_length = max(6.0, 0.35 * nx)

    # Set Re based on target Fn: Fn = U/sqrt(g_lu * L), g_lu = U^2/(Fn^2 * L)
    g_lu = u_in**2 / (fn_target**2 * hull_length)
    # For CG multiphase, gravity → density contrast body force on water phase
    # Re at this condition: nu = U*L/Re; use Re ~ 1e4-1e5 (model scale-ish for LBM)
    re = 5000  # moderate Re for stability with CG multiphase
    nu = u_in * hull_length / re
    tau = 3.0 * nu + 0.5

    # Hull
    hull = wigley_hull_mask(
        nx=nx,
        ny=ny,
        nz=nz,
        cx=int(0.4 * nx),
        cy=0.5 * (ny - 1),
        cz_keel=1.0,
        length=hull_length,
        beam=max(3.0, 0.25 * ny),
        draft=fill_height + 4,
        device=device,
    )
    solid_mask = hull.clone()
    zz = torch.arange(nz, device=device).view(nz, 1, 1)
    water_mask = (zz < fill_height).expand(nz, ny, nx)
    near_water_template = torch.zeros_like(solid_mask)
    for ax, sgn in [(2, 1), (2, -1), (1, 1), (1, -1), (0, 1), (0, -1)]:
        near_water_template |= torch.roll(solid_mask, sgn, dims=ax) & ~solid_mask

    # CG multiphase init
    rho_r0 = torch.where(water_mask, torch.ones((nz, ny, nx), device=device), 0.1)
    rho_b0 = torch.where(water_mask, 0.1, torch.ones((nz, ny, nx), device=device))
    ux0 = torch.where(water_mask, torch.full((nz, ny, nx), u_in, device=device), 0.0)
    f_r = equilibrium3d(rho_r0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0))
    f_b = equilibrium3d(rho_b0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0))
    zero3 = torch.zeros((nz, ny, nx), device=device)
    f_r_seq = equilibrium3d(rho_r0, zero3, zero3, zero3)
    f_b_seq = equilibrium3d(rho_b0, zero3, zero3, zero3)

    # Gravity body force on water (CG multiphase: denser phase gets downward force)
    grav_force = g_lu  # body force per unit mass
    # Far-field equilibrium
    rho_r_fs = torch.where(water_mask, torch.ones_like(zero3), torch.full_like(zero3, 0.1))
    rho_b_fs = torch.where(water_mask, torch.full_like(zero3, 0.1), torch.ones_like(zero3))
    ux_fs = torch.where(water_mask, torch.full_like(zero3, u_in), zero3)
    f_r_fs = equilibrium3d(rho_r_fs, ux_fs, zero3, zero3)
    f_b_fs = equilibrium3d(rho_b_fs, ux_fs, zero3, zero3)
    water_slice = water_mask[:, :, 0]

    S_wet = float((hull & water_mask).sum().item())
    dyn_p_S = 0.5 * u_in**2 * max(S_wet, 1.0)

    # Reference: Wigley hull experimental Ct at Fn
    ct_ref_map = {0.15: 0.0040, 0.20: 0.0048, 0.25: 0.0065, 0.30: 0.0095, 0.35: 0.014}
    ct_ref = ct_ref_map.get(fn_target, 0.006)

    cv_samples = []
    print(f"Wigley viscous baseline (Fn=0, no gravity): Re={re} grid={nx}x{ny}x{nz}")
    print(f"  hull_L={hull_length:.0f}  fill_z={fill_height}  S_wet={S_wet:.0f}")
    print(f"  Reference: ITTC Cf(Re={re}) = {0.075 / (math.log10(re) - 2) ** 2:.5f}\n")

    for step in range(1, n_steps + 1):
        # 1. CG collision
        f_r, f_b = color_gradient_step_3d(
            f_r, f_b, tau=tau, A=0.005, beta=0.7, solid_mask=solid_mask
        )
        # 2. Gravity on water phase (ramp-up over 500 steps to avoid CG instability)
        ramp = min(1.0, step / 500.0)
        grav_z = torch.zeros((nz, ny, nx), device=device)
        grav_z[water_mask & ~solid_mask] = -grav_force * ramp
        f_r = ibm_apply_body_force_3d(
            f_r, torch.zeros_like(grav_z), torch.zeros_like(grav_z), grav_z
        )
        # 3. Stream
        f_r = stream3d(f_r)
        f_b = stream3d(f_b)
        # 4. Wall function (water-side only)
        near_water = near_water_template & (f_r.sum(dim=0) > 0.5)
        f_r, f_b, _ = _wall_fn_cg(f_r, f_b, solid_mask, nu, near_water, y_val=0.5)
        # 5. Reset solid
        f_r = torch.where(solid_mask.unsqueeze(0), f_r_seq, f_r)
        f_b = torch.where(solid_mask.unsqueeze(0), f_b_seq, f_b)
        # 6. Far-field BC
        rho_ir = torch.where(water_slice, torch.ones_like(water_slice, dtype=torch.float32), 0.1)
        rho_ib = torch.where(water_slice, 0.1, torch.ones_like(water_slice, dtype=torch.float32))
        ux_in = torch.where(water_slice, torch.full_like(rho_ir, u_in), torch.zeros_like(rho_ir))
        feq_r_in = equilibrium3d(
            rho_ir.unsqueeze(-1),
            ux_in.unsqueeze(-1),
            torch.zeros_like(ux_in).unsqueeze(-1),
            torch.zeros_like(ux_in).unsqueeze(-1),
        )
        feq_b_in = equilibrium3d(
            rho_ib.unsqueeze(-1),
            ux_in.unsqueeze(-1),
            torch.zeros_like(ux_in).unsqueeze(-1),
            torch.zeros_like(ux_in).unsqueeze(-1),
        )
        f_r[:, :, :, 0] = feq_r_in[:, :, :, 0]
        f_b[:, :, :, 0] = feq_b_in[:, :, :, 0]
        f_r[:, :, :, -1] = f_r[:, :, :, -2]
        f_b[:, :, :, -1] = f_b[:, :, :, -2]
        f_r[:, 0, :] = f_r_fs[:, 0, :]
        f_r[:, -1, :] = f_r_fs[:, -1, :]
        f_b[:, 0, :] = f_b_fs[:, 0, :]
        f_b[:, -1, :] = f_b_fs[:, -1, :]
        f_r[0, :, :] = f_r_fs[0, :, :]
        f_r[-1, :, :] = f_r_fs[-1, :, :]
        f_b[0, :, :] = f_b_fs[0, :, :]
        f_b[-1, :, :] = f_b_fs[-1, :, :]

        if step > warmup:
            rho_r_now = f_r.sum(dim=0)
            _, ux, uy, uz = macroscopic3d(f_r + f_b)
            cv = _cv_drag_water(rho_r_now, ux, uy, uz, nx, ny, fill_height)
            if math.isfinite(cv):
                cv_samples.append(cv / dyn_p_S)
        if step % 1000 == 0 or step == n_steps:
            ct_cv = sum(cv_samples) / max(len(cv_samples), 1) if cv_samples else 0.0
            _, ux, uy, uz = macroscopic3d(f_r + f_b)
            ms = float(torch.sqrt(ux * ux + uy * uy + uz * uz).max().item())
            err = abs(ct_cv - ct_ref) / ct_ref * 100 if ct_ref > 0 else 0
            print(
                f"  step={step:5d}  Ct_CV={ct_cv:.5f}  (exp {ct_ref:.4f}, err {err:.1f}%)  max|u|={ms:.4f}  "
                f"{'UNSTABLE' if (not math.isfinite(ms) or ms > 0.5) else ''}"
            )

    ct_cv = sum(cv_samples) / max(len(cv_samples), 1) if cv_samples else 0.0
    print(
        f"\nFinal Ct_CV={ct_cv:.5f}  vs exp {ct_ref:.4f}  (err {abs(ct_cv - ct_ref) / ct_ref * 100:.1f}%)"
    )
    return {"Fn": fn_target, "Ct": ct_cv, "Ct_ref": ct_ref}


if __name__ == "__main__":
    print("=== Wigley hull free-surface benchmark (Fn=0.25) ===\n")
    run(fn_target=0.25)
