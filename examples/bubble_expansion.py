"""Bubble expansion test — gas bubble expanding in liquid.

Uses the validated compression approach (ρ=ρ₀×V₀/V + T=K^0.4)
but in REVERSE: bubble expands instead of compresses.

Setup:
  - Gas bubble at center, initial radius R0
  - Liquid surrounds bubble
  - Bubble expands: R increases → V increases → ρ decreases → p decreases
  - Liquid should be pushed outward by expanding bubble

Key difference from piston:
  - Piston: V known from piston position (1D)
  - Bubble: V known from bubble radius (3D, but still a scalar)
  - ρ_gas = ρ₀ × V₀/V = ρ₀ × (R0/R)³  (volume scales as R³)
  - T = K^0.4 = (R0/R)^(3×0.4) = (R0/R)^1.2
  - p_gas = ρ × cs² × T = ρ₀ × cs² × (R0/R)^(3×1.4) = ρ₀ × cs² × (R0/R)^4.2

The bubble radius R is tracked (like piston position).
Density set from R (not from f.sum()).
IBM pushes liquid outward (bubble surface = moving wall).
"""
import sys, math, torch
sys.path.insert(0, '/root/TensorLBM_test/src')
from tensorlbm.d3q19 import equilibrium3d, C as C3D, W as W3D
from tensorlbm.solver3d import stream3d, collide_bgk3d
from tensorlbm.ibm import ibm_direct_forcing_3d, ibm_force_spread_3d, ibm_velocity_interpolate_3d

def run_bubble_expansion(nx=64, ny=64, nz=64, n_steps=3000, device='sdaa:3'):
    dev = torch.device(device)
    cs2 = 1.0 / 3.0
    rho0_gas = 1.0   # initial gas density
    rho0_liq = 1.0   # liquid density (same for simplicity)
    gamma = 1.4
    T0 = 1.0

    # Bubble: starts at R0, expands to R = R0 * (1 + expansion_rate * step)
    cx, cy, cz = nx//2, ny//2, nz//2
    R0 = 8.0  # initial radius
    expansion_rate = 0.0005  # R increases by 0.05% per step

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=dev), torch.arange(ny, device=dev),
        torch.arange(nx, device=dev), indexing='ij')
    solid_wall = torch.zeros(nz, ny, nx, dtype=torch.bool, device=dev)
    solid_wall[:, 0, :] = True; solid_wall[:, -1, :] = True
    solid_wall[:, :, 0] = True; solid_wall[:, :, -1] = True
    solid_wall[0, :, :] = True; solid_wall[-1, :, :] = True

    opp = torch.tensor([0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17], device=dev)
    c = C3D.to(dev).float()
    cx3d = c[:, 0].view(19, 1, 1, 1)
    cy3d = c[:, 1].view(19, 1, 1, 1)
    cz3d = c[:, 2].view(19, 1, 1, 1)

    # Initial: liquid everywhere, gas bubble at center
    dist2 = (xx - cx)**2 + (yy - cy)**2 + (zz - cz)**2
    bubble_mask = dist2 < R0**2

    # Liquid at rest, gas at higher pressure (to drive expansion)
    rho_init = torch.ones(nz, ny, nx, device=dev)
    rho_init[bubble_mask] = rho0_gas  # gas density = 1.0 (same as liquid initially)
    f = equilibrium3d(rho_init, torch.zeros_like(rho_init),
                       torch.zeros_like(rho_init), torch.zeros_like(rho_init), device=dev)

    V0_bubble = float(bubble_mask.sum())  # initial bubble volume (in cells)
    R_current = R0

    print(f'=== Bubble Expansion ===', flush=True)
    print(f'Grid: {nx}x{ny}x{nz} R0={R0} expansion_rate={expansion_rate}', flush=True)
    print(f'V0={V0_bubble} cells', flush=True)
    print(f'Gas: rho=rho0*(R0/R)^3, T=(R0/R)^1.2, p=rho*cs²*T', flush=True)
    print(flush=True)

    for step in range(1, n_steps + 1):
        # Update bubble radius
        R_current = R0 * (1.0 + expansion_rate * step)
        dist2 = (xx - cx)**2 + (yy - cy)**2 + (zz - cz)**2
        bubble_now = dist2 < R_current**2
        liquid_now = ~bubble_now & ~solid_wall

        # Compression ratio for gas (expansion = compression ratio < 1)
        V_now = float(bubble_now.sum())
        cr = V0_bubble / max(V_now, 1)  # < 1 (expanding)
        K = cr  # K = V0/V < 1
        T_gas = K ** (gamma - 1)  # T = K^0.4 < 1 (cooling)
        rho_gas_target = rho0_gas * K  # density decreases
        p_gas = rho_gas_target * cs2 * T_gas  # pressure decreases

        # 1. LBM step: collision + streaming + bounce-back (liquid)
        f = collide_bgk3d(f, tau=0.8)
        f = f.clamp(min=0.0, max=3.0)
        f = stream3d(f)
        f_swapped = f[opp]
        f = torch.where(solid_wall.unsqueeze(0), f_swapped, f)

        # 2. Set gas density from compression ratio (NOT f.sum())
        # Gas region: set density to rho_gas_target
        rho_raw = f.sum(0)
        ux = (f * cx3d).sum(0) / rho_raw.clamp(min=1e-6)
        uy = (f * cy3d).sum(0) / rho_raw.clamp(min=1e-6)
        uz = (f * cz3d).sum(0) / rho_raw.clamp(min=1e-6)

        # Reconstruct gas region with target density
        feq_gas = equilibrium3d(
            torch.full_like(rho_raw, rho_gas_target).clamp(min=1e-6, max=5.0),
            ux.clamp(-0.5, 0.5), uy.clamp(-0.5, 0.5), uz.clamp(-0.5, 0.5), device=dev)
        feq_raw = equilibrium3d(
            rho_raw.clamp(min=1e-6, max=5.0),
            ux.clamp(-0.5, 0.5), uy.clamp(-0.5, 0.5), uz.clamp(-0.5, 0.5), device=dev)
        # Replace equilibrium in gas region, keep non-equilibrium
        f = torch.where(bubble_now.unsqueeze(0), f - feq_raw + feq_gas, f)
        f = f.clamp(min=0.0, max=5.0)

        # 3. Bubble surface: push liquid outward (IBM-like)
        # Find bubble surface cells (gas cells adjacent to liquid)
        bubble_surface = bubble_now & ~torch.roll(bubble_now, 1, dims=2)
        for d in [-1, 1]:
            bubble_surface |= bubble_now & ~torch.roll(bubble_now, d, dims=2)
            bubble_surface |= bubble_now & ~torch.roll(bubble_now, d, dims=1)
            bubble_surface |= bubble_now & ~torch.roll(bubble_now, d, dims=0)
        bubble_surface = bubble_surface & ~solid_wall

        # At bubble surface, set liquid velocity outward (radial)
        if bubble_surface.any():
            # Radial direction at surface cells
            sx = xx[bubble_surface].float() - cx
            sy = yy[bubble_surface].float() - cy
            sz = zz[bubble_surface].float() - cz
            sr = torch.sqrt(sx**2 + sy**2 + sz**2).clamp(min=1e-6)
            # Outward velocity proportional to expansion rate
            u_radial = expansion_rate * R_current  # surface velocity
            ux_surf = u_radial * sx / sr
            uy_surf = u_radial * sy / sr
            uz_surf = u_radial * sz / sr

            # Set velocity at liquid cells adjacent to bubble surface
            for dz, dy, dx in [(0,0,1),(0,0,-1),(0,1,0),(0,-1,0),(1,0,0),(-1,0,0)]:
                nbr_liquid = torch.roll(bubble_surface, shifts=(dz,dy,dx), dims=(0,1,2)) & liquid_now
                if nbr_liquid.any():
                    # Create full 3D fields for equilibrium
                    rho_push = rho_raw.clone()
                    ux_push = ux.clone()
                    uy_push = uy.clone()
                    uz_push = uz.clone()
                    # Set outward velocity at neighbor cells
                    ux_push[nbr_liquid] = float(ux_surf.mean())
                    uy_push[nbr_liquid] = float(uy_surf.mean())
                    uz_push[nbr_liquid] = float(uz_surf.mean())
                    feq_push = equilibrium3d(
                        rho_push.clamp(min=1e-6, max=5.0),
                        ux_push.clamp(-0.5, 0.5),
                        uy_push.clamp(-0.5, 0.5),
                        uz_push.clamp(-0.5, 0.5), device=dev)
                    feq_old = equilibrium3d(
                        rho_raw.clamp(min=1e-6, max=5.0),
                        ux.clamp(-0.5, 0.5), uy.clamp(-0.5, 0.5), uz.clamp(-0.5, 0.5), device=dev)
                    f = torch.where(nbr_liquid.unsqueeze(0), f - feq_old + feq_push, f)

        f = f.clamp(min=0.0, max=5.0)

        # 4. Measurement
        if step % 300 == 0 or step == n_steps:
            rg = f.sum(0)
            if not torch.isnan(rg).any():
                rho_gas = float(rg[bubble_now].mean()) if bubble_now.any() else 0
                rho_liq = float(rg[liquid_now].mean()) if liquid_now.any() else 0
                p_gas_actual = rho_gas * cs2
                p_gas_adi = rho_gas * cs2 * T_gas
                p_exp = rho0_gas * cs2 * K ** gamma  # expected adiabatic

                # Liquid velocity near bubble
                ux_liq = (f * cx3d).sum(0) / rg.clamp(min=1e-6)
                near_bubble = liquid_now & (dist2 < (R_current + 3)**2) & (dist2 > R_current**2)
                u_near = float(ux_liq[near_bubble].abs().mean()) if near_bubble.any() else 0

                print(f'step {step}: R={R_current:.2f} V_ratio={1/cr:.2f} '
                      f'rho_gas={rho_gas:.4f} (exp={rho_gas_target:.4f}) '
                      f'p_gas={p_gas_actual:.4f} p_adi={p_gas_adi:.4f} (exp={p_exp:.4f}) '
                      f'rho_liq={rho_liq:.4f} u_near={u_near:.4f} '
                      f'n_gas={int(bubble_now.sum())}', flush=True)
            else:
                print(f'step {step}: NaN!', flush=True); break

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--nx', type=int, default=64)
    p.add_argument('--ny', type=int, default=64)
    p.add_argument('--nz', type=int, default=64)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--device', default='sdaa:3')
    args = p.parse_args()
    run_bubble_expansion(args.nx, args.ny, args.nz, args.steps, args.device)
