"""Piston compression with K diffusion + temperature bridge.

K satisfies diffusion equation (∇²K=0), 1 Jacobi/step.
BC: K = V₀/V at piston face (from compression ratio, no f.sum()).
T = T₀ × K^(γ-1) (from K, no separate solve).
p = ρ × cs² × T₀ × K^γ (adiabatic pressure).

No f.sum(), no fluid mask, no D3Q7 temperature distribution.
"""
import sys, math, torch
sys.path.insert(0, '/root/TensorLBM_test/src')
from tensorlbm.d3q19 import equilibrium3d, C as C3D, W as W3D
from tensorlbm.solver3d import stream3d, collide_bgk3d

def run_piston_k_adiabatic(nx=64, ny=32, nz=32, n_steps=4000, device='sdaa:3', ps=0.01):
    dev = torch.device(device)
    cs2 = 1.0 / 3.0
    rho0 = 1.0
    gamma = 1.4
    T0 = 1.0

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

    rho_init = torch.ones(nz, ny, nx, device=dev)
    f = equilibrium3d(rho_init, torch.zeros_like(rho_init),
                       torch.zeros_like(rho_init), torch.zeros_like(rho_init), device=dev)

    piston_pos = float(nx - 1)
    V0 = (nx - 1) * ny * nz

    # K field (initial = 1.0 everywhere)
    K = torch.ones(nz, ny, nx, device=dev)

    print(f'=== K Diffusion + Temperature Bridge ===', flush=True)
    print(f'Grid: {nx}x{ny}x{nz} ps={ps} gamma={gamma}', flush=True)
    print(f'K: ∇²K=0, 1 Jacobi/step, BC: K=V₀/V at piston', flush=True)
    print(f'T = T₀ × K^(γ-1) = K^0.4', flush=True)
    print(f'p = ρ × cs² × T₀ × K^γ = ρ × cs² × K^1.4', flush=True)
    print(flush=True)

    for step in range(1, n_steps + 1):
        piston_pos = max(nx / 2, piston_pos - ps)
        px = int(piston_pos)
        solid = solid_wall.clone()
        solid[:, :, px:] = True
        fluid = ~solid

        # 1. LBM step: collision + streaming + bounce-back
        f = collide_bgk3d(f, tau=0.8)
        f = f.clamp(min=0.0, max=3.0)
        f = stream3d(f)
        f_swapped = f[opp]
        f = torch.where(solid.unsqueeze(0), f_swapped, f)
        f[:, 0, :] = f_swapped[:, 0, :]; f[:, -1, :] = f_swapped[:, -1, :]
        f[:, :, 0] = f_swapped[:, :, 0]; f[:, :, -1] = f_swapped[:, :, -1]
        f[:, :, :, 0] = f_swapped[:, :, :, 0]

        # 2. K diffusion: 10 Jacobi steps per LBM step
        V_now = px * ny * nz
        cr = V0 / max(V_now, 1)
        K_target = cr  # K = V₀/V at piston face
        for _ in range(10):
            K_avg = K.clone()
            for dz, dy, dx in [(0,0,1),(0,0,-1),(0,1,0),(0,-1,0),(1,0,0),(-1,0,0)]:
                K_avg += torch.roll(K, shifts=(dz,dy,dx), dims=(0,1,2))
            K_avg /= 7.0
            K_avg[:, :, px-1] = K_target  # BC at piston face
            # Neumann BC at walls (copy from interior)
            K_avg[:, 0, :] = K_avg[:, 1, :]
            K_avg[:, -1, :] = K_avg[:, -2, :]
            K_avg[:, :, 0] = K_avg[:, :, 1]
            K_avg[:, :, -1] = K_avg[:, :, -2]
            K_avg[0, :, :] = K_avg[1, :, :]
            K_avg[-1, :, :] = K_avg[-2, :, :]
            # Only zero piston cells (not walls!)
            K_avg[:, :, px:] = 0
            K = K_avg

        # 3. Apply K as mapping coefficient (NOT reconstruction!)
        # ρ = K × f.sum(0) — don't modify f, just correct density for pressure
        rho_raw = f.sum(0)
        rho_corrected = K * rho_raw  # mapped density

        # 4. Get velocity from f (preserved, no modification)
        ux = (f * cx3d).sum(0) / rho_raw.clamp(min=1e-6)
        uy = (f * cy3d).sum(0) / rho_raw.clamp(min=1e-6)
        uz = (f * cz3d).sum(0) / rho_raw.clamp(min=1e-6)
        # NO reconstruction — f evolves naturally

        # 6. Measurement
        if step % 400 == 0 or step == n_steps:
            rg = f.sum(0)
            if not torch.isnan(rg).any():
                rho = float(rho_corrected[fluid].mean())  # use mapped density!
                K_mean = float(K[fluid].mean())
                T = T0 * K_mean ** (gamma - 1)  # T = K^0.4
                p_iso = rho * cs2               # isothermal (no T)
                p_adi = rho * cs2 * T            # adiabatic (with T)
                p_exp_iso = rho0 * cs2 * cr      # expected isothermal
                p_exp_adi = rho0 * cs2 * T0 * cr ** gamma  # expected adiabatic
                mass = float(f[:, fluid].sum())
                rho_face = float(rg[nz//2, ny//2, px-1]) if px > 0 else 0
                rho_far = float(rg[nz//2, ny//2, 1])
                K_face = float(K[nz//2, ny//2, px-1]) if px > 0 else 0
                K_far = float(K[nz//2, ny//2, 1])
                print(f'step {step}: px={px} V_ratio={cr:.2f} '
                      f'K={K_mean:.4f}(face={K_face:.3f},far={K_far:.3f}) '
                      f'T={T:.4f} rho={rho:.4f} '
                      f'p_iso={p_iso:.4f}(exp={p_exp_iso:.4f} r={p_iso/p_exp_iso:.3f}) '
                      f'p_adi={p_adi:.4f}(exp={p_exp_adi:.4f} r={p_adi/p_exp_adi:.3f}) '
                      f'mass={mass:.0f} face/mean={rho_face/max(rho,1e-6):.3f}', flush=True)
            else:
                print(f'step {step}: NaN!', flush=True); break

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--nx', type=int, default=64)
    p.add_argument('--ny', type=int, default=32)
    p.add_argument('--nz', type=int, default=32)
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--device', default='sdaa:3')
    p.add_argument('--ps', type=float, default=0.01)
    args = p.parse_args()
    run_piston_k_adiabatic(args.nx, args.ny, args.nz, args.steps, args.device, args.ps)
