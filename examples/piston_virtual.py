"""Piston compression with virtual wall gas points.

Instead of global K (mass conservation scaling), use local mass transfer:
1. Virtual gas point on the piston wall
2. Piston moves → volume changes → mass changes
3. Mass difference transferred to nearest Eulerian gas cell
4. No global K, no volume tracking, no f.sum()

The virtual point carries the displaced gas mass.
When piston moves by delta_x, delta_mass = rho * delta_x * area.
This mass is injected at the nearest gas cell as equilibrium.
"""
import sys, math, torch
sys.path.insert(0, '/root/TensorLBM_test/src')
from tensorlbm.d3q19 import equilibrium3d, C as C3D, W as W3D
from tensorlbm.solver3d import stream3d, collide_bgk3d

def run_piston_virtual(nx=64, ny=32, nz=32, n_steps=4000, device='sdaa:3', ps=0.01):
    dev = torch.device(device)
    cs2 = 1.0 / 3.0
    rho0 = 1.0
    gamma = 1.4

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=dev), torch.arange(ny, device=dev),
        torch.arange(nx, device=dev), indexing='ij')
    solid_wall = torch.zeros(nz, ny, nx, dtype=torch.bool, device=dev)
    solid_wall[:, 0, :] = True; solid_wall[:, -1, :] = True
    solid_wall[:, :, 0] = True; solid_wall[:, :, -1] = True
    solid_wall[0, :, :] = True; solid_wall[-1, :, :] = True

    opp = torch.tensor([0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17], device=dev)
    c = C3D.to(dev).float()
    w = W3D.to(dev).float().view(19, 1, 1, 1)
    cx3d = c[:, 0].view(19, 1, 1, 1)

    rho_init = torch.ones(nz, ny, nx, device=dev)
    f = equilibrium3d(rho_init, torch.zeros_like(rho_init),
                       torch.zeros_like(rho_init), torch.zeros_like(rho_init), device=dev)

    piston_pos = float(nx - 1)
    V0 = (nx - 1) * ny * nz  # initial volume

    print(f'=== Piston Virtual Gas Points (no K) ===', flush=True)
    print(f'Grid: {nx}x{ny}x{nz} ps={ps} Ma={ps/math.sqrt(cs2):.4f}', flush=True)
    print(f'Local mass transfer: delta_m = rho * delta_x * area', flush=True)
    print(flush=True)

    for step in range(1, n_steps + 1):
        old_pos = piston_pos
        piston_pos = max(nx / 2, piston_pos - ps)
        px = int(piston_pos)
        solid = solid_wall.clone()
        solid[:, :, px:] = True
        fluid = ~solid

        # Collision
        f = collide_bgk3d(f, tau=0.8)
        f = f.clamp(min=0.0, max=3.0)
        f = stream3d(f)

        # Bounce-back
        f_swapped = f[opp]
        f = torch.where(solid.unsqueeze(0), f_swapped, f)
        f[:, 0, :] = f_swapped[:, 0, :]; f[:, -1, :] = f_swapped[:, -1, :]
        f[:, :, 0] = f_swapped[:, :, 0]; f[:, :, -1] = f_swapped[:, :, -1]
        f[:, :, :, 0] = f_swapped[:, :, :, 0]

        # Virtual gas point: mass transfer from piston displacement
        delta_x = old_pos - piston_pos  # distance moved (positive)
        if delta_x > 0:
            face_mask = (xx == px - 1) & ~solid
            if face_mask.any():
                rho_face = float(f.sum(0)[face_mask].mean())
                # Each face cell gets rho * delta_x (mass from sub-grid volume)
                # NOT divided by n_face — each cell gets its own share
                rho_local = f.sum(0).clamp(min=1e-6)
                ux_local = (f * cx3d).sum(0) / rho_local
                rho_add = torch.zeros_like(rho_init)
                rho_add[face_mask] = rho_face * delta_x  # 0.01 per cell per step
                feq_add = equilibrium3d(rho_add.clamp(min=0, max=3.0),
                                         ux_local.clamp(-0.5, 0.5),
                                         torch.zeros_like(rho_init),
                                         torch.zeros_like(rho_init), device=dev)
                f[:, face_mask] += feq_add[:, face_mask]

        # Measurement
        if step % 400 == 0 or step == n_steps:
            rg = f.sum(0)
            if not torch.isnan(rg).any():
                ra = float(rg[fluid].mean())
                V_now = px * ny * nz
                cr = V0 / max(V_now, 1)
                T = cr ** (gamma - 1)
                p_iso = ra * cs2
                p_adi = ra * cs2 * T
                p_exp_iso = rho0 * cs2 * cr
                p_exp_adi = rho0 * cs2 * T0 * cr ** gamma if (T0 := 1.0) else 0
                mass = float(f[:, fluid].sum())
                rho_min = float(rg[fluid].min())
                rho_max = float(rg[fluid].max())
                print(f'step {step}: px={px} V_ratio={cr:.2f} rho={ra:.4f} '
                      f'p_iso={p_iso:.4f} (exp={p_exp_iso:.4f} r={p_iso/p_exp_iso:.3f}) '
                      f'p_adi={p_adi:.4f} (exp={p_exp_adi:.4f} r={p_adi/p_exp_adi:.3f}) '
                      f'mass={mass:.0f} range=[{rho_min:.3f},{rho_max:.3f}]', flush=True)
            else:
                print(f'step {step}: NaN!', flush=True); break

    print(f'\nFinal: {time.time()-t0:.1f}s' if (t0 := locals().get('t0')) else '', flush=True)

if __name__ == '__main__':
    import time
    t0 = time.time()
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--nx', type=int, default=64)
    p.add_argument('--ny', type=int, default=32)
    p.add_argument('--nz', type=int, default=32)
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--device', default='sdaa:3')
    p.add_argument('--ps', type=float, default=0.01)
    args = p.parse_args()
    run_piston_virtual(args.nx, args.ny, args.nz, args.steps, args.device, args.ps)
