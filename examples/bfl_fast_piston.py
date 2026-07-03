"""BFL fast piston validation — test q != 1.0.

Previous test had q=1.0 always (slow piston). This script tests
faster piston speeds to exercise the BFL distance correction.
"""
import sys, math, torch
sys.path.insert(0, '/root/TensorLBM_test/src')
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d, C as C3D, W as W3D, OPPOSITE as OPP
from tensorlbm.solver3d import stream3d, collide_bgk3d

def run_bfl_fast(piston_speed=0.05, n_steps=4000, device='sdaa:9'):
    """BFL piston with faster speed to get q != 1.0."""
    dev = torch.device(device)
    nx, ny, nz = 64, 32, 32
    cs2 = 1.0/3.0; rho0 = 1.0; ps = piston_speed

    rho = torch.ones(nz, ny, nx, device=dev)
    f = equilibrium3d(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho), device=dev)
    zz, yy, xx = torch.meshgrid(torch.arange(nz, device=dev), torch.arange(ny, device=dev),
                                 torch.arange(nx, device=dev), indexing='ij')
    opp = OPP.to(dev)
    c = C3D.to(dev).float()
    w = W3D.to(dev).float().view(19, 1, 1, 1)

    piston_pos = float(nx - 1)
    initial_vol = (nx - 1) * ny * nz
    total_mass = rho0 * initial_vol

    print(f'=== BFL Fast Piston (ps={ps}) ===', flush=True)
    print(f'device={device} steps={n_steps}', flush=True)

    for step in range(1, n_steps + 1):
        piston_pos = max(nx / 2, piston_pos - ps)
        px = int(piston_pos)
        solid = xx >= px; fluid = ~solid
        q = piston_pos - (px - 1)  # fractional distance

        f = collide_bgk3d(f, tau=0.8)
        f_pre = f.clone()
        f = stream3d(f)

        # Standard bounce-back
        f_swapped = f[opp]
        f_bb = f_swapped.clone()

        # BFL on near-piston cells (x = px-1)
        near = (xx == px - 1)
        if bool(near.any()):
            rho_near = f.sum(0)[near].clamp(min=1e-6)
            cu_wall = c[:, 0] * (-ps)  # piston moving in -x
            corr = 2.0 * rho_near.unsqueeze(0) * w.view(19, 1) * cu_wall.view(19, 1) / cs2

            f_opp_near = f_swapped[:, near]
            f_pre_near = f_pre[:, near]
            f_pre_opp_near = f_pre[opp][:, near]

            q_val = float(q)
            if q_val < 0.5:
                f_bfl = 2.0 * q_val * f_opp_near + (1.0 - 2.0 * q_val) * f_pre_near - corr
            else:
                q_s = max(q_val, 0.5)
                f_bfl = f_opp_near / (2.0 * q_s) + (2.0 * q_s - 1.0) / (2.0 * q_s) * f_pre_opp_near - corr

            tmp = f_bb.clone()
            tmp[:, near] = f_bfl
            f_bb = tmp

        f = torch.where(solid.unsqueeze(0), f_swapped, f)
        f = torch.where(near.unsqueeze(0), f_bb, f)

        # Fixed walls
        f[:, 0, :] = f_swapped[:, 0, :]; f[:, -1, :] = f_swapped[:, -1, :]
        f[:, :, 0] = f_swapped[:, :, 0]; f[:, :, -1] = f_swapped[:, :, -1]
        f[:, :, :, 0] = f_swapped[:, :, :, 0]

        # Mass conservation
        if step % 10 == 0:
            cm = float(f[:, fluid].sum())
            if cm > 1e-6 and not math.isnan(cm):
                f[:, fluid] *= total_mass / cm

        if step % 400 == 0 or step == n_steps:
            rg = f.sum(0)
            if not torch.isnan(rg).any():
                ra = float(rg[fluid].mean()); pa = ra * cs2
                cr = initial_vol / max(px * ny * nz, 1)
                pe = rho0 * cs2 * cr
                print(f'step {step}: q={q:.3f} V={cr:.2f} rho={ra:.4f} p={pa:.4f} '
                      f'p_exp={pe:.4f} ratio={pa/pe:.4f}', flush=True)
            else:
                print(f'step {step}: NaN! q={q:.3f}', flush=True); break

    print(f'Final: ps={ps} done', flush=True)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ps', type=float, default=0.05)
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--device', default='sdaa:9')
    args = p.parse_args()
    run_bfl_fast(args.ps, args.steps, args.device)
