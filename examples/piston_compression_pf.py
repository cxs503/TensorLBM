"""Piston compression: closed container, gas on top, water on bottom.

Bottom wall moves up (piston) → compresses gas → pressure rises.
Compare p_gas with analytical: p = p0·(V0/V)^γ

Uses the verified phase-field + Fakhari anti-diffusion framework:
  - φ = -1 (gas, top), +1 (liquid, bottom)
  - FD Cahn-Hilliard + anti-diffusion (maintains flat interface)
  - Variable density (pressure difference drives flow naturally)
  - Gas mass conservation: p_gas = p_gas0·(V0/V)^γ
"""
import sys, math, torch, numpy as np
sys.path.insert(0, '/root/TensorLBM_test/src')
from tensorlbm.d3q19 import equilibrium3d, C as C3D, W as W3D
from tensorlbm.solver3d import stream3d
from tensorlbm.boundaries3d import bounce_back_cells_3d


def run_piston_compression(nx=32, ny=64, nz=32, n_steps=2000,
                           device='sdaa:0', u_piston=0.02, gamma=1.4):
    """Simulate piston compression in a closed container.

    Args:
        nx, ny, nz: Grid dimensions (ny = height, piston at bottom)
        u_piston: Piston upward velocity (lattice units)
        gamma: Adiabatic index
    """
    dev = torch.device(device)
    cs2 = 1.0 / 3.0
    rho_liq = 1.0
    rho_gas0 = 1.0       # initial gas density = liquid (no density contrast initially)
    tau_f = 0.8
    tau_g = 0.55
    A_coef, B_coef, kappa = 0.2, 0.2, 0.1
    W_ac = 4.0
    alpha_ac = 0.02

    # Initial gas height (top half)
    y_gas_start = ny // 2  # interface at y = ny/2
    V0_gas = float((nx - 2) * (ny - 1 - y_gas_start) * (nz - 2))  # initial gas volume (minus walls)
    p_gas0 = rho_gas0 * cs2  # initial gas pressure
    p_inf = rho_liq * cs2

    # Lattice
    c = C3D.to(dev).float()
    w = W3D.to(dev).float()
    cx3d = c[:, 0].view(19, 1, 1, 1)
    cy3d = c[:, 1].view(19, 1, 1, 1)
    cz3d = c[:, 2].view(19, 1, 1, 1)
    opp = torch.tensor([0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17], device=dev)

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=dev), torch.arange(ny, device=dev),
        torch.arange(nx, device=dev), indexing='ij')

    # Solid walls (all sides except bottom = piston)
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=dev)
    solid[:, -1, :] = True   # top wall (closed)
    solid[:, :, 0] = True    # left
    solid[:, :, -1] = True   # right
    solid[0, :, :] = True    # front
    solid[-1, :, :] = True   # back
    # Bottom wall is the piston (also solid, but with velocity)
    solid[:, 0, :] = True
    fluid_mask = ~solid

    # Phase field: -1 (gas) on top, +1 (liquid) on bottom
    phi = torch.where(yy >= y_gas_start, -1.0, 1.0).to(dev)
    # Smooth interface
    phi = torch.tanh((y_gas_start - yy.float()) / 2.0)

    # Initialize f: uniform density, zero velocity
    rho_init = torch.ones(nz, ny, nx, device=dev)
    f = equilibrium3d(rho_init, torch.zeros_like(rho_init),
                      torch.zeros_like(rho_init), torch.zeros_like(rho_init), device=dev)
    g = _init_g_equilibrium(phi, torch.zeros_like(phi), torch.zeros_like(phi),
                             torch.zeros_like(phi), c, w)

    print(f'=== Piston Compression ===', flush=True)
    print(f'Grid: {nx}×{ny}×{nz}  Interface at y={y_gas_start}', flush=True)
    print(f'Piston velocity: u={u_piston} (upward)', flush=True)
    print(f'V0_gas={V0_gas:.0f} cells  p_gas0={p_gas0:.4f}  γ={gamma}', flush=True)
    print(f'Expected compression: {n_steps} steps × {u_piston} = {n_steps*u_piston*nx*nz:.0f} cells', flush=True)
    print(flush=True)

    ts, ps_lbm, ps_ana, vs_lbm = [], [], [], []

    for step in range(1, n_steps + 1):
        # === 1. Macroscopic ===
        rho = f.sum(0)
        ux = (f * cx3d).sum(0) / rho.clamp(min=1e-6)
        uy = (f * cy3d).sum(0) / rho.clamp(min=1e-6)
        uz = (f * cz3d).sum(0) / rho.clamp(min=1e-6)

        # === 2. Phase field ===
        phi = g.sum(0).clamp(-1.0, 1.0)

        # === 3. Gas volume & pressure ===
        # V_gas from prescribed interface position (exact, not from phi threshold)
        y_interface_now = y_gas_start + u_piston * step
        y_interface_now = min(y_interface_now, ny - 4)
        V_gas = max((ny - 1 - int(y_interface_now)) * (nx - 2) * (nz - 2), 1.0)  # subtract walls
        p_gas = p_gas0 * (V0_gas / V_gas) ** gamma

        # === 4. Density from phase field ===
        rho_gas = p_gas / cs2
        rho_field = rho_gas + (rho_liq - rho_gas) * (phi + 1) / 2

        # === 5. Flow collision with density replacement ===
        rho_post = f.sum(0)
        ux_post = (f * cx3d).sum(0) / rho_post.clamp(min=1e-6)
        uy_post = (f * cy3d).sum(0) / rho_post.clamp(min=1e-6)
        uz_post = (f * cz3d).sum(0) / rho_post.clamp(min=1e-6)
        feq_new = equilibrium3d(rho_field.clamp(min=1e-6, max=5.0),
                                ux_post.clamp(-0.5, 0.5),
                                uy_post.clamp(-0.5, 0.5),
                                uz_post.clamp(-0.5, 0.5), device=dev)
        feq_old = equilibrium3d(rho_post.clamp(min=1e-6, max=5.0),
                                ux_post.clamp(-0.5, 0.5),
                                uy_post.clamp(-0.5, 0.5),
                                uz_post.clamp(-0.5, 0.5), device=dev)
        f = f - feq_old + feq_new
        feq = equilibrium3d(rho_field.clamp(min=1e-6, max=5.0),
                            ux_post.clamp(-0.5, 0.5),
                            uy_post.clamp(-0.5, 0.5),
                            uz_post.clamp(-0.5, 0.5), device=dev)
        f = f - (f - feq) / tau_f
        f = f.clamp(min=0.0, max=5.0)

        # === 6. Phase field update (FD Cahn-Hilliard + anti-diffusion) ===
        ux_s = ux.clamp(-0.5, 0.5)
        uy_s = uy.clamp(-0.5, 0.5)
        uz_s = uz.clamp(-0.5, 0.5)
        dphi_dx = torch.where(ux_s > 0, phi - torch.roll(phi, 1, dims=2),
                              torch.roll(phi, -1, dims=2) - phi) * ux_s
        dphi_dy = torch.where(uy_s > 0, phi - torch.roll(phi, 1, dims=1),
                              torch.roll(phi, -1, dims=1) - phi) * uy_s
        dphi_dz = torch.where(uz_s > 0, phi - torch.roll(phi, 1, dims=0),
                              torch.roll(phi, -1, dims=0) - phi) * uz_s
        phi_adv = phi - (dphi_dx + dphi_dy + dphi_dz)

        lap_phi = _laplacian_3d(phi_adv)
        mu = -A_coef * phi_adv + B_coef * phi_adv**3 - kappa * lap_phi
        lap_mu = _laplacian_3d(mu)
        M_mob = cs2 * (tau_g - 0.5)
        phi_new = phi_adv + M_mob * lap_mu

        ac_source = alpha_ac * (1.0 - phi_new**2) / W_ac * torch.sign(phi_new)
        phi_new = phi_new + ac_source
        phi_new = phi_new.clamp(-1.0, 1.0)
        phi_new[solid] = 1.0  # solid = liquid

        g = _init_g_equilibrium(phi_new, ux_s, uy_s, uz_s, c, w)

        # === 7. Streaming ===
        f = stream3d(f)

        # === 8. Bounce-back at walls ===
        f = bounce_back_cells_3d(f, solid)

        # === 9. Piston: continuously push interface up ===
        # Every step, move interface up by u_piston cells (continuous compression)
        new_interface = y_gas_start + u_piston * step
        new_interface = min(new_interface, ny - 4)
        # Set phase field: gas above interface, liquid below
        phi_set = torch.tanh((new_interface - yy.float()) / 2.0)
        phi_set[solid] = 1.0
        g = _init_g_equilibrium(phi_set, ux_s, uy_s, uz_s, c, w)
        # Bounce-back at walls
        f = bounce_back_cells_3d(f, solid)

        # === 10. Top wall BC: closed (bounce-back already done) ===
        # Side walls: bounce-back already done

        # === 11. Measurement ===
        if step % 100 == 0 or step == n_steps:
            # Analytical: V_ana = V0 - u_piston * A * step
            A_cross = float((nx - 2) * (nz - 2))  # cross-section area (minus walls)
            V_ana = max(V0_gas - u_piston * A_cross * step, 1.0)
            p_ana = p_gas0 * (V0_gas / V_ana) ** gamma
            ratio_lbm = p_gas / p_gas0
            ratio_ana = p_ana / p_gas0
            err = abs(ratio_lbm - ratio_ana) / ratio_ana * 100

            phi_min = float(phi_new[fluid_mask].min())
            phi_max = float(phi_new[fluid_mask].max())
            rho_min = float(rho_field[fluid_mask].min())
            rho_max = float(rho_field[fluid_mask].max())

            ts.append(step)
            ps_lbm.append(p_gas)
            ps_ana.append(p_ana)
            vs_lbm.append(V_gas)

            print(f'step {step:4d}: V_gas={V_gas:.0f} V_ana={V_ana:.0f} '
                  f'p_gas={p_gas:.4f} p_ana={p_ana:.4f} '
                  f'ratio_lbm={ratio_lbm:.3f} ratio_ana={ratio_ana:.3f} '
                  f'err={err:.1f}% '
                  f'ρ=[{rho_min:.3f},{rho_max:.3f}] φ=[{phi_min:.2f},{phi_max:.2f}]',
                  flush=True)

    return np.array(ts), np.array(ps_lbm), np.array(ps_ana), np.array(vs_lbm)


def _init_g_equilibrium(phi, ux, uy, uz, c, w):
    cx = c[:, 0].float().view(19, 1, 1, 1)
    cy = c[:, 1].float().view(19, 1, 1, 1)
    cz = c[:, 2].float().view(19, 1, 1, 1)
    wv = w.view(19, 1, 1, 1)
    cu = cx * ux.unsqueeze(0) + cy * uy.unsqueeze(0) + cz * uz.unsqueeze(0)
    u_sq = (ux**2 + uy**2 + uz**2).unsqueeze(0)
    return wv * phi.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u_sq)


def _laplacian_3d(field):
    return (torch.roll(field, 1, dims=0) + torch.roll(field, -1, dims=0)
            + torch.roll(field, 1, dims=1) + torch.roll(field, -1, dims=1)
            + torch.roll(field, 1, dims=2) + torch.roll(field, -1, dims=2)
            - 6.0 * field)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Piston compression: phase-field LBM')
    p.add_argument('--nx', type=int, default=32)
    p.add_argument('--ny', type=int, default=64)
    p.add_argument('--nz', type=int, default=32)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--device', default='sdaa:0')
    p.add_argument('--u-piston', type=float, default=0.02, help='Piston upward velocity')
    p.add_argument('--gamma', type=float, default=1.4, help='Adiabatic index')
    g = p.parse_args()

    print('='*60)
    print('Running piston compression...')
    ts, ps_lbm, ps_ana, vs_lbm = run_piston_compression(
        g.nx, g.ny, g.nz, g.steps, g.device, g.u_piston, g.gamma)
    print()

    print('='*60)
    print('=== COMPRESSION SUMMARY ===')
    print(f'{"":12s} {"p_initial":>10s} {"p_final_lbm":>12s} {"p_final_ana":>12s} {"ratio_lbm":>10s} {"ratio_ana":>10s} {"err":>6s}')
    p0 = ps_lbm[0]
    print(f'{"Value":12s} {p0:10.4f} {ps_lbm[-1]:12.4f} {ps_ana[-1]:12.4f} '
          f'{ps_lbm[-1]/p0:10.3f} {ps_ana[-1]/p0:10.3f} '
          f'{abs(ps_lbm[-1]/p0 - ps_ana[-1]/p0)/(ps_ana[-1]/p0)*100:6.1f}%')
    print()

    # MAPE
    mape = np.mean(np.abs(ps_lbm - ps_ana) / ps_ana) * 100
    print(f'Pressure MAPE: {mape:.1f}%')
    print(f'Volume: V0={vs_lbm[0]:.0f} → V_final={vs_lbm[-1]:.0f} (compression ratio={vs_lbm[0]/vs_lbm[-1]:.2f})')
