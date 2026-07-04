"""Bhaga & Weber (1981) 3D bubble rise benchmark.

Case B: Eo=10, Mo=0.001, Re_ref=7.0 (ellipsoidal regime)

LBM parameters:
  ρ_l=1.0, ρ_g=0.1, R=10, d=20, gz=0.001
  σ=0.036, ν_l=0.085 (τ=0.755)
  Eo = g(ρ_l-ρ_g)d²/σ = 10.0
  Mo = gμ_l⁴(ρ_l-ρ_g)/(ρ_l²σ³) = 0.001

Based on verified phase-field + Fakhari framework:
  - Variable density + hydrostatic gradient
  - Guo gravity on liquid only
  - Fakhari anti-diffusion
  - Surface tension via chemical potential force: F = μ·∇φ
  - Gas mass conservation
"""
import sys, math, torch, numpy as np
sys.path.insert(0, '/root/TensorLBM_test/src')
from tensorlbm.d3q19 import equilibrium3d, C as C3D, W as W3D
from tensorlbm.solver3d import stream3d
from tensorlbm.boundaries3d import bounce_back_cells_3d


def run_bhaga_benchmark(nx=48, ny=96, nz=48, n_steps=6000,
                        device='sdaa:0'):
    """Bhaga & Weber Case B: Eo=10, Mo=0.001, Re_ref=7."""
    dev = torch.device(device)
    cs2 = 1.0 / 3.0

    # Physical parameters (LBM units)
    rho_liq = 1.0
    rho_gas0 = 0.1        # density ratio 10:1
    R0 = 10.0             # bubble radius (d=20)
    d = 2 * R0            # diameter
    gz = 0.001            # gravity
    sigma = 0.036         # surface tension (Eo=10)
    nu_l = 0.0849         # liquid viscosity (Mo=0.001)
    tau_f = 3.0 * nu_l + 0.5  # = 0.7547
    tau_g = 0.55
    gamma = 1.4           # adiabatic index

    # Cahn-Hilliard / Fakhari surface tension parameters
    W = 4.0               # interface width
    Beta = 12.0 * sigma / W   # = 0.108
    k_grad = 1.5 * sigma * W  # = 0.216
    A_coef = Beta          # double-well coefficient
    B_coef = Beta
    kappa_ch = k_grad      # gradient penalty
    alpha_ac = 0.02        # Fakhari anti-diffusion

    # Dimensionless numbers
    Eo = gz * (rho_liq - rho_gas0) * d**2 / sigma
    Mo = gz * nu_l**4 * (rho_liq - rho_gas0) / (rho_liq**2 * sigma**3)
    Re_ref = 7.0
    v_ref = Re_ref * nu_l / (rho_liq * d)

    # Gas mass conservation
    p_gas0 = rho_gas0 * cs2
    p_inf = rho_liq * cs2

    # Bubble initial position (lower quarter)
    cx_b, cy_b, cz_b = nx//2, ny//4, nz//2

    # Lattice
    c = C3D.to(dev).float()
    w = W3D.to(dev).float().view(19, 1, 1, 1)
    cx3d = c[:, 0].view(19, 1, 1, 1)
    cy3d = c[:, 1].view(19, 1, 1, 1)
    cz3d = c[:, 2].view(19, 1, 1, 1)
    opp = torch.tensor([0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17], device=dev)

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=dev), torch.arange(ny, device=dev),
        torch.arange(nx, device=dev), indexing='ij')

    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=dev)
    solid[:, 0, :] = True; solid[:, -1, :] = True
    solid[:, :, 0] = True; solid[:, :, -1] = True
    solid[0, :, :] = True; solid[-1, :, :] = True
    fluid_mask = ~solid

    # Phase field: -1 (gas) inside bubble, +1 (liquid) outside
    dist = torch.sqrt((xx - cx_b)**2 + (yy - cy_b)**2 + (zz - cz_b)**2)
    phi = torch.tanh((dist - R0) / (W * 0.5))

    # Initialize f: uniform density
    rho_init = torch.ones(nz, ny, nx, device=dev)
    f = equilibrium3d(rho_init, torch.zeros_like(rho_init),
                      torch.zeros_like(rho_init), torch.zeros_like(rho_init), device=dev)
    g = _init_g_equilibrium(phi, torch.zeros_like(phi), torch.zeros_like(phi),
                             torch.zeros_like(phi), c, w)

    # Actual initial gas volume
    V0 = float((phi < 0).float().mul(fluid_mask.float()).sum())

    print(f'=== Bhaga & Weber Case B ===', flush=True)
    print(f'Eo={Eo:.2f}  Mo={Mo:.6f}  Re_ref={Re_ref:.1f}  v_ref={v_ref:.5f}', flush=True)
    print(f'Grid: {nx}×{ny}×{nz}  R0={R0}  d={d}', flush=True)
    print(f'ρ_l={rho_liq} ρ_g={rho_gas0} σ={sigma} ν_l={nu_l:.4f} τ={tau_f:.4f}', flush=True)
    print(f'gz={gz}  V0={V0:.0f}', flush=True)
    print(flush=True)

    ts, cy_track, v_track, R_track = [], [], [], []

    for step in range(1, n_steps + 1):
        # === 1. Macroscopic ===
        rho = f.sum(0)
        ux = (f * cx3d).sum(0) / rho.clamp(min=1e-6)
        uy = (f * cy3d).sum(0) / rho.clamp(min=1e-6)
        uz = (f * cz3d).sum(0) / rho.clamp(min=1e-6)

        # === 2. Phase field ===
        phi = g.sum(0).clamp(-1.0, 1.0)

        # === 3. Gas volume & pressure ===
        V_gas = float((phi < 0).float().mul(fluid_mask.float()).sum())
        V_gas = max(V_gas, 1.0)
        p_gas = p_gas0 * (V0 / V_gas) ** gamma

        # === 4. Density: variable + hydrostatic gradient ===
        rho_gas = p_gas / cs2
        rho_liq_h = rho_liq - rho_liq * gz * yy.float() / cs2
        rho_field = rho_gas + (rho_liq_h - rho_gas) * (phi + 1) / 2

        # === 5. Flow collision (density replacement + gravity + surface tension) ===
        rho_post = f.sum(0)
        ux_post = (f * cx3d).sum(0) / rho_post.clamp(min=1e-6)
        uy_post = (f * cy3d).sum(0) / rho_post.clamp(min=1e-6)
        uz_post = (f * cz3d).sum(0) / rho_post.clamp(min=1e-6)

        # Density replacement
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

        # Guo body forces: gravity (liquid) + surface tension (interface)
        # Gravity on liquid only
        Fy_grav = -rho_liq * gz * (phi + 1.0) / 2.0

        # Surface tension: F = μ·∇φ (chemical potential gradient)
        grad_phi_x = 0.5 * (torch.roll(phi, -1, dims=2) - torch.roll(phi, 1, dims=2))
        grad_phi_y = 0.5 * (torch.roll(phi, -1, dims=1) - torch.roll(phi, 1, dims=1))
        grad_phi_z = 0.5 * (torch.roll(phi, -1, dims=0) - torch.roll(phi, 1, dims=0))
        lap_phi = _laplacian_3d(phi)
        mu = -A_coef * phi + B_coef * phi**3 - kappa_ch * lap_phi
        Fx_st = mu * grad_phi_x
        Fy_st = mu * grad_phi_y
        Fz_st = mu * grad_phi_z

        # Total force
        Fx = Fx_st
        Fy = Fy_grav + Fy_st
        Fz = Fz_st

        cu_force = cx3d * Fx.unsqueeze(0) + cy3d * Fy.unsqueeze(0) + cz3d * Fz.unsqueeze(0)
        f = f + (1.0 - 0.5/tau_f) * w * cu_force / cs2
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

        lap_phi_adv = _laplacian_3d(phi_adv)
        mu_adv = -A_coef * phi_adv + B_coef * phi_adv**3 - kappa_ch * lap_phi_adv
        lap_mu = _laplacian_3d(mu_adv)
        M_mob = cs2 * (tau_g - 0.5)
        phi_new = phi_adv + M_mob * lap_mu

        ac_source = alpha_ac * (1.0 - phi_new**2) / W * torch.sign(phi_new)
        phi_new = phi_new + ac_source
        phi_new = phi_new.clamp(-1.0, 1.0)
        phi_new[solid] = 1.0
        g = _init_g_equilibrium(phi_new, ux_s, uy_s, uz_s, c, w)

        # === 7. Streaming + bounce-back ===
        f = stream3d(f)
        f = bounce_back_cells_3d(f, solid)

        # === 8. Measurement ===
        if step % 200 == 0 or step == n_steps:
            gas_mask = (phi_new < 0) & fluid_mask
            if gas_mask.any():
                cy_bubble = float(yy[gas_mask].float().mean())
                v_bubble = float(uy[gas_mask].mean())
                # Equivalent radius from volume
                V_now = float(gas_mask.sum())
                R_now = (3.0 * V_now / (4.0 * math.pi)) ** (1.0/3.0)
            else:
                cy_bubble = 0; v_bubble = 0; R_now = 0; V_now = 0

            # Reynolds number
            Re_now = rho_liq * abs(v_bubble) * d / nu_l if v_bubble != 0 else 0

            phi_min = float(phi_new[fluid_mask].min())
            phi_max = float(phi_new[fluid_mask].max())
            rho_min = float(rho_field[fluid_mask].min())
            rho_max = float(rho_field[fluid_mask].max())

            ts.append(step)
            cy_track.append(cy_bubble)
            v_track.append(v_bubble)
            R_track.append(R_now)

            print(f'step {step:4d}: cy={cy_bubble:.1f} v={v_bubble:+.5f} '
                  f'Re={Re_now:.2f}/{Re_ref:.0f} R={R_now:.2f} V={V_now:.0f} '
                  f'ρ=[{rho_min:.3f},{rho_max:.3f}] φ=[{phi_min:.2f},{phi_max:.2f}]',
                  flush=True)

    return np.array(ts), np.array(cy_track), np.array(v_track), np.array(R_track)


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
    p = argparse.ArgumentParser(description='Bhaga & Weber bubble rise benchmark')
    p.add_argument('--nx', type=int, default=48)
    p.add_argument('--ny', type=int, default=96)
    p.add_argument('--nz', type=int, default=48)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--device', default='sdaa:0')
    g = p.parse_args()

    print('='*60)
    ts, cy, v, R = run_bhaga_benchmark(g.nx, g.ny, g.nz, g.steps, g.device)
    print()
    print('='*60)
    print('=== BHAGA & WEBER CASE B SUMMARY ===')
    print(f'Eo=10.0  Mo=0.001  Re_ref=7.0')
    print(f'Bubble: cy {cy[0]:.1f} → {cy[-1]:.1f} (rise {cy[-1]-cy[0]:.1f} cells)')
    v_final = v[-1]
    Re_final = 1.0 * abs(v_final) * 20.0 / 0.0849
    print(f'Terminal velocity: v={v_final:.5f} (ref={0.02970:.5f})')
    print(f'Reynolds: Re={Re_final:.2f} (ref=7.0, error={abs(Re_final-7.0)/7.0*100:.1f}%)')
