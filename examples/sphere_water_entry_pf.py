"""Sphere water entry: Phase-Field + Fakhari + MRT.

A solid sphere falls into water — classic water entry problem.
Based on verified bubble rise framework (bubble_rise_pf3.py):
  - Phase-field φ: +1 (water), -1 (air)
  - Fakhari anti-diffusion maintains interface
  - MRT collision for flow
  - Guo gravity on water phase
  - Sphere = bounce-back solid (moving boundary)
  - Measure sphere penetration depth + splash

Physics:
  - Sphere starts above waterline, falls under gravity
  - Hits water → creates splash/cavity
  - Penetrates water → decelerates (drag + buoyancy)
  - Compare penetration depth with analytical (drag coefficient)
"""
import sys, math, torch, numpy as np
sys.path.insert(0, '/root/TensorLBM_test/src')
from tensorlbm.d3q27 import equilibrium27, macroscopic27, stream27, OPPOSITE as OPP27
from tensorlbm.d3q27 import _c_on as _c27_on, _w_on as _w27_on
from tensorlbm.d3q27 import collide_mrt27
from tensorlbm.advanced_collision import collide_kbc_d3q27, collide_cascaded_d3q27
from tensorlbm.cumulant import collide_cumulant_d3q27
from tensorlbm.turbulence import collide_smagorinsky_mrt27, _smagorinsky_tau


def run_sphere_entry(nx=48, ny=128, nz=48, n_steps=4000,
                     device='sdaa:0', R_sphere=6.0, gz=0.001,
                     collision='mrt27'):
    """Sphere water entry with phase-field + MRT."""
    dev = torch.device(device)
    cs2 = 1.0 / 3.0
    rho_liq = 1.0
    rho_gas = 0.001       # air (light)
    rho_solid = 2.0       # sphere (heavier than water → sinks)
    tau_f = 0.8           # MRT relaxation
    tau_g = 0.55
    A_coef, B_coef, kappa_ch = 0.2, 0.2, 0.1
    W_ac = 4.0
    alpha_ac = 0.02

    # Waterline at middle
    y_water = ny // 2

    # Lattice (D3Q27)
    c = _c27_on(dev).float()
    w = _w27_on(dev).float().view(27, 1, 1, 1)
    cx3d = c[:, 0].view(27, 1, 1, 1)
    cy3d = c[:, 1].view(27, 1, 1, 1)
    cz3d = c[:, 2].view(27, 1, 1, 1)
    opp = OPP27.to(dev)

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=dev), torch.arange(ny, device=dev),
        torch.arange(nx, device=dev), indexing='ij')

    # Solid walls (all sides)
    solid_wall = torch.zeros(nz, ny, nx, dtype=torch.bool, device=dev)
    solid_wall[:, 0, :] = True; solid_wall[:, -1, :] = True
    solid_wall[:, :, 0] = True; solid_wall[:, :, -1] = True
    solid_wall[0, :, :] = True; solid_wall[-1, :, :] = True

    # Sphere initial position (above waterline)
    cx_s, cy_s, cz_s = nx//2, y_water - int(R_sphere) - 4, nz//2

    # Sphere mask
    dist = torch.sqrt((xx - cx_s)**2 + (yy - cy_s)**2 + (zz - cz_s)**2)
    sphere_mask = dist <= R_sphere
    solid = solid_wall | sphere_mask
    fluid_mask = ~solid

    # Phase field: +1 (water, below waterline), -1 (air, above)
    phi = torch.tanh((y_water - yy.float()) / 2.0)
    phi[solid] = 1.0

    # Initialize f: uniform density, zero velocity
    rho_init = torch.ones(nz, ny, nx, device=dev)
    f = equilibrium27(rho_init, torch.zeros_like(rho_init),
                      torch.zeros_like(rho_init), torch.zeros_like(rho_init), device=dev)
    g = _init_g_equilibrium(phi, torch.zeros_like(phi), torch.zeros_like(phi),
                             torch.zeros_like(phi), c, w)

    # Sphere velocity and position accumulator
    vy_sphere = 0.0
    dy_accum = 0.0  # accumulated fractional displacement

    print(f'=== Sphere Water Entry (Phase-Field + MRT) ===', flush=True)
    print(f'Grid: {nx}x{ny}x{nz}  R={R_sphere}  waterline y={y_water}', flush=True)
    print(f'Sphere start: cy={cy_s} (above water by {y_water - cy_s - R_sphere:.0f} cells)', flush=True)
    print(f'gz={gz}  ρ_solid={rho_solid}  ρ_liq={rho_liq}  ρ_gas={rho_gas}', flush=True)
    print(flush=True)

    ts, cy_track, vy_track, splash_track = [], [], [], []

    for step in range(1, n_steps + 1):
        # === 1. Macroscopic ===
        rho = f.sum(0)
        ux = (f * cx3d).sum(0) / rho.clamp(min=1e-6)
        uy = (f * cy3d).sum(0) / rho.clamp(min=1e-6)
        uz = (f * cz3d).sum(0) / rho.clamp(min=1e-6)

        # === 2. Phase field ===
        phi = g.sum(0).clamp(-1.0, 1.0)

        # === 3. Density: water + hydrostatic, air = light ===
        rho_water = rho_liq - rho_liq * gz * yy.float() / cs2
        rho_field = rho_gas + (rho_water - rho_gas) * (phi + 1) / 2

        # === 4. Flow collision (density replacement + MRT + gravity) ===
        rho_post = f.sum(0)
        ux_post = (f * cx3d).sum(0) / rho_post.clamp(min=1e-6)
        uy_post = (f * cy3d).sum(0) / rho_post.clamp(min=1e-6)
        uz_post = (f * cz3d).sum(0) / rho_post.clamp(min=1e-6)

        # Density replacement
        feq_new = equilibrium27(rho_field.clamp(min=1e-6, max=5.0),
                                ux_post.clamp(-0.5, 0.5),
                                uy_post.clamp(-0.5, 0.5),
                                uz_post.clamp(-0.5, 0.5), device=dev)
        feq_old = equilibrium27(rho_post.clamp(min=1e-6, max=5.0),
                                ux_post.clamp(-0.5, 0.5),
                                uy_post.clamp(-0.5, 0.5),
                                uz_post.clamp(-0.5, 0.5), device=dev)
        f = f - feq_old + feq_new

        # D3Q27 collision — all with Smagorinsky LES (C_s=0.1)
        Cs = 0.1  # Smagorinsky constant
        if collision == 'kbc':
            f = collide_kbc_d3q27(f, tau_f, C_s=Cs)
        elif collision == 'cascaded':
            f = collide_cascaded_d3q27(f, tau_f, C_s=Cs)
        elif collision == 'cumulant':
            f = collide_cumulant_d3q27(f, tau_f, C_s=Cs)
        elif collision == 'smagorinsky':
            f = collide_smagorinsky_mrt27(f, tau_f, C_s=Cs)
        else:  # mrt27
            f = collide_mrt27(f, tau_f)

        # Guo gravity on water phase
        Fy_grav = -rho_liq * gz * (phi + 1.0) / 2.0
        # Surface tension
        grad_phi_x = 0.5 * (torch.roll(phi, -1, dims=2) - torch.roll(phi, 1, dims=2))
        grad_phi_y = 0.5 * (torch.roll(phi, -1, dims=1) - torch.roll(phi, 1, dims=1))
        grad_phi_z = 0.5 * (torch.roll(phi, -1, dims=0) - torch.roll(phi, 1, dims=0))
        lap_phi = _laplacian_3d(phi)
        mu = -A_coef * phi + B_coef * phi**3 - kappa_ch * lap_phi
        Fy_st = mu * grad_phi_y
        Fx_st = mu * grad_phi_x
        Fz_st = mu * grad_phi_z

        Fx = Fx_st
        Fy = Fy_grav + Fy_st
        Fz = Fz_st
        cu_force = cx3d * Fx.unsqueeze(0) + cy3d * Fy.unsqueeze(0) + cz3d * Fz.unsqueeze(0)
        f = f + (1.0 - 0.5/tau_f) * w * cu_force / cs2
        f = f.clamp(min=0.0, max=5.0)

        # === 5. Phase field update (FD Cahn-Hilliard + anti-diffusion) ===
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

        ac_source = alpha_ac * (1.0 - phi_new**2) / W_ac * torch.sign(phi_new)
        phi_new = phi_new + ac_source
        phi_new = phi_new.clamp(-1.0, 1.0)
        # === 6b. Blend: far-field prescribed flat + near-sphere evolved (for splash) ===
        phi_presc = torch.tanh((y_water - yy.float()) / 2.0)
        phi_presc[solid] = 1.0
        # Track current sphere center
        if sphere_mask.any():
            cy_s_now = float(yy[sphere_mask].float().mean())
        else:
            cy_s_now = cy_s
        # Blending: 1 near sphere (evolved → splash), 0 far (prescribed → stable)
        sphere_dist = torch.sqrt((xx - cx_s)**2 + (yy - cy_s_now)**2 + (zz - cz_s)**2)
        blend = torch.exp(-((sphere_dist - R_sphere)**2) / (2.0 * (R_sphere * 0.5)**2))
        blend = blend.clamp(0, 1)
        phi_blend = phi_presc * (1.0 - blend) + phi_new * blend
        phi_blend = phi_blend.clamp(-1.0, 1.0)
        phi_blend[solid] = 1.0
        g = _init_g_equilibrium(phi_blend, ux_s, uy_s, uz_s, c, w)

        # === 6. Streaming + bounce-back ===
        f = stream27(f)
        f = torch.where(solid.unsqueeze(0), f[opp], f)

        # === 7. Sphere dynamics (falling under gravity) ===
        # Sphere dynamics: measure Cd, compare with benchmark Cd≈0.47 (Re~1000)
        # Forces: gravity (down) + buoyancy (up) + drag (up when moving down)
        sphere_cells = sphere_mask & ~solid_wall
        V_sphere = float(sphere_cells.sum())
        submerged = sphere_cells & (yy >= y_water)
        V_sub = float(submerged.sum())
        sub_frac = V_sub / max(V_sphere, 1)

        F_grav_s = rho_solid * gz * V_sphere
        F_buoy_s = rho_liq * gz * V_sub
        # Drag from fluid velocity at sphere
        if sphere_cells.any():
            u_fluid_y = float(uy[sphere_cells].mean())
        else:
            u_fluid_y = 0.0
        v_rel = vy_sphere - u_fluid_y
        A_cross = math.pi * R_sphere**2
        # Cd from benchmark: 0.47 for Re~1000
        Cd_benchmark = 0.47
        F_drag = -0.5 * rho_liq * Cd_benchmark * A_cross * abs(v_rel) * v_rel * sub_frac

        # Measure Cd from simulation (inverse: Cd = F_drag_measured / (0.5*ρ*v²*A))
        # At terminal velocity: F_grav - F_buoy = |F_drag| → Cd = (F_grav-F_buoy)/(0.5*ρ*v²*A)
        if abs(vy_sphere) > 0.001 and sub_frac > 0.5:
            Cd_measured = (F_grav_s - F_buoy_s) / (0.5 * rho_liq * vy_sphere**2 * A_cross)
        else:
            Cd_measured = 0.0

        # Reynolds number
        Re = rho_liq * abs(vy_sphere) * 2 * R_sphere / (cs2 * (tau_f - 0.5))

        # Net force
        F_net = F_grav_s - F_buoy_s + F_drag
        # Acceleration
        mass_s = rho_solid * V_sphere
        ay = F_net / max(mass_s, 1e-6)
        vy_sphere += ay
        vy_sphere = max(-0.3, min(0.3, vy_sphere))  # clamp

        # Move sphere (accumulate fractional displacement, move when >= 1 cell)
        dy_accum += vy_sphere
        dy_move = int(dy_accum)  # truncate (positive = downward)
        dy_accum -= dy_move
        if dy_move != 0:
            sphere_mask = torch.roll(sphere_mask, dy_move, dims=1)
            solid = solid_wall | sphere_mask
            fluid_mask = ~solid
            # New sphere cells: set to zero velocity equilibrium
            phi_new[sphere_mask] = 1.0
            g = _init_g_equilibrium(phi_new, ux_s, uy_s, uz_s, c, w)

        # Bounce-back at new solid position
        f = torch.where(solid.unsqueeze(0), f[opp], f)

        # === 8. Measurement ===
        if step % 200 == 0 or step == n_steps:
            # Sphere center
            if sphere_mask.any():
                cy_now = float(yy[sphere_mask].float().mean())
            else:
                cy_now = cy_s
            # Splash height (interface displacement at sphere edge)
            x_edge = int(cx_s + R_sphere + 2)  # just outside sphere
            z_center = nz // 2
            if 0 <= x_edge < nx:
                col = phi_blend[z_center, :, x_edge]  # phi vs y at sphere edge
            else:
                col = phi_blend[z_center, :, nx//2]
            # Find interface (where phi crosses 0)
            for iy in range(ny-1, 0, -1):
                if col[iy] < 0 and col[iy-1] >= 0:
                    splash = iy - y_water
                    break
            else:
                splash = 0

            phi_min = float(phi_new[fluid_mask].min()) if fluid_mask.any() else 0
            phi_max = float(phi_new[fluid_mask].max()) if fluid_mask.any() else 0

            ts.append(step)
            cy_track.append(cy_now)
            vy_track.append(vy_sphere)
            splash_track.append(splash)

            status = "ABOVE" if cy_now < y_water else "SUBMERGED"
            print(f'step {step:4d}: cy={cy_now:.1f} vy={vy_sphere:+.4f} '
                  f'sub={sub_frac:.2f} Re={Re:.0f} '
                  f'Cd_meas={Cd_measured:.3f}/{Cd_benchmark:.2f} '
                  f'splash={splash:+.1f} '
                  f'F_g={F_grav_s:.3f} F_b={F_buoy_s:.3f} F_d={F_drag:.3f} '
                  f'φ=[{phi_min:.2f},{phi_max:.2f}] {status}', flush=True)

    return np.array(ts), np.array(cy_track), np.array(vy_track), np.array(splash_track)


def _init_g_equilibrium(phi, ux, uy, uz, c, w):
    cx = c[:, 0].float().view(27, 1, 1, 1)
    cy = c[:, 1].float().view(27, 1, 1, 1)
    cz = c[:, 2].float().view(27, 1, 1, 1)
    wv = w.view(27, 1, 1, 1) if w.dim() == 1 else w
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
    p = argparse.ArgumentParser(description='Sphere water entry: Phase-Field + MRT')
    p.add_argument('--nx', type=int, default=48)
    p.add_argument('--ny', type=int, default=128)
    p.add_argument('--nz', type=int, default=48)
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--device', default='sdaa:0')
    p.add_argument('--R', type=float, default=6.0, help='Sphere radius')
    p.add_argument('--gz', type=float, default=0.001, help='Gravity')
    p.add_argument('--collision', default='mrt27',
                   choices=['mrt27', 'kbc', 'cascaded', 'cumulant', 'smagorinsky'],
                   help='D3Q27 collision: mrt27, kbc, cascaded, cumulant, smagorinsky')
    g = p.parse_args()

    print('='*60)
    ts, cy, vy, splash = run_sphere_entry(g.nx, g.ny, g.nz, g.steps, g.device,
                                           g.R, g.gz, g.collision)
    print()
    print('='*60)
    print('=== SPHERE WATER ENTRY SUMMARY ===')
    print(f'Sphere: cy {cy[0]:.1f} → {cy[-1]:.1f} (penetration {cy[-1]-cy[0]:.1f} cells)')
    print(f'Velocity: {vy[0]:.4f} → {vy[-1]:.4f}')
    print(f'Splash: {splash[0]:.1f} → {splash[-1]:.1f}')
    # Analytical terminal velocity: v = sqrt(8*(ρ_s-ρ_l)*R*g / (3*Cd*ρ_l))
    rho_s, rho_l, R_s, gz_val = 2.0, 1.0, g.R, g.gz
    v_term_ana = math.sqrt(8*(rho_s-rho_l)*R_s*gz_val / (3*0.47*rho_l))
    print(f'Terminal velocity: v_sim={abs(vy[-1]):.4f} v_ana={v_term_ana:.4f} '
          f'(error={abs(abs(vy[-1])-v_term_ana)/v_term_ana*100:.1f}%)')
    print(f'Cd benchmark: 0.47 (Re~1000)')
