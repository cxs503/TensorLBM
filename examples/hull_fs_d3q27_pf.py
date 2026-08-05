"""Free-surface ship hull resistance: D3Q27 + Phase-Field + LES.

Based on verified sphere water entry framework:
  - D3Q27 lattice (27 velocities)
  - Phase-field φ: +1 (water), -1 (air)
  - Prescribed flat interface in far-field + blend near hull (captures waves)
  - 5 collision operators with Smagorinsky LES (mrt27/kbc/cascaded/cumulant/smagorinsky)
  - Guo gravity on water phase → hydrostatic gradient → waves
  - Log-law wall function on hull surface
  - Far-field BC (inlet/outlet/sides)
  - Ct = Cf (friction) + Cw (wave-making)

Usage:
    python examples/hull_fs_d3q27_pf.py --hull wigley --collision cumulant --device sdaa:0
"""
import sys, math, time, torch, numpy as np
sys.path.insert(0, '/root/TensorLBM_test/src')
from tensorlbm.d3q27 import equilibrium27, macroscopic27, stream27, OPPOSITE as OPP27
from tensorlbm.d3q27 import _c_on as _c27_on, _w_on as _w27_on, collide_mrt27
from tensorlbm.advanced_collision import collide_kbc_d3q27, collide_cascaded_d3q27
from tensorlbm.cumulant import collide_cumulant_d3q27
from tensorlbm.turbulence import collide_smagorinsky_mrt27
from tensorlbm.ship_cad import build_hull_mask
from tensorlbm.hydrodynamics import ittc57_friction_coefficient, voxel_wetted_area

KAPPA = 0.41
B_CONST = 5.0


def run_hull_fs_d3q27(hull_type="wigley", nx=192, ny=96, nz=96,
                       re=1e5, fr=0.25, u_in=0.06,
                       n_steps=2000, warmup=500, device='sdaa:0',
                       collision='cumulant', use_wall_function=True):
    """Free-surface ship hull resistance with D3Q27 + Phase-Field + LES."""
    dev = torch.device(device)
    cs2 = 1.0 / 3.0
    rho_liq = 1.0
    rho_gas = 0.001
    rho_solid = 1.0  # hull (fixed, not moving)

    # Hull geometry
    fill_height = nz // 2  # waterline
    solid, stats = build_hull_mask(
        hull_type, nx, ny, nz,
        cx=int(nx * 0.3), cy=ny // 2, cz_keel=fill_height - 10,
        device="cpu")
    solid = solid.to(dev)
    fluid = ~solid
    S = voxel_wetted_area(solid, 1.0)
    dyn_p_S = 0.5 * rho_liq * u_in**2 * S

    # Water mask
    zz_arr = torch.arange(nz, device=dev).view(nz, 1, 1)
    water_mask = (zz_arr < fill_height).expand(nz, ny, nx)

    # Near-wall mask (submerged hull cells)
    nbrs = torch.zeros_like(solid)
    for ax, sgn in [(2,1),(2,-1),(1,1),(1,-1),(0,1),(0,-1)]:
        nbrs |= (fluid & torch.roll(solid, sgn, dims=ax))
    near = nbrs & water_mask

    # LBM parameters
    hull_length = max(6.0, 0.35 * nx)
    nu_lat = u_in * hull_length / re
    tau_f = max(3.0 * nu_lat + 0.5, 0.51)  # clamp for stability at high Re
    gz = u_in**2 / (fr**2 * hull_length) if fr > 0.001 else 0.0
    Cs = 0.1  # Smagorinsky constant

    # Phase-field parameters
    tau_g = 0.55
    A_coef, B_coef, kappa_ch = 0.2, 0.2, 0.1
    W_ac = 4.0
    alpha_ac = 0.02

    # ITTC reference
    cf_ittc = ittc57_friction_coefficient(re)
    ct_ref = cf_ittc * 1.15  # form factor (1+k)

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

    # Phase field: +1 (water below waterline), -1 (air above)
    phi = torch.tanh((fill_height - zz.float()) / 2.0)
    phi[solid] = 1.0

    # Initialize f: uniform density, inlet velocity
    rho_init = torch.ones(nz, ny, nx, device=dev)
    ux_init = torch.full((nz, ny, nx), u_in, device=dev)
    f = equilibrium27(rho_init, ux_init, torch.zeros_like(ux_init),
                      torch.zeros_like(ux_init))
    g = _init_g_equilibrium(phi, ux_init, torch.zeros_like(ux_init),
                             torch.zeros_like(ux_init), c, w)

    print(f'=== Free-Surface {hull_type} (D3Q27 + PF + {collision}) ===', flush=True)
    print(f'Re={re:.0e} Fr={fr:.2f} grid={nx}x{ny}x{nz}', flush=True)
    print(f'tau={tau_f:.5f} nu={nu_lat:.2e} g={gz:.6f} Cs={Cs}', flush=True)
    print(f'S={S:.0f} Cf_ITTC={cf_ittc:.5f} Ct_ref={ct_ref:.5f}', flush=True)
    print(f'Waterline z={fill_height} LES={Cs}', flush=True)
    print(flush=True)

    fric_list = []
    wave_list = []
    wake_x = int(nx * 0.7)
    t0 = time.time()

    for step in range(1, n_steps + 1):
        # === 1. Macroscopic ===
        rho = f.sum(0)
        ux = (f * cx3d).sum(0) / rho.clamp(min=1e-6)
        uy = (f * cy3d).sum(0) / rho.clamp(min=1e-6)
        uz = (f * cz3d).sum(0) / rho.clamp(min=1e-6)

        # === 2. Phase field ===
        phi = g.sum(0).clamp(-1.0, 1.0)

        # === 3. Density: water + hydrostatic, air = light ===
        rho_water = rho_liq - rho_liq * gz * zz.float() / cs2
        rho_field = rho_gas + (rho_water - rho_gas) * (phi + 1) / 2

        # === 4. Flow collision (density replacement + D3Q27 LES + gravity) ===
        rho_post = f.sum(0)
        ux_post = (f * cx3d).sum(0) / rho_post.clamp(min=1e-6)
        uy_post = (f * cy3d).sum(0) / rho_post.clamp(min=1e-6)
        uz_post = (f * cz3d).sum(0) / rho_post.clamp(min=1e-6)

        # Density replacement
        feq_new = equilibrium27(rho_field.clamp(min=1e-6, max=5.0),
                                ux_post.clamp(-0.5, 0.5),
                                uy_post.clamp(-0.5, 0.5),
                                uz_post.clamp(-0.5, 0.5))
        feq_old = equilibrium27(rho_post.clamp(min=1e-6, max=5.0),
                                ux_post.clamp(-0.5, 0.5),
                                uy_post.clamp(-0.5, 0.5),
                                uz_post.clamp(-0.5, 0.5))
        f = f - feq_old + feq_new

        # D3Q27 collision with LES
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

        # Guo forces: gravity (water) + surface tension (interface)
        Fy_grav = -rho_liq * gz * (phi + 1.0) / 2.0
        grad_phi_x = 0.5 * (torch.roll(phi, -1, dims=2) - torch.roll(phi, 1, dims=2))
        grad_phi_y = 0.5 * (torch.roll(phi, -1, dims=1) - torch.roll(phi, 1, dims=1))
        grad_phi_z = 0.5 * (torch.roll(phi, -1, dims=0) - torch.roll(phi, 1, dims=0))
        lap_phi = _laplacian_3d(phi)
        mu = -A_coef * phi + B_coef * phi**3 - kappa_ch * lap_phi
        Fx_st = mu * grad_phi_x
        Fy_st = mu * grad_phi_y
        Fz_st = mu * grad_phi_z
        Fx = Fx_st
        Fy = Fy_grav + Fy_st
        Fz = Fz_st
        cu_force = cx3d * Fx.unsqueeze(0) + cy3d * Fy.unsqueeze(0) + cz3d * Fz.unsqueeze(0)
        f = f + (1.0 - 0.5/tau_f) * w * cu_force / cs2
        f = f.clamp(min=0.0, max=5.0)

        # === 5. Wall function (on submerged hull) ===
        df = 0.0
        if use_wall_function:
            near_water = near & (phi > 0)
            f, df = _wall_function_3d(f, solid, near_water, fluid,
                                       c, cx3d, cy3d, cz3d, w, cs2, nu_lat)

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
        ac_source = alpha_ac * (1.0 - phi_new**2) / W_ac * torch.sign(phi_new)
        phi_new = phi_new + ac_source
        phi_new = phi_new.clamp(-1.0, 1.0)

        # Blend: far-field prescribed flat + near-hull evolved (captures waves)
        phi_presc = torch.tanh((fill_height - zz.float()) / 2.0)
        phi_presc[solid] = 1.0
        hull_cx, hull_cy, hull_cz = int(nx * 0.3), ny // 2, fill_height
        hull_dist = torch.sqrt((xx - hull_cx)**2 + (yy - hull_cy)**2 + (zz - hull_cz)**2)
        blend = torch.exp(-hull_dist**2 / (2.0 * (hull_length * 0.3)**2))
        blend = blend.clamp(0, 1)
        phi_blend = phi_presc * (1.0 - blend) + phi_new * blend
        phi_blend = phi_blend.clamp(-1.0, 1.0)
        phi_blend[solid] = 1.0
        g = _init_g_equilibrium(phi_blend, ux_s, uy_s, uz_s, c, w)

        # === 7. Streaming + bounce-back ===
        f = stream27(f)
        f = torch.where(solid.unsqueeze(0), f[opp], f)

        # === 8. Far-field BC ===
        r1 = torch.ones(nz, ny, nx, dtype=f.dtype, device=dev)
        feq_in = equilibrium27(r1, torch.full_like(r1, u_in),
                               torch.zeros_like(r1), torch.zeros_like(r1))
        f[:, :, :, 0] = feq_in[:, :, :, 0]      # inlet
        f[:, :, :, -1] = f[:, :, :, -2]          # outlet (convective)
        f[:, 0, :, :] = feq_in[:, 0, :, :]       # side walls
        f[:, -1, :, :] = feq_in[:, -1, :, :]
        f[:, :, 0, :] = feq_in[:, :, 0, :]       # top/bottom
        f[:, :, -1, :] = feq_in[:, :, -1, :]

        # === 9. Diagnostics ===
        if step > warmup:
            if use_wall_function:
                fric_list.append(df)
            if step % 10 == 0:
                # Wave amplitude at wake plane
                for_j = phi_blend[:, :, wake_x]  # (nz, ny)
                surface_h = torch.zeros(ny, device=dev)
                for j in range(ny):
                    col = for_j[:, j]
                    for iz in range(nz-1, 0, -1):
                        if col[iz] < 0 and col[iz-1] >= 0:
                            surface_h[j] = float(iz)
                            break
                    else:
                        surface_h[j] = float(fill_height)
                eta_rms = float(torch.sqrt(((surface_h - fill_height)**2).mean()))
                wave_list.append(eta_rms)

        if step % 200 == 0 or step == n_steps:
            cf = abs(sum(fric_list)/max(len(fric_list),1)) / dyn_p_S if fric_list else 0.0
            eta_avg = sum(wave_list)/max(len(wave_list),1) if wave_list else 0.0
            cw = gz * eta_avg**2 * ny / (u_in**2 * S) if gz > 0 else 0.0
            ct = cf + cw
            phi_min = float(phi_blend[fluid].min()) if fluid.any() else 0
            phi_max = float(phi_blend[fluid].max()) if fluid.any() else 0
            print(f'  step {step:5d}: Cf={cf:.5f} Cw={cw:.5f} Ct={ct:.5f} '
                  f'eta={eta_avg:.2f} (ref={ct_ref:.5f}) '
                  f'φ=[{phi_min:.2f},{phi_max:.2f}]', flush=True)

    dt = time.time() - t0
    cf = abs(sum(fric_list)/max(len(fric_list),1)) / dyn_p_S if fric_list else 0.0
    eta_avg = sum(wave_list)/max(len(wave_list),1) if wave_list else 0.0
    cw = gz * eta_avg**2 * ny / (u_in**2 * S) if gz > 0 else 0.0
    ct = cf + cw

    print(f'\n=== Final Results ({collision}) ===', flush=True)
    print(f'Cf (friction)    = {cf:.5f}', flush=True)
    print(f'Cw (wave-making) = {cw:.5f}  eta_rms={eta_avg:.2f}', flush=True)
    print(f'Ct (total)       = {ct:.5f} (ref={ct_ref:.5f}, ratio={ct/ct_ref:.2f}x)', flush=True)
    print(f'Time: {dt:.1f}s ({dt/n_steps*1000:.0f}ms/step)', flush=True)
    return {"cf": cf, "cw": cw, "ct": ct, "ct_ref": ct_ref}


def _wall_function_3d(f, solid, near, fluid, c, cx, cy, cz, w, cs2, nu_lat, y_val=0.5):
    """Log-law wall function."""
    rho, ux, uy, uz = macroscopic27(f)
    u_mag = torch.sqrt(ux*ux + uy*uy + uz*uz).clamp(min=1e-12)
    u_tau = torch.sqrt(nu_lat * u_mag / y_val).clamp(min=1e-12)
    y_plus = y_val * u_tau / nu_lat
    turb = (y_plus > 11.6) & near
    if bool(turb.any()):
        ut = u_tau[turb].clone()
        um = u_mag[turb]
        for _ in range(8):
            lyp = torch.log(y_val * ut / nu_lat)
            fv = ut * (lyp / KAPPA + B_CONST) - um
            fp = (lyp / KAPPA + B_CONST) + 1.0 / KAPPA
            ut = (ut - fv / fp.clamp(min=1e-10)).clamp(min=1e-12)
        u_tau[turb] = ut
    tau_w = u_tau * u_tau
    inv_umag = 1.0 / u_mag
    coef = -(tau_w / y_val) * near.to(f.dtype)
    fx = coef * (ux * inv_umag)
    fy = coef * (uy * inv_umag)
    fz = coef * (uz * inv_umag)
    cu_force = cx * fx.unsqueeze(0) + cy * fy.unsqueeze(0) + cz * fz.unsqueeze(0)
    tau_val = 3.0 * nu_lat + 0.5
    f = f + (1.0 - 0.5/tau_val) * w * cu_force / cs2
    df = float(tau_w[near].sum()) if near.any() else 0.0
    return f, df


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
    p = argparse.ArgumentParser(description='Free-surface hull: D3Q27 + Phase-Field + LES')
    p.add_argument('--hull', default='wigley', choices=['wigley', 'series60', 'kcs', 'kvlcc2'])
    p.add_argument('--nx', type=int, default=192)
    p.add_argument('--ny', type=int, default=96)
    p.add_argument('--nz', type=int, default=96)
    p.add_argument('--re', type=float, default=1e5)
    p.add_argument('--fr', type=float, default=0.25)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--device', default='sdaa:0')
    p.add_argument('--collision', default='cumulant',
                   choices=['mrt27', 'kbc', 'cascaded', 'cumulant', 'smagorinsky'])
    g = p.parse_args()

    run_hull_fs_d3q27(g.hull, g.nx, g.ny, g.nz, g.re, g.fr,
                      n_steps=g.steps, device=g.device, collision=g.collision)
