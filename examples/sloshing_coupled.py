"""Sloshing benchmark for coupled liquid-gas model.

Liquid sloshes in a closed tank with gas above.
Tests: oscillatory interface motion, gas compression/expansion.

Physics:
  - Tank partially filled with liquid (bottom half)
  - Gas above liquid (top half)
  - Initial tilt → liquid sloshes back and forth
  - Gas gets compressed alternately on each side
  - BFL at the oscillating interface

Validation:
  - Sloshing frequency vs analytical (omega = sqrt(g/L*tanh(h/L)))
  - Gas pressure oscillation amplitude
"""
import sys, math, time, torch
sys.path.insert(0, '/root/TensorLBM_test/src')
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d, C as C3D, W as W3D, OPPOSITE as OPP
from tensorlbm.solver3d import stream3d, collide_bgk3d
from tensorlbm.free_surface_lbm import init_flags_from_fill, _init_new, GAS, LIQUID, INTERFACE, SOLID
from tensorlbm.boundaries3d import bounce_back_cells_3d

_SHIFT_SPECS = [(a, s) for a in [0, 1, 2] for s in [-1, 1]]

def liquid_collide(f, fill, flags, solid, tau, gz, rho_liq=1.0, rho_gas=0.001):
    non_gas = ~(flags == GAS)
    rho, ux, uy, uz = macroscopic3d(f)
    rho_s = rho.clamp(min=rho_gas * 0.01, max=rho_liq * 3.0)
    ux_eq = ux.clamp(-0.5, 0.5)
    uy_eq = (uy + tau * gz).clamp(-0.5, 0.5)
    uz_eq = uz.clamp(-0.5, 0.5)
    feq = equilibrium3d(rho_s, ux_eq, uy_eq, uz_eq)
    f = f - (f - feq) / tau
    f = f.clamp(min=0.0, max=rho_liq * 3.0)
    f = torch.where(non_gas.unsqueeze(0), f, torch.zeros_like(f))
    return f, ux, uy, uz

def liquid_update(f, fill, flags, solid, rho_liq=1.0, rho_gas=0.001, ux=None, uy=None, uz=None):
    device = f.device
    rho_new = f.sum(dim=0)
    fill = torch.where(~solid, (rho_new / rho_liq).clamp(0.0, 1.0), fill)
    gas_mask = (flags == GAS); iface_mask = (flags == INTERFACE); liq_mask = (flags == LIQUID)
    to_iface = gas_mask & (fill > 0.01) & (~solid)
    if to_iface.any():
        f = _init_new(f, flags, to_iface, rho_gas, device, ux, uy, uz)
        flags = torch.where(to_iface, torch.full_like(flags, INTERFACE), flags)
    to_liq = iface_mask & (fill >= 0.999) & (~solid)
    if to_liq.any():
        flags = torch.where(to_liq, torch.full_like(flags, LIQUID), flags)
        fill = torch.where(to_liq, torch.ones_like(fill), fill)
    to_gas = (iface_mask | liq_mask) & (fill <= 0.01) & (~solid)
    if to_gas.any():
        flags = torch.where(to_gas, torch.full_like(flags, GAS), flags)
        fill = torch.where(to_gas, torch.zeros_like(fill), fill)
        f = torch.where(to_gas.unsqueeze(0), torch.zeros_like(f), f)
    shifted = torch.stack([flags.roll(s, dims=a) for a, s in _SHIFT_SPECS])
    is_nbr = ((shifted == LIQUID) | (shifted == INTERFACE)).any(dim=0)
    to_i = gas_mask & is_nbr & (~solid)
    if to_i.any():
        f = _init_new(f, flags, to_i, rho_gas, device, ux, uy, uz)
        flags = torch.where(to_i, torch.full_like(flags, INTERFACE), flags)
        fill = torch.where(to_i, torch.full_like(fill, 0.01), fill)
    flags = torch.where(solid, torch.full_like(flags, SOLID), flags)
    return f, fill, flags

def gas_bfl_interface(f_g, f_g_pre, fill, ux_l, uy_l, uz_l, opp, c, w, cs2, gas_mask):
    f_out = f_g.clone()
    for d in range(1, 19):
        o = int(opp[d].item())
        cz, cy, cx = int(c[d, 2].item()), int(c[d, 1].item()), int(c[d, 0].item())
        fill_nbr = torch.roll(fill, shifts=(cz, cy, cx), dims=(0, 1, 2))
        near_iface = gas_mask & (fill_nbr > 0.5)
        if not near_iface.any(): continue
        q = fill_nbr[near_iface].clamp(0.01, 1.0)
        ux_w = torch.roll(ux_l, shifts=(cz, cy, cx), dims=(0, 1, 2))[near_iface]
        uy_w = torch.roll(uy_l, shifts=(cz, cy, cx), dims=(0, 1, 2))[near_iface]
        uz_w = torch.roll(uz_l, shifts=(cz, cy, cx), dims=(0, 1, 2))[near_iface]
        cu_wall = float(c[d,0])*ux_w + float(c[d,1])*uy_w + float(c[d,2])*uz_w
        rho_gas = f_g.sum(0)[near_iface].clamp(min=1e-6)
        corr = 2.0 * rho_gas * float(w[d]) * cu_wall / cs2
        f_opp = f_g[o][near_iface]; f_pre_d = f_g_pre[d][near_iface]; f_pre_opp = f_g_pre[o][near_iface]
        lin = q < 0.5
        f_lin = 2.0*q*f_opp + (1.0-2.0*q)*f_pre_d - corr
        q_safe = torch.where(lin, torch.ones_like(q), q)
        f_quad = f_opp/(2.0*q_safe) + (2.0*q_safe-1.0)/(2.0*q_safe)*f_pre_opp - corr
        f_bc = torch.where(lin, f_lin, f_quad)
        tmp = f_out[d].clone(); tmp[near_iface] = f_bc; f_out[d] = tmp
    return f_out

def run_sloshing(nx=96, ny=64, nz=8, n_steps=3000, device='sdaa:0', use_gas=True):
    """Sloshing tank with coupled liquid-gas."""
    dev = torch.device(device)
    cs2 = 1.0/3.0; tau = 0.8; gz = 0.001

    zz, yy, xx = torch.meshgrid(torch.arange(nz, device=dev), torch.arange(ny, device=dev),
                                 torch.arange(nx, device=dev), indexing='ij')
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=dev)
    solid[:, 0, :] = True; solid[:, -1, :] = True
    solid[:, :, 0] = True; solid[:, :, -1] = True
    solid[0, :, :] = True; solid[-1, :, :] = True

    # Initial tilted water surface + initial velocity (kick-start sloshing)
    fill_height = ny // 2
    tilt = ny // 8  # tilt amplitude
    water_h = (fill_height + tilt * (1.0 - 2.0 * xx.float() / nx)).int()
    water_mask = (yy < water_h) & ~solid

    fill = torch.where(water_mask, 1.0, 0.0).to(dev)
    flags = init_flags_from_fill(fill, solid)

    # Initial horizontal velocity to kick-start sloshing
    ux_init = torch.full((nz, ny, nx), 0.01, device=dev)
    ux_init[~water_mask] = 0
    rho0 = torch.ones(nz, ny, nx, device=dev)
    f_l = equilibrium3d(rho0, ux_init, torch.zeros_like(rho0), torch.zeros_like(rho0), device=dev)

    gas_init = ~water_mask & ~solid
    f_g = equilibrium3d(torch.ones_like(rho0), torch.zeros_like(rho0), torch.zeros_like(rho0), torch.zeros_like(rho0), device=dev)
    f_g[:, ~gas_init] = 0
    total_gas_mass = float(f_g[:, gas_init].sum())

    opp = OPP.to(dev); c = C3D.to(dev).float(); w = W3D.to(dev).float().view(19,1,1,1)
    cx3d = c[:,0].view(19,1,1,1); cy3d = c[:,1].view(19,1,1,1); cz3d = c[:,2].view(19,1,1,1)

    # Analytical sloshing frequency (1st mode)
    L = nx; h = fill_height
    omega_analytical = math.sqrt(gz * math.pi / L * math.tanh(math.pi * h / L))

    print(f'=== Sloshing (coupled) ===', flush=True)
    print(f'Grid: {nx}x{ny}x{nz} steps={n_steps} device={device}', flush=True)
    print(f'Water: tilted (tilt={tilt}) Gas: {"ON" if use_gas else "OFF"}', flush=True)
    print(f'Analytical omega={omega_analytical:.6f} period={2*math.pi/omega_analytical:.0f} steps', flush=True)
    print(flush=True)

    t0 = time.time()
    for step in range(1, n_steps + 1):
        f_l, ux_l, uy_l, uz_l = liquid_collide(f_l, fill, flags, solid, tau, gz)
        f_l = stream3d(f_l)
        f_l = bounce_back_cells_3d(f_l, solid)
        gas_mask_now = (flags == GAS)
        f_l = torch.where(gas_mask_now.unsqueeze(0), torch.zeros_like(f_l), f_l)
        f_l, fill, flags = liquid_update(f_l, fill, flags, solid, ux=ux_l, uy=uy_l, uz=uz_l)

        if use_gas:
            f_g_pre = f_g.clone()
            gas_domain = (fill < 0.5) & ~solid
            rho_g = f_g.sum(0)
            ux_g = (f_g * cx3d).sum(0) / rho_g.clamp(min=1e-6)
            uy_g = (f_g * cy3d).sum(0) / rho_g.clamp(min=1e-6)
            uz_g = (f_g * cz3d).sum(0) / rho_g.clamp(min=1e-6)
            feq_g = equilibrium3d(rho_g.clamp(min=1e-6, max=3.0), ux_g, uy_g, uz_g)
            f_g = f_g - (f_g - feq_g) / tau
            f_g = f_g.clamp(min=0.0, max=3.0)
            f_g = torch.where(gas_domain.unsqueeze(0), f_g, torch.zeros_like(f_g))
            f_g = stream3d(f_g)
            f_g = gas_bfl_interface(f_g, f_g_pre, fill, ux_l, uy_l, uz_l, opp, c, w, cs2, gas_domain)
            f_g = bounce_back_cells_3d(f_g, solid)
            if step % 10 == 0:
                cm = float(f_g[:, gas_domain].sum())
                if cm > 1e-6 and not math.isnan(cm):
                    scale = total_gas_mass / cm
                    if scale < 100: f_g[:, gas_domain] *= scale
                f_g = f_g.clamp(min=0.0, max=3.0)
            # Gas → Liquid
            p_gas_field = f_g.sum(0) * cs2
            gas_full = (fill < 0.5) & ~solid
            p_nbr_sum = torch.zeros_like(p_gas_field); gas_nbr_cnt = torch.zeros_like(p_gas_field)
            for ax, sgn in [(2,1),(2,-1),(1,1),(1,-1),(0,1),(0,-1)]:
                nbr_gas = torch.roll(gas_full, sgn, dims=ax)
                nbr_p = torch.roll(p_gas_field, sgn, dims=ax)
                p_nbr_sum += torch.where(nbr_gas, nbr_p, torch.zeros_like(nbr_p))
                gas_nbr_cnt += nbr_gas.float()
            p_at_iface = torch.where(gas_nbr_cnt > 0, p_nbr_sum / gas_nbr_cnt.clamp(min=1), torch.ones_like(p_gas_field) * cs2)
            iface = (fill > 0.01) & (fill < 0.99) & ~solid
            # Don't apply coupling at wall-adjacent cells (prevents liquid depletion at walls)
            wall_adj = torch.zeros_like(iface)
            for ax, sgn in [(2,1),(2,-1),(1,1),(1,-1),(0,1),(0,-1)]:
                wall_adj |= torch.roll(solid, sgn, dims=ax)
            iface = iface & ~wall_adj
            if bool(iface.any()):
                rho_iface = (p_at_iface / cs2).clamp(min=1e-6, max=3.0)
                feq_iface = equilibrium3d(rho_iface, ux_l, uy_l, uz_l)
                f_l = torch.where(iface.unsqueeze(0), feq_iface, f_l)

        if step % 300 == 0 or step == n_steps:
            # Water height: count water cells at left/right walls (robust)
            water_cells = (fill > 0.5).sum(dim=0)  # count over z for each (y, x)
            left_h = float(water_cells[:, 1].sum()) / nz  # total water at x=1
            right_h = float(water_cells[:, -2].sum()) / nz  # total water at x=-2
            diff = left_h - right_h  # should oscillate

            if use_gas:
                gas_d = (fill < 0.5) & ~solid
                p_gas = float(f_g.sum(0)[gas_d].mean()) * cs2 if gas_d.any() else 0
                gas_mass = float(f_g[:, gas_d].sum()) if gas_d.any() else 0
            else:
                p_gas = 0; gas_mass = 0

            print(f'step {step:4d}: L={left_h:.0f} R={right_h:.0f} diff={diff:.1f} '
                  f'p_gas={p_gas:.4f} gas_mass={gas_mass:.0f}', flush=True)

    dt = time.time() - t0
    print(f'\nFinal: {dt:.1f}s ({dt/n_steps*1000:.0f}ms/step)', flush=True)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--nx', type=int, default=96)
    p.add_argument('--ny', type=int, default=64)
    p.add_argument('--nz', type=int, default=8)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--device', default='sdaa:0')
    p.add_argument('--no-gas', action='store_true')
    args = p.parse_args()
    run_sloshing(args.nx, args.ny, args.nz, args.steps, args.device, use_gas=not args.no_gas)
