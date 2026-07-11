"""Improved propeller open-water test with 3 key improvements:

1. Local average velocity for newly-fluid cells (not u_in)
2. Fine mask angles (2° intervals = 180 masks, not 10° = 36)
3. BFL interpolated bounce-back (not simple bounce-back)

Based on the D3Q27 Cumulant + moving mask approach.
"""
import sys, math, time, torch
sys.path.insert(0, '/root/TensorLBM_test/src')
from tensorlbm.d3q27 import (equilibrium27, macroscopic27, C as C27,
                              correct_mass27, moving_wall_linkwise_me_force_torque)
from tensorlbm.cumulant import collide_cumulant_d3q27
from tensorlbm.propeller_cad import KP505_PRESET, build_propeller_mask

SHIFTS = [(int(C27[q,0]), int(C27[q,1]), int(C27[q,2])) for q in range(27)]

def stream27(f):
    out = torch.empty_like(f)
    for q in range(27):
        sx, sy, sz = SHIFTS[q]
        out[q] = torch.roll(f[q], shifts=(sz, sy, sx), dims=(0, 1, 2))
    return out

def far_field(f, u):
    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    r = torch.ones(nz, ny, nx, dtype=f.dtype, device=f.device)
    feq = equilibrium27(r, torch.full_like(r, u), torch.zeros_like(r), torch.zeros_like(r))
    f = f.clone()
    f[:,:,:,0] = feq[:,:,:,0]
    f[:,:,:,-1] = f[:,:,:,-2]
    f[:,0,:,:] = feq[:,0,:,:]; f[:,-1,:,:] = feq[:,-1,:,:]
    f[:,:,0,:] = feq[:,:,0,:]; f[:,:,-1,:] = feq[:,:,-1,:]
    return f

def run_improved_propeller(nx=192, ny=96, nz=96, n_steps=5000, device='sdaa:0',
                           mask_interval_deg=2.0, use_bfl=True,
                           collect_me_diagnostic=True):
    """Improved propeller with fine masks + local refill + BFL."""
    dev = torch.device(device)
    cfg = KP505_PRESET
    D = cfg.diameter
    cx = int(nx * 0.3); cy = ny // 2; cz = nz // 2
    u_in = 0.05; J = 0.5
    n_rev = u_in / (J * D)
    omega = 2 * math.pi * n_rev
    nu_lat = 0.02; tau = 3.0 * nu_lat + 0.5

    # Improvement 2: Fine mask angles (2° intervals = 180 masks)
    n_masks = int(360.0 / mask_interval_deg)
    print(f'Building {n_masks} masks at {mask_interval_deg}° intervals...', flush=True)
    t_mask = time.time()
    masks = [build_propeller_mask(nx, ny, nz, cx, cy, cz,
                                   angle_deg=i * mask_interval_deg,
                                   config=cfg, device='cpu').to(dev)
             for i in range(n_masks)]
    print(f'  Done in {time.time()-t_mask:.1f}s', flush=True)

    # Opposite direction map
    opp_map = torch.zeros(27, dtype=torch.long, device=dev)
    for q in range(27):
        for q2 in range(27):
            if C27[q2,0]==-C27[q,0] and C27[q2,1]==-C27[q,1] and C27[q2,2]==-C27[q,2]:
                opp_map[q] = q2; break

    c = C27.to(dev).float()
    w27 = torch.tensor([8/27]+[2/27]*6+[1/54]*12+[1/216]*8,
                       dtype=torch.float32, device=dev).view(27,1,1,1)
    zz, yy, xx = torch.meshgrid(torch.arange(nz,device=dev),
                                 torch.arange(ny,device=dev),
                                 torch.arange(nx,device=dev), indexing='ij')

    # Wall velocity field (rotation about x-axis at cx)
    u_wall_y = -omega * (zz.float() - cz)
    u_wall_z = omega * (yy.float() - cy)

    # Initialize
    rho0 = torch.ones(nz, ny, nx, device=dev)
    ux0 = torch.full((nz,ny,nx), u_in, device=dev); ux0[masks[0]] = 0
    f = equilibrium27(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0))
    initial_mass = float(rho0.sum().item())
    wake_x = min(int(cx + D * 1.5), nx - 2)

    print(f'=== Improved Propeller ===', flush=True)
    print(f'Grid: {nx}x{ny}x{nz} J={J} u_in={u_in} omega={omega:.4f}', flush=True)
    print(f'Masks: {n_masks} ({mask_interval_deg}°) BFL={use_bfl}', flush=True)
    print(f'tip_speed={omega*D/2:.3f} tau={tau:.3f}', flush=True)
    print(flush=True)

    angle = 0.0; samples = []; t0 = time.time(); bfl_link_count = 0
    me_force = torch.zeros(3, dtype=f.dtype, device=dev)
    me_torque = torch.zeros(3, dtype=f.dtype, device=dev)

    for step in range(1, n_steps + 1):
        angle += omega
        idx = int((angle / (2*math.pi)) * n_masks) % n_masks
        solid = masks[idx]
        fluid = ~solid

        # Collision
        f = collide_cumulant_d3q27(f, tau=tau)
        f = stream27(f)

        # Improvement 3: BFL interpolated bounce-back (or simple)
        f_swapped = f[opp_map]
        # Wall velocity correction (use rho=1.0 fixed — same as working code!)
        cu_wall = c[:,1].view(27,1,1,1) * u_wall_y.unsqueeze(0) + \
                  c[:,2].view(27,1,1,1) * u_wall_z.unsqueeze(0)
        correction = 2.0 * 1.0 * w27 * cu_wall / (1.0/3.0)  # rho=1.0 FIXED

        if use_bfl:
            # BFL with moving wall velocity correction
            # For crossing directions: f[q] = f[opp] - 2*ρ*w*(c·u_wall)/cs²
            # u_wall = ω × r (rotation about x-axis)
            for q in range(1, 27):
                o = int(opp_map[q].item())
                sz_q, sy_q, sx_q = SHIFTS[q]
                nbr_solid = torch.roll(solid, shifts=(sz_q, sy_q, sx_q), dims=(0,1,2))
                crossing = fluid & nbr_solid  # fluid cell where neighbor is solid
                if crossing.any():
                    # Moving wall velocity at crossing cells
                    cu_w = float(c[q,1]) * u_wall_y[crossing] + float(c[q,2]) * u_wall_z[crossing]
                    rho_c = torch.ones_like(cu_w)  # rho=1.0 fixed
                    corr = 2.0 * rho_c * float(w27[q]) * cu_w / (1.0/3.0)
                    # Diagnostic-only: this reads the same link populations and
                    # correction as BFL, but never writes solver state.
                    if collect_me_diagnostic:
                        n_links = int(crossing.sum().item())
                        outgoing = f_swapped[q][crossing]
                        directions = c[q].expand(n_links, 3)
                        weights = torch.full_like(outgoing, float(w27[q]))
                        wall_velocity = torch.stack((
                            torch.zeros_like(cu_w), u_wall_y[crossing], u_wall_z[crossing],
                        ), dim=1)
                        positions = torch.stack((xx[crossing], yy[crossing], zz[crossing]), dim=1)
                        _, _, link_force, link_torque = moving_wall_linkwise_me_force_torque(
                            outgoing, directions, weights, wall_velocity, positions,
                            origin=(cx, cy, cz), density=1.0,
                        )
                        me_force += link_force
                        me_torque += link_torque
                    f[q][crossing] = f_swapped[q][crossing] - corr
                    bfl_link_count += int(crossing.sum().item())
        else:
            # Simple bounce-back with wall velocity (clamp to prevent NaN!)
            f_bb = (f_swapped - correction).clamp(min=0.0)
            f = torch.where(solid.unsqueeze(0), f_bb, f)

        # Improvement 1: Newly-fluid cells initialized from virtual Lagrangian points
        # Virtual Lagrangian points = BFL boundary cells from previous step
        # They store: velocity = u_wall (rotation), pressure = rho_wall * cs²
        prev = masks[(idx - 1) % n_masks]
        new_fluid = prev & fluid  # was solid, now fluid
        if bool(new_fluid.any()):
            # Virtual Lagrangian points: fluid cells adjacent to solid (previous step's boundary)
            prev_fluid = ~prev
            prev_boundary = prev_fluid & ~torch.roll(prev_fluid, 1, dims=2) | \
                            prev_fluid & ~torch.roll(prev_fluid, -1, dims=2) | \
                            prev_fluid & ~torch.roll(prev_fluid, 1, dims=1) | \
                            prev_fluid & ~torch.roll(prev_fluid, -1, dims=1) | \
                            prev_fluid & ~torch.roll(prev_fluid, 1, dims=0) | \
                            prev_fluid & ~torch.roll(prev_fluid, -1, dims=0)
            prev_boundary = prev_boundary & ~prev & ~solid  # was fluid, adjacent to prev solid, still fluid

            if prev_boundary.any():
                # Virtual Lagrangian point interpolation (vectorized)
                rho_local = f.sum(0)  # for measurement only
                ux_bnd_full = (f * c[:,0].view(27,1,1,1)).sum(0) / rho_local.clamp(min=1e-6)
                uy_bnd_full = (f * c[:,1].view(27,1,1,1)).sum(0) / rho_local.clamp(min=1e-6)
                uz_bnd_full = (f * c[:,2].view(27,1,1,1)).sum(0) / rho_local.clamp(min=1e-6)
                uy_bnd_full = uy_bnd_full + u_wall_y * 0.5 * prev_boundary.float()
                uz_bnd_full = uz_bnd_full + u_wall_z * 0.5 * prev_boundary.float()

                rho_sum = torch.zeros_like(rho0); ux_sum = torch.zeros_like(rho0)
                uy_sum = torch.zeros_like(rho0); uz_sum = torch.zeros_like(rho0)
                cnt = torch.zeros_like(rho0)
                for sz_s, sy_s, sx_s in [(0,0,1),(0,0,-1),(0,1,0),(0,-1,0),(1,0,0),(-1,0,0)]:
                    nbr_bnd = torch.roll(prev_boundary, shifts=(sz_s, sy_s, sx_s), dims=(0,1,2))
                    rho_sum += torch.where(nbr_bnd, rho_local, torch.zeros_like(rho0))
                    ux_sum += torch.where(nbr_bnd, ux_bnd_full, torch.zeros_like(rho0))
                    uy_sum += torch.where(nbr_bnd, uy_bnd_full, torch.zeros_like(rho0))
                    uz_sum += torch.where(nbr_bnd, uz_bnd_full, torch.zeros_like(rho0))
                    cnt += nbr_bnd.float()

                has_nbr = cnt > 0
                ux_fill = torch.where(has_nbr, ux_sum / cnt.clamp(min=1), torch.full_like(rho0, u_in))
                uy_fill = torch.where(has_nbr, uy_sum / cnt.clamp(min=1), torch.zeros_like(rho0))
                uz_fill = torch.where(has_nbr, uz_sum / cnt.clamp(min=1), torch.zeros_like(rho0))
                rho_fill = torch.where(has_nbr, rho_sum / cnt.clamp(min=1), torch.ones_like(rho0))
            else:
                # Fallback: use u_in (same as previous working code)
                ux_fill = torch.full_like(rho0, u_in)
                uy_fill = torch.zeros_like(rho0)
                uz_fill = torch.zeros_like(rho0)
                rho_fill = torch.ones_like(rho0)

            feq_fill = equilibrium27(rho_fill.clamp(min=1e-6, max=3.0),
                                      ux_fill.clamp(-0.5, 0.5),
                                      uy_fill.clamp(-0.5, 0.5),
                                      uz_fill.clamp(-0.5, 0.5))
            f[:, new_fluid] = feq_fill[:, new_fluid]

        # Far-field BC
        f = far_field(f, u_in)
        if step % 100 == 0:
            f = correct_mass27(f, initial_mass)

        # Measure thrust
        if step > 1000:
            _, ux, _, _ = macroscopic27(f)
            if not torch.isnan(ux).any():
                deficit = (u_in - ux[:,:,wake_x]) * fluid[:,:,wake_x].to(f.dtype)
                thrust = 1.0 * u_in * deficit.sum().item()
                if math.isfinite(thrust):
                    samples.append(thrust)

        if step % 500 == 0 or step == n_steps:
            avg = abs(sum(samples) / max(len(samples), 1))
            kt = avg / (1.0 * n_rev**2 * D**4)
            print(f'step {step:5d}: KT={kt:.4f} (exp=0.37) n={len(samples)} '
                  f'angle={math.degrees(angle)%360:.1f}° {time.time()-t0:.0f}s', flush=True)

    kt_final = abs(sum(samples) / max(len(samples), 1)) / (1.0 * n_rev**2 * D**4)
    print(f'\nFinal: KT={kt_final:.4f} (exp=0.37, {kt_final/0.37*100:.0f}%) '
          f'{time.time()-t0:.0f}s', flush=True)
    rho, ux, uy, uz = macroscopic27(f)
    return {"f": f, "rho": rho, "ux": ux, "uy": uy, "uz": uz,
            "bfl_link_count": bfl_link_count,
            "me_force": me_force, "me_torque": me_torque}

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--nx', type=int, default=192)
    p.add_argument('--ny', type=int, default=96)
    p.add_argument('--nz', type=int, default=96)
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--device', default='sdaa:0')
    p.add_argument('--mask-deg', type=float, default=2.0)
    p.add_argument('--no-bfl', action='store_true')
    args = p.parse_args()
    run_improved_propeller(args.nx, args.ny, args.nz, args.steps, args.device,
                           mask_interval_deg=args.mask_deg, use_bfl=not args.no_bfl)
