"""Moving sphere test — BFL + virtual Lagrangian refill.

Sphere moves through fluid, compare drag with stationary sphere benchmark.
- Sphere = solid mask that translates each step
- BFL bounce-back at sphere surface (with wall velocity)
- Newly-fluid cells from virtual Lagrangian points
- Measure: drag force, compare with stationary sphere Cd

Stationary sphere Cd ref: ~0.47 (Re~1000)
"""
import sys, math, time, torch
sys.path.insert(0, '/root/TensorLBM_test/src')
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d, C as C3D, W as W3D, OPPOSITE as OPP
from tensorlbm.solver3d import stream3d, collide_bgk3d
from tensorlbm.boundaries3d import bounce_back_cells_3d

def run_moving_sphere(nx=128, ny=64, nz=64, n_steps=3000, device='sdaa:0', stationary=False):
    dev = torch.device(device)
    cs2 = 1.0 / 3.0
    tau = 0.8
    u_sphere = 0.05  # sphere moves in +x (or fluid moves if stationary)
    R = 8  # sphere radius

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=dev), torch.arange(ny, device=dev),
        torch.arange(nx, device=dev), indexing='ij')
    solid_wall = torch.zeros(nz, ny, nx, dtype=torch.bool, device=dev)
    solid_wall[:, 0, :] = True; solid_wall[:, -1, :] = True
    solid_wall[:, :, 0] = True; solid_wall[:, :, -1] = True
    solid_wall[0, :, :] = True; solid_wall[-1, :, :] = True

    cx_s = 32.0; cy_s = ny // 2; cz_s = nz // 2  # sphere center (float for smooth motion)

    opp = OPP.to(dev)
    c = C3D.to(dev).float()
    w = W3D.to(dev).float().view(19, 1, 1, 1)
    cx3d = c[:, 0].view(19, 1, 1, 1)
    cy3d = c[:, 1].view(19, 1, 1, 1)
    cz3d = c[:, 2].view(19, 1, 1, 1)

    # Initial: fluid at rest (moving sphere) or uniform flow (stationary sphere)
    rho0 = torch.ones(nz, ny, nx, device=dev)
    if stationary:
        # Stationary sphere in uniform flow (u_in)
        u_init = torch.full_like(rho0, u_sphere)
        f = equilibrium3d(rho0, u_init, torch.zeros_like(rho0), torch.zeros_like(rho0), device=dev)
    else:
        # Moving sphere in fluid at rest
        f = equilibrium3d(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0),
                           torch.zeros_like(rho0), device=dev)

    # Sphere mask function
    def sphere_mask(cx_val):
        cx_int = int(cx_val)
        dist2 = (xx - cx_int)**2 + (yy - cy_s)**2 + (zz - cz_s)**2
        mask = solid_wall.clone()
        mask[dist2 < R**2] = True
        return mask

    print(f'=== {"Stationary" if stationary else "Moving"} Sphere Test ===', flush=True)
    print(f'Grid: {nx}x{ny}x{nz} R={R} u={u_sphere} Re={u_sphere*2*R/((tau-0.5)/3.0):.0f}', flush=True)
    print(f'BFL + virtual Lagrangian refill', flush=True)
    print(f'Expected Cd ≈ 0.47 (Re~1000)', flush=True)
    print(flush=True)

    t0 = time.time()
    for step in range(1, n_steps + 1):
        # Update sphere position
        if not stationary:
            cx_s += u_sphere
        cx_int = int(cx_s)

        # Current and previous solid masks
        solid = sphere_mask(cx_s)
        if not stationary:
            prev_solid = sphere_mask(cx_s - u_sphere)
        else:
            prev_solid = solid  # no movement
        fluid = ~solid

        # 1. Collision
        rho = f.sum(0)
        ux = (f * cx3d).sum(0) / rho.clamp(min=1e-6)
        uy = (f * cy3d).sum(0) / rho.clamp(min=1e-6)
        uz = (f * cz3d).sum(0) / rho.clamp(min=1e-6)
        feq = equilibrium3d(rho.clamp(min=1e-6, max=3.0),
                             ux.clamp(-0.5, 0.5), uy.clamp(-0.5, 0.5), uz.clamp(-0.5, 0.5))
        f = f - (f - feq) / tau
        f = f.clamp(min=0.0, max=3.0)

        # 2. Save pre-streaming
        f_pre = f.clone()

        # 3. Streaming
        f = stream3d(f)
        f = bounce_back_cells_3d(f, solid_wall)

        # 4. BFL bounce-back at sphere surface (with wall velocity)
        rho_local = f.sum(0).clamp(min=1e-6)
        for q in range(1, 19):
            o = int(opp[q].item())
            cz_q, cy_q, cx_q = int(c[q, 2].item()), int(c[q, 1].item()), int(c[q, 0].item())
            nbr_solid = torch.roll(solid, shifts=(cz_q, cy_q, cx_q), dims=(0, 1, 2))
            crossing = fluid & nbr_solid
            if crossing.any():
                # Wall velocity = u_sphere in x (moving sphere) or 0 (stationary)
                u_wall = u_sphere if not stationary else 0.0
                cu_wall = float(c[q, 0]) * u_wall
                corr = 2.0 * rho_local[crossing] * float(w[q]) * cu_wall / cs2
                f[q][crossing] = f_pre[o][crossing] - corr

        # 5. Newly-fluid cells: virtual Lagrangian refill
        new_fluid = prev_solid & fluid
        if bool(new_fluid.any()):
            prev_fluid = ~prev_solid
            prev_boundary = torch.zeros_like(prev_solid)
            for sz_s, sy_s, sx_s in [(0,0,1),(0,0,-1),(0,1,0),(0,-1,0),(1,0,0),(-1,0,0)]:
                prev_boundary |= (prev_fluid & torch.roll(prev_solid, shifts=(sz_s, sy_s, sx_s), dims=(0,1,2)))
            prev_boundary = prev_boundary & ~solid

            if prev_boundary.any():
                rho_bnd = f.sum(0)
                ux_bnd = (f * cx3d).sum(0) / rho_bnd.clamp(min=1e-6)
                uy_bnd = (f * cy3d).sum(0) / rho_bnd.clamp(min=1e-6)
                uz_bnd = (f * cz3d).sum(0) / rho_bnd.clamp(min=1e-6)
                # Add wall velocity at boundary
                ux_bnd = ux_bnd + u_sphere * prev_boundary.float()

                rho_sum = torch.zeros_like(rho0)
                ux_sum = torch.zeros_like(rho0)
                uy_sum = torch.zeros_like(rho0)
                uz_sum = torch.zeros_like(rho0)
                cnt = torch.zeros_like(rho0)
                for sz_s, sy_s, sx_s in [(0,0,1),(0,0,-1),(0,1,0),(0,-1,0),(1,0,0),(-1,0,0)]:
                    nbr_bnd = torch.roll(prev_boundary, shifts=(sz_s, sy_s, sx_s), dims=(0,1,2))
                    rho_sum += torch.where(nbr_bnd, rho_bnd, torch.zeros_like(rho0))
                    ux_sum += torch.where(nbr_bnd, ux_bnd, torch.zeros_like(rho0))
                    uy_sum += torch.where(nbr_bnd, uy_bnd, torch.zeros_like(rho0))
                    uz_sum += torch.where(nbr_bnd, uz_bnd, torch.zeros_like(rho0))
                    cnt += nbr_bnd.float()

                has_nbr = cnt > 0
                ux_fill = torch.where(has_nbr, ux_sum / cnt.clamp(min=1), torch.zeros_like(rho0))
                uy_fill = torch.where(has_nbr, uy_sum / cnt.clamp(min=1), torch.zeros_like(rho0))
                uz_fill = torch.where(has_nbr, uz_sum / cnt.clamp(min=1), torch.zeros_like(rho0))
                rho_fill = torch.where(has_nbr, rho_sum / cnt.clamp(min=1), torch.ones_like(rho0))

                feq_fill = equilibrium3d(rho_fill.clamp(min=1e-6, max=3.0),
                                          ux_fill.clamp(-0.5, 0.5),
                                          uy_fill.clamp(-0.5, 0.5),
                                          uz_fill.clamp(-0.5, 0.5))
                f[:, new_fluid] = feq_fill[:, new_fluid]

        # 6. Inlet BC (for stationary sphere)
        if stationary:
            rho_in = torch.ones(nz, ny, nx, device=dev)
            feq_in = equilibrium3d(rho_in, torch.full_like(rho0, u_sphere),
                                    torch.zeros_like(rho0), torch.zeros_like(rho0), device=dev)
            f[:, :, :, 0] = feq_in[:, :, :, 0]
            f[:, :, :, -1] = f[:, :, :, -2]  # outlet

        # 7. Measurement
        if step % 300 == 0 or step == n_steps:
            # Drag force: momentum exchange
            f_swapped = f[opp]
            force_x = 0.0
            for q in range(1, 19):
                cz_q, cy_q, cx_q = int(c[q, 2].item()), int(c[q, 1].item()), int(c[q, 0].item())
                nbr_solid = torch.roll(solid, shifts=(cz_q, cy_q, cx_q), dims=(0, 1, 2))
                bnd = fluid & nbr_solid
                if bnd.any():
                    force_x += float((f[q][bnd] - f_swapped[q][bnd]).sum()) * float(c[q, 0])

            # Cd = Fx / (0.5 * rho * U² * A)
            A = math.pi * R**2  # projected area
            cd = force_x / (0.5 * 1.0 * u_sphere**2 * A) if u_sphere > 0 else 0

            # Wake velocity
            wake_x = min(int(cx_s) + R + 5, nx - 2)
            rho_now = f.sum(0)
            ux_now = (f * cx3d).sum(0) / rho_now.clamp(min=1e-6)
            ux_wake = float(ux_now[:, :, wake_x].mean())

            print(f'step {step:4d}: cx={cx_int} Fx={force_x:.2f} Cd={cd:.3f} '
                  f'ux_wake={ux_wake:.4f} nf={int(new_fluid.sum())} '
                  f'{time.time()-t0:.0f}s', flush=True)

    print(f'\nFinal: {time.time()-t0:.1f}s', flush=True)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--nx', type=int, default=128)
    p.add_argument('--ny', type=int, default=64)
    p.add_argument('--nz', type=int, default=64)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--device', default='sdaa:0')
    p.add_argument('--stationary', action='store_true')
    args = p.parse_args()
    run_moving_sphere(args.nx, args.ny, args.nz, args.steps, args.device, args.stationary)
