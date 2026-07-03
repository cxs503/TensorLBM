"""Bubble rise with IBM (Immersed Boundary Method).

Bubble surface = Lagrangian marker points (moving solid wall)
Liquid: IBM direct forcing enforces no-slip at bubble surface
Gas: inside bubble, analytical pressure (mass conservation + temperature bridge)
Bubble motion: buoyancy = (rho_liq - rho_gas) * g * V

No gas leakage (hard wall), no Körner cell conversion, no streaming issues.
"""
import sys, math, time, torch
sys.path.insert(0, '/root/TensorLBM_test/src')
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d, C as C3D, W as W3D, OPPOSITE as OPP
from tensorlbm.solver3d import stream3d, collide_bgk3d
from tensorlbm.boundaries3d import bounce_back_cells_3d
from tensorlbm.ibm import ibm_direct_forcing_3d

def run_ibm_bubble(nx=64, ny=96, nz=16, n_steps=3000, device='sdaa:0'):
    dev = torch.device(device)
    cs2 = 1.0 / 3.0
    tau = 0.8
    gz = 0.01  # gravity
    rho_liq = 1.0
    rho_gas = 0.5  # gas density (for buoyancy)

    zz, yy, xx = torch.meshgrid(torch.arange(nz, device=dev), torch.arange(ny, device=dev),
                                 torch.arange(nx, device=dev), indexing='ij')
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=dev)
    solid[:, 0, :] = True; solid[:, -1, :] = True
    solid[:, :, 0] = True; solid[:, :, -1] = True
    solid[0, :, :] = True; solid[-1, :, :] = True

    # Bubble: rigid sphere, starts at center-bottom
    cx, cy, cz = nx // 2, int(ny * 0.25), nz // 2
    R = max(min(nx, nz) // 4, 6)
    V_bubble = (4.0/3.0) * math.pi * R**3  # bubble volume

    # Lagrangian markers on sphere surface
    n_markers = max(int(4 * math.pi * R**2 / 2), 50)  # ~1 marker per 2 cells²
    theta = torch.linspace(0, math.pi, int(math.sqrt(n_markers)), device=dev)
    phi = torch.linspace(0, 2*math.pi, int(n_markers / len(theta)) + 1, device=dev)[:-1]
    th, ph = torch.meshgrid(theta, phi, indexing='ij')
    mx = cx + R * torch.sin(th) * torch.cos(ph)
    my = cy + R * torch.cos(th)
    mz = cz + R * torch.sin(th) * torch.sin(ph)
    mx = mx.flatten(); my = my.flatten(); mz = mz.flatten()
    n_markers = len(mx)

    # Bubble velocity (starts at rest)
    vy_bubble = 0.0

    # Liquid: density=1.0, zero velocity
    rho0 = torch.ones(nz, ny, nx, device=dev)
    f = equilibrium3d(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0), torch.zeros_like(rho0), device=dev)

    opp = OPP.to(dev); c = C3D.to(dev).float(); w = W3D.to(dev).float().view(19,1,1,1)
    cx3d = c[:,0].view(19,1,1,1); cy3d = c[:,1].view(19,1,1,1); cz3d = c[:,2].view(19,1,1,1)

    # Bubble mask (for gas pressure measurement)
    bubble_mask = (xx - cx)**2 + (yy - cy)**2 + (zz - cz)**2 < R**2

    print(f'=== IBM Bubble Rise ===', flush=True)
    print(f'Grid: {nx}x{ny}x{nz} R={R} markers={n_markers} steps={n_steps}', flush=True)
    print(f'rho_liq={rho_liq} rho_gas={rho_gas} gz={gz}', flush=True)
    print(f'V_bubble={V_bubble:.0f} buoyancy={(rho_liq-rho_gas)*gz*V_bubble:.2f}', flush=True)
    print(flush=True)

    t0 = time.time()
    for step in range(1, n_steps + 1):
        # 1. Liquid collision (with gravity)
        rho = f.sum(0)
        ux = (f * cx3d).sum(0) / rho.clamp(min=1e-6)
        uy = (f * cy3d).sum(0) / rho.clamp(min=1e-6)
        uz = (f * cz3d).sum(0) / rho.clamp(min=1e-6)
        feq = equilibrium3d(rho.clamp(min=1e-6, max=3.0),
                             ux.clamp(-0.5, 0.5),
                             (uy + tau * gz).clamp(-0.5, 0.5),
                             uz.clamp(-0.5, 0.5))
        f = f - (f - feq) / tau
        f = f.clamp(min=0.0, max=3.0)

        # 2. Streaming
        f = stream3d(f)
        f = bounce_back_cells_3d(f, solid)

        # 3. IBM: enforce no-slip at bubble surface
        # Target velocity at markers = bubble velocity (moving wall)
        rho_now = f.sum(0)
        ux_now = (f * cx3d).sum(0) / rho_now.clamp(min=1e-6)
        uy_now = (f * cy3d).sum(0) / rho_now.clamp(min=1e-6)
        uz_now = (f * cz3d).sum(0) / rho_now.clamp(min=1e-6)

        # Target: bubble moves with vy_bubble (in y direction)
        u_target_x = torch.zeros_like(mx)
        u_target_y = torch.full_like(mx, vy_bubble)
        u_target_z = torch.zeros_like(mz)

        fx, fy, fz = ibm_direct_forcing_3d(ux_now, uy_now, uz_now,
                                            mx, my, mz,
                                            u_target_x, u_target_y, u_target_z,
                                            kernel='hat')
        # Apply IBM force to distribution
        cu = cx3d * ux_now + cy3d * uy_now + cz3d * uz_now
        forcing = w * (1.0 + cu / cs2) * (cx3d * fx + cy3d * fy + cz3d * fz) / cs2
        f = f + forcing

        # 4. Bubble dynamics: buoyancy drives motion
        # F_buoyancy = (rho_liq - rho_gas) * g * V
        # Also drag from IBM force (reaction force)
        F_buoyancy = (rho_liq - rho_gas) * gz * V_bubble
        F_drag = -float(fy.sum())  # reaction force from IBM
        # Update velocity (simple Euler, mass = rho_gas * V)
        mass_bubble = rho_gas * V_bubble
        ay = (F_buoyancy + F_drag) / mass_bubble
        vy_bubble += ay * 1.0  # dt=1
        vy_bubble = max(vy_bubble, 0.0)  # can only rise (simplified)

        # 5. Move markers
        my += vy_bubble
        cy += vy_bubble

        # 6. Update bubble mask
        bubble_mask = (xx - cx)**2 + (yy - cy)**2 + (zz - cz)**2 < R**2

        # Measurement
        if step % 300 == 0 or step == n_steps:
            # Liquid pressure around bubble
            above = (yy > cy + R) & ~solid & ~bubble_mask
            below = (yy < cy - R) & ~solid & ~bubble_mask
            p_top = float(rho_now[above].mean()) * cs2 if above.any() else 0
            p_bot = float(rho_now[below].mean()) * cs2 if below.any() else 0
            rho_min = float(rho_now[~solid].min())
            rho_max = float(rho_now[~solid].max())

            print(f'step {step:4d}: cy={cy:.1f} vy={vy_bubble:.4f} '
                  f'F_buoy={F_buoyancy:.2f} F_drag={F_drag:.2f} '
                  f'p_top={p_top:.4f} p_bot={p_bot:.4f} '
                  f'rho_l=[{rho_min:.3f},{rho_max:.3f}]', flush=True)

    dt = time.time() - t0
    print(f'\nFinal: cy={cy:.1f} vy={vy_bubble:.4f} {dt:.1f}s ({dt/n_steps*1000:.0f}ms/step)', flush=True)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--nx', type=int, default=64)
    p.add_argument('--ny', type=int, default=96)
    p.add_argument('--nz', type=int, default=16)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--device', default='sdaa:0')
    args = p.parse_args()
    run_ibm_bubble(args.nx, args.ny, args.nz, args.steps, args.device)
