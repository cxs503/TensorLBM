"""Quick test of updated wall-surface BFL (after stream) + ME on D=24, 500 steps."""
import sys
sys.path.insert(0, 'src')
import torch
torch.sdaa.set_device(int(sys.argv[1]) if len(sys.argv) > 1 else 0)

from tensorlbm.d3q19 import C, OPPOSITE, equilibrium3d, macroscopic3d
from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d
from tensorlbm.bfl_d3q19 import compute_q_cylinder_d3q19
from tensorlbm.wall_surface_bfl import bouzidi_bounce_back_wallsurface, drag_momentum_exchange_bfl
from tensorlbm.solver3d import stream3d, correct_mass3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.drag_pressure import SurfaceMesh, drag_total, get_near_wall_2d
from tensorlbm.drag_momentum import drag_momentum_exchange
import math, time

D = 24; nx, ny, nz = 200, 80, 4
diameter = float(D); radius = diameter/2.0; u_in = 0.08; Re = 200.0
nu = u_in*diameter/Re; tau = 3.0*nu+0.5; cs_smag = 0.05
cx_c = nx*0.25; cy_c = ny*0.5
A_frontal = diameter*nz; dpS = 0.5*1.0*u_in**2*A_frontal
n_steps = 500; warmup = 100
device = torch.device(f'sdaa:{int(sys.argv[1]) if len(sys.argv) > 1 else 0}')

from grid_conv_bfl_worker import build_cylinder_mask
solid = build_cylinder_mask(nx, ny, nz, cx_c, cy_c, radius, device)
sm = solid.unsqueeze(0).expand(19, nz, ny, nx)
near = get_near_wall_2d(solid)
fbm, qf = compute_q_cylinder_d3q19(nx, ny, nz, cx_c, cy_c, radius, device)
mesh = SurfaceMesh.from_cylinder(solid, near, cx_c, cy_c, radius)

rho0 = torch.ones(nz, ny, nx, device=device)
ux0 = torch.full((nz, ny, nx), u_in, device=device); ux0[solid] = 0.0
f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
initial_mass = float(rho0.sum().item())

tag = f"[BFL-v2 SDAA:{device.index}]"
print(f"{tag} After-stream BFL + ME, 500 steps", flush=True)
t0 = time.time()

cd_bfl = []; cd_pf = []; cd_std = []

for step in range(1, n_steps+1):
    f_pre = f.clone()
    f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)
    for q in range(19):
        f[q] = torch.where(sm[q], f_pre[q], f[q])
    f_post_coll = f.clone()
    
    # Stream
    f_streamed = stream3d(f_post_coll)
    f_streamed = far_field_bc_3d(f_streamed, u_in=u_in)
    
    # BFL after stream
    f_bfl = bouzidi_bounce_back_wallsurface(f_streamed, f_post_coll, fbm, qf)
    
    # ME: F = (fp_d + f_wall) * c_i, no /q
    cd_me = drag_momentum_exchange_bfl(f_bfl, f_post_coll, fbm, qf, dpS, use_q_scaling=False)
    
    # Also compute BB+PF for comparison
    f_bb = bounce_back_cells_3d(f_streamed, solid)
    cd_tot, _, _ = drag_total(f_bb, mesh, dpS, nu)
    cd_s = drag_momentum_exchange(f_bb, near, solid, dpS)
    
    if step > warmup:
        if math.isfinite(cd_me): cd_bfl.append(cd_me)
        if math.isfinite(cd_tot): cd_pf.append(cd_tot)
        if math.isfinite(cd_s): cd_std.append(cd_s)
    
    # Use BFL for dynamics
    f = f_bfl
    if step % 200 == 0: f = correct_mass3d(f, initial_mass)
    
    if not torch.isfinite(f).all():
        print(f"{tag} DIVERGED at step {step}", flush=True)
        break
    
    if step % 100 == 0:
        el = time.time() - t0
        print(f"{tag} step={step} BFL+ME={sum(cd_bfl)/max(len(cd_bfl),1):.4f} "
              f"BB+PF={sum(cd_pf)/max(len(cd_pf),1):.4f} "
              f"BB+ME={sum(cd_std)/max(len(cd_std),1):.4f} ({el:.0f}s)", flush=True)

el = time.time() - t0
print(f"\n{tag} === FINAL (ref=1.30) ===", flush=True)
print(f"{tag} BFL+ME  = {sum(cd_bfl)/max(len(cd_bfl),1):.4f}", flush=True)
print(f"{tag} BB+PF   = {sum(cd_pf)/max(len(cd_pf),1):.4f}", flush=True)
print(f"{tag} BB+ME   = {sum(cd_std)/max(len(cd_std),1):.4f}", flush=True)
print(f"{tag} time={el:.0f}s", flush=True)
