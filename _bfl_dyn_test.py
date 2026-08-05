"""Test BFL dynamics + various drag methods, 500 steps on D=24."""
import sys
sys.path.insert(0, 'src')
import torch
torch.sdaa.set_device(int(sys.argv[1]))

from tensorlbm.d3q19 import C, OPPOSITE, equilibrium3d, macroscopic3d
from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d
from tensorlbm.bfl_d3q19 import compute_q_cylinder_d3q19
from tensorlbm.wall_surface_bfl import bouzidi_bounce_back_wallsurface
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
device = torch.device(f'sdaa:{int(sys.argv[1])}')

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

c = C.to(device).float()
opp = OPPOSITE.to(device)

def me_wall_interp(f_prev, fbm, qf, dpS):
    """Wall-surface: F = 2*((1-q)*fp_d + q*feq[d](x_s))*c_i"""
    fx = torch.tensor(0.0, device=device)
    for d in range(1, 19):
        opp_d = int(opp[d].item())
        mask = fbm[d]
        if not mask.any(): continue
        q_cell = qf[d][mask]
        fp_d = f_prev[d][mask]
        cx_i = int(c[d,0].item()); cy_i = int(c[d,1].item()); cz_i = int(c[d,2].item())
        feq_d_xs = torch.roll(f_prev[opp_d], shifts=(-cz_i, -cy_i, -cx_i), dims=(0,1,2))[mask]
        f_i_wall = (1.0 - q_cell) * fp_d + q_cell * feq_d_xs
        c_d_x = float(c[d, 0].item())
        fx = fx + (2.0 * f_i_wall * c_d_x).sum()
    return float(fx.item() / dpS)

tag = f"[BFL-dyn SDAA:{device.index}]"
print(f"{tag} BFL dynamics + wall-ME + PF, 500 steps", flush=True)
t0 = time.time()

cd_wall = []; cd_pf = []; cd_std_me = []

for step in range(1, n_steps+1):
    f_pre = f.clone()
    f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)
    for q in range(19):
        f[q] = torch.where(sm[q], f_pre[q], f[q])
    f_post_coll = f.clone()
    
    # BFL before stream
    f_bfl = bouzidi_bounce_back_wallsurface(f_post_coll, f_post_coll, fbm, qf)
    
    # Wall-surface ME
    cd_w = me_wall_interp(f_post_coll, fbm, qf, dpS)
    # Standard ME (before stream, on BFL-modified dist)
    cd_s = drag_momentum_exchange(f_bfl, near, solid, dpS)
    # Pressure + friction
    cd_tot, _, _ = drag_total(f_bfl, mesh, dpS, nu)
    
    if step > warmup:
        if math.isfinite(cd_w): cd_wall.append(cd_w)
        if math.isfinite(cd_s): cd_std_me.append(cd_s)
        if math.isfinite(cd_tot): cd_pf.append(cd_tot)
    
    # Use BFL for dynamics
    f = stream3d(f_bfl)
    f = far_field_bc_3d(f, u_in=u_in)
    if step % 200 == 0: f = correct_mass3d(f, initial_mass)
    
    if not torch.isfinite(f).all():
        print(f"{tag} DIVERGED at step {step}", flush=True)
        break
    
    if step % 100 == 0:
        el = time.time() - t0
        print(f"{tag} step={step} WallME={sum(cd_wall)/max(len(cd_wall),1):.4f} "
              f"StdME={sum(cd_std_me)/max(len(cd_std_me),1):.4f} "
              f"PF={sum(cd_pf)/max(len(cd_pf),1):.4f} ({el:.0f}s)", flush=True)

el = time.time() - t0
print(f"\n{tag} === FINAL (ref=1.30) ===", flush=True)
print(f"{tag} Wall-ME  = {sum(cd_wall)/max(len(cd_wall),1):.4f}", flush=True)
print(f"{tag} Std-ME   = {sum(cd_std_me)/max(len(cd_std_me),1):.4f}", flush=True)
print(f"{tag} PF       = {sum(cd_pf)/max(len(cd_pf),1):.4f}", flush=True)
print(f"{tag} time={el:.0f}s", flush=True)
