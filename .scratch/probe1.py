import sys; sys.path.insert(0,'/root/tp-agent/work/m9874/src')
import torch, math, time
torch.set_num_threads(8)
from tensorlbm.d2q9 import C, W, equilibrium, macroscopic
from tensorlbm.multiphase import collide_sc_single_component, psi_exp
from tensorlbm.solver import stream
from tensorlbm.contact_line.engine import SIGMA_DEFAULT

def pressure_tensor(rho, G=5.0, solid=None, wall_psi=None):
    psi = psi_exp(rho)
    ny, nx = rho.shape
    Pxx = torch.zeros_like(rho); Pyy = torch.zeros_like(rho)
    for i in range(9):
        cx, cy = int(C[i,0]), int(C[i,1])
        ps = torch.roll(psi, shifts=(-cy,-cx), dims=(0,1))  # psi(x+c_i)
        if solid is not None:
            ps = torch.where(solid, torch.full_like(ps, wall_psi), ps)
        d = ps - psi
        Pxx += W[i]*d*cx*cx
        Pyy += W[i]*d*cy*cy
    return 0.5*G*psi*Pxx, 0.5*G*psi*Pyy

# 1D planar liquid slab, periodic in y
ny, nx = 128, 4
yy = torch.arange(ny, dtype=torch.float32).view(ny,1).expand(ny,nx)
dist = torch.abs(yy - ny/2) - 24.0
rho = 0.1596 + (1.957-0.1596)*0.5*(1-torch.tanh(dist/1.5))
f = equilibrium(rho, torch.zeros_like(rho), torch.zeros_like(rho))
t0=time.time()
for n in range(4001):
    f = collide_sc_single_component(f, 5.0, tau=1.0, psi_fn=psi_exp, forcing="edm")
    f = stream(f)
    if n % 1000 == 0:
        rho = macroscopic(f)[0]
        Pt_x, Pt_y = pressure_tensor(rho)
        g = float(((Pt_x - Pt_y).mean(1)).sum())
        print(n, "rho_mid", float(rho.mean()), "gamma_pt", g)
rho = macroscopic(f)[0]
Pt_x, Pt_y = pressure_tensor(rho)
print("final gamma_pt", float(((Pt_x-Pt_y).mean(1)).sum()), "sigma_lap", SIGMA_DEFAULT, "t", time.time()-t0)
