import sys; sys.path.insert(0,'/root/tp-agent/work/m9874/src')
import torch, math, time
torch.set_num_threads(8)
from tensorlbm.d2q9 import C, W, equilibrium, macroscopic
from tensorlbm.multiphase import collide_sc_single_component, psi_exp
from tensorlbm.solver import stream

def pressure_tensor(rho, G=5.0, solid=None, wall_psi=None):
    psi = psi_exp(rho)
    Pxx = torch.zeros_like(rho); Pyy = torch.zeros_like(rho)
    for i in range(9):
        cx, cy = int(C[i,0]), int(C[i,1])
        ps = torch.roll(psi, shifts=(-cy,-cx), dims=(0,1))
        if solid is not None:
            ps = torch.where(solid, torch.full_like(ps, wall_psi), ps)
        d = ps - psi
        Pxx += W[i]*d*cx*cx
        Pyy += W[i]*d*cy*cy
    return 0.5*G*psi*Pxx, 0.5*G*psi*Pyy

def run(f, n):
    for _ in range(n):
        f = collide_sc_single_component(f, 5.0, tau=1.0, psi_fn=psi_exp, forcing="edm")
        f = stream(f)
    return f

# flat interface: check P_yy constant across
ny, nx = 128, 4
yy = torch.arange(ny, dtype=torch.float32).view(ny,1).expand(ny,nx)
dist = torch.abs(yy - ny/2) - 24.0
rho = 0.1596 + (1.957-0.1596)*0.5*(1-torch.tanh(dist/1.5))
f = equilibrium(rho, torch.zeros_like(rho), torch.zeros_like(rho))
f = run(f, 4000)
rho = macroscopic(f)[0]
Ix, Iy = pressure_tensor(rho)
prof = Iy.mean(1)
print("P_yy int: left", float(prof[:20].mean()), " right", float(prof[-20:].mean()), " min/max", float(prof.min()), float(prof.max()))
print("gamma flat", float(((Ix-Iy).mean(1)).sum()))
# total pressure incl. EOS part? we only have interaction; also print rho_l/rho_v
print("rho profile ends", float(rho.mean(1)[:5].mean()), float(rho.mean(1)[-5:].mean()), float(rho.mean(1).max()))

# big drop Laplace
N=160; R=30.0
yy, xx = torch.meshgrid(torch.arange(N,dtype=torch.float32), torch.arange(N,dtype=torch.float32), indexing='ij')
d = torch.sqrt((xx-N/2)**2+(yy-N/2)**2) - R
rho = 0.1596 + (1.957-0.1596)*0.5*(1-torch.tanh(d/1.5))
f = equilibrium(rho, torch.zeros_like(rho), torch.zeros_like(rho))
t0=time.time(); f = run(f, 4001); print("drop run s", time.time()-t0)
rho = macroscopic(f)[0]
Ix, Iy = pressure_tensor(rho)
# isotropic-ish pressure: use Ptot = p0 + int ; p0 unknown but cancels in difference.
c = N//2
pin = float(Ix[c-5:c+5, c-5:c+5].mean()); pout = float(Ix[5:20,5:20].mean())
print("dP int part", pin-pout, "sigma=dP*R", (pin-pout)*R)
# also radial measure of rho=mid crossing
mid = 0.5*(1.957+0.1596)
r = torch.sqrt((xx-c)**2+(yy-c)**2)
line = rho[c, c:]
import numpy as np
l = line.numpy(); rr = np.arange(N-c)
idx = np.where(l < mid)[0][0]
print("R_half from rho=mid:", rr[idx])
