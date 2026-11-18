import sys; sys.path.insert(0,'src')
import numpy as np, torch
torch.set_num_threads(4)
from tensorlbm.elbm import entropic_equilibrium_gauss_d3q27, gauss_entropy
from tensorlbm.d3q27 import W_EXACT64 as W27, macroscopic27
nx,ny,nz = 10,12,9
torch.manual_seed(3)
rho = torch.ones(nz,ny,nx); ux=0.05*torch.randn(nz,ny,nx); uy=0.02*torch.randn(nz,ny,nx); uz=0.02*torch.randn(nz,ny,nx)
f = entropic_equilibrium_gauss_d3q27(rho,ux,uy,uz)
f = torch.clamp(f*(1+0.08*torch.randn_like(f)), min=1e-6)
zi,yi,xi=4,6,3
fc = f[:, zi, yi, xi].view(27,1,1,1)
r,u1,u2,u3 = macroscopic27(fc); feq = entropic_equilibrium_gauss_d3q27(r,u1,u2,u3)
w = W27.view(27,1,1,1); tau=0.504
H0 = float(gauss_entropy(fc, w)); print("H0", H0)
for a in (0.0,0.2,0.4,0.53,0.8,1.0,1.00997,1.1,1.1895,1.5,2.0):
    fs = fc + (a/tau)*(feq-fc)
    print(f"alpha={a:8.5f} H={float(gauss_entropy(fs,w)):.12f}  H-H0={float(gauss_entropy(fs,w))-H0:.3e}  minf={float(fs.min()):.3e}")
