import sys; sys.path.insert(0,'src')
import numpy as np, torch
torch.set_num_threads(8)
from tensorlbm.elbm import entropic_equilibrium_gauss_d3q27, gauss_entropy, solve_entropy_condition
from tensorlbm.d3q27 import W_EXACT64 as W27
from tensorlbm.elbm_channel_native import native_channel_step
nx,ny,nz = 10,12,9
torch.manual_seed(3)
rho = torch.ones(nz,ny,nx); ux=0.05*torch.randn(nz,ny,nx); uy=0.02*torch.randn(nz,ny,nx); uz=0.02*torch.randn(nz,ny,nx)
f = entropic_equilibrium_gauss_d3q27(rho,ux,uy,uz)
f = torch.clamp(f*(1+0.08*torch.randn_like(f)), min=1e-6)
quiet = entropic_equilibrium_gauss_d3q27(torch.ones(nz,1,nx),torch.zeros(nz,1,nx),torch.zeros(nz,1,nx),torch.zeros(nz,1,nx))
f[:,:,0]=quiet[:,:,0]; f[:,:,ny-1]=quiet[:,:,0]
tau=0.504
from tensorlbm.d3q27 import macroscopic27
r,u1,u2,u3 = macroscopic27(f)
feq = entropic_equilibrium_gauss_d3q27(r,u1,u2,u3)
w = W27.view(27,1,1,1)
H0 = gauss_entropy(f, w)
infos = {}
for ni in (2,5,20):
    infos[ni] = solve_entropy_condition(f, feq, tau, w=w, n_iter=ni)
# native alpha via a full step (n_iter=5) -- reuse its alpha array output
fnp, anp, st = native_channel_step(f.numpy(), torch.ones(nz,ny,nx).numpy(), nx=nx,ny=ny,nz=nz,tau=tau,force=0.0,n_iter=5)
alphas = {f"py{ni}": infos[ni].alpha for ni in infos}
alphas["native"] = torch.from_numpy(anp)
for name, a in alphas.items():
    fstar = f + (a/tau)*(feq-f)
    H1 = gauss_entropy(fstar, w)
    d = (H1-H0)
    fl = (slice(None), slice(1,ny-1))
    print(f"{name:8s} alpha[{float(a[fl].min()):.4f},{float(a[fl].max()):.4f}]  dH max={float(d[fl].max()):.3e}  |dH| max={float(d[fl].abs().max()):.3e}")
