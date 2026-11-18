import sys, time; sys.path.insert(0,'src')
import numpy as np, torch
from tensorlbm.elbm_channel_native import build_kernel, native_channel_step, native_channel_run
p = build_kernel(verbose=True); print("lib", p)
# 1) native vs python reference: one step from identical state
from tensorlbm.elbm import entropic_equilibrium_gauss_d3q27
from tensorlbm.elbm_channel import stream_channel_d3q27, _guo_force
nx,ny,nz = 12,10,8
torch.set_num_threads(8)
torch.manual_seed(1)
rho = torch.ones(nz,ny,nx); ux = 0.05*torch.randn(nz,ny,nx); uy=0.01*torch.randn(nz,ny,nx); uz=0.01*torch.randn(nz,ny,nx)
f = entropic_equilibrium_gauss_d3q27(rho,ux,uy,uz)
f[:,:,0]=0.0; f[:,:,ny-1]=0.0
tau=0.504; force=1e-5
alpha0 = torch.ones(nz,ny,nx)
# python reference step (mirrors run_elbm_channel loop body)
from tensorlbm.d3q27 import macroscopic27
from tensorlbm.elbm import collide_elbm_d3q27
r,u1,u2,u3 = macroscopic27(f)
fp, info = collide_elbm_d3q27(f, tau, alpha_init=alpha0, n_iter=5, return_info=True)
fp = _guo_force(fp, r, u1,u2,u3, force, tau)
fpy = stream_channel_d3q27(fp)
fnp, anp, stats = native_channel_step(f.numpy(), alpha0.numpy(), nx=nx,ny=ny,nz=nz,tau=tau,force=force,n_iter=5)
d = np.abs(fnp - fpy.numpy())
print("max |f_native - f_python| =", d.max(), " rel", d.max()/np.abs(fpy.numpy()).max())
print("alpha diff", np.abs(anp - info.alpha.numpy()).max(), "stats", stats)
from tensorlbm.elbm import gauss_entropy
from tensorlbm.d3q27 import W_EXACT64 as W27
Hf = gauss_entropy(f, W27.view(27,1,1,1)); Hp = gauss_entropy(fpy, W27.view(27,1,1,1))
print("H increase max (python)", float((Hp-Hf).max()))
