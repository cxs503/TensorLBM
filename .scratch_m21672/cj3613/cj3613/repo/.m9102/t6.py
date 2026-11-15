import sys; sys.path.insert(0,'src')
import numpy as np, torch
torch.set_num_threads(8)
from tensorlbm.elbm import entropic_equilibrium_gauss_d3q27, collide_elbm_d3q27
from tensorlbm.elbm_channel import stream_channel_d3q27, _guo_force
from tensorlbm.elbm_channel_native import native_channel_step
from tensorlbm.d3q27 import macroscopic27
nx,ny,nz = 6,6,5
# (a) uniform quiescent field, zero force
f = entropic_equilibrium_gauss_d3q27(torch.ones(nz,ny,nx), torch.zeros(nz,ny,nx),torch.zeros(nz,ny,nx),torch.zeros(nz,ny,nx))
r,u1,u2,u3 = macroscopic27(f)
fp, info = collide_elbm_d3q27(f, 0.504, alpha_init=torch.ones(nz,ny,nx), n_iter=0, return_info=True)
fnp, anp, st = native_channel_step(f.numpy(), torch.ones(nz,ny,nx).numpy(), nx=nx,ny=ny,nz=nz,tau=0.504,force=0.0,n_iter=0)
print("(a) uniform: max|df|", np.abs(fnp-f.numpy()).max(), "alpha py", float(info.alpha.max()), "nat", anp.max())
# (b) uniform flow +x
ux = 0.1*torch.ones(nz,ny,nx)
f = entropic_equilibrium_gauss_d3q27(torch.ones(nz,ny,nx), ux, torch.zeros_like(ux), torch.zeros_like(ux))
f[:,:,0]=0; f[:,:,ny-1]=0
fp, info = collide_elbm_d3q27(f, 0.504, alpha_init=torch.ones(nz,ny,nx), n_iter=0, return_info=True)
fpy = stream_channel_d3q27(fp)
fnp, anp, st = native_channel_step(f.numpy(), torch.ones(nz,ny,nx).numpy(), nx=nx,ny=ny,nz=nz,tau=0.504,force=0.0,n_iter=0)
d=np.abs(fnp-fpy.numpy())
idx = np.unravel_index(np.argmax(d), d.shape)
print("(b) uniform flow: max|df|", d.max(), "at", idx, "alpha diff", np.abs(anp-info.alpha.numpy()).max())
print("   py f at idx", fpy.numpy()[idx[0],idx[1],idx[2],idx[3]], "nat", fnp[idx[0],idx[1],idx[2],idx[3]])
print("   q=1 (c=+x) native vs py rows y=1:", fnp[1,0,2,3], fpy.numpy()[1,0,2,3], " y=2:", fnp[1,1,2,3], fpy.numpy()[1,1,2,3])
