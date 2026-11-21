import sys; sys.path.insert(0,'src')
import numpy as np, torch
torch.set_num_threads(4)
from tensorlbm.elbm import entropic_equilibrium_gauss_d3q27, collide_elbm_d3q27
from tensorlbm.elbm_channel import stream_channel_d3q27
from tensorlbm.elbm_channel_native import native_channel_step
nx,ny,nz = 6,6,5
ux = 0.1*torch.ones(nz,ny,nx)
f = entropic_equilibrium_gauss_d3q27(torch.ones(nz,ny,nx), ux, torch.zeros_like(ux), torch.zeros_like(ux))
f[:,:,0]=0; f[:,:,ny-1]=0
fp, info = collide_elbm_d3q27(f, 0.504, alpha_init=torch.ones(nz,ny,nx), n_iter=0, return_info=True)
fpy = stream_channel_d3q27(fp).numpy()
fnp, anp, st = native_channel_step(f.numpy(), torch.ones(nz,ny,nx).numpy(), nx=nx,ny=ny,nz=nz,tau=0.504,force=0.0,n_iter=0)
d = np.abs(fnp-fpy)
print("diff per (q summed) over z,y,x:")
print(np.round(d.sum(axis=(1,2,3)),4))
print("per y (sum over q,z,x):", np.round(d.sum(axis=(0,1,3)),4))
print("per z:", np.round(d.sum(axis=(0,2,3)),4))
print("post-collision py q=1 row y=2 z=2:", fp[1,2,2,:].numpy())
print("py streamed q=1 row y=2 z=2:", fpy[1,2,2,:])
print("nat        q=1 row y=2 z=2:", fnp[1,2,2,:])
print("nat alpha map z=2:", anp[2])
print("py  alpha map z=2:", info.alpha[2].numpy())
