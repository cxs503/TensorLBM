import sys; sys.path.insert(0,'src')
import numpy as np, torch
torch.set_num_threads(4)
from tensorlbm.elbm_channel import stream_channel_d3q27
from tensorlbm.elbm_channel_native import native_channel_step
nx,ny,nz = 6,7,5
f = torch.zeros(27,nz,ny,nx)
f[:,3,5,2] = 0.001   # (z=3,y=5,x=2)
# no collision: tau huge, rho tiny -> equilibrium tiny but nonzero; set alpha so rate~0
fp, info = None, None
fnp, anp, st = native_channel_step(f.numpy(), torch.ones(nz,ny,nx).numpy(), nx=nx,ny=ny,nz=nz,tau=1e9,force=0.0,n_iter=0)
fpy = stream_channel_d3q27(f).numpy()
print("native nonzero nodes (q,z,y,x):", np.argwhere(np.abs(fnp)>1e-9)[:12].tolist())
print("python nonzero nodes (q,z,y,x):", np.argwhere(np.abs(fpy)>1e-9)[:12].tolist())
print("max diff", np.abs(fnp-fpy).max())
