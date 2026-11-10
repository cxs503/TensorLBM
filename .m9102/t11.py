import sys; sys.path.insert(0,'src')
import numpy as np, torch
torch.set_num_threads(8)
from tensorlbm.elbm import entropic_equilibrium_gauss_d3q27, collide_elbm_d3q27
from tensorlbm.elbm_channel import stream_channel_d3q27, _guo_force
from tensorlbm.elbm_channel_native import native_channel_step
from tensorlbm.d3q27 import macroscopic27
nx,ny,nz = 10,12,9
torch.manual_seed(3)
rho = torch.ones(nz,ny,nx); ux=0.05*torch.randn(nz,ny,nx); uy=0.02*torch.randn(nz,ny,nx); uz=0.02*torch.randn(nz,ny,nx)
f = entropic_equilibrium_gauss_d3q27(rho,ux,uy,uz)
f = torch.clamp(f*(1+0.08*torch.randn_like(f)), min=1e-6)
# keep walls quiescent-equilibrium (as the driver does)
quiet = entropic_equilibrium_gauss_d3q27(torch.ones(nz,1,nx),torch.zeros(nz,1,nx),torch.zeros(nz,1,nx),torch.zeros(nz,1,nx))
f[:,:,0]=quiet[:,:,0]; 
f[:,:,ny-1]=quiet[:,:,0]
tau=0.504; force=3e-6
alpha0=torch.ones(nz,ny,nx)
for ni in (10,24):
    r,u1,u2,u3 = macroscopic27(f)
    fp, info = collide_elbm_d3q27(f, tau, alpha_init=alpha0, n_iter=ni, return_info=True)
    fp = _guo_force(fp, r,u1,u2,u3, force, tau)
    fpy = stream_channel_d3q27(fp).numpy()
    fnp, anp, st = native_channel_step(f.numpy(), alpha0.numpy(), nx=nx,ny=ny,nz=nz,tau=tau,force=force,n_iter=ni)
    fl = slice(1,ny-1)
    d = np.abs(fnp-fpy)[:, :, fl]; da = np.abs(anp-info.alpha.numpy())[:, fl]
    print(f"n_iter={ni}: max|df| fluid={d.max():.3e} rel={d.max()/np.abs(fpy).max():.3e}  max|dalpha|={da.max():.3e}  native: {st}")
    print(f"   python alpha fluid [{float(info.alpha[:,fl].min()):.6f}, {float(info.alpha[:,fl].max()):.6f}]  native [{anp[:,fl].min():.6f}, {anp[:,fl].max():.6f}]")
