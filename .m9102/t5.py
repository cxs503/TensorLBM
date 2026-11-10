import sys; sys.path.insert(0,'src')
import numpy as np, torch
torch.set_num_threads(8)
from tensorlbm.elbm import entropic_equilibrium_gauss_d3q27, collide_elbm_d3q27, gauss_entropy
from tensorlbm.elbm_channel import stream_channel_d3q27, _guo_force
from tensorlbm.elbm_channel_native import native_channel_step
from tensorlbm.d3q27 import macroscopic27, W_EXACT64 as W27
nx,ny,nz = 12,10,8
torch.manual_seed(1)
rho = torch.ones(nz,ny,nx); ux = 0.05*torch.randn(nz,ny,nx); uy=0.01*torch.randn(nz,ny,nx); uz=0.01*torch.randn(nz,ny,nx)
f = entropic_equilibrium_gauss_d3q27(rho,ux,uy,uz)
f[:,:,0]=0.0; f[:,:,ny-1]=0.0
tau=0.504; force=1e-5; alpha0=torch.ones(nz,ny,nx)
for ni in (0,5):
    r,u1,u2,u3 = macroscopic27(f)
    fp, info = collide_elbm_d3q27(f, tau, alpha_init=alpha0, n_iter=ni, return_info=True)
    fp = _guo_force(fp, r,u1,u2,u3, force, tau)
    fpy = stream_channel_d3q27(fp)
    fnp, anp, stats = native_channel_step(f.numpy(), alpha0.numpy(), nx=nx,ny=ny,nz=nz,tau=tau,force=force,n_iter=ni)
    fluid = slice(1,ny-1)
    d = np.abs(fnp - fpy.numpy())[:, :, fluid]
    da = np.abs(anp - info.alpha.numpy())[:, fluid]
    print(f"n_iter={ni}: max|df| fluid={d.max():.3e}  max|dalpha| fluid={da.max():.3e} c stats {stats}")
    # collision-only comparison (before force/stream)
    import ctypes
    print("  python alpha fluid range", float(info.alpha[:, fluid].min()), float(info.alpha[:, fluid].max()),
          "| native", float(anp[:, fluid].min()), float(anp[:, fluid].max()))
