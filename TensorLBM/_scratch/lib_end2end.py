import sys,time; sys.path.insert(0,"src")
import torch; torch.set_default_dtype(torch.float64)
from tensorlbm.d2q9 import C, equilibrium, macroscopic
from tensorlbm.multiphase import step_sc
from tensorlbm.multiphase.laplace import laplace_pressure_calibration
N=64;R=15.0;RHOL=2.1;RHOV=0.15;G=4.7
yy,xx=torch.meshgrid(torch.arange(N,dtype=torch.float64),torch.arange(N,dtype=torch.float64),indexing="ij")
ins=((yy-N/2.)**2+(xx-N/2.)**2)<=R*R
rho=torch.where(ins,torch.tensor(RHOL,dtype=torch.float64),torch.tensor(RHOV,dtype=torch.float64))
f=equilibrium(rho,torch.zeros_like(rho),torch.zeros_like(rho))
t0=time.time()
for s in range(1,12001):
    f=step_sc(f,G=G,tau=1.0,forcing="guo")
    if not torch.isfinite(f).all(): print("DIV",s); break
    if s%2000==0:
        r=f.sum(0);_,ux,uy=macroscopic(f)
        try:
            c=laplace_pressure_calibration(r,G)
            print(f"s={s:6d} rg={c['rho_v']:.4f} rl={c['rho_l']:.4f} dp={c['dp']:+.5e} r={c['radius']:.3f} gamma={c['gamma']:+.5e} umax={float(torch.sqrt(ux**2+uy**2).max()):.4f} ({time.time()-t0:.0f}s)",flush=True)
        except Exception as e:
            print(f"s={s:6d} rho[{float(r.min()):.4f},{float(r.max()):.4f}] cal={type(e).__name__} ({time.time()-t0:.0f}s)",flush=True)
