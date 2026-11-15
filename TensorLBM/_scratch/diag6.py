import sys; sys.path.insert(0,"src"); sys.path.insert(0,"_scratch")
import torch; torch.set_default_dtype(torch.float64)
import tensorlbm.multiphase as mp
from tensorlbm.d2q9 import equilibrium, stream, macroscopic
from sc_proto import sc_force_fixed
n=48
yy,xx=torch.meshgrid(torch.arange(n,dtype=torch.float64),torch.arange(n,dtype=torch.float64),indexing="ij")
rho=(0.1529+1.7761*0.5*(1.0+torch.tanh((yy-n/2)/6.0)))
f=equilibrium(rho,torch.zeros_like(rho),torch.zeros_like(rho))
for s in range(1,12):
    r=f.sum(0); fx,fy=sc_force_fixed(r,5.0)
    r2,ux,uy=macroscopic(f)
    print(f"s={s} rho[{float(r.min()):.4f},{float(r.max()):.4f}] Fmax=({float(fx.abs().max()):.4f},{float(fy.abs().max()):.4f}) umax={float(torch.sqrt(ux**2+uy**2).max()):.5f} f[{float(f.min()):.3e},{float(f.max()):.3e}] sum={float(f.sum()):.6f}")
    f=stream(mp.collide_guo(f,fx,fy,1.0))
    if not torch.isfinite(f).all(): print("nan"); break
# check: force on a pure y-varying field vs analytic (G/6) d(psi^2)/dy
psi=mp.psi_exp(rho)
dpsi=(torch.roll(psi,-1,0)-torch.roll(psi,1,0))/2
fx,fy=sc_force_fixed(rho,5.0)
print("sy vs psi'/3: max rel diff", float(((fy/(5*psi)-dpsi/3).abs()/(dpsi.abs()/3+1e-12))[10:38].max()))
