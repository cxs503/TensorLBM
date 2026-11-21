import sys; sys.path.insert(0,"src"); sys.path.insert(0,"_scratch")
import torch; torch.set_default_dtype(torch.float64)
import tensorlbm.multiphase as mp
from tensorlbm.d2q9 import macroscopic, stream
from sc_proto import droplet, sc_force_fixed

n=64
f = droplet(16,n,3.0); rho=f.sum(0)
fx,fy = sc_force_fixed(rho,5.0)
print("sum Fx, Fy =", float(fx.sum()), float(fy.sum()))
yy,xx=torch.meshgrid(torch.arange(n,dtype=torch.float64),torch.arange(n,dtype=torch.float64),indexing="ij")
dx=xx-n/2+0.5; dy=yy-n/2+0.5; r=torch.sqrt(dx*dx+dy*dy)
# radial force component on the interface band
band=(r>14)&(r<18)
Fr = (fx*dx+fy*dy)/(r+1e-12)
print("radial F on interface band: mean=%.4e  (negative=inward=attraction)"%float(Fr[band].mean()))
print("F magnitude in band:", float(torch.sqrt(fx**2+fy**2)[band].mean()))
# mean momentum tracking, library vs fixed force, G=5 and G=0.2
for G in (0.2, 5.0):
  for force,name in ((sc_force_fixed,"fixed"),(mp.sc_force,"lib")):
    f = droplet(16,n,3.0); 
    for s in range(1,201):
        rho=f.sum(0); a,b=force(rho,G)
        f=stream(mp.collide_guo(f,a,b,1.0))
        if not torch.isfinite(f).all(): print(name,G,"DIV",s); break
    else:
        rho=f.sum(0); _,ux,uy=macroscopic(f)
        print(f"{name} G={G}: rho[{float(rho.min()):.4f},{float(rho.max()):.4f}] umax={float(torch.sqrt(ux**2+uy**2).max()):.4f} umeanx={float((ux*rho).sum()/rho.sum()):.3e}")
