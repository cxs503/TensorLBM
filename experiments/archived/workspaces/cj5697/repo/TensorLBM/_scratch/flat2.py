import sys,math,time; sys.path.insert(0,"src"); sys.path.insert(0,"_scratch")
import torch; torch.set_default_dtype(torch.float64)
import tensorlbm.multiphase as mp
from tensorlbm.d2q9 import equilibrium, stream, macroscopic
from sc_proto import sc_force_fixed
n=48
yy,xx=torch.meshgrid(torch.arange(n,dtype=torch.float64),torch.arange(n,dtype=torch.float64),indexing="ij")
def band(w,rhol,rhov,half=8.0):
    # liquid slab centred at y=n/2, thickness 2*half, periodic in y
    rho=rhov+(rhol-rhov)*0.5*(torch.tanh((yy-(n/2-half))/w)-torch.tanh((yy-(n/2+half))/w))
    return equilibrium(rho,torch.zeros_like(rho),torch.zeros_like(rho))
for tag,G,w,rhol,rhov in [
  ("G5 w2 maxwell",5.0,2.0,1.929,0.1529),
  ("G5 w3 maxwell",5.0,3.0,1.929,0.1529),
  ("G5 w6 maxwell",5.0,6.0,1.929,0.1529),
  ("G4.7 w3 refcp",4.7,3.0,1.6707,0.18473),
]:
    f=band(w,rhol,rhov); t0=time.time(); out=[]
    for s in range(1,3001):
        r=f.sum(0); fx,fy=sc_force_fixed(r,G)
        f=stream(mp.collide_guo(f,fx,fy,1.0))
        if not torch.isfinite(f).all(): out.append(f"DIV@{s}"); break
        if s%600==0:
            r,ux,uy=macroscopic(f); out.append(f"s{s}[{float(r.min()):.3f},{float(r.max()):.3f}]u{float(torch.sqrt(ux**2+uy**2).max()):.3f}")
    print(f"{tag:16s} {' '.join(out)} ({time.time()-t0:.0f}s)",flush=True)
