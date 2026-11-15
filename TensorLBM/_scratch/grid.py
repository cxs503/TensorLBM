import sys,math,time; sys.path.insert(0,"src"); sys.path.insert(0,"_scratch")
import torch; torch.set_default_dtype(torch.float64)
import tensorlbm.multiphase as mp
from tensorlbm.d2q9 import equilibrium, stream, macroscopic
from sc_proto import sc_force_fixed

def droplet(R,n,width,rhol,rhov):
    yy,xx=torch.meshgrid(torch.arange(n,dtype=torch.float64),torch.arange(n,dtype=torch.float64),indexing="ij")
    d=torch.sqrt((xx-n/2+0.5)**2+(yy-n/2+0.5)**2)-R
    rho=rhov+(rhol-rhov)*0.5*(1.0-torch.tanh(d/width))
    return equilibrium(rho,torch.zeros_like(rho),torch.zeros_like(rho))

n=48; R=12
cases=[
 ("a G5 w3 r0",      5.0,3.0,1.957,0.1596,0,   1500),
 ("b G5 w3 ramp500", 5.0,3.0,1.957,0.1596,500, 2500),
 ("c G5 w5 ramp500", 5.0,5.0,1.957,0.1596,500, 2500),
 ("d G4.7 w3 r0",    4.7,3.0,1.6707,0.18473,0, 1500),
 ("e G4.7 w3 ramp500",4.7,3.0,1.6707,0.18473,500,2500),
 ("f G6 w3 r0",      6.0,3.0,1.957,0.1596,0,   1500),
]
for tag,G,w,rhol,rhov,ramp,steps in cases:
    f=droplet(R,n,w,rhol,rhov); t0=time.time(); out=[]
    for s in range(1,steps+1):
        g=G*min(1.0,s/ramp) if ramp else G
        r=f.sum(0); fx,fy=sc_force_fixed(r,g)
        f=stream(mp.collide_guo(f,fx,fy,1.0))
        if not torch.isfinite(f).all(): out.append(f"DIV@{s}"); break
        if s%(steps//5)==0:
            r,ux,uy=macroscopic(f); out.append(f"s{s}[{float(r.min()):.3f},{float(r.max()):.3f}]u{float(torch.sqrt(ux**2+uy**2).max()):.3f}")
    print(f"{tag:18s} {' '.join(out)}  ({time.time()-t0:.0f}s)",flush=True)
