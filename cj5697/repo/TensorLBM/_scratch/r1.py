import sys,time; sys.path.insert(0,"src"); sys.path.insert(0,"_scratch")
import torch; torch.set_default_dtype(torch.float64)
import tensorlbm.multiphase as mp
from tensorlbm.d2q9 import macroscopic, stream
from sc_proto import droplet, sc_force_fixed
import os
G=float(os.environ.get("G","5")); R=float(os.environ.get("R","16")); n=int(os.environ.get("N","64"))
w=float(os.environ.get("W","6")); tau=float(os.environ.get("TAU","1.0")); steps=int(os.environ.get("STEPS","8000"))
ramp=int(os.environ.get("RAMP","4000")); forcing=os.environ.get("FORCING","guo")
f=droplet(R,n,w); t0=time.time()
for s in range(1,steps+1):
    g=G*min(1.0,s/ramp)
    rho=f.sum(0); fx,fy=sc_force_fixed(rho,g)
    fpost = mp.collide_guo(f,fx,fy,tau) if forcing=="guo" else (mp.collide_edm(f,fx,fy,tau) if forcing=="edm" else mp.collide_velocity_shift(f,fx,fy,tau))
    f=stream(fpost)
    if not torch.isfinite(f).all():
        print(f"DIV at {s}",flush=True); break
    if s%500==0:
        rho,ux,uy=macroscopic(f); u=torch.sqrt(ux**2+uy**2)
        print(f"s={s:6d} g={g:.2f} rho[{float(rho.min()):.4f},{float(rho.max()):.4f}] umax={float(u.max()):.4f} t={time.time()-t0:.0f}s",flush=True)
